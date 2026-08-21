"""Cliente genérico para la API de OpenRouter.

Soporta mensajes multimodales (texto + imágenes en base64) y reintentos con
backoff exponencial. La API de OpenRouter es compatible con el formato de
chat completions de OpenAI.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

try:
    # Carga automática de variables desde un archivo .env si existe.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv es opcional
    pass

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 180.0
MAX_RETRIES = 4

# Códigos que sí vale la pena reintentar (saturación o fallo pasajero del
# proveedor). Cualquier otro 4xx es un problema del payload.
CODIGOS_REINTENTABLES = frozenset({429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """Error irrecuperable tras agotar reintentos."""


class ErrorAPIPermanente(OpenRouterError):
    """La API rechazó la request de forma determinista (4xx, o un 200 cuyo
    cuerpo trae "error" en vez de "choices").

    No se reintenta: el mismo payload va a fallar igual las veces que haga
    falta. Antes esto se veía como un KeyError('choices') que agotaba los 4
    intentos y perdía por el camino el mensaje del proveedor, que es justo el
    que dice qué hay que arreglar.
    """


class _ErrorTransitorio(Exception):
    """Error recuperable señalado en el cuerpo de una respuesta 200."""


class RespuestaVaciaError(OpenRouterError):
    """El modelo terminó sin error pero `message.content` vino vacío.

    Pasa con los modelos de razonamiento cuando todo el output se queda en el
    campo `reasoning` y no llega a emitirse la respuesta. No se reintenta: la
    generación ya consumió tokens (y minutos) y volver a pedir lo mismo suele
    dar igual; hay que subir max_tokens o cambiar de modelo.
    """


class RespuestaTruncadaError(OpenRouterError):
    """El modelo agotó max_tokens y devolvió una respuesta incompleta.

    Se distingue del JSON inválido porque no tiene sentido reintentar con los
    mismos parámetros: hay que subir max_tokens (ojo: en los modelos de
    razonamiento los tokens de thinking también consumen ese presupuesto).
    """


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError(
                "Falta OPENROUTER_API_KEY. Defínela en el archivo .env o "
                "como variable de entorno."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    # -- construcción de contenido multimodal -----------------------------

    @staticmethod
    def text_part(texto: str) -> dict[str, Any]:
        return {"type": "text", "text": texto}

    @staticmethod
    def image_part(imagen_base64: str, media_type: str = "image/png") -> dict[str, Any]:
        """Empaqueta una imagen base64 como data URL para la API."""
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{imagen_base64}"},
        }

    # -- llamada principal -------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        response_format_json: bool = True,
        max_tokens: Optional[int] = None,
        dump_base: Optional[Path] = None,
        image_refs: Optional[dict[str, str]] = None,
    ) -> str:
        """Ejecuta una completion y devuelve el contenido de texto de la
        respuesta. Reintenta ante errores transitorios (429 / 5xx / red).

        Si `dump_base` se pasa, guarda para observabilidad:
          - `{dump_base}.request.json`  — el payload enviado (imágenes
            reemplazadas por su ruta en disco según `image_refs`).
          - `{dump_base}.response.json` — la respuesta cruda del modelo.
        """

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if dump_base is not None:
            _escribir_json(
                Path(f"{dump_base}.request.json"),
                {
                    "model": model,
                    "temperature": temperature,
                    "messages": _sanitizar_messages(messages, image_refs),
                },
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Cabeceras opcionales recomendadas por OpenRouter.
            "HTTP-Referer": "https://localhost/anatomy-question-generator",
            "X-Title": "anatomy-question-generator",
        }

        ultimo_error: Optional[Exception] = None
        for intento in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(OPENROUTER_URL, headers=headers, json=payload)
                if resp.status_code in CODIGOS_REINTENTABLES:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                if resp.status_code >= 400:
                    codigo, mensaje = _detalle_error_http(resp)
                    self._fallo_permanente(model, codigo, mensaje, dump_base)
                data = resp.json()
                # OpenRouter devuelve los errores del proveedor con HTTP 200 y
                # un cuerpo {"error": {...}}, sin "choices".
                error = _error_de_cuerpo(data)
                if error is not None:
                    codigo, mensaje = error
                    if codigo in CODIGOS_REINTENTABLES:
                        raise _ErrorTransitorio(f"HTTP {codigo}: {mensaje}")
                    self._fallo_permanente(model, codigo, mensaje, dump_base)
                choice = data["choices"][0]
                mensaje = choice.get("message") or {}
                content = _texto_de_content(mensaje.get("content"))
                razonamiento = mensaje.get("reasoning")
                finish_reason = choice.get("finish_reason")
                uso = data.get("usage") or {}
                if dump_base is not None:
                    # finish_reason y usage se guardan siempre: sin ellos, una
                    # respuesta truncada solo se ve como un JSON corrupto.
                    registro: dict[str, Any] = {
                        "model": model,
                        "finish_reason": finish_reason,
                        "usage": uso,
                        "respuesta": _quizas_json(content),
                    }
                    if not content.strip() and razonamiento:
                        # Sin contenido, lo único que explica qué hizo el
                        # modelo durante la llamada es el razonamiento.
                        registro["razonamiento"] = razonamiento
                    _escribir_json(Path(f"{dump_base}.response.json"), registro)
                if finish_reason == "length":
                    razonamiento = (uso.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                    raise RespuestaTruncadaError(
                        f"El modelo {model} agotó max_tokens"
                        + (f"={max_tokens}" if max_tokens is not None else "")
                        + " y devolvió una respuesta incompleta"
                        + (
                            f" ({uso.get('completion_tokens')} tokens de salida, "
                            f"{razonamiento} de razonamiento)"
                            if razonamiento is not None
                            else ""
                        )
                        + ". Sube el max_tokens de la etapa en config.py."
                    )
                if not content.strip():
                    if razonamiento:
                        raise RespuestaVaciaError(
                            f"El modelo {model} terminó "
                            f"(finish_reason={finish_reason!r}) sin contenido: "
                            "todo el output se quedó en el campo 'reasoning' "
                            f"({uso.get('completion_tokens')} tokens de salida). "
                            "Sube el max_tokens de la etapa en config.py o usa "
                            "otro modelo."
                        )
                    raise _ErrorTransitorio(
                        f"El modelo {model} devolvió contenido vacío "
                        f"(finish_reason={finish_reason!r})"
                    )
                return content
            except (
                httpx.HTTPStatusError,
                httpx.TransportError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
                _ErrorTransitorio,
            ) as exc:
                ultimo_error = exc
                if intento < self.max_retries:
                    espera = min(2 ** intento, 30)
                    logger.warning(
                        "OpenRouter intento %d/%d falló (%s). Reintentando en %ds...",
                        intento,
                        self.max_retries,
                        exc,
                        espera,
                    )
                    time.sleep(espera)
                else:
                    logger.warning(
                        "OpenRouter intento %d/%d falló (%s). Sin más reintentos.",
                        intento,
                        self.max_retries,
                        exc,
                    )

        raise OpenRouterError(
            f"Falló la llamada a OpenRouter (modelo={model}) tras "
            f"{self.max_retries} intentos: {ultimo_error}"
        )

    @staticmethod
    def _fallo_permanente(
        model: str,
        codigo: Any,
        mensaje: str,
        dump_base: Optional[Path],
    ) -> None:
        """Registra el rechazo y aborta sin reintentar."""
        if dump_base is not None:
            _escribir_json(
                Path(f"{dump_base}.error.json"),
                {"model": model, "code": codigo, "error": mensaje},
            )
        raise ErrorAPIPermanente(
            f"OpenRouter rechazó la request (modelo={model}, code={codigo}): "
            f"{mensaje}"
        )

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        dump_base: Optional[Path] = None,
        image_refs: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Como chat() pero parsea la respuesta como JSON. Tolera bloques
        ```json ... ``` que algunos modelos añaden."""
        raw = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format_json=True,
            max_tokens=max_tokens,
            dump_base=dump_base,
            image_refs=image_refs,
        )
        return _parse_json_laxo(raw)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _escribir_json(path: Path, obj: Any) -> None:
    """Escribe `obj` como JSON legible, creando la carpeta si hace falta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _detalle_error_http(resp: httpx.Response) -> tuple[Any, str]:
    """(código, mensaje) de una respuesta 4xx, prefiriendo el cuerpo JSON."""
    try:
        error = _error_de_cuerpo(resp.json())
    except (json.JSONDecodeError, ValueError):
        error = None
    if error is not None:
        codigo, mensaje = error
        return codigo or resp.status_code, mensaje
    return resp.status_code, resp.text[:2000]


def _error_de_cuerpo(data: Any) -> Optional[tuple[Any, str]]:
    """Devuelve (código, mensaje) si el cuerpo trae un error, o None.

    El detalle útil suele venir en `error.metadata.raw` (el mensaje crudo del
    proveedor), no en `error.message`, así que se concatenan los dos.
    """
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not error:
        return None
    if not isinstance(error, dict):
        return None, str(error)

    mensaje = str(error.get("message") or json.dumps(error, ensure_ascii=False))
    metadata = error.get("metadata")
    if isinstance(metadata, dict) and metadata.get("raw"):
        mensaje = f"{mensaje} | {metadata['raw']}"
    return error.get("code"), mensaje


def _texto_de_content(content: Any) -> str:
    """Normaliza `message.content` a texto.

    Algunos proveedores devuelven None (cuando no llegaron a emitir respuesta)
    y otros una lista de partes {"type": "text", ...} en vez de un string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            parte.get("text", "")
            for parte in content
            if isinstance(parte, dict)
        )
    return str(content)


def _quizas_json(raw: str) -> Any:
    """Devuelve el contenido parseado como JSON si se puede; si no, el texto
    crudo. Sirve para que la respuesta guardada sea legible."""
    try:
        return _parse_json_laxo(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _sanitizar_messages(
    messages: list[dict[str, Any]], image_refs: Optional[dict[str, str]]
) -> list[dict[str, Any]]:
    """Copia los mensajes reemplazando cada imagen (data URL base64) por una
    referencia al PNG en disco, para que el JSON guardado sea legible."""
    refs = image_refs or {}
    out = copy.deepcopy(messages)
    for msg in out:
        contenido = msg.get("content")
        if not isinstance(contenido, list):
            continue
        for parte in contenido:
            if isinstance(parte, dict) and parte.get("type") == "image_url":
                url = parte.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    ref = refs.get(url)
                    parte["image_url"] = (
                        {"_archivo": ref} if ref else {"_imagen_omitida": True}
                    )
    return out


def _parse_json_laxo(raw: str) -> dict[str, Any]:
    """Parsea JSON tolerando fences de markdown y texto circundante."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("La respuesta del modelo no trae texto que parsear")
    texto = raw.strip()
    if texto.startswith("```"):
        # Quita ```json ... ``` o ``` ... ```
        texto = texto.split("```", 2)
        texto = texto[1] if len(texto) >= 2 else raw
        if texto.lstrip().lower().startswith("json"):
            texto = texto.lstrip()[4:]
        texto = texto.strip("` \n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        # Último recurso: extraer el primer objeto {...} balanceado.
        inicio = texto.find("{")
        fin = texto.rfind("}")
        if inicio != -1 and fin != -1 and fin > inicio:
            return json.loads(texto[inicio : fin + 1])
        raise
