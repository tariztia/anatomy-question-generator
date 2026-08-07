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


class OpenRouterError(RuntimeError):
    """Error irrecuperable tras agotar reintentos."""


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
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")
                uso = data.get("usage") or {}
                if dump_base is not None:
                    # finish_reason y usage se guardan siempre: sin ellos, una
                    # respuesta truncada solo se ve como un JSON corrupto.
                    _escribir_json(
                        Path(f"{dump_base}.response.json"),
                        {
                            "model": model,
                            "finish_reason": finish_reason,
                            "usage": uso,
                            "respuesta": _quizas_json(content),
                        },
                    )
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
                return content
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError, json.JSONDecodeError) as exc:
                ultimo_error = exc
                espera = min(2 ** intento, 30)
                logger.warning(
                    "OpenRouter intento %d/%d falló (%s). Reintentando en %ds...",
                    intento,
                    self.max_retries,
                    exc,
                    espera,
                )
                if intento < self.max_retries:
                    time.sleep(espera)

        raise OpenRouterError(
            f"Falló la llamada a OpenRouter (modelo={model}) tras "
            f"{self.max_retries} intentos: {ultimo_error}"
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
