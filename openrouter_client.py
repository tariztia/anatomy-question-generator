"""Cliente genérico para la API de OpenRouter.

Soporta mensajes multimodales (texto + imágenes en base64) y reintentos con
backoff exponencial. La API de OpenRouter es compatible con el formato de
chat completions de OpenAI.
"""

from __future__ import annotations

import json
import logging
import os
import time
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
    ) -> str:
        """Ejecuta una completion y devuelve el contenido de texto de la
        respuesta. Reintenta ante errores transitorios (429 / 5xx / red)."""

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

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
                return data["choices"][0]["message"]["content"]
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
    ) -> dict[str, Any]:
        """Como chat() pero parsea la respuesta como JSON. Tolera bloques
        ```json ... ``` que algunos modelos añaden."""
        raw = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format_json=True,
            max_tokens=max_tokens,
        )
        return _parse_json_laxo(raw)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


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
