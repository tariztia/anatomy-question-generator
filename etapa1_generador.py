"""Etapa 1 — Generación de preguntas (LLM: Claude Opus 4.7 vía OpenRouter).

También implementa la corrección de preguntas de la Etapa 2.5 (mismo agente,
con el feedback del evaluador).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from openrouter_client import OpenRouterClient
from schemas import PaquetePaper, PreguntaGenerada, SalidaGenerador

logger = logging.getLogger(__name__)

MODELO_GENERADOR = "anthropic/claude-opus-4-7"
TEMPERATURA = 0.7
MAX_CHARS_TEXTO = 60_000  # recorte defensivo del texto del paper

SYSTEM_PROMPT = """\
Eres un experto en anatomía humana y en diseño de evaluaciones. Generas \
preguntas de estudio a partir de papers académicos de anatomía.

REGLAS ESTRICTAS:
1. Genera EXACTAMENTE 15 preguntas.
2. Todas son preguntas ABIERTAS BREVES: la respuesta es un término, estructura, \
condición o concepto puntual de 1 a 5 palabras. NO generes preguntas de alternativas, \
NO de verdadero/falso, NO de desarrollo.
3. AL MENOS 5 preguntas deben referirse a figuras del paper (tipo "imagen").
4. Cada pregunta debe incluir la EVIDENCIA que respalda la respuesta: una cita \
textual del paper o una referencia explícita a una figura.
5. Si "tipo" es "imagen", "figura_id" DEBE ser el id de una figura real del \
input (por ejemplo "fig_1"). Si "tipo" es "texto", "figura_id" debe ser null.
6. La respuesta debe ser correcta, unívoca y verificable con la evidencia.
7. Las preguntas deben ser autocontenidas (entendibles sin ver el paper).

Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional, con este \
formato:
{
  "paper_id": "<paper_id>",
  "preguntas": [
    {
      "pregunta_id": "q_01",
      "pregunta": "string",
      "respuesta": "string (1-5 palabras)",
      "tipo": "texto" | "imagen",
      "figura_id": "fig_1" | null,
      "evidencia": "cita textual o referencia a figura",
      "tema": "string (ej: 'osteología - miembro inferior')",
      "dificultad_estimada": "baja" | "media" | "alta"
    }
  ]
}
Los pregunta_id van de "q_01" a "q_15"."""

SYSTEM_PROMPT_CORRECCION = """\
Eres un experto en anatomía humana. Recibes preguntas que fueron rechazadas o \
marcadas como corregibles por un evaluador, junto con el feedback específico. \
Corrige cada pregunta respetando el feedback.

REGLAS ESTRICTAS (idénticas a la generación):
- Preguntas ABIERTAS BREVES, respuesta de 1 a 5 palabras.
- Mantén el mismo "pregunta_id" de cada pregunta que corriges.
- Si el feedback indica ambigüedad, haz la pregunta más específica y unívoca.
- Respeta las reglas de "tipo"/"figura_id"/"evidencia".

Devuelve ÚNICAMENTE un objeto JSON con el mismo formato del generador:
{ "paper_id": "<paper_id>", "preguntas": [ ... ] }
Incluye SOLO las preguntas corregidas."""


def _construir_contenido_paper(
    paquete: PaquetePaper, instruccion_final: str
) -> list[dict[str, Any]]:
    """Construye el contenido multimodal: texto del paper + figuras."""
    texto = paquete.texto_completo[:MAX_CHARS_TEXTO]

    partes: list[dict[str, Any]] = [
        OpenRouterClient.text_part(
            f"PAPER_ID: {paquete.paper_id}\n"
            f"TÍTULO: {paquete.titulo}\n\n"
            f"=== TEXTO DEL PAPER ===\n{texto}\n"
        )
    ]

    if paquete.figuras:
        partes.append(
            OpenRouterClient.text_part(
                "\n=== FIGURAS DEL PAPER ===\n"
                "A continuación las figuras disponibles (con su id y caption). "
                "Úsalas para las preguntas de tipo 'imagen'."
            )
        )
        for fig in paquete.figuras:
            partes.append(
                OpenRouterClient.text_part(
                    f"[{fig.figura_id}] (página {fig.pagina}) caption: {fig.caption or '(sin caption)'}"
                )
            )
            partes.append(OpenRouterClient.image_part(fig.imagen_base64))
    else:
        partes.append(
            OpenRouterClient.text_part(
                "\n(Este paper no tiene figuras extraídas; genera las 15 "
                "preguntas de tipo 'texto'.)"
            )
        )

    partes.append(OpenRouterClient.text_part("\n" + instruccion_final))
    return partes


def generar_preguntas(
    client: OpenRouterClient, paquete: PaquetePaper
) -> SalidaGenerador:
    """Genera las 15 preguntas para un paper."""
    ids_validos = paquete.figura_ids()
    instruccion = (
        "Genera ahora las 15 preguntas siguiendo TODAS las reglas. "
        f"Figuras válidas: {sorted(ids_validos) or 'ninguna'}."
    )
    contenido = _construir_contenido_paper(paquete, instruccion)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": contenido},
    ]

    logger.info("[%s] generando preguntas con %s...", paquete.paper_id, MODELO_GENERADOR)
    data = client.chat_json(
        model=MODELO_GENERADOR,
        messages=messages,
        temperature=TEMPERATURA,
    )
    data.setdefault("paper_id", paquete.paper_id)
    salida = SalidaGenerador.model_validate(data)
    salida = _sanear(salida, ids_validos)
    logger.info(
        "[%s] generadas %d preguntas (%d de tipo imagen)",
        paquete.paper_id,
        len(salida.preguntas),
        sum(1 for p in salida.preguntas if p.tipo == "imagen"),
    )
    return salida


def corregir_preguntas(
    client: OpenRouterClient,
    paquete: PaquetePaper,
    preguntas_a_corregir: list[dict[str, Any]],
) -> list[PreguntaGenerada]:
    """Reenvía preguntas corregibles al generador con el feedback del
    evaluador. Devuelve la lista de preguntas corregidas."""
    if not preguntas_a_corregir:
        return []

    ids_validos = paquete.figura_ids()
    lineas = []
    for p in preguntas_a_corregir:
        lineas.append(
            f"- pregunta_id: {p['pregunta_id']}\n"
            f"  pregunta_original: {p['pregunta_original']}\n"
            f"  respuesta_original: {p['respuesta_original']}\n"
            f"  feedback: {p['feedback']}"
        )
    instruccion = (
        "Corrige las siguientes preguntas según su feedback. "
        f"Figuras válidas: {sorted(ids_validos) or 'ninguna'}.\n\n"
        "PREGUNTAS A CORREGIR:\n" + "\n".join(lineas)
    )
    contenido = _construir_contenido_paper(paquete, instruccion)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CORRECCION},
        {"role": "user", "content": contenido},
    ]

    logger.info(
        "[%s] corrigiendo %d preguntas...",
        paquete.paper_id,
        len(preguntas_a_corregir),
    )
    data = client.chat_json(
        model=MODELO_GENERADOR,
        messages=messages,
        temperature=TEMPERATURA,
    )
    data.setdefault("paper_id", paquete.paper_id)
    salida = SalidaGenerador.model_validate(data)
    salida = _sanear(salida, ids_validos)
    return salida.preguntas


def _sanear(
    salida: SalidaGenerador, ids_figuras_validas: set[str]
) -> SalidaGenerador:
    """Corrige inconsistencias obvias en la salida del LLM sin descartar
    preguntas (la validación dura está en la Etapa 3)."""
    for p in salida.preguntas:
        if p.tipo == "imagen":
            if not p.figura_id or p.figura_id not in ids_figuras_validas:
                # Referencia a figura inexistente: la degradamos a texto para
                # no romper el pipeline; el evaluador puede rechazarla.
                logger.debug(
                    "[%s] %s tipo imagen con figura_id inválido (%s) -> texto",
                    salida.paper_id,
                    p.pregunta_id,
                    p.figura_id,
                )
                p.tipo = "texto"
                p.figura_id = None
        else:
            p.figura_id = None
    return salida
