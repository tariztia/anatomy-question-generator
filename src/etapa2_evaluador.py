"""Etapa 2 — Evaluación (LLM: Gemini 3.1 Pro vía OpenRouter).

Verifica veracidad, calidad y dificultad de las 15 preguntas y selecciona las
10 mejores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from openrouter_client import OpenRouterClient
from schemas import PaquetePaper, PreguntaGenerada, SalidaEvaluador

logger = logging.getLogger(__name__)

# "google/gemini-3.1-pro"
# "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

MODELO_EVALUADOR = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
TEMPERATURA = 0.0
MAX_CHARS_TEXTO = 60_000

SYSTEM_PROMPT = """\
Eres un evaluador experto en anatomía humana. Recibes un paper y 15 preguntas \
abiertas breves generadas a partir de él. Debes evaluar cada pregunta con \
rigor y seleccionar las 10 mejores.

Para CADA pregunta evalúa:

1. VERACIDAD:
   - evidencia_verificada: ¿la evidencia citada existe realmente en el paper?
   - respuesta_univoca: ¿la respuesta es correcta y ÚNICA? CRÍTICO: si la \
pregunta admite múltiples respuestas válidas, respuesta_univoca=false y el \
veredicto debe ser "corregible".
   - aprobada: true solo si evidencia_verificada Y respuesta_univoca son true.

2. CALIDAD:
   - clara: ¿se entiende sin ambigüedad?
   - autocontenida: ¿se puede responder sin ver el paper?
   - especificidad_adecuada: ¿ni demasiado vaga ni demasiado trivial?

3. DIFICULTAD: reclasifícala de forma independiente ("baja"|"media"|"alta").

Para preguntas de tipo "imagen", usa la figura correspondiente (te la paso con \
su figura_id) para verificar que la respuesta se deduce de la imagen.

VEREDICTO por pregunta:
   - "aprobada": veraz y de buena calidad.
   - "corregible": tiene un defecto subsanable (p. ej. respuesta no unívoca, \
poco específica). Incluye "feedback_correccion" con la instrucción concreta.
   - "rechazada": defecto grave (evidencia inexistente, respuesta incorrecta).

SELECCIÓN FINAL: elige hasta 10 pregunta_ids, priorizando las "aprobada" de \
mayor calidad y con buena cobertura de temas y figuras.

Devuelve ÚNICAMENTE un objeto JSON con este formato:
{
  "paper_id": "<paper_id>",
  "evaluaciones": [
    {
      "pregunta_id": "q_01",
      "veracidad": {"aprobada": true, "evidencia_verificada": true, "respuesta_univoca": true, "comentario": null},
      "calidad": {"clara": true, "autocontenida": true, "especificidad_adecuada": true, "comentario": null},
      "dificultad_reclasificada": "media",
      "veredicto": "aprobada",
      "feedback_correccion": null
    }
  ],
  "seleccion_final": {"pregunta_ids": ["q_01", "q_03"], "criterio_seleccion": "string"}
}"""


def _serializar_preguntas(preguntas: list[PreguntaGenerada]) -> str:
    items = [
        {
            "pregunta_id": p.pregunta_id,
            "pregunta": p.pregunta,
            "respuesta": p.respuesta,
            "tipo": p.tipo,
            "figura_id": p.figura_id,
            "evidencia": p.evidencia,
            "tema": p.tema,
            "dificultad_estimada": p.dificultad_estimada,
        }
        for p in preguntas
    ]
    return json.dumps(items, ensure_ascii=False, indent=2)


def _figuras_referenciadas(
    paquete: PaquetePaper, preguntas: list[PreguntaGenerada]
) -> list:
    ids = {p.figura_id for p in preguntas if p.tipo == "imagen" and p.figura_id}
    return [f for f in paquete.figuras if f.figura_id in ids]


def evaluar_preguntas(
    client: OpenRouterClient,
    paquete: PaquetePaper,
    preguntas: list[PreguntaGenerada],
    dir_figuras: Path,
    dump_base: Optional[Path] = None,
) -> SalidaEvaluador:
    """Evalúa las preguntas de un paper y devuelve la salida del evaluador."""
    texto = paquete.texto_completo[:MAX_CHARS_TEXTO]

    partes: list[dict[str, Any]] = [
        OpenRouterClient.text_part(
            f"PAPER_ID: {paquete.paper_id}\n"
            f"TÍTULO: {paquete.titulo}\n\n"
            f"=== TEXTO DEL PAPER ===\n{texto}\n"
        )
    ]
    image_refs: dict[str, str] = {}

    figuras = _figuras_referenciadas(paquete, preguntas)
    if figuras:
        partes.append(
            OpenRouterClient.text_part(
                "\n=== FIGURAS REFERENCIADAS POR LAS PREGUNTAS ==="
            )
        )
        for fig in figuras:
            partes.append(
                OpenRouterClient.text_part(
                    f"[{fig.figura_id}] (página {fig.pagina}) caption: {fig.caption or '(sin caption)'}"
                )
            )
            parte_img = OpenRouterClient.image_part(fig.imagen_base64)
            partes.append(parte_img)
            image_refs[parte_img["image_url"]["url"]] = str(
                dir_figuras / f"{paquete.paper_id}_{fig.figura_id}.png"
            )

    partes.append(
        OpenRouterClient.text_part(
            "\n=== PREGUNTAS A EVALUAR ===\n"
            + _serializar_preguntas(preguntas)
            + "\n\nEvalúa las 15 y devuelve el JSON."
        )
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": partes},
    ]

    logger.info("[%s] evaluando con %s...", paquete.paper_id, MODELO_EVALUADOR)
    data = client.chat_json(
        model=MODELO_EVALUADOR,
        messages=messages,
        temperature=TEMPERATURA,
        dump_base=dump_base,
        image_refs=image_refs,
    )
    data.setdefault("paper_id", paquete.paper_id)
    salida = SalidaEvaluador.model_validate(data)

    conteo = {"aprobada": 0, "corregible": 0, "rechazada": 0}
    for e in salida.evaluaciones:
        conteo[e.veredicto] = conteo.get(e.veredicto, 0) + 1
    logger.info(
        "[%s] evaluación: %d aprobadas, %d corregibles, %d rechazadas",
        paquete.paper_id,
        conteo["aprobada"],
        conteo["corregible"],
        conteo["rechazada"],
    )
    return salida
