"""Etapa 2 — Evaluación (LLM: Gemini 3.1 Pro vía OpenRouter).

Verifica veracidad, calidad y dificultad de las preguntas generadas y
selecciona las mejores (como máximo config.N_SELECCION_MAX; pueden ser menos).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from string import Template
from typing import Any, Optional

import config
from etapa0_preprocesamiento import ruta_figura
from openrouter_client import OpenRouterClient  # carga el .env al importarse
from schemas import PaquetePaper, PreguntaGenerada, SalidaEvaluador

logger = logging.getLogger(__name__)

# Configurable con la variable de entorno MODELO_EVALUADOR (o en el .env).
# DEBE ser de una familia distinta a MODELO_GENERADOR. Recomendado:
#   MODELO_EVALUADOR=google/gemini-3.1-pro   (multimodal, buen seguimiento de rúbricas)
# Otras opciones probadas:
#   "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "openai/gpt-4o-mini".

MODELO_EVALUADOR = os.getenv("MODELO_EVALUADOR", "openai/gpt-4o-mini")
TEMPERATURA = 0.0
MAX_CHARS_TEXTO = 60_000

_SYSTEM_PROMPT_TMPL = Template("""\
Eres un evaluador experto en anatomía humana. Recibes un paper y un conjunto de \
preguntas abiertas breves generadas a partir de él. Debes evaluar cada pregunta \
con rigor y seleccionar las mejores, como máximo $n_seleccion_max.

Las preguntas alimentarán un dataset de entrenamiento de un modelo multimodal \
de anatomía. El paper es solo la FUENTE del conocimiento y de las imágenes: las \
preguntas deben ser de ANATOMÍA GENERAL, para alguien que nunca verá el paper. \
Sé severo: una pregunta mediocre contamina el dataset, una pregunta menos no.

Para CADA pregunta evalúa:

1. VERACIDAD:
   - evidencia_verificada: ¿la evidencia citada aparece LITERALMENTE en el \
paper, en su idioma original? Una paráfrasis o una traducción cuenta como NO \
verificada.
   - respuesta_univoca: ¿la respuesta es correcta y ÚNICA? CRÍTICO: si la \
pregunta admite múltiples respuestas válidas, respuesta_univoca=false y el \
veredicto debe ser "corregible".
   - aprobada: true solo si evidencia_verificada Y respuesta_univoca son true.

2. CALIDAD:
   - clara: ¿se entiende sin ambigüedad?
   - autocontenida: ¿se puede responder sin ver el paper?
   - especificidad_adecuada: ¿ni demasiado vaga ni demasiado trivial?
   - independiente_del_paper: ¿es una pregunta de anatomía general? Pon false si \
menciona el estudio, sus autores, su muestra o su método; si pregunta por una \
estadística del estudio (porcentaje, prevalencia, n, media, rango, medida de la \
muestra); si pregunta por una figura como objeto ("¿Qué figura muestra X?"); o \
si la respuesta es un número del paper o un id de figura.
   - imagen_como_estimulo: solo para tipo "imagen" (null si es "texto"). Pon \
true únicamente si la figura es el ESTÍMULO y no la respuesta: es imposible \
responder sin mirar la imagen, la respuesta es una estructura, variante o \
consecuencia clínica (nunca "fig_N"), la imagen no se identifica por su número, \
las marcas mencionadas (flechas, etiquetas, calipers) existen de verdad en ella, \
y la respuesta no se limita a leer un texto o una medición impresa en la imagen.

3. DIFICULTAD: reclasifícala de forma independiente ("baja"|"media"|"alta"). Una \
pregunta cuya respuesta se recita de memoria es "baja" por muy técnica que suene.

Para preguntas de tipo "imagen", MIRA la figura correspondiente (te la paso con \
su figura_id) y comprueba que la respuesta se deduce de la imagen y que lo que \
el enunciado dice ver está realmente ahí.

VEREDICTO por pregunta:
   - "aprobada": todos los criterios de veracidad y calidad en true.
   - "corregible": defecto subsanable manteniendo la idea (respuesta no unívoca, \
poco específica, evidencia parafraseada, o una pregunta de figura reformulable \
como "¿qué estructura/variante se observa en la imagen?"). Incluye \
"feedback_correccion" con la instrucción concreta de cómo reescribirla.
   - "rechazada": defecto grave e insalvable: evidencia inexistente, respuesta \
incorrecta, o la pregunta solo existe por ser un dato del estudio (una \
estadística de la muestra no se puede convertir en pregunta de anatomía).

SELECCIÓN FINAL: elige COMO MÁXIMO $n_seleccion_max pregunta_ids, solo entre las \
"aprobada", priorizando la mayor calidad, la variedad de temas y de ejes \
(identificación, relaciones, variantes, correlato clínico) y una buena \
proporción de preguntas de imagen. Seleccionar MENOS de $n_seleccion_max es un \
resultado perfectamente válido y esperable: no completes el cupo con preguntas \
débiles. Es preferible devolver 14 preguntas excelentes que $n_seleccion_max \
con relleno. Ordena los ids de mejor a peor.

Devuelve ÚNICAMENTE un objeto JSON con este formato:
{
  "paper_id": "<paper_id>",
  "evaluaciones": [
    {
      "pregunta_id": "q_01",
      "veracidad": {"aprobada": true, "evidencia_verificada": true, "respuesta_univoca": true, "comentario": null},
      "calidad": {"clara": true, "autocontenida": true, "especificidad_adecuada": true, "independiente_del_paper": true, "imagen_como_estimulo": null, "comentario": null},
      "dificultad_reclasificada": "media",
      "veredicto": "aprobada",
      "feedback_correccion": null
    }
  ],
  "seleccion_final": {"pregunta_ids": ["q_01", "q_03"], "criterio_seleccion": "string"}
}"""
)

SYSTEM_PROMPT = _SYSTEM_PROMPT_TMPL.substitute(
    n_seleccion_max=config.N_SELECCION_MAX,
)


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
                ruta_figura(dir_figuras, paquete.paper_id, fig.figura_id)
            )

    partes.append(
        OpenRouterClient.text_part(
            "\n=== PREGUNTAS A EVALUAR ===\n"
            + _serializar_preguntas(preguntas)
            + f"\n\nEvalúa las {len(preguntas)} y devuelve el JSON."
        )
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": partes},
    ]

    logger.info(
        "[%s] evaluando %d preguntas con %s (%d figuras adjuntas)...",
        paquete.paper_id,
        len(preguntas),
        MODELO_EVALUADOR,
        len(figuras),
    )
    data = client.chat_json(
        model=MODELO_EVALUADOR,
        messages=messages,
        temperature=TEMPERATURA,
        max_tokens=config.MAX_TOKENS_EVALUADOR,
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
