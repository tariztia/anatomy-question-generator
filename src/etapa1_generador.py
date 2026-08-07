"""Etapa 1 — Generación de preguntas (LLM: Claude Opus 4.7 vía OpenRouter).

También implementa la corrección de preguntas de la Etapa 2.5 (mismo agente,
con el feedback del evaluador).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from string import Template
from typing import Any, Optional

import config
from etapa0_preprocesamiento import ruta_figura
from openrouter_client import OpenRouterClient  # carga el .env al importarse
from schemas import PaquetePaper, PreguntaGenerada, SalidaGenerador

logger = logging.getLogger(__name__)

# Configurable con la variable de entorno MODELO_GENERADOR (o en el .env).
# DEBE ser distinto de MODELO_EVALUADOR: si un modelo se evalúa a sí mismo,
# aprueba sus propios errores. Recomendado en producción:
#   MODELO_GENERADOR=anthropic/claude-opus-4-7   (multimodal fuerte)
# Otras opciones probadas: "google/gemma-4-31b-it:free", "openai/gpt-4o-mini".

MODELO_GENERADOR = os.getenv("MODELO_GENERADOR", "openai/gpt-4o-mini")
TEMPERATURA = 0.7
MAX_CHARS_TEXTO = 60_000  # recorte defensivo del texto del paper

# Esquema de salida compartido por los dos system prompts (generación y
# corrección). Vive en una constante porque cuando cada prompt lo describía por
# su cuenta divergieron: el de corrección decía "el mismo formato del generador"
# sin listar los campos, y el modelo devolvía preguntas sin "tema" ni
# "dificultad_estimada".
_ESQUEMA_SALIDA = """\
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
Los ocho campos son OBLIGATORIOS en CADA pregunta; omitir uno invalida la \
respuesta completa."""

_SYSTEM_PROMPT_TMPL = Template("""\
Eres un experto en anatomía humana y en diseño de evaluaciones. Usas papers \
académicos como FUENTE DE CONOCIMIENTO para escribir preguntas de ANATOMÍA \
GENERAL. No escribes preguntas SOBRE el paper.

PRINCIPIO RECTOR: quien responde nunca ha visto el paper y nunca lo verá. Cada \
pregunta debe tener sentido tal cual en un examen de anatomía de cualquier \
universidad. El paper solo te sirve para saber qué es anatómicamente cierto y \
para aportarte las imágenes.

PROHIBIDO (descarta la pregunta y escribe otra):
- Mencionar el estudio, sus autores, su muestra o su método ("en este estudio", \
"según los autores", "la serie analizada", "en la tomografía realizada").
- Preguntar por resultados estadísticos del estudio: porcentajes, prevalencias, \
n, medias, rangos o medidas de la muestra. Ejemplos de lo que NO debes generar: \
"¿Qué porcentaje presentó el tronco hepatogástrico?", "¿Cuál es la longitud \
media del tronco celíaco?", "¿Cuál fue la distancia media entre X e Y?".
- Preguntar por una figura como objeto de la pregunta: "¿Qué figura muestra X?", \
"¿En cuál de las imágenes se ve Y?". Una figura NUNCA es la respuesta.
- Que la respuesta sea un número tomado del paper, un id de figura ("fig_3") o \
"Figura 2".

PREGUNTAS DE TIPO "imagen" (las más valiosas):
La figura es el ESTÍMULO de la pregunta, jamás la respuesta. Quien responde ve \
la imagen y el enunciado, nada más.
- Patrones correctos: "¿Qué estructura señala la punta de flecha?", "¿Qué \
variante anatómica se observa en la imagen?", "¿De qué vaso nace la estructura \
que cruza la línea media?", "¿Qué territorio quedaría isquémico si se ocluyera \
el vaso más grueso de la imagen?", "¿Qué órgano se vería comprometido por un \
trombo en la estructura señalada?".
- Debe ser IMPOSIBLE responder sin mirar la imagen. Si el enunciado por sí solo \
basta para responder, es una pregunta de texto mal etiquetada.
- Refiérete a la imagen como "la imagen" o "la figura", NUNCA por su número.
- Solo puedes apuntar a marcas que REALMENTE existan en esa imagen (flechas, \
punteros, etiquetas, calipers). Si no las hay, localiza la estructura en \
términos anatómicos o de posición ("el vaso más superior", "la estructura que \
cruza la línea media").
- No preguntes por valores escritos sobre la imagen (mediciones, texto del \
equipo): eso se lee, no se razona.
- Si una figura no permite una pregunta honesta y respondible desde la imagen \
(es un gráfico, una tabla o es ilegible), NO la uses.
- Cada figura da pie a VARIAS preguntas (la instrucción final te dice cuántas \
por figura). Esas preguntas deben atacar aspectos DISTINTOS de la imagen y \
tener respuestas DISTINTAS entre sí: por ejemplo, una de identificación de una \
estructura, otra de relación espacial con lo que la rodea y otra de \
consecuencia clínica. Reformular la misma pregunta con otras palabras, o que \
dos preguntas de la misma figura compartan respuesta, cuenta como repetición y \
está prohibido.

CONTENIDO Y VARIEDAD de las $n_generadas preguntas:
- Mezcla ejes: identificación de estructuras, origen/trayecto/ramas, relaciones \
espaciales (anterior, posterior, medial, superior), variantes anatómicas y su \
nombre propio, y correlato clínico-quirúrgico ("qué se lesionaría si...", "qué \
se infartaría si...").
- Al menos $n_alta preguntas de dificultad "alta", que exijan razonar (consecuencia \
funcional, relación espacial, distinguir una variante de la disposición \
habitual), no recordar un dato.
- No repitas la misma estructura como respuesta más de dos veces.
- Usa nomenclatura anatómica estándar (Terminologia Anatomica) en español.

REGLAS DE FORMATO:
1. Genera EXACTAMENTE $n_generadas preguntas, con pregunta_id de "q_01" a "$id_max".
2. Todas son ABIERTAS BREVES: la respuesta es un término, estructura, variante \
o concepto puntual de 1 a 5 palabras. NO alternativas, NO verdadero/falso, NO \
desarrollo.
3. El número EXACTO de preguntas de tipo "imagen" y de tipo "texto", y cuántas \
preguntas corresponden a cada figura, te los indica la instrucción final. \
Respétalos al pie de la letra.
4. Si "tipo" es "imagen", "figura_id" DEBE ser el id de una figura real del \
input (por ejemplo "fig_1"). Si "tipo" es "texto", "figura_id" debe ser null.
5. La respuesta debe ser UNÍVOCA: si un estudiante informado pudiera dar otra \
respuesta igualmente válida, reformula la pregunta hasta que solo quepa una.
6. EVIDENCIA: cita TEXTUAL y literal del paper, en su idioma original, copiada \
carácter por carácter. Nunca traduzcas ni parafrasees. Prefiere pasajes de \
introducción o discusión que describan anatomía general, no filas de resultados. \
Para preguntas de imagen: la cita literal del caption MÁS una descripción de qué \
se ve en la imagen que justifica la respuesta.

Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional, con este \
formato:
$esquema
Los pregunta_id van de "q_01" a "$id_max"."""
)

SYSTEM_PROMPT = _SYSTEM_PROMPT_TMPL.substitute(
    n_generadas=config.N_GENERADAS,
    id_max=f"q_{config.N_GENERADAS:02d}",
    n_alta=config.N_DIFICULTAD_ALTA,
    esquema=_ESQUEMA_SALIDA,
)

SYSTEM_PROMPT_CORRECCION = """\
Eres un experto en anatomía humana. Recibes preguntas que fueron rechazadas o \
marcadas como corregibles por un evaluador, junto con el feedback específico. \
Corrige cada pregunta respetando el feedback.

REGLAS ESTRICTAS (idénticas a la generación):
- Las preguntas son de ANATOMÍA GENERAL, no sobre el paper. Quien responde nunca \
verá el paper. Prohibido mencionar el estudio, su muestra o su método, y \
prohibido preguntar por sus estadísticas (porcentajes, medias, prevalencias).
- En las preguntas de tipo "imagen", la figura es el ESTÍMULO, nunca la \
respuesta. Prohibido "¿Qué figura muestra X?" y prohibido responder con un id de \
figura. Reformúlalas como "¿Qué estructura/variante se observa en la imagen?" o \
como una consecuencia clínica de lo que se ve. Refiérete a la imagen como "la \
imagen" o "la figura", nunca por su número, y solo apunta a marcas que \
realmente existan en ella.
- Preguntas ABIERTAS BREVES, respuesta de 1 a 5 palabras, unívoca.
- Evidencia: cita literal del paper en su idioma original, sin traducir ni \
parafrasear.
- Mantén el mismo "pregunta_id" de cada pregunta que corriges.
- Si el feedback indica ambigüedad, haz la pregunta más específica y unívoca.
- Respeta las reglas de "tipo"/"figura_id"/"evidencia".
- Cada pregunta a corregir viene con su "tema" y su "dificultad_estimada" \
originales: cópialos tal cual en tu respuesta, salvo que la corrección los \
vuelva incorrectos.

Devuelve ÚNICAMENTE un objeto JSON con este formato (el mismo del generador):
""" + _ESQUEMA_SALIDA + """
Incluye SOLO las preguntas corregidas."""


def _construir_contenido_paper(
    paquete: PaquetePaper, instruccion_final: str, dir_figuras: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Construye el contenido multimodal (texto + figuras) y el mapa de
    referencias imagen->ruta PNG para el volcado de observabilidad."""
    texto = paquete.texto_completo[:MAX_CHARS_TEXTO]

    partes: list[dict[str, Any]] = [
        OpenRouterClient.text_part(
            f"PAPER_ID: {paquete.paper_id}\n"
            f"TÍTULO: {paquete.titulo}\n\n"
            f"=== TEXTO DEL PAPER ===\n{texto}\n"
        )
    ]
    image_refs: dict[str, str] = {}

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
            parte_img = OpenRouterClient.image_part(fig.imagen_base64)
            partes.append(parte_img)
            image_refs[parte_img["image_url"]["url"]] = str(
                ruta_figura(dir_figuras, paquete.paper_id, fig.figura_id)
            )
    else:
        partes.append(
            OpenRouterClient.text_part(
                "\n(Este paper no tiene figuras extraídas; todas las preguntas "
                "deben ser de tipo 'texto'.)"
            )
        )

    partes.append(OpenRouterClient.text_part("\n" + instruccion_final))
    return partes, image_refs


def generar_preguntas(
    client: OpenRouterClient,
    paquete: PaquetePaper,
    dir_figuras: Path,
    dump_base: Optional[Path] = None,
) -> SalidaGenerador:
    """Genera las preguntas de un paper (config.N_GENERADAS en total).

    El reparto imagen/texto depende de cuántas figuras tenga el paper: se piden
    config.PREGUNTAS_POR_FIGURA por figura, con el tope que impone el suelo de
    preguntas de texto. Cuando no caben 3 por figura se reparten de forma
    pareja entre todas para no dejar figuras sin usar."""
    ids_validos = paquete.figura_ids()
    figura_ids = [f.figura_id for f in paquete.figuras]
    n_imagen, n_texto = config.reparto_preguntas(len(figura_ids))
    reparto = config.reparto_por_figura(figura_ids, n_imagen)

    if reparto:
        detalle = ", ".join(f"{fid}: {k}" for fid, k in reparto.items())
        instruccion_figuras = (
            f"De esas, EXACTAMENTE {n_imagen} deben ser de tipo 'imagen' y "
            f"{n_texto} de tipo 'texto'.\n"
            "Reparto obligatorio de las preguntas de imagen por figura "
            f"(figura_id: nº de preguntas) -> {detalle}.\n"
            "Cada figura debe recibir exactamente ese número de preguntas, y "
            "las preguntas de una misma figura deben enfocar aspectos "
            "distintos y tener respuestas distintas.\n"
            "Si alguna de esas figuras no admite una pregunta honesta "
            "(es un gráfico, una tabla o es ilegible), genera en su lugar "
            "preguntas de tipo 'texto' hasta completar el total."
        )
    else:
        instruccion_figuras = (
            f"Este paper no tiene figuras utilizables: las {n_texto} preguntas "
            "deben ser de tipo 'texto'."
        )

    instruccion = (
        f"Genera ahora las {config.N_GENERADAS} preguntas siguiendo TODAS las "
        "reglas. Recuerda: son preguntas de anatomía general para alguien que "
        "nunca verá este paper, y en las de tipo 'imagen' la figura es el "
        "estímulo, nunca la respuesta.\n"
        f"{instruccion_figuras}\n"
        f"Figuras válidas: {sorted(ids_validos) or 'ninguna'}."
    )
    contenido, image_refs = _construir_contenido_paper(paquete, instruccion, dir_figuras)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": contenido},
    ]

    logger.info(
        "[%s] generando %d preguntas con %s (%d imagen sobre %d figuras, %d texto)...",
        paquete.paper_id,
        config.N_GENERADAS,
        MODELO_GENERADOR,
        n_imagen,
        len(figura_ids),
        n_texto,
    )
    data = client.chat_json(
        model=MODELO_GENERADOR,
        messages=messages,
        temperature=TEMPERATURA,
        max_tokens=config.MAX_TOKENS_GENERADOR,
        dump_base=dump_base,
        image_refs=image_refs,
    )
    data.setdefault("paper_id", paquete.paper_id)
    salida = SalidaGenerador.model_validate(data)
    salida = _sanear(salida, ids_validos)
    obtenidas_imagen = sum(1 for p in salida.preguntas if p.tipo == "imagen")
    logger.info(
        "[%s] generadas %d/%d preguntas (%d de tipo imagen, pedidas %d)",
        paquete.paper_id,
        len(salida.preguntas),
        config.N_GENERADAS,
        obtenidas_imagen,
        n_imagen,
    )
    return salida


def corregir_preguntas(
    client: OpenRouterClient,
    paquete: PaquetePaper,
    preguntas_a_corregir: list[dict[str, Any]],
    dir_figuras: Path,
    dump_base: Optional[Path] = None,
) -> list[PreguntaGenerada]:
    """Reenvía preguntas corregibles al generador con el feedback del
    evaluador. Devuelve la lista de preguntas corregidas."""
    if not preguntas_a_corregir:
        return []

    ids_validos = paquete.figura_ids()
    originales = {p["pregunta_id"]: p for p in preguntas_a_corregir}
    lineas = []
    for p in preguntas_a_corregir:
        lineas.append(
            f"- pregunta_id: {p['pregunta_id']}\n"
            f"  pregunta_original: {p['pregunta_original']}\n"
            f"  respuesta_original: {p['respuesta_original']}\n"
            f"  tema: {p.get('tema', '')}\n"
            f"  dificultad_estimada: {p.get('dificultad_estimada', '')}\n"
            f"  feedback: {p['feedback']}"
        )
    instruccion = (
        "Corrige las siguientes preguntas según su feedback. "
        f"Figuras válidas: {sorted(ids_validos) or 'ninguna'}.\n\n"
        "PREGUNTAS A CORREGIR:\n" + "\n".join(lineas)
    )
    contenido, image_refs = _construir_contenido_paper(paquete, instruccion, dir_figuras)

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
        max_tokens=config.MAX_TOKENS_GENERADOR,
        dump_base=dump_base,
        image_refs=image_refs,
    )
    data.setdefault("paper_id", paquete.paper_id)
    _completar_campos_heredados(data, originales, paquete.paper_id)
    salida = SalidaGenerador.model_validate(data)
    salida = _sanear(salida, ids_validos)
    return salida.preguntas


def _completar_campos_heredados(
    data: dict[str, Any], originales: dict[str, dict[str, Any]], paper_id: str
) -> None:
    """Rellena in situ los campos que el corrector suele omitir por venir de la
    pregunta original y no del feedback.

    El prompt ya los pide explícitamente, pero un solo campo ausente hace
    fallar la validación del lote entero y con él el paper completo. Como el
    valor original es de todos modos la mejor fuente, se hereda en silencio.
    """
    for pregunta in data.get("preguntas") or []:
        if not isinstance(pregunta, dict):
            continue
        original = originales.get(pregunta.get("pregunta_id"))
        if original is None:
            continue
        for campo in ("tema", "dificultad_estimada"):
            if not pregunta.get(campo) and original.get(campo):
                logger.debug(
                    "[%s] %s: '%s' ausente en la corrección; se hereda el original",
                    paper_id,
                    pregunta.get("pregunta_id"),
                    campo,
                )
                pregunta[campo] = original[campo]


def _sanear(
    salida: SalidaGenerador, ids_figuras_validas: set[str]
) -> SalidaGenerador:
    """Corrige inconsistencias obvias en la salida del LLM sin descartar
    preguntas (la validación dura está en la Etapa 3)."""
    degradadas = 0
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
                degradadas += 1
        else:
            p.figura_id = None
    if degradadas:
        # Señal de que el generador está inventando figura_ids: cada
        # degradación resta una pregunta de imagen del reparto pedido.
        logger.info(
            "[%s] %d preguntas de imagen degradadas a texto por figura_id inválido",
            salida.paper_id,
            degradadas,
        )
    return salida
