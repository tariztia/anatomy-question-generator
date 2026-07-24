"""Orquestador del pipeline de generación de preguntas de anatomía.

Ejecuta las 5 etapas para cada PDF de la carpeta Papers/ y produce
dataset.jsonl.

La OPENROUTER_API_KEY se lee del archivo .env (o de una variable de entorno).
Uso:

    python main.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from etapa0_preprocesamiento import preprocesar_carpeta
from etapa1_generador import corregir_preguntas, generar_preguntas
from etapa2_evaluador import evaluar_preguntas
from etapa3_validacion import ValidacionError, construir_registro
from openrouter_client import OpenRouterClient, OpenRouterError
from schemas import Evaluacion, PaquetePaper, PreguntaGenerada, SalidaEvaluador

logger = logging.getLogger("pipeline")

DIR_PAPERS = Path("Papers")
DIR_FIGURAS = Path("figuras")
ARCHIVO_SALIDA = Path("dataset.jsonl")

OBJETIVO_APROBADAS = 10
MAX_REINTENTOS = 2


def _configurar_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _contar_aprobadas(evals: dict[str, Evaluacion]) -> int:
    return sum(1 for e in evals.values() if e.veredicto == "aprobada")


def procesar_paper(
    client: OpenRouterClient,
    paquete: PaquetePaper,
    dir_figuras: Path,
) -> list:
    """Ejecuta etapas 1, 2, 2.5 y 3 para un paper. Devuelve la lista de
    registros (RegistroDataset) aprobados y validados."""

    # --- Etapa 1: generación -------------------------------------------
    salida_gen = generar_preguntas(client, paquete)
    preguntas: dict[str, PreguntaGenerada] = {
        p.pregunta_id: p for p in salida_gen.preguntas
    }

    # --- Etapa 2: evaluación inicial -----------------------------------
    salida_eval: SalidaEvaluador = evaluar_preguntas(
        client, paquete, list(preguntas.values())
    )
    evaluaciones: dict[str, Evaluacion] = {
        e.pregunta_id: e for e in salida_eval.evaluaciones
    }
    # intento en el que se generó cada pregunta (para metadata).
    intento_de: dict[str, int] = {pid: 1 for pid in preguntas}

    # --- Etapa 2.5: regeneración condicional ---------------------------
    for reintento in range(1, MAX_REINTENTOS + 1):
        aprobadas = _contar_aprobadas(evaluaciones)
        if aprobadas >= OBJETIVO_APROBADAS:
            break

        corregibles = [
            preguntas[pid]
            for pid, e in evaluaciones.items()
            if e.veredicto == "corregible" and pid in preguntas
        ]
        if not corregibles:
            logger.info(
                "[%s] no hay preguntas corregibles; se aceptan %d aprobadas",
                paquete.paper_id,
                aprobadas,
            )
            break

        logger.info(
            "[%s] reintento %d/%d: %d aprobadas (<%d), corrigiendo %d",
            paquete.paper_id,
            reintento,
            MAX_REINTENTOS,
            aprobadas,
            OBJETIVO_APROBADAS,
            len(corregibles),
        )

        payload = [
            {
                "pregunta_id": p.pregunta_id,
                "pregunta_original": p.pregunta,
                "respuesta_original": p.respuesta,
                "feedback": (
                    evaluaciones[p.pregunta_id].feedback_correccion
                    or evaluaciones[p.pregunta_id].veracidad.comentario
                    or "Mejorar claridad y unicidad de la respuesta."
                ),
            }
            for p in corregibles
        ]

        try:
            corregidas = corregir_preguntas(client, paquete, payload)
        except OpenRouterError as exc:
            logger.error("[%s] fallo en corrección: %s", paquete.paper_id, exc)
            break

        if not corregidas:
            logger.info("[%s] la corrección no devolvió preguntas", paquete.paper_id)
            break

        # Actualiza preguntas corregidas y su intento.
        for p in corregidas:
            preguntas[p.pregunta_id] = p
            intento_de[p.pregunta_id] = reintento + 1

        # Re-evalúa SOLO las corregidas (Etapa 2 de nuevo).
        try:
            reeval = evaluar_preguntas(client, paquete, corregidas)
        except OpenRouterError as exc:
            logger.error("[%s] fallo en re-evaluación: %s", paquete.paper_id, exc)
            break
        for e in reeval.evaluaciones:
            evaluaciones[e.pregunta_id] = e

    # --- Selección final -----------------------------------------------
    aprobadas_ids = [
        pid for pid, e in evaluaciones.items() if e.veredicto == "aprobada"
    ]
    aprobadas_ids.sort()
    seleccion = aprobadas_ids[:OBJETIVO_APROBADAS]

    final = _contar_aprobadas(evaluaciones)
    if len(seleccion) < OBJETIVO_APROBADAS:
        logger.warning(
            "[%s] solo %d preguntas aprobadas (objetivo %d); se registran las disponibles",
            paquete.paper_id,
            len(seleccion),
            OBJETIVO_APROBADAS,
        )
    else:
        logger.info(
            "[%s] %d aprobadas; se seleccionan %d",
            paquete.paper_id,
            final,
            len(seleccion),
        )

    # --- Etapa 3: validación de formato --------------------------------
    ids_globales: set[str] = set()  # unicidad se garantiza a nivel global fuera
    registros = []
    for pid in seleccion:
        pregunta = preguntas[pid]
        evaluacion = evaluaciones[pid]
        try:
            registro = construir_registro(
                pregunta=pregunta,
                paquete=paquete,
                dificultad_final=evaluacion.dificultad_reclasificada,
                intento_generacion=intento_de.get(pid, 1),
                dir_figuras=dir_figuras,
                ids_globales=ids_globales,
            )
            registros.append(registro)
        except (ValidacionError, ValueError) as exc:
            logger.warning(
                "[%s] pregunta %s descartada en validación: %s",
                paquete.paper_id,
                pid,
                exc,
            )

    return registros


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline de preguntas de anatomía")
    parser.add_argument("--papers", type=Path, default=DIR_PAPERS)
    parser.add_argument("--figuras", type=Path, default=DIR_FIGURAS)
    parser.add_argument("--salida", type=Path, default=ARCHIVO_SALIDA)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configurar_logging(args.verbose)

    if not args.papers.is_dir():
        logger.error("La carpeta de papers no existe: %s", args.papers)
        return 1

    # --- Etapa 0: preprocesamiento de todos los PDFs -------------------
    logger.info("=== Etapa 0: preprocesamiento de %s ===", args.papers)
    paquetes = preprocesar_carpeta(args.papers, args.figuras)
    if not paquetes:
        logger.error("No hay papers para procesar. Fin.")
        return 1
    logger.info("Preprocesados %d papers", len(paquetes))

    try:
        client = OpenRouterClient()
    except OpenRouterError as exc:
        logger.error("%s", exc)
        return 1

    ids_globales: set[str] = set()
    total = 0
    with client, args.salida.open("w", encoding="utf-8") as fh:
        for paquete in paquetes:
            logger.info("========== Paper: %s ==========", paquete.paper_id)
            try:
                registros = procesar_paper(client, paquete, args.figuras)
            except OpenRouterError as exc:
                logger.error("[%s] error de API, se omite el paper: %s", paquete.paper_id, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%s] error inesperado, se omite: %s", paquete.paper_id, exc)
                continue

            for registro in registros:
                # Unicidad global de pregunta_id entre papers.
                if registro.pregunta_id in ids_globales:
                    logger.warning(
                        "pregunta_id duplicado global %s, se omite",
                        registro.pregunta_id,
                    )
                    continue
                ids_globales.add(registro.pregunta_id)
                fh.write(
                    json.dumps(
                        registro.model_dump(), ensure_ascii=False
                    )
                    + "\n"
                )
                total += 1

            logger.info("[%s] %d registros escritos", paquete.paper_id, len(registros))

    logger.info("=== Listo. %d preguntas escritas en %s ===", total, args.salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
