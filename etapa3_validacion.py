"""Etapa 3 — Validación de formato (sin LLM).

Construye los registros finales del dataset y los valida programáticamente con
Pydantic (schema RegistroDataset). Verifica además la unicidad global de
pregunta_id y la existencia del archivo de figura.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from etapa1_generador import MODELO_GENERADOR
from etapa2_evaluador import MODELO_EVALUADOR
from schemas import (
    MetadataRegistro,
    PaquetePaper,
    PreguntaGenerada,
    RegistroDataset,
)

logger = logging.getLogger(__name__)


class ValidacionError(ValueError):
    pass


def construir_registro(
    pregunta: PreguntaGenerada,
    paquete: PaquetePaper,
    dificultad_final: str,
    intento_generacion: int,
    dir_figuras: Path,
    ids_globales: set[str],
) -> RegistroDataset:
    """Construye y valida un registro del dataset a partir de una pregunta
    aprobada. Lanza ValidacionError si no cumple el formato."""

    pregunta_id_global = f"{paquete.paper_id}_{pregunta.pregunta_id}"

    if pregunta_id_global in ids_globales:
        raise ValidacionError(
            f"pregunta_id duplicado globalmente: {pregunta_id_global}"
        )

    figura_archivo: Optional[str] = None
    if pregunta.tipo == "imagen":
        if not pregunta.figura_id:
            raise ValidacionError(
                f"{pregunta_id_global}: tipo 'imagen' sin figura_id"
            )
        nombre = f"{paquete.paper_id}_{pregunta.figura_id}.png"
        ruta = dir_figuras / nombre
        if not ruta.exists():
            raise ValidacionError(
                f"{pregunta_id_global}: figura_archivo no existe: {ruta}"
            )
        figura_archivo = f"{dir_figuras.name}/{nombre}"

    metadata = MetadataRegistro(
        paper_id=paquete.paper_id,
        paper_titulo=paquete.titulo,
        modelo_generador=MODELO_GENERADOR,
        modelo_evaluador=MODELO_EVALUADOR,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        intento_generacion=intento_generacion,
    )

    # RegistroDataset aplica las validaciones de campo (respuesta <= 5 palabras,
    # dificultad válida, tipos correctos, etc.).
    registro = RegistroDataset(
        pregunta_id=pregunta_id_global,
        pregunta=pregunta.pregunta,
        respuesta=pregunta.respuesta,
        tipo=pregunta.tipo,
        figura_id=pregunta.figura_id if pregunta.tipo == "imagen" else None,
        figura_archivo=figura_archivo,
        evidencia=pregunta.evidencia,
        tema=pregunta.tema,
        dificultad=dificultad_final,  # type: ignore[arg-type]
        metadata=metadata,
    )

    ids_globales.add(pregunta_id_global)
    return registro
