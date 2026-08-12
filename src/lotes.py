"""Reparto de figuras y preguntas en lotes de llamada al modelo.

Anthropic aplica un límite de tamaño distinto según cuántas imágenes lleve la
request: con 20 o menos, cada imagen puede medir hasta 8000 px de lado; a
partir de 21 el máximo baja a 2000 px y la API rechaza la request ENTERA con un
400 ("At least one of the image dimensions exceed max allowed size for
many-image requests: 2000 pixels").

Trocear en lotes de 20 imágenes es preferible a reescalar las figuras: el
número de tokens de imagen (y con él el detalle que el modelo llega a ver)
crece con los píxeles, así que encoger una figura sí degrada el desempeño. Un
montaje de varios cortes de TC en una sola figura pierde legibilidad panel a
panel al reducirlo, y son justo esas las figuras que sostienen las preguntas de
tipo "imagen".
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")


def dividir_en_lotes(items: Sequence[T], tam: int) -> list[list[T]]:
    """Trocea `items` en grupos de como máximo `tam` elementos."""
    if tam <= 0:
        raise ValueError("el tamaño de lote debe ser positivo")
    return [list(items[i : i + tam]) for i in range(0, len(items), tam)]


def lotes_por_figura(
    items: Sequence[T],
    clave_figura: Callable[[T], Optional[str]],
    max_figuras: int,
) -> list[list[T]]:
    """Agrupa `items` (preguntas) en lotes que referencien como máximo
    `max_figuras` figuras distintas cada uno.

    Los items de una misma figura nunca se separan: al modelo hay que pasarle
    la imagen junto a todas las preguntas que dependen de ella. Los items sin
    figura (preguntas de texto) van al primer lote, que es el único que existe
    cuando el paper no tiene figuras.
    """
    sin_figura: list[T] = []
    por_figura: dict[str, list[T]] = {}
    for item in items:
        fid = clave_figura(item)
        if fid is None:
            sin_figura.append(item)
        else:
            por_figura.setdefault(fid, []).append(item)

    lotes = [
        [item for fid in grupo for item in por_figura[fid]]
        for grupo in dividir_en_lotes(list(por_figura), max_figuras)
    ]
    if not lotes:
        lotes = [[]]
    lotes[0] = sin_figura + lotes[0]
    return lotes
