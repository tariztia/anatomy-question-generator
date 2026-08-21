"""Parámetros de volumen y reparto de preguntas del pipeline.

Fuente única de verdad para los conteos que antes vivían dispersos en los
prompts de cada etapa. Los prompts se construyen sustituyendo estos valores,
de modo que cambiar un número aquí lo cambia en todo el pipeline.
"""

from __future__ import annotations

from typing import Sequence

# --- Etapa 1: generación ---------------------------------------------------

# Preguntas que se piden al generador por paper.
N_GENERADAS = 40

# Preguntas de imagen que se piden por cada figura del paper.
PREGUNTAS_POR_FIGURA = 3

# Suelo de preguntas de texto: aunque el paper tenga muchas figuras, siempre se
# reservan al menos estas para tipo "texto".
MIN_PREGUNTAS_TEXTO = 5

# Preguntas de dificultad "alta" que se exigen (proporcional a N_GENERADAS).
N_DIFICULTAD_ALTA = 10

# --- Etapa 2: evaluación y selección ---------------------------------------

# Tope de preguntas seleccionadas por paper. Pueden ser menos: el evaluador no
# debe completar el cupo con preguntas débiles.
N_SELECCION_MAX = 30

# --- Límites de entrada de los modelos -------------------------------------

# Figuras (imágenes) que puede llevar UNA llamada. A partir de 21 imágenes
# Anthropic baja el lado máximo permitido por imagen de 8000 a 2000 px y
# rechaza la request entera con un 400. Se trocea en lotes en vez de reescalar
# las figuras porque reescalar sí cuesta detalle: los tokens de imagen (y lo
# que el modelo ve) escalan con los píxeles. Ver src/lotes.py.
MAX_FIGURAS_POR_LLAMADA = 20

# --- Límites de salida de los modelos --------------------------------------

# Con 40 preguntas por paper la respuesta ronda los 8k tokens. Se fija un tope
# explícito y holgado porque el default del proveedor puede ser mucho menor, y
# una respuesta truncada es JSON inválido que consume todos los reintentos.
MAX_TOKENS_GENERADOR = 32_000

# El evaluador necesita bastante más que el generador: en los modelos de
# razonamiento los tokens de thinking cuentan contra max_tokens y no aparecen
# en la respuesta. Medido con gemini-3.1-pro sobre un paper de 40 preguntas:
# 9.149 tokens de razonamiento + ~10k de JSON, o sea que con 16.000 la
# respuesta se cortaba a mitad de camino (finish_reason="length").
MAX_TOKENS_EVALUADOR = 32_000


def reparto_preguntas(n_figuras: int) -> tuple[int, int]:
    """Devuelve (n_imagen, n_texto) para un paper con `n_figuras` figuras.

    Se piden PREGUNTAS_POR_FIGURA por figura, con el tope que impone el suelo
    de preguntas de texto. Un paper sin figuras genera solo preguntas de texto.
    """
    max_imagen = N_GENERADAS - MIN_PREGUNTAS_TEXTO
    n_imagen = min(PREGUNTAS_POR_FIGURA * max(n_figuras, 0), max_imagen)
    return n_imagen, N_GENERADAS - n_imagen


def reparto_por_figura(
    figura_ids: Sequence[str], n_imagen: int
) -> dict[str, int]:
    """Reparte `n_imagen` preguntas entre las figuras lo más parejo posible.

    Cuando no alcanzan 3 por figura se prefiere cubrir TODAS las figuras con
    menos preguntas cada una antes que dar 3 a unas pocas: con 12 figuras y 35
    preguntas salen 11 figuras con 3 y 1 figura con 2. Las figuras que quedan
    a cero (solo si hay más figuras que preguntas) se omiten del reparto.
    """
    n = len(figura_ids)
    if n == 0 or n_imagen <= 0:
        return {}
    base, resto = divmod(n_imagen, n)
    reparto = {
        fid: base + (1 if i < resto else 0) for i, fid in enumerate(figura_ids)
    }
    return {fid: k for fid, k in reparto.items() if k > 0}
