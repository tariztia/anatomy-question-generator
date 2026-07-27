"""Etapa 0 — Preprocesamiento (sin LLM).

Recorre los PDFs de una carpeta y, para cada uno, extrae el texto completo y
las imágenes con su caption, número de figura y página. Las imágenes se
guardan como PNG en `figuras/{paper_id}_{figura_id}.png`.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from schemas import Figura, PaquetePaper

logger = logging.getLogger(__name__)

# Detecta captions del tipo "Figure 1", "Fig. 2:", "Figura 3 -", etc.
CAPTION_RE = re.compile(r"^\s*(fig(?:ure|ura)?\.?\s*\d+)", re.IGNORECASE)

# Dimensiones mínimas para considerar una imagen relevante (evita iconos/logos).
MIN_ANCHO = 80
MIN_ALTO = 80

# Los metadatos de muchos PDFs traen el nombre del archivo del maquetador
# (p. ej. "rb2015v48n6p358-362_en.p65") en lugar del título del paper.
NOMBRE_ARCHIVO_RE = re.compile(r"\.(p65|pdf|docx?|indd|qxd|qxp|fm|tex|ai|eps)$", re.I)


def _titulo_plausible(titulo: str) -> bool:
    """Descarta nombres de archivo, códigos y fragmentos demasiado cortos."""
    t = " ".join(titulo.split())
    if len(t) < 15 or " " not in t:
        return False
    if NOMBRE_ARCHIVO_RE.search(t):
        return False
    letras = sum(1 for c in t if c.isalpha())
    return letras >= 0.6 * len(t)


def _titulo_por_fuente(pagina: fitz.Page) -> str:
    """Reconstruye el título con los spans de mayor tamaño de fuente de la
    mitad superior de la primera página (suele venir partido en varias líneas)."""
    alto = pagina.rect.height or 1.0
    spans: list[tuple[float, float, float, str]] = []  # (tamaño, y, x, texto)
    for bloque in pagina.get_text("dict").get("blocks", []):
        for linea in bloque.get("lines", []):
            for span in linea.get("spans", []):
                texto = " ".join(span.get("text", "").split())
                if len(texto) < 2:  # asteriscos y marcas de nota al pie
                    continue
                x0, y0 = span["bbox"][0], span["bbox"][1]
                if y0 > alto * 0.5:
                    continue
                spans.append((round(span.get("size", 0.0), 1), y0, x0, texto))

    if not spans:
        return ""
    tam_max = max(s[0] for s in spans)
    del_titulo = [s for s in spans if s[0] >= tam_max - 0.5]
    del_titulo.sort(key=lambda s: (s[1], s[2]))  # orden de lectura
    return " ".join(s[3] for s in del_titulo)


def _titulo_desde_pdf(doc: fitz.Document, paper_id: str) -> str:
    """Título del paper, en orden de confianza: metadatos (si no son el nombre
    del archivo), texto de mayor tamaño de la portada, primera línea del texto."""
    candidatos = [(doc.metadata or {}).get("title") or ""]

    if doc.page_count:
        candidatos.append(_titulo_por_fuente(doc[0]))
        for linea in doc[0].get_text("text").splitlines():
            if len(linea.strip()) > 5:
                candidatos.append(linea.strip())
                break

    for candidato in candidatos:
        if _titulo_plausible(candidato):
            return " ".join(candidato.split())

    logger.warning("[%s] no se pudo determinar el título; se usa el paper_id", paper_id)
    return paper_id


def _buscar_caption(pagina: fitz.Page, bbox: fitz.Rect) -> str:
    """Busca el bloque de texto tipo caption más cercano por debajo (o encima)
    del bounding box de la imagen."""
    bloques = pagina.get_text("blocks")  # (x0, y0, x1, y1, texto, ...)
    candidatos: list[tuple[float, str]] = []
    for x0, y0, x1, y1, texto, *_ in bloques:
        texto_limpio = " ".join(texto.split())
        if not texto_limpio:
            continue
        if CAPTION_RE.match(texto_limpio):
            # Distancia vertical entre la imagen y el bloque de texto.
            if y0 >= bbox.y1:  # texto debajo de la imagen
                dist = y0 - bbox.y1
            elif y1 <= bbox.y0:  # texto encima de la imagen
                dist = bbox.y0 - y1
            else:
                dist = 0.0
            candidatos.append((dist, texto_limpio))

    if not candidatos:
        return ""
    candidatos.sort(key=lambda c: c[0])
    return candidatos[0][1]


def _numero_figura(caption: str, fallback: int) -> Optional[int]:
    m = re.search(r"(\d+)", caption)
    return int(m.group(1)) if m else fallback


def procesar_pdf(pdf_path: Path, dir_figuras: Path) -> PaquetePaper:
    """Extrae texto e imágenes de un PDF y devuelve el paquete de la etapa 0."""
    paper_id = pdf_path.stem
    # Normaliza el paper_id para nombres de archivo seguros.
    paper_id_safe = re.sub(r"[^\w\-.]+", "_", paper_id).strip("_")

    doc = fitz.open(pdf_path)
    try:
        titulo = _titulo_desde_pdf(doc, paper_id_safe)

        partes_texto: list[str] = []
        figuras: list[Figura] = []
        vistos: set[int] = set()  # xrefs ya extraídos (evita duplicados)
        contador = 0

        for num_pagina in range(doc.page_count):
            pagina = doc[num_pagina]
            partes_texto.append(pagina.get_text("text"))

            for img in pagina.get_images(full=True):
                xref = img[0]
                if xref in vistos:
                    continue
                vistos.add(xref)

                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < MIN_ANCHO or pix.height < MIN_ALTO:
                        pix = None
                        continue
                    # Convierte CMYK / con alpha a RGB para PNG estándar.
                    if pix.n - pix.alpha >= 4 or pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    png_bytes = pix.tobytes("png")
                    pix = None
                except Exception as exc:  # imagen corrupta / no rasterizable
                    logger.warning(
                        "[%s] no se pudo extraer imagen xref=%s: %s",
                        paper_id_safe,
                        xref,
                        exc,
                    )
                    continue

                contador += 1
                figura_id = f"fig_{contador}"

                # Caption: usa el rectángulo de la imagen en la página.
                caption = ""
                try:
                    rects = pagina.get_image_rects(xref)
                    if rects:
                        caption = _buscar_caption(pagina, rects[0])
                except Exception:  # noqa: BLE001
                    pass

                nombre_archivo = f"{paper_id_safe}_{figura_id}.png"
                (dir_figuras / nombre_archivo).write_bytes(png_bytes)

                figuras.append(
                    Figura(
                        figura_id=figura_id,
                        imagen_base64=base64.b64encode(png_bytes).decode("ascii"),
                        caption=caption,
                        pagina=num_pagina + 1,
                    )
                )

        texto_completo = "\n".join(partes_texto).strip()
        logger.info(
            "[%s] extraído: %d páginas, %d figuras, %d caracteres",
            paper_id_safe,
            doc.page_count,
            len(figuras),
            len(texto_completo),
        )

        return PaquetePaper(
            paper_id=paper_id_safe,
            titulo=titulo,
            texto_completo=texto_completo,
            figuras=figuras,
        )
    finally:
        doc.close()


def preprocesar_carpeta(
    dir_papers: Path, dir_figuras: Path
) -> list[PaquetePaper]:
    """Procesa todos los PDFs de una carpeta. Errores por paper no detienen
    el resto."""
    dir_figuras.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(dir_papers.glob("*.pdf"))
    if not pdfs:
        logger.warning("No se encontraron PDFs en %s", dir_papers)

    paquetes: list[PaquetePaper] = []
    for pdf_path in pdfs:
        try:
            paquetes.append(procesar_pdf(pdf_path, dir_figuras))
        except Exception as exc:  # noqa: BLE001
            logger.error("Fallo procesando %s: %s", pdf_path.name, exc)
    return paquetes
