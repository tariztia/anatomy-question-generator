"""Todos los schemas Pydantic del pipeline.

Se agrupan aquí los modelos de las distintas etapas para tener una única
fuente de verdad. Las etapas importan desde este módulo.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Tipos comunes
# ---------------------------------------------------------------------------

Tipo = Literal["texto", "imagen"]
Dificultad = Literal["baja", "media", "alta"]
Veredicto = Literal["aprobada", "corregible", "rechazada"]


# ---------------------------------------------------------------------------
# Etapa 0 — Preprocesamiento
# ---------------------------------------------------------------------------


class Figura(BaseModel):
    figura_id: str
    imagen_base64: str
    caption: str = ""
    pagina: int


class PaquetePaper(BaseModel):
    """Paquete producido por la etapa 0 para cada paper."""

    paper_id: str
    titulo: str
    texto_completo: str
    figuras: list[Figura] = Field(default_factory=list)

    def figura_ids(self) -> set[str]:
        return {f.figura_id for f in self.figuras}


# ---------------------------------------------------------------------------
# Etapa 1 — Generación de preguntas
# ---------------------------------------------------------------------------


class PreguntaGenerada(BaseModel):
    pregunta_id: str
    pregunta: str
    respuesta: str
    tipo: Tipo
    figura_id: Optional[str] = None
    evidencia: str
    tema: str
    dificultad_estimada: Dificultad


class SalidaGenerador(BaseModel):
    paper_id: str
    preguntas: list[PreguntaGenerada]


# ---------------------------------------------------------------------------
# Etapa 2 — Evaluación
# ---------------------------------------------------------------------------


class Veracidad(BaseModel):
    aprobada: bool
    evidencia_verificada: bool
    respuesta_univoca: bool
    comentario: Optional[str] = None


class Calidad(BaseModel):
    clara: bool
    autocontenida: bool
    especificidad_adecuada: bool
    # ¿Es una pregunta de anatomía general y no una pregunta sobre el paper?
    independiente_del_paper: Optional[bool] = None
    # Solo para tipo "imagen": la figura es el estímulo y no la respuesta.
    imagen_como_estimulo: Optional[bool] = None
    comentario: Optional[str] = None


class Evaluacion(BaseModel):
    pregunta_id: str
    veracidad: Veracidad
    calidad: Calidad
    dificultad_reclasificada: Dificultad
    veredicto: Veredicto
    feedback_correccion: Optional[str] = None


class SeleccionFinal(BaseModel):
    pregunta_ids: list[str] = Field(default_factory=list)
    criterio_seleccion: str = ""


class SalidaEvaluador(BaseModel):
    paper_id: str
    evaluaciones: list[Evaluacion]
    seleccion_final: SeleccionFinal


# ---------------------------------------------------------------------------
# Etapa 2.5 — Regeneración condicional
# ---------------------------------------------------------------------------


class PreguntaACorregir(BaseModel):
    pregunta_id: str
    pregunta_original: str
    respuesta_original: str
    feedback: str


class InputRegeneracion(BaseModel):
    paper_id: str
    preguntas_a_corregir: list[PreguntaACorregir]


# ---------------------------------------------------------------------------
# Salida final — registro del dataset JSONL (Etapa 3 valida esto)
# ---------------------------------------------------------------------------


class MetadataRegistro(BaseModel):
    paper_id: str
    paper_titulo: str
    modelo_generador: str
    modelo_evaluador: str
    timestamp: str
    intento_generacion: int


class RegistroDataset(BaseModel):
    """Schema de cada línea del dataset.jsonl. Aquí viven las validaciones
    de formato de la Etapa 3."""

    pregunta_id: str
    pregunta: str
    respuesta: str
    tipo: Tipo
    figura_id: Optional[str] = None
    figura_archivo: Optional[str] = None
    evidencia: str
    tema: str
    dificultad: Dificultad
    metadata: MetadataRegistro

    @field_validator("respuesta")
    @classmethod
    def respuesta_valida(cls, v: str) -> str:
        v_limpia = v.strip()
        if not v_limpia:
            raise ValueError("respuesta no puede estar vacía")
        if len(v_limpia.split()) > 5:
            raise ValueError("respuesta debe tener como máximo 5 palabras")
        return v_limpia

    @field_validator("pregunta")
    @classmethod
    def pregunta_no_vacia(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pregunta no puede estar vacía")
        return v
