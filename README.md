# anatomy-question-generator

Pipeline en Python que genera datasets de **preguntas abiertas breves de anatomía**
a partir de papers académicos en PDF. Extrae texto e imágenes de cada paper, genera
preguntas con un LLM, las evalúa y filtra con un segundo LLM, y produce un archivo de
preguntas validadas por paper.

Las respuestas son términos, estructuras o conceptos puntuales de 1 a 5 palabras
(no alternativas, no verdadero/falso, no desarrollo).

---

## Arquitectura

El pipeline tiene 5 etapas. Dos usan LLMs vía [OpenRouter](https://openrouter.ai);
el resto es código puro.

| Etapa | Qué hace | LLM |
|-------|----------|-----|
| **0 — Preprocesamiento** | Extrae texto e imágenes (con caption y página) de cada PDF con PyMuPDF. Guarda las imágenes en `figuras/`. | No |
| **1 — Generación** | Genera 15 preguntas abiertas breves por paper, usando texto + imágenes como input multimodal. | Sí (generador) |
| **2 — Evaluación** | Verifica veracidad, calidad y dificultad de las 15 preguntas y selecciona las 10 mejores. | Sí (evaluador) |
| **2.5 — Regeneración** | Si hay menos de 10 aprobadas, reenvía las "corregibles" al generador con el feedback del evaluador y las re-evalúa. Máximo 2 reintentos. | Sí |
| **3 — Validación** | Valida el formato de cada pregunta con Pydantic (campos, tipos, respuesta ≤ 5 palabras, figura existente, id único) y escribe el resultado. | No |

La salida es **un archivo `.jsonl` por paper** en la carpeta de resultados, donde cada
línea es una pregunta validada. Además, para **observabilidad**, cada llamada a un modelo
guarda en `payloads/` el request enviado y la respuesta recibida (ver más abajo).

---

## Requisitos

- Python 3.10+
- Una cuenta de OpenRouter con **saldo** (los modelos se cobran por token)
- Dependencias: PyMuPDF, Pydantic, httpx, python-dotenv (ver `requirements.txt`)

---

## Instalación

```bash
# 1. Crea el entorno virtual e instala dependencias
make install

# 2. Crea el archivo .env con tu API key de OpenRouter
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env
```

La `OPENROUTER_API_KEY` se lee automáticamente del archivo `.env` (o de una variable de
entorno del sistema). El `.env` está en `.gitignore`, así que **no se sube al repo**.

Consigue tu key en <https://openrouter.ai/keys> y carga saldo en
<https://openrouter.ai/credits>.

---

## Uso

Coloca los PDFs de entrada en la carpeta `papers/` (o `papers_test/` para pruebas) y
usa los comandos del `Makefile`:

| Comando | Qué hace | Entrada | Salida |
|---------|----------|---------|--------|
| `make help` | Lista los comandos disponibles | — | — |
| `make install` | Crea `.venv` e instala las dependencias | — | `.venv/` |
| `make pipeline` | **Pipeline completo** (etapas 0 → 3). Consume tokens de OpenRouter | `papers/` | preguntas → `resultados/`, observabilidad → `payloads/`, imágenes → `figuras/` |
| `make pipeline-test` | Pipeline completo sobre la carpeta de prueba | `papers_test/` | preguntas → `resultados_test/`, observabilidad → `payloads_test/`, imágenes → `figuras_test/` |
| `make clean` | Borra todas las salidas generadas (imágenes, payloads, resultados) | — | — |

Las carpetas de salida se crean automáticamente; no hace falta crearlas a mano.

El pipeline hace todo de una sola vez (incluida la extracción de texto e imágenes), así
que no hay un paso previo que ejecutar.

### Flujo recomendado para probar

```bash
make install         # una sola vez
# ... crea el .env con tu key ...
make pipeline-test   # corre el pipeline completo sobre 1 paper de prueba
```

### Uso directo (sin Makefile)

Los comandos del `Makefile` son atajos de `src/main.py`, que acepta estos flags:

```bash
./.venv/bin/python src/main.py \
  --papers papers \            # carpeta de PDFs de entrada
  --figuras figuras \          # carpeta donde guardar las imágenes
  --payloads-dir payloads \    # carpeta de observabilidad (request/respuesta por llamada)
  --salida-dir resultados \    # carpeta de resultados (un .jsonl por paper)
  -v                           # (opcional) logging detallado
```

---

## Estructura del proyecto

```
anatomy-question-generator/
├── papers/                 # PDFs de entrada (oficial)
├── papers_test/            # PDFs de entrada (prueba)
├── figuras/                # imágenes extraídas (oficial)        [generado]
├── figuras_test/           # imágenes extraídas (prueba)         [generado]
├── payloads/               # observabilidad: request/respuesta   [generado]
├── payloads_test/          #   por llamada a modelo (prueba)     [generado]
├── resultados/             # preguntas por paper (oficial)       [generado]
├── resultados_test/        # preguntas por paper (prueba)        [generado]
├── .env                    # OPENROUTER_API_KEY (no versionado)
├── Makefile                # atajos de los comandos
├── requirements.txt
└── src/                    # código del pipeline
    ├── main.py                 # orquestador del pipeline
    ├── etapa0_preprocesamiento.py
    ├── etapa1_generador.py     # generación y corrección (LLM)
    ├── etapa2_evaluador.py     # evaluación y selección (LLM)
    ├── etapa3_validacion.py    # validación de formato
    ├── openrouter_client.py    # cliente de la API de OpenRouter
    └── schemas.py              # modelos Pydantic (todos los schemas)
```

Las carpetas marcadas como `[generado]` se crean al ejecutar los comandos y están en
`.gitignore`.

---

## Formato de salida

Cada línea de `resultados/{paper_id}.jsonl` es una pregunta validada:

```json
{
  "pregunta_id": "paper001_q_01",
  "pregunta": "¿Qué hueso largo articula proximalmente con el fémur y distalmente con el astrágalo?",
  "respuesta": "tibia",
  "tipo": "texto",
  "figura_id": null,
  "figura_archivo": null,
  "evidencia": "Sección 3.2: 'La tibia se articula en su extremo proximal...'",
  "tema": "osteología - miembro inferior",
  "dificultad": "media",
  "metadata": {
    "paper_id": "paper001",
    "paper_titulo": "Anatomía del miembro inferior",
    "modelo_generador": "...",
    "modelo_evaluador": "...",
    "timestamp": "2026-07-24T14:30:00Z",
    "intento_generacion": 1
  }
}
```

Para preguntas de tipo `imagen`, `figura_archivo` apunta al PNG extraído
(`figuras/{paper_id}_{figura_id}.png`).

---

## Observabilidad (`payloads/`)

Cada llamada a un modelo guarda en `payloads/{paper_id}/` **exactamente lo que se envió y
lo que respondió**, para poder auditar el pipeline sin volver a correrlo:

```
payloads/{paper_id}/
├── 01_generador.request.json     # lo enviado al generador (Etapa 1)
├── 01_generador.response.json    # lo que respondió el generador
├── 02_evaluador.request.json     # lo enviado al evaluador (Etapa 2)
├── 02_evaluador.response.json
├── 03_correccion_r1.request.json # reintento 1 (Etapa 2.5), si lo hubo
├── 03_correccion_r1.response.json
├── 04_evaluador_r1.request.json
└── 04_evaluador_r1.response.json
```

- El `request.json` contiene el prompt de sistema, el texto del paper y las preguntas tal
  como se mandaron. **Las imágenes no se guardan en base64**: se reemplazan por una
  referencia al PNG real (`{"_archivo": "figuras/..._fig_1.png"}`), que puedes abrir
  directamente. Esto no altera lo que se envía a OpenRouter, solo la copia en disco.
- El `response.json` contiene la respuesta cruda del modelo (por ejemplo, los veredictos
  completos del evaluador, que no aparecen en el resultado final).

---

## Configuración de los modelos

Los modelos de OpenRouter se definen al inicio de cada etapa:

- **Generador**: constante `MODELO_GENERADOR` en `etapa1_generador.py`
- **Evaluador**: constante `MODELO_EVALUADOR` en `etapa2_evaluador.py`

Puedes cambiarlos por cualquier slug disponible en <https://openrouter.ai/models>.
Ten en cuenta que los modelos grandes (p. ej. Claude Opus) cuestan bastante más por
token y requieren más saldo que los modelos pequeños o `:free`.

---

## Notas y limitaciones

- **Errores por paper no detienen el pipeline**: si un PDF falla, se registra en el log
  y se continúa con el siguiente.
- **Captions heurísticos**: cuando hay varias imágenes en la misma página, pueden
  compartir el caption más cercano. El LLM igual recibe la imagen completa.
- **Objetivo de 10 preguntas por paper**: si tras 2 reintentos no se alcanzan 10
  aprobadas, se registran las que haya y se avisa en el log.
- **Errores comunes de OpenRouter**: `401` = key inválida; `402` = sin saldo suficiente
  para ese modelo (los modelos caros requieren más saldo que los baratos).
