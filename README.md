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
| **1 — Generación** | Genera 40 preguntas abiertas breves por paper, usando texto + imágenes como input multimodal. El reparto imagen/texto depende de las figuras del paper (ver abajo). | Sí (generador) |
| **2 — Evaluación** | Verifica veracidad, calidad y dificultad de las 40 preguntas y selecciona las mejores, como máximo 30 (pueden ser menos). | Sí (evaluador) |
| **2.5 — Regeneración** | Si hay menos de 30 aprobadas, reenvía las "corregibles" al generador con el feedback del evaluador y las re-evalúa. Máximo 2 reintentos. | Sí |
| **3 — Validación** | Valida el formato de cada pregunta con Pydantic (campos, tipos, respuesta ≤ 5 palabras, figura existente, id único) y escribe el resultado. | No |

La salida es **un archivo `.jsonl` por paper** en la carpeta de resultados, donde cada
línea es una pregunta validada. Además, para **observabilidad**, cada llamada a un modelo
guarda en `payloads/` el request enviado y la respuesta recibida (ver más abajo).

### Cuántas preguntas de imagen se piden

Los conteos viven en `src/config.py`. El generador siempre produce **40 preguntas** por
paper; cuántas son de imagen depende de las figuras que tenga:

- Se piden **3 preguntas por figura**, repartidas entre **todas** las figuras.
- Se reservan siempre **al menos 5 preguntas de texto**, así que el tope de imagen es 35.
- Cuando 3 por figura no caben en ese tope, se baja el número por figura (3 → 2 → 1) antes
  que dejar figuras sin usar: 12 figuras dan 35 preguntas de imagen repartidas como 11
  figuras con 3 y 1 figura con 2.
- Si hay más figuras que preguntas de imagen (36+), las que quedan a cero se omiten del
  reparto y **no se envían al modelo**: serían tokens de imagen pagados para nada.

| Figuras del paper | Preguntas de imagen | Preguntas de texto |
|---|---|---|
| 0 | 0 | 40 |
| 1 | 3 | 37 |
| 6 | 18 | 22 |
| 11 | 33 | 7 |
| 12 | 35 | 5 |
| 20+ | 35 | 5 |

### Lotes de figuras por llamada

Una request no puede llevar más de **20 imágenes** (`MAX_FIGURAS_POR_LLAMADA`). A partir
de 21, Anthropic baja el lado máximo permitido por imagen de 8000 a 2000 px y rechaza la
request **entera** con un `400`. Se trocea en lotes en vez de reescalar las figuras porque
reescalar sí cuesta detalle: los tokens de imagen —y con ellos lo que el modelo llega a
ver— escalan con los píxeles, y un montaje de varios cortes de TC pierde legibilidad panel
a panel al reducirlo.

Cuando un paper supera ese límite (ver `src/lotes.py`):

- **Etapa 1**: las figuras se reparten en grupos de 20 y cada llamada pide solo su cuota de
  preguntas. Las de texto van enteras en el primer lote. Los `pregunta_id` se renumeran al
  final, porque cada lote numera desde `q_01`.
- **Etapa 2**: las preguntas se agrupan por figura (las de una misma figura nunca se
  separan, porque la imagen tiene que viajar con ellas) en lotes de ≤ 20 figuras. Cada lote
  ordena su propia selección y las selecciones se mezclan en *round-robin*, para que el
  primer lote no cope el cupo global y sesgue el dataset hacia sus figuras.
- En `payloads/` cada lote deja sus propios archivos, con el sufijo `_lote{n}`.

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
    ├── config.py               # conteos y límites (fuente única de verdad)
    ├── etapa0_preprocesamiento.py
    ├── etapa1_generador.py     # generación y corrección (LLM)
    ├── etapa2_evaluador.py     # evaluación y selección (LLM)
    ├── etapa3_validacion.py    # validación de formato
    ├── lotes.py                # reparto de figuras en lotes por llamada
    ├── openrouter_client.py    # cliente de la API de OpenRouter
    └── schemas.py              # modelos Pydantic (todos los schemas)
```

Las carpetas marcadas como `[generado]` se crean al ejecutar los comandos y están en
`.gitignore`. Las carpetas de PDFs (`papers/`, `papers_test/`) también están ignoradas:
los papers no se versionan.

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
(`figuras/{paper_id}/{figura_id}.png`: una carpeta por paper).

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

Cuando el paper tiene más de 20 figuras y la llamada se trocea en lotes, cada lote escribe
sus propios archivos con el sufijo `_lote{n}`
(`01_generador_lote1.request.json`, `01_generador_lote2.request.json`, …).

- El `request.json` contiene el prompt de sistema, el texto del paper y las preguntas tal
  como se mandaron. **Las imágenes no se guardan en base64**: se reemplazan por una
  referencia al PNG real (`{"_archivo": "figuras/{paper_id}/fig_1.png"}`), que puedes abrir
  directamente. Esto no altera lo que se envía a OpenRouter, solo la copia en disco.
- El `response.json` contiene la respuesta cruda del modelo (por ejemplo, los veredictos
  completos del evaluador, que no aparecen en el resultado final), más el `finish_reason`
  y el `usage` de la llamada. Un `finish_reason: "length"` significa que la respuesta se
  cortó por `max_tokens`; en los modelos de razonamiento revisa
  `usage.completion_tokens_details.reasoning_tokens`, porque el thinking consume ese mismo
  presupuesto y no aparece en la respuesta.

---

## Configuración de los modelos

Los modelos de OpenRouter se configuran con variables de entorno (en el `.env` o en
la línea de comandos), con la constante de cada etapa como valor por defecto:

- **Generador**: `MODELO_GENERADOR` (constante en `etapa1_generador.py`)
- **Evaluador**: `MODELO_EVALUADOR` (constante en `etapa2_evaluador.py`)

Si no defines ninguna de las dos, ambas etapas caen en el mismo default
(`openai/gpt-4o-mini`), que es justo la autoevaluación que hay que evitar: define las dos
en el `.env`.

```bash
MODELO_GENERADOR=anthropic/claude-opus-4-7 MODELO_EVALUADOR=google/gemini-3.1-pro make pipeline
```

**Usa siempre modelos de familias distintas.** Si el generador y el evaluador son el
mismo modelo, la etapa 2 es una autoevaluación: el modelo aprueba sus propios errores
(evidencias parafraseadas, respuestas no unívocas) y el control de veracidad deja de
funcionar. El pipeline emite un `WARNING` al arrancar cuando detecta esta situación.

Puedes usar cualquier slug disponible en <https://openrouter.ai/models>.
Ten en cuenta que los modelos grandes (p. ej. Claude Opus) cuestan bastante más por
token y requieren más saldo que los modelos pequeños o `:free`. Ambas etapas envían
imágenes, así que los dos modelos deben ser multimodales.

---

## Parámetros del pipeline (`src/config.py`)

Los conteos y límites viven en un solo lugar; cambiar un número ahí lo cambia en todos los
prompts y en todas las etapas.

| Constante | Valor | Qué controla |
|---|---|---|
| `N_GENERADAS` | 40 | Preguntas que se piden al generador por paper |
| `PREGUNTAS_POR_FIGURA` | 3 | Preguntas de imagen por cada figura |
| `MIN_PREGUNTAS_TEXTO` | 5 | Suelo de preguntas de texto (tope de imagen = 35) |
| `N_DIFICULTAD_ALTA` | 10 | Preguntas de dificultad "alta" exigidas |
| `N_SELECCION_MAX` | 30 | Tope de preguntas seleccionadas por paper |
| `MAX_FIGURAS_POR_LLAMADA` | 20 | Imágenes por request antes de trocear en lotes |
| `MAX_TOKENS_GENERADOR` | 32 000 | Tope de salida del generador |
| `MAX_TOKENS_EVALUADOR` | 32 000 | Tope de salida del evaluador (el thinking cuenta aquí) |

Los reintentos de red y el timeout están en `openrouter_client.py`
(`MAX_RETRIES = 4`, `DEFAULT_TIMEOUT = 180 s`); los reintentos de regeneración de la
etapa 2.5, en `main.py` (`MAX_REINTENTOS = 2`).

---

## Notas y limitaciones

- **Errores por paper no detienen el pipeline**: si un PDF falla, se registra en el log
  y se continúa con el siguiente.
- **Captions heurísticos**: cuando hay varias imágenes en la misma página, pueden
  compartir el caption más cercano. El LLM igual recibe la imagen completa.
- **Objetivo de 30 preguntas por paper**: si tras 2 reintentos no se alcanzan 30
  aprobadas, se registran las que haya y se avisa en el log. Seleccionar menos del tope es
  un resultado válido: el evaluador no debe completar el cupo con preguntas débiles.
- **Errores comunes de OpenRouter**: `401` = key inválida; `402` = sin saldo suficiente
  para ese modelo (los modelos caros requieren más saldo que los baratos).
- **Reintentos selectivos**: solo se reintentan `429`, `500`, `502`, `503` y `504`
  (saturación o fallo pasajero). Cualquier otro `4xx` es un problema del payload y se
  aborta de inmediato (`ErrorAPIPermanente`) conservando el mensaje del proveedor.
- **Respuestas cortadas o vacías**: `RespuestaTruncadaError` (se agotó `max_tokens`) y
  `RespuestaVaciaError` (el modelo de razonamiento dejó todo el output en `reasoning`) no
  se reintentan; hay que subir el tope de tokens o cambiar de modelo.
