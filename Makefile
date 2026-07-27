# Makefile del pipeline de generación de preguntas de anatomía.
# La OPENROUTER_API_KEY se lee del archivo .env automáticamente.

PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

# --- Carpetas ---------------------------------------------------------------
# Oficial
PAPERS        := papers
FIGURAS       := figuras
PAYLOADS      := payloads
RESULTADOS    := resultados
# Test
PAPERS_T      := papers_test
FIGURAS_T     := figuras_test
PAYLOADS_T    := payloads_test
RESULTADOS_T  := resultados_test

.PHONY: help install pipeline pipeline-test clean

help:
	@echo "Comandos disponibles:"
	@echo "  make install        Crea .venv e instala dependencias"
	@echo "  make pipeline       Pipeline completo de $(PAPERS)/ -> preguntas en $(RESULTADOS)/"
	@echo "  make pipeline-test  Pipeline completo de $(PAPERS_T)/ -> preguntas en $(RESULTADOS_T)/"
	@echo "  make clean          Borra salidas generadas (figuras, payloads, resultados)"

install:
	python3 -m venv .venv
	$(PIP) install -q -r requirements.txt

# --- Pipeline completo ------------------------------------------------------
# Extrae texto e imágenes, genera y evalúa preguntas, y guarda:
#   - preguntas finales en la carpeta de resultados (un .jsonl por paper)
#   - request y respuesta de cada modelo en la carpeta de payloads (observabilidad)
pipeline:
	$(PY) src/main.py \
		--papers $(PAPERS) --figuras $(FIGURAS) \
		--payloads-dir $(PAYLOADS) --salida-dir $(RESULTADOS) -v

pipeline-test:
	$(PY) src/main.py \
		--papers $(PAPERS_T) --figuras $(FIGURAS_T) \
		--payloads-dir $(PAYLOADS_T) --salida-dir $(RESULTADOS_T) -v

clean:
	rm -rf $(FIGURAS) $(FIGURAS_T) $(PAYLOADS) $(PAYLOADS_T) $(RESULTADOS) $(RESULTADOS_T)
