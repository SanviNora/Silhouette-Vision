# Silhouette Vision pipeline. Each stage reads the previous stage's outputs (see docs/pipeline.md).
PY ?= .venv/bin/python

.PHONY: catalog embed colour attributes clusters match models demand public eval app test lint

catalog:     ; $(PY) scripts/01_build_catalog.py
embed:       ; $(PY) scripts/02_embed.py --model marqo_fashion_siglip
colour:      ; $(PY) scripts/03_colour_features.py
attributes:  ; $(PY) scripts/04_attributes.py
clusters:    ; $(PY) scripts/05_style_clusters.py
match:       ; $(PY) scripts/06_match_confidence.py
models:      ; $(PY) scripts/07_named_models.py
demand:      ; $(PY) scripts/08_visuelle_prepare.py && $(PY) scripts/09_demand.py
public:      ; $(PY) scripts/10_build_public.py
eval:        ; $(PY) scripts/eval_lookbench.py && $(PY) scripts/eval_catalog.py

app:         ; $(PY) -m streamlit run app/ui.py
test:        ; $(PY) -m pytest -q
lint:        ; $(PY) -m ruff check src scripts app tests
