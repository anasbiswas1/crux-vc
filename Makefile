.PHONY: install validate test notebooks zip

install:
	python -m pip install -e .

validate:
	python tools/validate_repo.py

notebooks:
	python tools/validate_notebooks.py

test:
	pytest

zip:
	cd .. && zip -r crux-vc-notebook-suite.zip crux-vc -x 'crux-vc/.git/*' '*/__pycache__/*' '*.pyc'
