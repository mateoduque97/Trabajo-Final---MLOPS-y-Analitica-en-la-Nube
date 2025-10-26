.PHONY: install train test lint

install:
	python3 -m pip install -r requirements.txt

train:
	python3 src/train.py

test:
	pytest -q

lint:
	flake8 src
