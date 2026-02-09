PYTHON = python3
PIP = pip3

.PHONY: install lint test clean run-kyle run-almgren

install:
	$(PIP) install -r requirements.txt

lint:
	@echo "Linting..."
	flake8 src/lob_modeling tests

test:
	@echo "Running tests..."
	python3 -m unittest discover tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-kyle:
	$(PYTHON) -c "from src.lob_modeling.models.kyle import KyleModel; KyleModel()"

run-almgren:
	$(PYTHON) -c "from src.lob_modeling.models.almgren_chriss import AlmgrenChriss2000; AlmgrenChriss2000()"

run-glosten:
	$(PYTHON) -c "from src.lob_modeling.models.glosten_milgrom import GlostenAndMilgromSimplest; GlostenAndMilgromSimplest()"

run-criscuolo:
	$(PYTHON) -c "from src.lob_modeling.models.criscuolo_waehlbroeck import Criscuolo2014; Criscuolo2014()"
