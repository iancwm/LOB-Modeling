PYTHON = python3
PIP = pip3

.PHONY: install lint format check-format check-docstrings test clean run-kyle run-almgren run-glosten run-criscuolo

install:
	$(PIP) install -r requirements.txt

lint:
	@echo "Running flake8..."
	flake8 src/lob_modeling tests
	@echo "Running black check..."
	black --check src/lob_modeling tests
	@echo "Running isort check..."
	isort --check-only src/lob_modeling tests
	@echo "Running pydocstyle check..."
	pydocstyle src/lob_modeling

format:
	@echo "Formatting code with black..."
	black src/lob_modeling tests
	@echo "Sorting imports with isort..."
	isort src/lob_modeling tests

check-format:
	@echo "Checking code formatting..."
	black --check src/lob_modeling tests
	isort --check-only src/lob_modeling tests

check-docstrings:
	@echo "Checking docstring conventions..."
	pydocstyle src/lob_modeling

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
