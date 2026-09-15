.PHONY: all setup run api dashboard test calibrate clean help

PYTHON := $(shell command -v python3 2>/dev/null || echo python)
VENV_PYTHON := venv/bin/python
VENV_PIP := venv/bin/pip

# Use venv python if it exists, else system python
ifeq ($(wildcard venv/bin/python),)
    PY := $(PYTHON)
else
    PY := $(VENV_PYTHON)
endif

all:            ## Setup + run pipeline
	$(PYTHON) setup_env.py

setup:          ## Create venv and install deps
	$(PYTHON) setup_env.py

run:            ## Run CLI pipeline
	$(PY) main.py

api:            ## Start FastAPI server
	$(PY) -m uvicorn api.server:app --port 8000 --reload

dashboard:      ## Start Streamlit dashboard
	$(PY) -m streamlit run dashboard/app.py

run-all:        ## Start API + dashboard together
	@$(PY) -m uvicorn api.server:app --port 8000 --reload & \
	sleep 2 && $(PY) -m streamlit run dashboard/app.py

test:           ## Run all tests
	$(PY) -m pytest tests/ -v

test-fast:      ## Run tests, stop on first failure
	$(PY) -m pytest tests/ -x -v

test-cov:       ## Run tests with coverage
	$(PY) -m pytest tests/ --cov=. --cov-report=term-missing

calibrate-demo: ## Calibrate model on synthetic data
	$(PY) models/calibration.py --demo

structure:      ## Print project file structure
	$(PY) file_structure.py

clean:          ## Remove venv, caches, build artifacts
	rm -rf venv __pycache__ .pytest_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

kill-ports:     ## Kill processes on port 8000
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@pkill -f streamlit 2>/dev/null || true

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'