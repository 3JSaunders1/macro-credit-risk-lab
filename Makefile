SHELL := /bin/bash

RUN_ID  := $(shell date +%Y%m%d_%H%M%S)
RUN_DIR := reports/runs/$(RUN_ID)
LOG     := $(RUN_DIR)/run_log.txt

# `set -o pipefail;` makes a failing step stop `make` even when its output is piped to `tee`
# (macOS ships GNU Make 3.81, which ignores .SHELLFLAGS).

.PHONY: all run-dir setup data estimate backtest scenarios run api dashboard test archive \
	docker-build docker-test docker-up docker-down

all: run-dir estimate backtest scenarios run archive
	@echo "Run complete: $(RUN_DIR)"

run-dir:
	@mkdir -p $(RUN_DIR)

setup:          ## Create venv and install pinned dependencies
	python3 setup_env.py

data:           ## Rebuild macro data from FRED (requires FRED_API_KEY in .env)
	python -m pipeline.fetch_data

estimate: run-dir   ## Static credit model, dynamic loss model, and its backtests
	set -o pipefail; python -m models.credit_estimation 2>&1 | tee -a $(LOG)
	set -o pipefail; python -m models.loss_validation 2>&1 | tee -a $(LOG)

backtest: run-dir   ## Rolling out-of-sample forecast backtest
	set -o pipefail; python -m analysis.forecast_backtest 2>&1 | tee -a $(LOG)

scenarios: run-dir  ## Model-based stress scenarios
	set -o pipefail; python -m pipeline.scenario_engine 2>&1 | tee -a $(LOG)

run: run-dir        ## CLI pipeline: forecast, credit, and stress test
	set -o pipefail; python main.py 2>&1 | tee -a $(LOG)

api:            ## Start the FastAPI server
	python -m uvicorn api.server:app --port 8000 --reload

dashboard:      ## Start the Streamlit dashboard
	python -m streamlit run dashboard/app.py

test:           ## Run the test suite
	python -m pytest -v

archive: run-dir
	cp reports/figures/* $(RUN_DIR)/
	cp data/macro_data_meta.json config/*.json $(RUN_DIR)/
	@echo "Run ID:     $(RUN_ID)" > $(RUN_DIR)/run_info.txt
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'not committed')" >> $(RUN_DIR)/run_info.txt

docker-build:   ## Build the Docker image
	docker build -t macro-credit-risk-lab .

docker-test: docker-build   ## Run the test suite inside the container
	docker run --rm macro-credit-risk-lab

docker-up: docker-build   ## Build once, then start the API (port 8000) and dashboard (port 8501)
	docker-compose up -d
	@echo "API:       http://localhost:8000/docs"
	@echo "Dashboard: http://localhost:8501"

docker-down:    ## Stop the containers
	docker-compose down
