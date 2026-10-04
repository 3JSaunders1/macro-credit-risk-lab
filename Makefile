SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

RUN_ID  := $(shell date +%Y%m%d_%H%M%S)
RUN_DIR := reports/runs/$(RUN_ID)
LOG     := $(RUN_DIR)/run_log.txt

.PHONY: all run-dir setup data estimate backtest scenarios run api dashboard test archive

all: run-dir estimate backtest scenarios run archive
	@echo "Run complete: $(RUN_DIR)"

run-dir:
	@mkdir -p $(RUN_DIR)

setup:          ## Create venv and install pinned dependencies
	python3 setup_env.py

data:           ## Rebuild macro data from FRED (requires FRED_API_KEY in .env)
	python -m pipeline.fetch_data

estimate: run-dir   ## Static credit model, dynamic loss model, and its backtests
	python -m models.credit_estimation 2>&1 | tee -a $(LOG)
	python -m models.loss_validation 2>&1 | tee -a $(LOG)

backtest: run-dir   ## Rolling out-of-sample forecast backtest
	python -m analysis.forecast_backtest 2>&1 | tee -a $(LOG)

scenarios: run-dir  ## Model-based stress scenarios
	python -m pipeline.scenario_engine 2>&1 | tee -a $(LOG)

run: run-dir        ## CLI pipeline: forecast, credit, and stress test
	python main.py 2>&1 | tee -a $(LOG)

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
