.PHONY: help install dev data eval report tools ask serve mcp test lint fmt clean demo

PY ?= python3
export MVB_FORCE_MOCK ?= 1

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## editable install
	$(PY) -m pip install -e .

dev:  ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

data:  ## generate and curate the 1,500-example suite
	$(PY) -m mvb.cli data

eval:  ## full comparison across all adapters
	$(PY) eval/run_eval.py

report:  ## machine-readable eval output
	$(PY) eval/run_eval.py --json

tools:  ## print the tool manifest
	$(PY) -m mvb.cli tools

ask:  ## run the agent loop (Q="...")
	$(PY) -m mvb.cli ask "$(or $(Q),which cameras saw a blocked keep-clear aisle)"

serve:  ## run the API on :8000
	MVB_FORCE_MOCK=$(MVB_FORCE_MOCK) $(PY) -m mvb.cli serve

mcp:  ## print an MCP client config stanza
	$(PY) -m mvb.cli mcp --print-config

demo:  ## the rehearsed walkthrough
	$(PY) scripts/demo.py

test:  ## run the test suite
	$(PY) -m pytest -q

lint:  ## ruff
	$(PY) -m ruff check src tests eval scripts

fmt:  ## ruff --fix
	$(PY) -m ruff check --fix src tests eval scripts

clean:
	rm -rf .pytest_cache .ruff_cache eval/reports runs checkpoints
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
