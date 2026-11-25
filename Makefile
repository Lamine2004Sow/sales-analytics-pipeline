.PHONY: install run validate test lint format clean up down load dashboard notebook all nuke

install:
	uv sync

run:
	uv run retail-pipeline run

validate:
	uv run retail-pipeline validate

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts && uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests

clean:
	rm -rf data/processed data/marts
	rm -f reports/*.json reports/*.csv

# Local warehouse (Postgres) and dashboard (Metabase): see docs/metabase.md.
up:
	docker compose up -d --wait

down:
	docker compose down

load:
	uv run retail-pipeline load

dashboard:
	uv run python scripts/metabase_setup.py

# Feasibility audit notebook: generate from scripts/, then execute in place (about 2 minutes).
notebook:
	uv run python scripts/build_feasibility_notebook.py
	uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 --inplace notebooks/what_can_this_data_support.ipynb

all: run load dashboard

nuke:
	@echo "WARNING: this deletes the warehouse volume (pgdata) and all Metabase state."
	docker compose down -v
