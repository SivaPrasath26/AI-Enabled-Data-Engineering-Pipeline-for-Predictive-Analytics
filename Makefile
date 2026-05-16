.PHONY: install lint test synthetic run api docker clean

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest tests/ -v --cov=src

synthetic:
	python scripts/generate_synthetic.py --target data/raw --students 3000

run:
	python scripts/run_pipeline.py --config configs/pipeline.yaml

api:
	uvicorn src.serving.app:app --reload --port 8000

docker:
	docker compose up --build

clean:
	rm -rf data/bronze/* data/silver/* data/gold/* models/artifacts/* logs/*
