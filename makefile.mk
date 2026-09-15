.PHONY: install train evaluate api app test lint format docker-build docker-up

install:
	pip install -e ".[dev,app]"

train:
	python scripts/train.py --config configs/config.yaml

evaluate:
	python scripts/evaluate.py --config configs/config.yaml

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v

lint:
	flake8 src api scripts

format:
	black src api scripts

docker-build:
	docker build -t dbci-tnbc:latest .

docker-up:
	docker compose up --build
export-onnx:
	python scripts/export.py --config configs/config.yaml --torchscript

benchmark:
	python scripts/benchmark.py

explain:
	python scripts/explain.py

track-wandb:
	python scripts/train.py --tracker wandb --run-name dbci-v1

helm-lint:
	helm lint deploy/helm/dbci-tnbc

helm-install:
	helm upgrade --install dbci deploy/helm/dbci-tnbc \
		--set global.imageRegistry=ghcr.io/your-user/deep-bayesian-causal-tnbc

helm-uninstall:
	helm uninstall dbci
