FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py pyproject.toml ./

RUN pip install -e .

ENV PYTHONPATH=/app/src:/app
ENV CONFIG_PATH=/app/configs/config.yaml
ENV MODEL_PATH=/app/checkpoints/best_model.pt

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]