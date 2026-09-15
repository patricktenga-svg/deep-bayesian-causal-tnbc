"""FastAPI application entry point."""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dbci_tnbc.inference.predictor import Predictor
from dbci_tnbc.utils.config import load_config

from . import routes

app = FastAPI(
    title="DBCI-TNBC API",
    description="Deep Bayesian Causal Inference for TNBC pCR prediction",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
checkpoint = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")

if Path(config_path).exists():
    cfg = load_config(config_path)
    predictor = Predictor(
        checkpoint_path=checkpoint,
        device=os.getenv("DEVICE", "cpu"),
        imaging_channels=cfg.data["imaging_channels"],
        microbial_dim=cfg.data["microbial_dim"],
        transcriptomic_dim=cfg.data["transcriptomic_dim"],
        latent_dim=cfg.model["latent_dim"],
        hidden_dim=cfg.model["hidden_dim"],
        num_mc_samples=cfg.model["num_mc_samples"],
        image_size=cfg.data["imaging_size"],
    )
    routes.set_predictor(predictor)

app.include_router(routes.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "DBCI-TNBC API", "docs": "/docs"}