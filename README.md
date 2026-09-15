# Deep Bayesian Causal Inference Framework for TNBC pCR Prediction

A modular, deployment-ready implementation of a **Deep Bayesian Causal Inference Framework** for predicting pathological complete response (pCR) to neoadjuvant chemotherapy (NAC) in triple-negative breast cancer (TNBC), designed for African and LMIC contexts.

## Architecture

```
Imaging (MRI)  ─► CNN-ViT ─────────────┐
                                       │
Microbial ─────► VAE ──┐               ├─► Bayesian Causal Module ─► pCR + Uncertainty
                       ├─► Causal Graph│
Transcriptomic ► VAE ──┘               │
```

The framework integrates:
- **CNN-Vision Transformer** for radiomic feature extraction
- **Variational Autoencoders** for omics compression
- **Bayesian Structural Causal Model**: `Microbial → TIL ← Transcriptomic → Radiomic → pCR`
- **Monte Carlo uncertainty quantification**

## Installation

```bash
git clone https://github.com/your-user/deep-bayesian-causal-tnbc.git
cd deep-bayesian-causal-tnbc
pip install -e ".[dev,app]"
```

## Quickstart

### Train
```bash
python scripts/train.py --config configs/config.yaml
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### API
```bash
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

### Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

### Docker
```bash
docker compose up --build
# API  → http://localhost:8000
# App  → http://localhost:8501
```

## Project Structure

```
├── src/dbci_tnbc/       # Core Python package
│   ├── data/            # Datasets, loaders, preprocessing
│   ├── models/          # CNN-ViT, VAE, causal module, framework
│   ├── training/        # Trainer, losses, metrics
│   ├── inference/       # Predictor wrapper
│   ├── utils/           # Config, logging, seeding
│   └── visualization/   # Plots
├── api/                 # FastAPI backend
├── app/                 # Streamlit frontend
├── scripts/             # CLI entry points
├── configs/             # YAML configs
├── tests/               # Pytest suite
├── Dockerfile
└── docker-compose.yml
```

## Mathematical Highlights

- **ELBO**: $\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$
- **Causal effect**: $P(Y | \text{do}(X_M = x_M)) = \sum_{z_T, z_R} P(Y|z_R)P(z_R|z_T)P(z_T|x_M)$
- **Uncertainty**: $\sigma^2_{pred} = \frac{1}{S}\sum_s (y_s - \mu)^2 + \frac{1}{S}\sum_s \sigma_s^2$

## Citation

```bibtex
@software{dbci_tnbc2026,
  title  = {Deep Bayesian Causal Inference for TNBC pCR Prediction},
  author = {Patrick Tenga Shako},
  year   = {2026},
  url    = {https://github.com/your-user/deep-bayesian-causal-tnbc}
}
```

## License

MIT — see [LICENSE](LICENSE).
