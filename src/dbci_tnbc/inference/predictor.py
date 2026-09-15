"""Inference wrapper for deployment."""
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch

from ..models.framework import DeepBayesianCausalFramework


class Predictor:
    """High-level predictor wrapping the trained model."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
        **model_kwargs,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DeepBayesianCausalFramework(**model_kwargs).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        imaging: np.ndarray,
        microbial: np.ndarray,
        transcriptomic: np.ndarray,
        return_causal: bool = True,
    ) -> Dict:
        """Run a single-batch prediction."""
        x = torch.as_tensor(imaging, dtype=torch.float32).to(self.device)
        m = torch.as_tensor(microbial, dtype=torch.float32).to(self.device)
        t = torch.as_tensor(transcriptomic, dtype=torch.float32).to(self.device)

        _, mean, std = self.model(x, m, t, return_uncertainty=True)

        out = {
            "probability": mean.squeeze(-1).cpu().numpy().tolist(),
            "uncertainty": std.squeeze(-1).cpu().numpy().tolist(),
        }

        if return_causal:
            effects = self.model.get_causal_effects(x, m, t)
            out["causal_effects"] = {
                k: v.squeeze(-1).cpu().numpy().tolist() for k, v in effects.items()
            }
        return out