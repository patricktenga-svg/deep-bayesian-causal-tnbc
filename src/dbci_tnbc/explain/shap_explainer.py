"""SHAP-based explainability for tabular and image modalities."""
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


class ShapExplainer:
    """
    Wraps KernelSHAP / DeepSHAP for the omics branches and GradSHAP
    for the imaging branch of the framework.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    # ---------- Omics (DeepSHAP) ----------
    def explain_omics(
        self,
        background_microbial: np.ndarray,
        background_transcriptomic: np.ndarray,
        sample_microbial: np.ndarray,
        sample_transcriptomic: np.ndarray,
        dummy_imaging: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        import shap

        # Wrap the model to accept concatenated omics features
        class OmicsWrapper(nn.Module):
            def __init__(self, model, dummy_imaging):
                super().__init__()
                self.model = model
                self.register_buffer("dummy_imaging", torch.as_tensor(dummy_imaging, dtype=torch.float32))

            def forward(self, x):
                m = x[:, : self.model.microbial_vae.encoder[0].in_features]
                t = x[:, self.model.microbial_vae.encoder[0].in_features :]
                img = self.dummy_imaging.expand(x.size(0), -1, -1, -1)
                logits, _ = self.model(img, m, t)
                return torch.sigmoid(logits).squeeze(-1)

        wrapper = OmicsWrapper(self.model, dummy_imaging).to(self.device)

        bg = np.concatenate([background_microbial, background_transcriptomic], axis=1)
        x = np.concatenate([sample_microbial, sample_transcriptomic], axis=1)

        bg_t = torch.as_tensor(bg, dtype=torch.float32, device=self.device)
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        explainer = shap.DeepExplainer(wrapper, bg_t)
        shap_values = explainer.shap_values(x_t)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)

        return {
            "microbial": shap_values[:, : background_microbial.shape[1]],
            "transcriptomic": shap_values[:, background_microbial.shape[1] :],
        }

    # ---------- Imaging (GradientSHAP) ----------
    def explain_imaging(
        self,
        imaging: np.ndarray,
        microbial: np.ndarray,
        transcriptomic: np.ndarray,
        nsamples: int = 32,
    ) -> np.ndarray:
        import shap

        img = torch.as_tensor(imaging, dtype=torch.float32, device=self.device)
        mic = torch.as_tensor(microbial, dtype=torch.float32, device=self.device)
        trn = torch.as_tensor(transcriptomic, dtype=torch.float32, device=self.device)

        def f(x):
            logits, _ = self.model(x, mic, trn)
            return torch.sigmoid(logits).squeeze(-1)

        explainer = shap.GradientExplainer(f, img)
        sv = explainer.shap_values(img, nsamples=nsamples)
        if isinstance(sv, list):
            sv = sv[0]
        return np.asarray(sv)