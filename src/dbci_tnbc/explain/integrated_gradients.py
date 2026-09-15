"""Integrated Gradients for imaging modality (Sundararajan et al., 2017)."""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class IntegratedGradients:
    """Computes IG attributions for imaging input."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    def attribute(
        self,
        imaging: np.ndarray,
        microbial: np.ndarray,
        transcriptomic: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        steps: int = 32,
    ) -> Tuple[np.ndarray, float]:
        """
        Returns:
            attributions: same shape as imaging
            delta: difference in prediction (f(x) - f(baseline))
        """
        img = torch.as_tensor(imaging, dtype=torch.float32, device=self.device)
        mic = torch.as_tensor(microbial, dtype=torch.float32, device=self.device)
        trn = torch.as_tensor(transcriptomic, dtype=torch.float32, device=self.device)

        if baseline is None:
            baseline_t = torch.zeros_like(img)
        else:
            baseline_t = torch.as_tensor(baseline, dtype=torch.float32, device=self.device)

        # Scaled inputs along the path
        alphas = torch.linspace(0, 1, steps, device=self.device)
        alphas = alphas.view(-1, 1, 1, 1, 1) if img.dim() == 4 else alphas.view(-1, 1, 1)
        interpolated = baseline_t.unsqueeze(0) + alphas * (img - baseline_t).unsqueeze(0)

        # Require grad on the interpolated tensor
        interpolated = interpolated.clone().requires_grad_(True)

        # Forward pass
        logits_list = []
        for i in range(interpolated.size(0)):
            logits, _ = self.model(interpolated[i], mic, trn)
            logits_list.append(torch.sigmoid(logits))
        preds = torch.cat(logits_list, dim=0)

        # Gradient of the target class
        target = preds[:, 0].sum()
        grads = torch.autograd.grad(target, interpolated, create_graph=False)[0]

        # Trapezoidal rule
        avg_grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = avg_grads.mean(dim=0, keepdim=True)
        attributions = (img - baseline_t) * avg_grads.squeeze(0)

        # Delta = f(x) - f(baseline)
        with torch.no_grad():
            f_x, _ = self.model(img, mic, trn)
            f_base, _ = self.model(baseline_t, mic, trn)
            delta = torch.sigmoid(f_x).item() - torch.sigmoid(f_base).item()

        return attributions.detach().cpu().numpy(), delta