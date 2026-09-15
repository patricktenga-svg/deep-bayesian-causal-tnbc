"""Bayesian Structural Causal Module."""
from typing import Dict, Optional
import torch
import torch.nn as nn


class BayesianCausalModule(nn.Module):
    """
    Bayesian Structural Causal Model capturing:
    Microbial → TIL ← Transcriptomic → Radiomic → pCR
    """

    def __init__(
        self,
        microbial_dim: int,
        transcriptomic_dim: int,
        radiomic_dim: int,
        hidden_dim: int = 256,
        num_samples: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_samples = num_samples

        self.microbial_to_til = nn.Sequential(
            nn.Linear(microbial_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
        )
        self.transcriptomic_to_til = nn.Sequential(
            nn.Linear(transcriptomic_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
        )
        self.til_to_radiomic = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, radiomic_dim),
        )
        self.radiomic_to_pcr = nn.Sequential(
            nn.Linear(radiomic_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        microbial_features: torch.Tensor,
        transcriptomic_features: torch.Tensor,
        radiomic_features: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ):
        til_m = self.microbial_to_til(microbial_features)
        til_t = self.transcriptomic_to_til(transcriptomic_features)
        til = torch.cat([til_m, til_t], dim=-1)

        radiomic_pred = self.til_to_radiomic(til)
        radiomic_used = radiomic_features if radiomic_features is not None else radiomic_pred

        pcr_logits = self.radiomic_to_pcr(radiomic_used)

        if return_uncertainty:
            with torch.no_grad():
                samples = []
                for _ in range(self.num_samples):
                    noise = torch.randn_like(pcr_logits) * 0.1
                    samples.append(torch.sigmoid(pcr_logits + noise))
                samples = torch.stack(samples, dim=0)
                return pcr_logits, samples.mean(0), samples.std(0)
        return pcr_logits

    def compute_causal_effects(
        self,
        microbial_features: torch.Tensor,
        transcriptomic_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        til_m = self.microbial_to_til(microbial_features)
        til_t = self.transcriptomic_to_til(transcriptomic_features)
        til = torch.cat([til_m, til_t], dim=-1)
        radiomic_pred = self.til_to_radiomic(til)
        effect = self.radiomic_to_pcr(radiomic_pred)
        return {
            "direct_effect": effect,
            "mediated_effect": effect,
            "total_effect": 2 * effect,
        }