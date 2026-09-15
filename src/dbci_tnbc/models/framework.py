"""Full Deep Bayesian Causal Inference Framework."""
from typing import Tuple, Union
import torch
import torch.nn as nn

from .cnn_vit import CNNVisionTransformer
from .vae import VariationalAutoencoder
from .causal import BayesianCausalModule


class DeepBayesianCausalFramework(nn.Module):
    """End-to-end multi-modal framework for TNBC pCR prediction."""

    def __init__(
        self,
        imaging_channels: int = 1,
        microbial_dim: int = 500,
        transcriptomic_dim: int = 1000,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_mc_samples: int = 100,
        image_size: int = 224,
    ):
        super().__init__()

        self.imaging_encoder = CNNVisionTransformer(
            input_channels=imaging_channels, image_size=image_size
        )
        self.microbial_vae = VariationalAutoencoder(microbial_dim, latent_dim)
        self.transcriptomic_vae = VariationalAutoencoder(transcriptomic_dim, latent_dim)

        self.causal_module = BayesianCausalModule(
            microbial_dim=latent_dim,
            transcriptomic_dim=latent_dim,
            radiomic_dim=256,
            hidden_dim=hidden_dim,
            num_samples=num_mc_samples,
        )

        self.predictor = nn.Sequential(
            nn.Linear(256 + 2 * latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        imaging: torch.Tensor,
        microbial: torch.Tensor,
        transcriptomic: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        imaging_features = self.imaging_encoder(imaging)
        _, _, _, microbial_latent = self.microbial_vae(microbial)
        _, _, _, transcriptomic_latent = self.transcriptomic_vae(transcriptomic)

        if return_uncertainty:
            pcr_logits, pcr_mean, pcr_std = self.causal_module(
                microbial_latent,
                transcriptomic_latent,
                imaging_features,
                return_uncertainty=True,
            )
            return pcr_logits, pcr_mean, pcr_std

        pcr_logits = self.causal_module(
            microbial_latent, transcriptomic_latent, imaging_features
        )
        combined = torch.cat(
            [imaging_features, microbial_latent, transcriptomic_latent], dim=-1
        )
        final_output = self.predictor(combined)
        return pcr_logits, final_output

    def get_causal_effects(self, imaging, microbial, transcriptomic):
        _, _, _, microbial_latent = self.microbial_vae(microbial)
        _, _, _, transcriptomic_latent = self.transcriptomic_vae(transcriptomic)
        return self.causal_module.compute_causal_effects(
            microbial_latent, transcriptomic_latent
        )