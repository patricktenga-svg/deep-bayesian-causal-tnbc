"""Synthetic data generation for testing/demo purposes."""
import numpy as np
import torch
from typing import Dict


def generate_synthetic_data(
    num_samples: int = 500,
    imaging_size: int = 224,
    imaging_channels: int = 1,
    microbial_dim: int = 500,
    transcriptomic_dim: int = 1000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic multi-modal data mirroring the real data structure."""
    rng = np.random.default_rng(seed)

    imaging = rng.standard_normal(
        (num_samples, imaging_channels, imaging_size, imaging_size)
    ).astype(np.float32)

    microbial = rng.standard_normal((num_samples, microbial_dim)).astype(np.float32)
    transcriptomic = rng.standard_normal(
        (num_samples, transcriptomic_dim)
    ).astype(np.float32)

    scores = (
        0.3 * microbial[:, 0]
        + 0.2 * transcriptomic[:, 0]
        + rng.standard_normal(num_samples) * 0.5
    )
    labels = (1 / (1 + np.exp(-scores)) > 0.5).astype(np.float32)

    return {
        "imaging": imaging,
        "microbial": microbial,
        "transcriptomic": transcriptomic,
        "labels": labels,
    }