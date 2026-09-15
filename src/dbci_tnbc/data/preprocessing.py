"""Preprocessing utilities (scaling, filtering)."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class MultiModalPreprocessor:
    """Standardize microbial and transcriptomic features."""

    def __init__(self):
        self.microbial_scaler = StandardScaler()
        self.transcriptomic_scaler = StandardScaler()

    def fit(self, microbial: np.ndarray, transcriptomic: np.ndarray) -> "MultiModalPreprocessor":
        self.microbial_scaler.fit(microbial)
        self.transcriptomic_scaler.fit(transcriptomic)
        return self

    def transform(self, microbial: np.ndarray, transcriptomic: np.ndarray):
        return (
            self.microbial_scaler.transform(microbial).astype(np.float32),
            self.transcriptomic_scaler.transform(transcriptomic).astype(np.float32),
        )

    def fit_transform(self, microbial, transcriptomic):
        self.fit(microbial, transcriptomic)
        return self.transform(microbial, transcriptomic)