"""Dataset definitions for multi-modal breast cancer data."""
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """PyTorch dataset for imaging + microbial + transcriptomic data."""

    def __init__(
        self,
        imaging_data: np.ndarray,
        microbial_data: np.ndarray,
        transcriptomic_data: np.ndarray,
        labels: np.ndarray,
        clinical_data: Optional[np.ndarray] = None,
    ):
        self.imaging_data = torch.as_tensor(imaging_data, dtype=torch.float32)
        self.microbial_data = torch.as_tensor(microbial_data, dtype=torch.float32)
        self.transcriptomic_data = torch.as_tensor(
            transcriptomic_data, dtype=torch.float32
        )
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.clinical_data = (
            torch.as_tensor(clinical_data, dtype=torch.float32)
            if clinical_data is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "imaging": self.imaging_data[idx],
            "microbial": self.microbial_data[idx],
            "transcriptomic": self.transcriptomic_data[idx],
            "label": self.labels[idx],
        }
        if self.clinical_data is not None:
            sample["clinical"] = self.clinical_data[idx]
        return sample