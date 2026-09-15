import numpy as np
from dbci_tnbc.data import MultiModalDataset, create_dataloaders
from dbci_tnbc.utils.synthetic import generate_synthetic_data


def test_dataset():
    data = generate_synthetic_data(num_samples=20, imaging_size=32,
                                   microbial_dim=16, transcriptomic_dim=24)
    ds = MultiModalDataset(data["imaging"], data["microbial"],
                           data["transcriptomic"], data["labels"])
    sample = ds[0]
    assert sample["imaging"].shape == (1, 32, 32)
    assert sample["microbial"].shape == (16,)


def test_loaders():
    data = generate_synthetic_data(num_samples=20, imaging_size=32,
                                   microbial_dim=16, transcriptomic_dim=24)
    train, test = create_dataloaders(data, batch_size=4)
    assert len(train) > 0 and len(test) > 0