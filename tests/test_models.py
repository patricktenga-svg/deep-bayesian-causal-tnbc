import torch
from dbci_tnbc.models import (
    CNNVisionTransformer,
    VariationalAutoencoder,
    DeepBayesianCausalFramework,
)


def test_cnn_vit():
    model = CNNVisionTransformer(input_channels=1, image_size=64)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 256)


def test_vae():
    vae = VariationalAutoencoder(input_dim=100, latent_dim=16)
    x = torch.randn(4, 100)
    recon, mu, logvar, z = vae(x)
    assert recon.shape == x.shape
    assert z.shape == (4, 16)


def test_framework_forward():
    model = DeepBayesianCausalFramework(
        microbial_dim=50, transcriptomic_dim=80, image_size=64, num_mc_samples=4
    )
    imaging = torch.randn(2, 1, 64, 64)
    microbial = torch.randn(2, 50)
    transcriptomic = torch.randn(2, 80)
    logits, out = model(imaging, microbial, transcriptomic)
    assert logits.shape == (2, 1)
    assert out.shape == (2, 1)