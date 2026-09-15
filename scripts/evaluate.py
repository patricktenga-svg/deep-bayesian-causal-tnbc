"""Evaluation entry point."""
import argparse
from dbci_tnbc.utils.config import load_config
from dbci_tnbc.utils.synthetic import generate_synthetic_data
from dbci_tnbc.data.loaders import create_dataloaders
from dbci_tnbc.models.framework import DeepBayesianCausalFramework
from dbci_tnbc.training.trainer import Trainer
from dbci_tnbc.visualization.plots import plot_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    data = generate_synthetic_data(
        num_samples=cfg.data["num_samples"],
        imaging_size=cfg.data["imaging_size"],
        microbial_dim=cfg.data["microbial_dim"],
        transcriptomic_dim=cfg.data["transcriptomic_dim"],
    )
    _, test_loader = create_dataloaders(data, batch_size=cfg.data["batch_size"])

    model = DeepBayesianCausalFramework(
        imaging_channels=cfg.data["imaging_channels"],
        microbial_dim=cfg.data["microbial_dim"],
        transcriptomic_dim=cfg.data["transcriptomic_dim"],
        latent_dim=cfg.model["latent_dim"],
        hidden_dim=cfg.model["hidden_dim"],
        num_mc_samples=cfg.model["num_mc_samples"],
        image_size=cfg.data["imaging_size"],
    )
    trainer = Trainer(model, device="cpu", checkpoint_dir="checkpoints")
    trainer.load_checkpoint(args.checkpoint)

    metrics = trainer.evaluate(test_loader)
    print({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    plot_results(metrics, save_path="results.png")


if __name__ == "__main__":
    main()