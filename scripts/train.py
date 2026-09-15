"""Training entry point."""
import argparse
from pathlib import Path

from dbci_tnbc.utils.config import load_config
from dbci_tnbc.utils.logging import get_logger
from dbci_tnbc.utils.seed import set_seed
from dbci_tnbc.utils.synthetic import generate_synthetic_data
from dbci_tnbc.data.loaders import create_dataloaders
from dbci_tnbc.models.framework import DeepBayesianCausalFramework
from dbci_tnbc.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    logger = get_logger(
        "dbci_tnbc.train",
        level="INFO",
        log_file=Path(cfg.training.get("log_dir", "logs")) / "train.log",
    )
    set_seed(cfg.project.get("seed", 42))

    logger.info("Generating synthetic data (replace with real loaders).")
    data = generate_synthetic_data(
        num_samples=cfg.data["num_samples"],
        imaging_size=cfg.data["imaging_size"],
        imaging_channels=cfg.data["imaging_channels"],
        microbial_dim=cfg.data["microbial_dim"],
        transcriptomic_dim=cfg.data["transcriptomic_dim"],
        seed=cfg.project.get("seed", 42),
    )

    train_loader, test_loader = create_dataloaders(
        data,
        batch_size=cfg.data["batch_size"],
        test_size=cfg.data["test_size"],
        num_workers=cfg.data.get("num_workers", 0),
        seed=cfg.project.get("seed", 42),
    )

    model = DeepBayesianCausalFramework(
        imaging_channels=cfg.data["imaging_channels"],
        microbial_dim=cfg.data["microbial_dim"],
        transcriptomic_dim=cfg.data["transcriptomic_dim"],
        latent_dim=cfg.model["latent_dim"],
        hidden_dim=cfg.model["hidden_dim"],
        num_mc_samples=cfg.model["num_mc_samples"],
        image_size=cfg.data["imaging_size"],
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model,
        learning_rate=cfg.training["learning_rate"],
        weight_decay=cfg.training["weight_decay"],
        device=cfg.project.get("device", "cpu"),
        checkpoint_dir=cfg.training.get("checkpoint_dir", "checkpoints"),
        logger=logger,
    )

    epochs = args.epochs or cfg.training["epochs"]
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(train_loader, vae_weight=cfg.training["vae_weight"])
        logger.info(
            f"Epoch {epoch}/{epochs} | loss={metrics['loss']:.4f} "
            f"pcr={metrics['pcr_loss']:.4f} vae={metrics['vae_loss']:.4f}"
        )

    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: { {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))} }")
    trainer.save_checkpoint("best_model.pt", metrics=test_metrics)


if __name__ == "__main__":
    main()