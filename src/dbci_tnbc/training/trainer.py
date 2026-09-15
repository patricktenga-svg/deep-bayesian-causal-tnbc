"""Training and evaluation loop."""
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import vae_loss
from .metrics import compute_metrics


class Trainer:
    """Handles training, evaluation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        logger=None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.best_metric = -float("inf")

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def train_epoch(self, loader: DataLoader, vae_weight: float = 0.1) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "pcr_loss": 0.0, "vae_loss": 0.0}
        n = 0
        for batch in tqdm(loader, desc="train", leave=False):
            imaging = batch["imaging"].to(self.device)
            microbial = batch["microbial"].to(self.device)
            transcriptomic = batch["transcriptomic"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            pcr_logits, _ = self.model(imaging, microbial, transcriptomic)
            pcr_loss = self.criterion(pcr_logits.squeeze(-1), labels)

            recon_m, mu_m, logvar_m, _ = self.model.microbial_vae(microbial)
            recon_t, mu_t, logvar_t, _ = self.model.transcriptomic_vae(transcriptomic)
            vm = vae_loss(recon_m, microbial, mu_m, logvar_m)
            vt = vae_loss(recon_t, transcriptomic, mu_t, logvar_t)
            loss = pcr_loss + vae_weight * (vm + vt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            totals["loss"] += loss.item()
            totals["pcr_loss"] += pcr_loss.item()
            totals["vae_loss"] += (vm + vt).item()
            n += 1

        self.scheduler.step()
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_probs, all_labels, all_stds = [], [], []
        for batch in loader:
            imaging = batch["imaging"].to(self.device)
            microbial = batch["microbial"].to(self.device)
            transcriptomic = batch["transcriptomic"].to(self.device)
            labels = batch["label"].to(self.device)

            _, pcr_mean, pcr_std = self.model(
                imaging, microbial, transcriptomic, return_uncertainty=True
            )
            all_probs.extend(pcr_mean.squeeze(-1).cpu().numpy().tolist())
            all_stds.extend(pcr_std.squeeze(-1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        metrics = compute_metrics(all_labels, all_probs)
        metrics["mean_uncertainty"] = float(sum(all_stds) / max(len(all_stds), 1))
        metrics["predictions"] = all_probs
        metrics["labels"] = all_labels
        metrics["uncertainties"] = all_stds
        return metrics

    def save_checkpoint(self, name: str = "best_model.pt", **extra):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                **extra,
            },
            path,
        )
        self._log(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._log(f"Loaded checkpoint from {path}")