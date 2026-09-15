"""Experiment tracking backends (W&B, TensorBoard, MLflow-compatible)."""
from pathlib import Path
from typing import Any, Dict, Optional
import os


class Tracker:
    """Unified tracker wrapping W&B, TensorBoard, or a no-op logger."""

    def __init__(
        self,
        backend: str = "none",  # "wandb" | "tensorboard" | "none"
        project: str = "dbci-tnbc",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
    ):
        self.backend = backend.lower()
        self.run = None
        self.writer = None

        if self.backend == "wandb":
            try:
                import wandb
                self.run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    reinit=True,
                )
            except ImportError:
                print("[Tracker] wandb not installed — falling back to none.")
                self.backend = "none"

        elif self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("[Tracker] tensorboard not installed — falling back to none.")
                self.backend = "none"

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.backend == "wandb" and self.run is not None:
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "tensorboard" and self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step or 0)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model"):
        if self.backend == "wandb" and self.run is not None:
            import wandb
            art = wandb.Artifact(name, type=artifact_type)
            art.add_file(path)
            self.run.log_artifact(art)

    def finish(self):
        if self.backend == "wandb" and self.run is not None:
            self.run.finish()
        if self.backend == "tensorboard" and self.writer is not None:
            self.writer.close()