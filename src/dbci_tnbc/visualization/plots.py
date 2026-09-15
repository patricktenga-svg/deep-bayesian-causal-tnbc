"""Visualization helpers."""
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_results(results: Dict, save_path: str = "results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    axes[0].bar(list(metrics.keys()), list(metrics.values()))
    axes[0].set_title("Metrics")
    axes[0].tick_params(axis="x", rotation=45)

    uncertainties = np.array(results.get("uncertainties", []))
    if uncertainties.size:
        axes[1].hist(uncertainties, bins=20, alpha=0.7)
        axes[1].set_title("Uncertainty Distribution")
        axes[1].set_xlabel("std")

    labels = results.get("labels", [])
    preds = (np.array(results.get("predictions", [])) > 0.5).astype(int)
    if len(labels) and len(preds):
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        axes[2].imshow(cm, cmap="Blues")
        axes[2].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()