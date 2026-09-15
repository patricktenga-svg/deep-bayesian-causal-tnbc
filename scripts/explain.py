"""Generate SHAP and Integrated Gradients explanations."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dbci_tnbc.utils.config import load_config
from dbci_tnbc.utils.synthetic import generate_synthetic_data
from dbci_tnbc.models.framework import DeepBayesianCausalFramework
from dbci_tnbc.explain import ShapExplainer, IntegratedGradients


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--out", default="explanations")
    p.add_argument("--n-background", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_data(
        num_samples=100,
        imaging_size=cfg.data["imaging_size"],
        microbial_dim=cfg.data["microbial_dim"],
        transcriptomic_dim=cfg.data["transcriptomic_dim"],
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
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model_state_dict"])
    model.eval()

    # --- Omics SHAP ---
    shap_exp = ShapExplainer(model, device="cpu")
    bg_m = data["microbial"][: args.n_background]
    bg_t = data["transcriptomic"][: args.n_background]
    s_m = data["microbial"][args.n_background : args.n_background + 1]
    s_t = data["transcriptomic"][args.n_background : args.n_background + 1]
    dummy_img = np.zeros((1, 1, cfg.data["imaging_size"], cfg.data["imaging_size"]), dtype="float32")

    shap_vals = shap_exp.explain_omics(bg_m, bg_t, s_m, s_t, dummy_img)

    np.save(out_dir / "shap_microbial.npy", shap_vals["microbial"])
    np.save(out_dir / "shap_transcriptomic.npy", shap_vals["transcriptomic"])

    # Save plot (top features)
    import pandas as pd
    mic_df = pd.DataFrame({
        "feature": [f"M{i}" for i in range(shap_vals["microbial"].shape[1])],
        "shap": shap_vals["microbial"][0],
    }).sort_values("shap", key=np.abs, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(mic_df["feature"], mic_df["shap"])
    ax.set_title("Top microbial SHAP features")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "shap_microbial_top.png")
    plt.close()

    # --- Imaging IG ---
    ig = IntegratedGradients(model, device="cpu")
    img = data["imaging"][args.n_background : args.n_background + 1]
    attributions, delta = ig.attribute(img, s_m, s_t, steps=32)

    np.save(out_dir / "ig_attributions.npy", attributions)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img[0, 0], cmap="gray"); axes[0].set_title("Input MRI")
    axes[1].imshow(attributions[0, 0], cmap="RdBu_r"); axes[1].set_title("IG Attribution")
    axes[2].imshow(np.abs(attributions[0, 0]), cmap="hot"); axes[2].set_title("|Attribution|")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "ig_imaging.png")
    plt.close()

    print(f"Δ prediction (x − baseline) = {delta:+.4f}")
    print(f"Saved explanations to {out_dir}")


if __name__ == "__main__":
    import torch
    main()