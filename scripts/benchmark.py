"""Benchmark PyTorch vs. ONNX inference latency."""
import argparse
import time
import numpy as np
import torch

from dbci_tnbc.inference.predictor import Predictor
from dbci_tnbc.inference.onnx_predictor import OnnxPredictor


def timeit(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--onnx", default="exports/dbci_tnbc.onnx")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    img = np.random.randn(args.batch, 1, args.image_size, args.image_size).astype("float32")
    mic = np.random.randn(args.batch, 500).astype("float32")
    trn = np.random.randn(args.batch, 1000).astype("float32")

    pt = Predictor(
        checkpoint_path=args.checkpoint,
        device="cpu",
        imaging_channels=1,
        microbial_dim=500,
        transcriptomic_dim=1000,
        latent_dim=128,
        hidden_dim=256,
        num_mc_samples=10,
        image_size=args.image_size,
    )
    onnx_pred = OnnxPredictor(args.onnx)

    pt_ms = timeit(lambda: pt.predict(img, mic, trn, return_causal=False))
    onnx_ms = timeit(lambda: onnx_pred.predict(img, mic, trn))

    print(f"PyTorch : {pt_ms:.2f} ms/batch (B={args.batch})")
    print(f"ONNX    : {onnx_ms:.2f} ms/batch (B={args.batch})")
    print(f"Speedup : {pt_ms / onnx_ms:.2f}x")


if __name__ == "__main__":
    main()