"""Export trained model to ONNX (and TorchScript)."""
import argparse
from pathlib import Path

import torch

from dbci_tnbc.utils.config import load_config
from dbci_tnbc.models.framework import DeepBayesianCausalFramework


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--output-dir", default="exports")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--torchscript", action="store_true")
    return p.parse_args()


class ExportWrapper(torch.nn.Module):
    """Wraps the framework to return a single tensor for ONNX export."""

    def __init__(self, model: DeepBayesianCausalFramework):
        super().__init__()
        self.model = model

    def forward(self, imaging, microbial, transcriptomic):
        logits, _ = self.model(imaging, microbial, transcriptomic)
        return torch.sigmoid(logits)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild model
    model = DeepBayesianCausalFramework(
        imaging_channels=cfg.data["imaging_channels"],
        microbial_dim=cfg.data["microbial_dim"],
        transcriptomic_dim=cfg.data["transcriptomic_dim"],
        latent_dim=cfg.model["latent_dim"],
        hidden_dim=cfg.model["hidden_dim"],
        num_mc_samples=cfg.model["num_mc_samples"],
        image_size=cfg.data["imaging_size"],
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = ExportWrapper(model).eval()

    # Dummy inputs
    img_size = cfg.data["imaging_size"]
    dummy_img = torch.randn(1, cfg.data["imaging_channels"], img_size, img_size)
    dummy_mic = torch.randn(1, cfg.data["microbial_dim"])
    dummy_trn = torch.randn(1, cfg.data["transcriptomic_dim"])

    onnx_path = out_dir / "dbci_tnbc.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_mic, dummy_trn),
        str(onnx_path),
        input_names=["imaging", "microbial", "transcriptomic"],
        output_names=["pcr_probability"],
        dynamic_axes={
            "imaging": {0: "batch"},
            "microbial": {0: "batch"},
            "transcriptomic": {0: "batch"},
            "pcr_probability": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[ONNX] exported → {onnx_path}")

    if args.torchscript:
        ts_path = out_dir / "dbci_tnbc.pt"
        traced = torch.jit.trace(wrapper, (dummy_img, dummy_mic, dummy_trn))
        traced.save(str(ts_path))
        print(f"[TorchScript] exported → {ts_path}")

    # Verify ONNX
    try:
        import onnx
        onnx.checker.check_model(str(onnx_path))
        print("[ONNX] model check passed ✓")
    except ImportError:
        print("[ONNX] install `onnx` for verification.")

    # Optional: quantize
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = out_dir / "dbci_tnbc_int8.onnx"
        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QUInt8)
        print(f"[ONNX] dynamic-quantized → {quant_path}")
    except ImportError:
        print("[ONNX] install `onnxruntime` for dynamic quantization.")


if __name__ == "__main__":
    main()