"""ONNX Runtime predictor (CPU/GPU)."""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class OnnxPredictor:
    """Fast ONNX-based inference with optional quantization."""

    def __init__(
        self,
        onnx_path: str | Path,
        providers: Optional[List[str]] = None,
    ):
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        if providers is None:
            providers = ort.get_available_providers()
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def predict(
        self,
        imaging: np.ndarray,
        microbial: np.ndarray,
        transcriptomic: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Predict pCR probability."""
        imaging = imaging.astype(np.float32)
        microbial = microbial.astype(np.float32)
        transcriptomic = transcriptomic.astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_names[0]: imaging,
                self.input_names[1]: microbial,
                self.input_names[2]: transcriptomic,
            },
        )[0]

        return {"probability": outputs.squeeze(-1).tolist()}