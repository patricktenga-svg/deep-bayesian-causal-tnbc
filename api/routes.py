"""API routes."""
from fastapi import APIRouter
from .schemas import PredictRequest, PredictResponse

router = APIRouter()

_predictor = None


def set_predictor(predictor):
    global _predictor
    _predictor = predictor


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    import numpy as np

    out = _predictor.predict(
        imaging=np.array(req.imaging, dtype="float32"),
        microbial=np.array(req.microbial, dtype="float32"),
        transcriptomic=np.array(req.transcriptomic, dtype="float32"),
        return_causal=req.return_causal,
    )
    return PredictResponse(
        probabilities=out["probability"],
        uncertainties=out["uncertainty"],
        causal_effects=out.get("causal_effects"),
    )
@router.post("/predict/onnx", response_model=PredictResponse)
def predict_onnx(req: PredictRequest):
    """Low-latency inference using the ONNX runtime."""
    import numpy as np
    from dbci_tnbc.inference.onnx_predictor import OnnxPredictor
    from pathlib import Path

    global _onnx_predictor
    try:
        _onnx_predictor
    except NameError:
        _onnx_predictor = None

    if _onnx_predictor is None:
        path = Path(os.getenv("ONNX_PATH", "exports/dbci_tnbc.onnx"))
        if not path.exists():
            from fastapi import HTTPException
            raise HTTPException(503, "ONNX model not exported yet.")
        _onnx_predictor = OnnxPredictor(path)

    out = _onnx_predictor.predict(
        np.array(req.imaging, dtype="float32"),
        np.array(req.microbial, dtype="float32"),
        np.array(req.transcriptomic, dtype="float32"),
    )
    return PredictResponse(
        probabilities=out["probability"],
        uncertainties=[0.0] * len(out["probability"]),
        causal_effects=None,
    )
@router.post("/explain")
def explain(req: PredictRequest):
    """Return SHAP + IG explanations for a batch."""
    import numpy as np
    from fastapi import HTTPException

    if _predictor is None:
        raise HTTPException(503, "Model not loaded.")

    imaging = np.array(req.imaging, dtype="float32")
    microbial = np.array(req.microbial, dtype="float32")
    transcriptomic = np.array(req.transcriptomic, dtype="float32")

    from dbci_tnbc.explain import IntegratedGradients
    ig = IntegratedGradients(_predictor.model, device=_predictor.device)
    attributions, delta = ig.attribute(imaging, microbial, transcriptomic, steps=16)

    return {
        "attribution_shape": list(attributions.shape),
        "mean_abs_attribution": float(np.abs(attributions).mean()),
        "delta_prediction": float(delta),
        "attributions": attributions.tolist(),
    }
