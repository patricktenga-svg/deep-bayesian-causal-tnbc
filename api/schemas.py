"""Pydantic schemas for the API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    imaging: List[List[List[List[float]]]] = Field(
        ..., description="Shape [N, C, H, W]"
    )
    microbial: List[List[float]] = Field(..., description="Shape [N, M]")
    transcriptomic: List[List[float]] = Field(..., description="Shape [N, G]")
    return_causal: bool = True


class PredictResponse(BaseModel):
    probabilities: List[float]
    uncertainties: List[float]
    causal_effects: Optional[dict] = None