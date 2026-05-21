from __future__ import annotations
from .base import PredictionRequest, PredictionResult

class LearnedPredictor:
    is_async = False
    uses_modality = False
    uses_prompting = False
    uses_history = False

    def __init__(self, name: str, variant: str, checkpoint: str | None = None):
        self.name = name
        self.variant = variant
        self.checkpoint = checkpoint

    def predict(self, req: PredictionRequest) -> PredictionResult:
        return PredictionResult(
            pred_theta=[float("nan")] * req.cell.k,
            pred_omega=[float("nan")] * req.cell.k,
            success=False,
            error=f"learned predictor {self.variant} not implemented",
        )

    async def apredict(self, req):
        return self.predict(req)
