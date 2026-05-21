from __future__ import annotations

import time
import numpy as np

from ..simulator import integrate, PendulumParams
from .base import PredictionRequest, PredictionResult

class NumericalPredictor:
    is_async = False
    uses_modality = False
    uses_prompting = False
    uses_history = False

    def __init__(self, name: str, method: str, step: float = 0.01):
        self.name = name
        self.method = method
        self.step = step

    def predict(self, req: PredictionRequest) -> PredictionResult:
        p: PendulumParams = req.params
        t0 = time.perf_counter()
        _times, states = integrate(req.state0, p, t_end=req.horizon,
                                   dt=self.step, method=self.method,
                                   t_start=0.0)
        latency = time.perf_counter() - t0
        final = states[-1]
        k = p.k
        return PredictionResult(
            pred_theta=final[:k].tolist(),
            pred_omega=final[k:].tolist(),
            raw_response="",
            latency_s=latency,
            success=True,
        )

    async def apredict(self, req):
        return self.predict(req)
