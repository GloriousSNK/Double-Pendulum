from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional
import numpy as np

from ..simulator import PendulumParams
from ..schema import EvalCell


@dataclass
class PredictionRequest:
    cell: EvalCell
    params: PendulumParams
    disclosed_params: Optional[PendulumParams]
    state0: np.ndarray
    horizon: float
    image_b64: Optional[str] = None
    history_times: Optional[np.ndarray] = None
    history_states: Optional[np.ndarray] = None


@dataclass
class PredictionResult:
    pred_theta: list[float]
    pred_omega: list[float]
    raw_response: str = ""
    cot_text: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: str = ""


class Predictor(Protocol):
    name: str
    is_async: bool
    uses_modality: bool
    uses_prompting: bool
    uses_history: bool

    async def apredict(self, req: PredictionRequest) -> PredictionResult: ...
    def predict(self, req: PredictionRequest) -> PredictionResult: ...
