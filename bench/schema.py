from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json
import numpy as np

def _np2list(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _np2list(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np2list(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o

@dataclass
class Trajectory:
    movement_id: str
    k: int
    regime: str
    times: np.ndarray
    states: np.ndarray
    xy: np.ndarray
    ke: np.ndarray
    pe: np.ndarray
    e_total: np.ndarray
    constants: dict
    initial_conditions: dict

    def slice_at(self, t: float) -> dict:
        idx = int(np.argmin(np.abs(self.times - t)))
        k = self.k
        return {
            "t": float(self.times[idx]),
            "theta": self.states[idx, :k].tolist(),
            "omega": self.states[idx, k:].tolist(),
            "xy":    self.xy[idx].tolist(),
            "ke":    float(self.ke[idx]),
            "pe":    float(self.pe[idx]),
            "e":     float(self.e_total[idx]),
        }

    def to_dict(self) -> dict:
        return _np2list({
            "movement_id": self.movement_id,
            "number_of_pendulums": self.k,
            "regime": self.regime,
            "time": self.times,
            "initial_conditions": self.initial_conditions,
            "constants": self.constants,
            "x": self.xy[..., 0],
            "y": self.xy[..., 1],
            "potential_energy": self.pe,
            "total_energy": self.e_total,
            "kinetic_energy": self.ke,
            "theta": self.states[:, :self.k],
            "omega": self.states[:, self.k:],
        })

@dataclass(frozen=True)
class EvalCell:
    model_name: str
    k: int
    regime: str
    modality: str
    horizon: float
    prompting: str
    movement_id: str

    def slug(self) -> str:
        return (f"{self.model_name}__k{self.k}__{self.regime}__"
                f"{self.modality}__h{self.horizon:g}__{self.prompting}__"
                f"{self.movement_id}")

@dataclass
class Prediction:
    cell: EvalCell
    pred_theta: list[float]
    pred_omega: list[float]
    raw_response: str = ""
    cot_text: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cell"] = asdict(self.cell)
        return _np2list(d)
