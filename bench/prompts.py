from __future__ import annotations

import json
import numpy as np

from .simulator import PendulumParams

SYSTEM_NOCOT = (
    "You are a careful physicist predicting the future state of a "
    "k-pendulum (a chain of k point masses on massless rods, hinged in a "
    "planar chain, swinging under gravity with viscous damping). "
    "Respond with ONLY a single JSON object on one line, no prose, no "
    "code fences. Schema: {\"theta\": [..k floats..], \"omega\": [..k floats..]} "
    "where theta_i is in radians (link i measured from straight-down, "
    "positive counter-clockwise) and omega_i is angular velocity in rad/s."
)

SYSTEM_COT = (
    "You are a careful physicist predicting the future state of a "
    "k-pendulum (a chain of k point masses on massless rods, hinged in a "
    "planar chain, swinging under gravity with viscous damping). "
    "Reason step-by-step about the dynamics, then output your final answer. "
    "Wrap the final answer in <answer>...</answer> tags containing ONLY a "
    "single JSON object: {\"theta\": [..k floats..], \"omega\": [..k floats..]}."
)

def _format_constants(p: PendulumParams | None) -> str:
    if p is None:
        return ("Physical constants: HIDDEN. The numerical values of gravity, "
                "rod lengths, point masses, and damping are NOT disclosed. "
                "You must infer dynamics from the initial conditions alone.")
    return (
        f"Physical constants:\n"
        f"  g (gravity, m/s^2): {p.g}\n"
        f"  L (rod lengths, m): {p.L.tolist()}\n"
        f"  m (point masses, kg): {p.m.tolist()}\n"
        f"  damping (viscous coefficient on omega): {p.damping}\n"
    )

def build_user_prompt(req, k: int, modality: str) -> str:
    state0 = np.asarray(req.state0)
    theta0 = state0[:k].tolist()
    omega0 = state0[k:].tolist()

    constants_block = _format_constants(req.disclosed_params)

    if modality == "coords":
        repr_block = (
            f"Initial state at t=0:\n"
            f"  theta_0 (rad): {theta0}\n"
            f"  omega_0 (rad/s): {omega0}\n"
        )
    elif modality == "images":
        repr_block = (
            "Initial state is shown in the attached image. Pivot is the white "
            "dot at center; rods extend to colored bobs. Angles are measured "
            "from straight-down, positive counter-clockwise. Initial angular "
            "velocities are all reported below in case the image alone is "
            "insufficient:\n"
            f"  omega_0 (rad/s): {omega0}\n"
        )
    elif modality == "images_coords":
        repr_block = (
            "Initial state is shown in the attached image AND given numerically.\n"
            f"  theta_0 (rad): {theta0}\n"
            f"  omega_0 (rad/s): {omega0}\n"
        )
    else:
        raise ValueError(f"unknown modality {modality}")

    return (
        f"Number of links k = {k}\n"
        f"{constants_block}"
        f"{repr_block}"
        f"Predict the state at t = {req.horizon} seconds.\n"
        f"Return theta and omega as length-{k} JSON arrays."
    )

ANSWER_TAG_OPEN = "<answer>"
ANSWER_TAG_CLOSE = "</answer>"

def parse_response(text: str, k: int, used_cot: bool) -> tuple[list[float], list[float], str]:
    cot_text = ""
    payload = text.strip()
    if used_cot:
        i = payload.find(ANSWER_TAG_OPEN)
        j = payload.rfind(ANSWER_TAG_CLOSE)
        if i != -1 and j != -1 and j > i:
            cot_text = payload[:i].strip()
            payload = payload[i + len(ANSWER_TAG_OPEN):j].strip()

    if payload.startswith("```"):
        payload = payload.strip("`")
        nl = payload.find("\n")
        if nl != -1:
            payload = payload[nl + 1:]
        if payload.endswith("```"):
            payload = payload[:-3]
        payload = payload.strip()

    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    obj = json.loads(payload[start:end + 1])

    theta = obj.get("theta")
    omega = obj.get("omega")
    if not isinstance(theta, list) or not isinstance(omega, list):
        raise ValueError(f"theta/omega missing or wrong type: {obj}")
    if len(theta) != k or len(omega) != k:
        raise ValueError(f"expected length {k}, got theta={len(theta)} omega={len(omega)}")
    theta = [float(x) for x in theta]
    omega = [float(x) for x in omega]
    return theta, omega, cot_text
