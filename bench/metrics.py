from __future__ import annotations

import numpy as np
from .simulator import (PendulumParams, bob_positions, bob_positions_batch,
                        kinetic_energy, potential_energy, total_energy)

def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

def angle_error(pred_theta, true_theta):
    diff = wrap_angle(np.asarray(pred_theta) - np.asarray(true_theta))
    return np.abs(diff)

def coord_error(pred_state, true_state, p: PendulumParams):
    k = p.k
    pos_pred = bob_positions(pred_state[:k], p)
    pos_true = bob_positions(true_state[:k], p)
    return np.linalg.norm(pos_pred - pos_true, axis=-1)

def direction_velocity_error(pred_state, true_state, p: PendulumParams):
    k = p.k
    pred_w = pred_state[k:]
    true_w = true_state[k:]
    same_sign = np.sign(pred_w) == np.sign(true_w)
    mag_err = np.abs(pred_w - true_w)
    return {"sign_match": same_sign.astype(float), "mag_error": mag_err}

def energy_errors(pred_state, true_state, p: PendulumParams):
    return {
        "delta_KE": kinetic_energy(pred_state, p) - kinetic_energy(true_state, p),
        "delta_PE": potential_energy(pred_state, p) - potential_energy(true_state, p),
        "delta_E":  total_energy(pred_state, p) - total_energy(true_state, p),
    }

def long_run_deviance(pred_traj, true_traj, p: PendulumParams):
    k = p.k
    pos_pred = bob_positions_batch(pred_traj[:, :k], p)
    pos_true = bob_positions_batch(true_traj[:, :k], p)
    return float(np.mean(np.linalg.norm(pos_pred - pos_true, axis=-1)))

def time_to_divergence(pred_traj, true_traj, p: PendulumParams,
                       times: np.ndarray, threshold_rad: float = 0.5,
                       threshold_xy: float = 0.5) -> dict:
    k = p.k
    ang = angle_error(pred_traj[:, :k], true_traj[:, :k])
    pos_pred = bob_positions_batch(pred_traj[:, :k], p)
    pos_true = bob_positions_batch(true_traj[:, :k], p)
    xy = np.linalg.norm(pos_pred - pos_true, axis=-1)
    exceeds = (ang.max(axis=1) > threshold_rad) | (xy.max(axis=1) > threshold_xy)
    idx = np.argmax(exceeds) if exceeds.any() else -1
    return {
        "t_div_rad": float(times[idx]) if exceeds.any() else float("inf"),
        "threshold_rad": threshold_rad,
        "threshold_xy": threshold_xy,
    }

def all_pointwise_metrics(pred_state, true_state, p: PendulumParams) -> dict:
    k = p.k
    cerr = coord_error(pred_state, true_state, p)
    aerr = angle_error(pred_state[:k], true_state[:k])
    dv = direction_velocity_error(pred_state, true_state, p)
    en = energy_errors(pred_state, true_state, p)
    return {
        "coord_error_per_bob":  cerr.tolist(),
        "coord_error_mean":     float(cerr.mean()),
        "coord_error_max":      float(cerr.max()),
        "angle_error_per_link": aerr.tolist(),
        "angle_error_mean":     float(aerr.mean()),
        "sign_match_per_link":  dv["sign_match"].tolist(),
        "omega_mag_error":      dv["mag_error"].tolist(),
        "delta_KE":             en["delta_KE"],
        "delta_PE":             en["delta_PE"],
        "delta_E":              en["delta_E"],
    }
