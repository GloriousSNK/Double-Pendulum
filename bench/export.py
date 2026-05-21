from __future__ import annotations

import json
import os
import csv
from .schema import Trajectory

def write_trajectory_json(traj: Trajectory, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traj.to_dict(), f)

def write_trajectory_csv(traj: Trajectory, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    k = traj.k
    fieldnames = ["movement_id", "number_of_pendulums", "regime", "time",
                  "link", "theta", "omega", "x", "y",
                  "potential_energy", "kinetic_energy", "total_energy",
                  "g", "L", "m", "damping",
                  "theta0", "omega0"]
    consts = traj.constants
    ic = traj.initial_conditions
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ti, t in enumerate(traj.times):
            for li in range(k):
                w.writerow({
                    "movement_id": traj.movement_id,
                    "number_of_pendulums": k,
                    "regime": traj.regime,
                    "time": float(t),
                    "link": li,
                    "theta": float(traj.states[ti, li]),
                    "omega": float(traj.states[ti, k + li]),
                    "x": float(traj.xy[ti, li, 0]),
                    "y": float(traj.xy[ti, li, 1]),
                    "potential_energy": float(traj.pe[ti]),
                    "kinetic_energy": float(traj.ke[ti]),
                    "total_energy": float(traj.e_total[ti]),
                    "g": consts["g"],
                    "L": consts["L"][li],
                    "m": consts["m"][li],
                    "damping": consts["damping"],
                    "theta0": ic["theta0"][li],
                    "omega0": ic["omega0"][li],
                })
