from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass
class PendulumParams:
    k: int
    L: np.ndarray
    m: np.ndarray
    g: float
    damping: float = 0.0

    @classmethod
    def make(cls, k: int, L, m, g: float, damping: float = 0.0) -> "PendulumParams":
        L = np.asarray(L, dtype=float)
        m = np.asarray(m, dtype=float)
        if L.ndim == 0 or len(L) == 1:
            L = np.full(k, float(L if L.ndim == 0 else L[0]))
        if m.ndim == 0 or len(m) == 1:
            m = np.full(k, float(m if m.ndim == 0 else m[0]))
        assert len(L) == k and len(m) == k, f"L/m length must match k={k}"
        return cls(k=k, L=L, m=m, g=float(g), damping=float(damping))

    def to_dict(self) -> dict:
        return {"k": self.k, "L": self.L.tolist(), "m": self.m.tolist(),
                "g": self.g, "damping": self.damping}

def _mass_coeffs(p: PendulumParams) -> tuple[np.ndarray, np.ndarray]:
    k = p.k
    suffix_mass = np.cumsum(p.m[::-1])[::-1]
    a = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            a[i, j] = suffix_mass[max(i, j)] * p.L[i] * p.L[j]
    b = suffix_mass * p.L
    return a, b

def dynamics(state: np.ndarray, p: PendulumParams,
             _cache: dict | None = None) -> np.ndarray:
    k = p.k
    theta = state[:k]
    omega = state[k:]

    if _cache is not None and "ab" in _cache:
        a, b = _cache["ab"]
    else:
        a, b = _mass_coeffs(p)
        if _cache is not None:
            _cache["ab"] = (a, b)

    dth = theta[:, None] - theta[None, :]
    M = a * np.cos(dth)
    C = a * np.sin(dth)
    rhs = -(C @ (omega ** 2)) - p.g * b * np.sin(theta) - p.damping * omega
    alpha = np.linalg.solve(M, rhs)
    return np.concatenate([omega, alpha])

def step_euler(state, p, dt, cache):
    return state + dt * dynamics(state, p, cache)

def step_rk4(state, p, dt, cache):
    k1 = dynamics(state, p, cache)
    k2 = dynamics(state + 0.5 * dt * k1, p, cache)
    k3 = dynamics(state + 0.5 * dt * k2, p, cache)
    k4 = dynamics(state + dt * k3, p, cache)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def step_leapfrog(state, p, dt, cache):
    k = p.k
    theta = state[:k]
    omega = state[k:]
    a = dynamics(state, p, cache)[k:]
    theta_new = theta + dt * omega + 0.5 * dt * dt * a
    state_mid = np.concatenate([theta_new, omega])
    a_new = dynamics(state_mid, p, cache)[k:]
    omega_new = omega + 0.5 * dt * (a + a_new)
    return np.concatenate([theta_new, omega_new])

INTEGRATORS = {"euler": step_euler, "rk4": step_rk4, "leapfrog": step_leapfrog}

def integrate(state0: np.ndarray, p: PendulumParams,
              t_end: float, dt: float, method: str = "rk4",
              t_start: float = 0.0,
              record_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    step = INTEGRATORS[method]
    cache: dict = {}
    state = state0.astype(float).copy()
    n_total = max(1, int(np.ceil((t_end - t_start) / dt)))
    times_list = [t_start]
    states_list = [state.copy()]
    t = t_start
    for i in range(n_total):
        h = min(dt, t_end - t)
        if h <= 0:
            break
        state = step(state, p, h, cache)
        t += h
        if (i + 1) % record_every == 0 or i == n_total - 1:
            times_list.append(t)
            states_list.append(state.copy())
    return np.asarray(times_list), np.asarray(states_list)

def bob_positions(theta: np.ndarray, p: PendulumParams) -> np.ndarray:
    x = np.cumsum(p.L * np.sin(theta))
    y = -np.cumsum(p.L * np.cos(theta))
    return np.stack([x, y], axis=-1)

def bob_positions_batch(thetas: np.ndarray, p: PendulumParams) -> np.ndarray:
    x = np.cumsum(p.L[None, :] * np.sin(thetas), axis=1)
    y = -np.cumsum(p.L[None, :] * np.cos(thetas), axis=1)
    return np.stack([x, y], axis=-1)

def kinetic_energy(state: np.ndarray, p: PendulumParams) -> float:
    k = p.k
    theta = state[:k]
    omega = state[k:]
    x_dot = np.cumsum(p.L * np.cos(theta) * omega)
    y_dot = np.cumsum(p.L * np.sin(theta) * omega)
    v2 = x_dot ** 2 + y_dot ** 2
    return 0.5 * float(np.sum(p.m * v2))

def potential_energy(state: np.ndarray, p: PendulumParams) -> float:
    k = p.k
    theta = state[:k]
    y = -np.cumsum(p.L * np.cos(theta))
    return float(np.sum(p.m * p.g * y))

def total_energy(state: np.ndarray, p: PendulumParams) -> float:
    return kinetic_energy(state, p) + potential_energy(state, p)

def energy_batch(states: np.ndarray, p: PendulumParams) -> dict:
    T = states.shape[0]
    ke = np.empty(T); pe = np.empty(T)
    for i in range(T):
        ke[i] = kinetic_energy(states[i], p)
        pe[i] = potential_energy(states[i], p)
    return {"KE": ke, "PE": pe, "E": ke + pe}
