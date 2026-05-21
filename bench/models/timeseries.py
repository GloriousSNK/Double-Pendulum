from __future__ import annotations

import asyncio
import os
import time
import numpy as np

from .base import PredictionRequest, PredictionResult

try:
    import httpx
    _HAVE_HTTPX = True
except ImportError:
    _HAVE_HTTPX = False


VARIANTS = {"chronos", "timesfm", "moirai"}


def _resolve_endpoint_and_key(model_name: str) -> tuple[str, str]:
    upper = model_name.upper().replace("-", "_").replace(".", "_")
    ep = os.getenv(f"AZURE_{upper}_ENDPOINT")
    key = os.getenv(f"AZURE_{upper}_KEY")
    if not ep or not key:
        raise RuntimeError(
            f"Missing endpoint/key for time-series model {model_name!r}. "
            f"Set AZURE_{upper}_ENDPOINT (scoring URI) and AZURE_{upper}_KEY "
            f"in .env. Each AzureML real-time endpoint has its own URI + key."
        )
    return ep, key


def _resample(times: np.ndarray, values: np.ndarray, dt: float,
              t_start: float, t_end: float) -> np.ndarray:
    grid = np.arange(t_start, t_end + 1e-9, dt)
    return np.interp(grid, times, values)


def _build_chronos_body(context_channels: np.ndarray, prediction_length: int) -> dict:
    return {
        "inputs": {
            "target": [ch.tolist() for ch in context_channels],
            "prediction_length": prediction_length,
            "num_samples": 20,
        }
    }


def _build_timesfm_body(context_channels: np.ndarray, prediction_length: int) -> dict:
    return {
        "inputs": [
            {"context": ch.tolist(), "horizon_len": prediction_length, "freq": 0}
            for ch in context_channels
        ]
    }


def _build_moirai_body(context_channels: np.ndarray, prediction_length: int) -> dict:
    return {
        "context": context_channels.tolist(),
        "prediction_length": prediction_length,
        "patch_size": "auto",
        "num_samples": 20,
    }


def _parse_chronos(resp_json, n_channels: int, prediction_length: int) -> np.ndarray:
    outs = resp_json.get("outputs") or resp_json.get("predictions") or resp_json
    if isinstance(outs, dict) and "mean" in outs:
        outs = [outs]
    arr = np.array([np.asarray(o["mean"]) for o in outs])
    if arr.shape != (n_channels, prediction_length):
        raise ValueError(f"chronos response shape {arr.shape} != "
                         f"({n_channels}, {prediction_length})")
    return arr


def _parse_timesfm(resp_json, n_channels: int, prediction_length: int) -> np.ndarray:
    if isinstance(resp_json, dict) and "point_forecast" in resp_json:
        pf = np.asarray(resp_json["point_forecast"])
        if pf.ndim == 1:
            pf = pf.reshape(1, -1)
    else:
        pf = np.array([np.asarray(r["point_forecast"]) for r in resp_json])
    if pf.shape != (n_channels, prediction_length):
        raise ValueError(f"timesfm response shape {pf.shape} != "
                         f"({n_channels}, {prediction_length})")
    return pf


def _parse_moirai(resp_json, n_channels: int, prediction_length: int) -> np.ndarray:
    if "median" in resp_json:
        pred = np.asarray(resp_json["median"])
    elif "mean" in resp_json:
        pred = np.asarray(resp_json["mean"])
    elif "samples" in resp_json:
        pred = np.asarray(resp_json["samples"]).mean(axis=0)
    else:
        raise ValueError(f"unrecognized moirai response keys: {list(resp_json)}")
    if pred.shape != (n_channels, prediction_length):
        raise ValueError(f"moirai response shape {pred.shape} != "
                         f"({n_channels}, {prediction_length})")
    return pred


_BUILDERS = {
    "chronos": _build_chronos_body,
    "timesfm": _build_timesfm_body,
    "moirai":  _build_moirai_body,
}
_PARSERS = {
    "chronos": _parse_chronos,
    "timesfm": _parse_timesfm,
    "moirai":  _parse_moirai,
}


class TimeSeriesPredictor:
    is_async = True
    uses_modality = False
    uses_prompting = False
    uses_history = True

    def __init__(self, name: str, variant: str, *, max_prediction_length: int = 256,
                 max_context_length: int = 512,
                 concurrency: int = 4, request_timeout: float = 120.0,
                 max_retries: int = 4):
        if not _HAVE_HTTPX:
            raise RuntimeError("httpx not installed. Run `pip install -r requirements.txt`.")
        if variant not in VARIANTS:
            raise ValueError(f"unknown time-series variant {variant!r}; "
                             f"expected one of {sorted(VARIANTS)}")
        self.name = name
        self.variant = variant
        self.max_prediction_length = max_prediction_length
        self.max_context_length = max_context_length
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout)

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _resample_context(self, req: PredictionRequest) -> tuple[np.ndarray, float, int]:
        k = req.cell.k
        n_channels = 2 * k
        horizon = req.horizon

        ht = req.history_times
        hs = req.history_states
        if ht is None or hs is None or len(ht) < 2:
            raise ValueError("time-series predictor requires history_times/states")
        ctx_seconds = float(ht[-1] - ht[0])
        if ctx_seconds <= 0:
            raise ValueError("history window has non-positive duration")

        dt = max(horizon / self.max_prediction_length,
                 ctx_seconds / self.max_context_length)
        prediction_length = max(1, int(round(horizon / dt)))
        ctx_len = max(8, min(self.max_context_length,
                             int(round(ctx_seconds / dt))))
        t_end = float(ht[-1])
        t_start = max(float(ht[0]), t_end - ctx_len * dt)

        channels = np.empty((n_channels, ctx_len))
        grid = np.linspace(t_start, t_end, ctx_len, endpoint=True)
        for ch_idx in range(n_channels):
            channels[ch_idx] = np.interp(grid, ht, hs[:, ch_idx])
        return channels, dt, prediction_length

    async def apredict(self, req: PredictionRequest) -> PredictionResult:
        await self._ensure_client()
        k = req.cell.k
        n_channels = 2 * k
        try:
            context, dt, prediction_length = self._resample_context(req)
        except Exception as e:
            return PredictionResult(
                pred_theta=[float("nan")] * k,
                pred_omega=[float("nan")] * k,
                success=False, error=f"context: {e!r}",
            )

        body = _BUILDERS[self.variant](context, prediction_length)
        try:
            ep, key = _resolve_endpoint_and_key(self.name)
        except Exception as e:
            return PredictionResult(
                pred_theta=[float("nan")] * k,
                pred_omega=[float("nan")] * k,
                success=False, error=f"config: {e}",
            )
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json"}

        last_err: BaseException | None = None
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    r = await self._client.post(ep, json=body, headers=headers)
                    r.raise_for_status()
                    resp = r.json()
                    latency = time.perf_counter() - t0
                    forecast = _PARSERS[self.variant](resp, n_channels, prediction_length)
                    final = forecast[:, -1]
                    theta = final[:k].tolist()
                    omega = final[k:].tolist()
                    return PredictionResult(
                        pred_theta=theta, pred_omega=omega,
                        raw_response=str(resp)[:2000],
                        latency_s=latency, success=True,
                    )
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt + 0.5, 30.0))

        return PredictionResult(
            pred_theta=[float("nan")] * k,
            pred_omega=[float("nan")] * k,
            success=False, error=f"api: {last_err!r}",
        )

    def predict(self, req: PredictionRequest) -> PredictionResult:
        return asyncio.run(self.apredict(req))
