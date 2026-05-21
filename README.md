# Pendulum Physics Benchmark for LLMs / VLMs

End-to-end harness for **"Can LLMs Predict Physics?"** (see [SPEC.md](SPEC.md)).
Includes a k-pendulum simulator, numerical baselines, Azure AI Foundry
LLM + VLM adapters, and a parallel, checkpointed evaluation runner.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env             # then fill in AZURE_AI_FOUNDRY_KEY / endpoint

# 1. Generate the ground-truth dataset (deterministic from config.yaml seed)
python scripts/generate_dataset.py                   # or --smoke for 1/cell

# 2. Run the evaluation (resumable; re-running skips completed cells)
python scripts/run_eval.py                           # all models
python scripts/run_eval.py --models rk4 euler        # baselines only
python scripts/run_eval.py --smoke                   # tiny smoke run

# 3. Aggregate per-cell checkpoints into a leaderboard
python scripts/aggregate.py
```

Outputs land in `results/`:

```
results/
  dataset/          # ground-truth trajectories (JSON; CSV with --csv)
  checkpoints/      # one JSON per (model x cell x trajectory) — resume safe
  summary/          # results_long.csv + leaderboard.csv / .json
  renders/          # (reserved)
```

## What the runner does

For every cell in the Cartesian product
**model × k ∈ {1,2,3} × regime × modality × horizon × prompting × trajectory**:

1. Pull the ground-truth trajectory from `results/dataset/`.
2. Build a prompt (text-only, image, or image+text).
3. Call the predictor (Azure model, or a local integrator).
4. Parse the JSON answer, compute all SPEC metrics against the true state
   at `t=horizon`.
5. Write a single JSON checkpoint to `results/checkpoints/`. Re-running the
   script skips any cell whose checkpoint already exists.

Across models, eval runs concurrently. Each Azure predictor holds its own
semaphore (`concurrency:` in config), so per-model rate limits are
independent.

## Configuration (`config.yaml`)

- `dataset.trajectories_per_cell`: how many trajectories per (k, regime). 20
  is a reasonable starting point; the cell count scales linearly.
- `regimes`: physics constants for `normal`, `changed_disclosed`,
  `changed_hidden`. The first two reveal the constants to the model; the
  third does not (the prompt explicitly tells the model they are hidden).
- `horizons_seconds`: [0.01, 1.0, 10.0, 60.0] per SPEC.
- `modalities`: `coords`, `images`, `images_coords`.
- `prompting`: `no_cot`, `cot` (CoT wraps the final answer in
  `<answer>{...}</answer>` tags; see `bench/prompts.py`).
- `models`: each entry has `kind` ∈ {`llm`, `numerical`, `timeseries`,
  `learned`} and is wired to the matching adapter.

## Azure AI Foundry setup

1. Create a Foundry project, deploy your five chosen models:
   `gpt-5.5`, `kimi-k2.6`, `deepseek-v4`, `qwen`, `grok-4-fast-reasoning`.
2. Copy the project endpoint (looks like
   `https://YOUR-RESOURCE.services.ai.azure.com/models`) and an API key into
   `.env`.
3. Make sure the `deployment:` value in `config.yaml` matches what you
   named each deployment in the Azure portal. Change `vision: true/false`
   if your specific deployment is/isn't multimodal.

Per-model overrides are supported via env vars
(`AZURE_<UPPER_NAME>_ENDPOINT` / `AZURE_<UPPER_NAME>_KEY`) — useful if a
model lives in a different region/project.

## Metrics

All metrics from SPEC are computed per cell and saved alongside the
prediction (`metrics` field in each checkpoint):

| Metric | Symbol |
|---|---|
| Coordinate error per bob | `coord_error_per_bob` |
| Mean coordinate error    | `coord_error_mean` |
| Max coordinate error     | `coord_error_max` |
| Angle error per link     | `angle_error_per_link` |
| Mean angle error         | `angle_error_mean` |
| ω sign-match per link    | `sign_match_per_link` |
| \|Δω\| per link          | `omega_mag_error` |
| ΔKE / ΔPE / ΔE_total     | `delta_KE`, `delta_PE`, `delta_E` |

`long_run_deviance` and `time_to_divergence` are available in
`bench/metrics.py` and are computed over **trajectories**, so they require
that the predictor emit a full trajectory rather than a single end-state;
wire these in via a future "rollout" predictor type if needed for the
learned-model comparison.

## Time-series foundation models (Chronos / TimesFM / Moirai)

These are wired up as real Azure adapters but need to be **deployed
separately** from the chat models. Each one becomes an Azure ML real-time
endpoint with its own scoring URI + API key:

1. In Azure AI Foundry / Azure ML Studio, deploy each model from the
   catalog (Chronos: `amazon/chronos-t5-large` or `chronos-bolt-base`;
   TimesFM: `google/timesfm-1.0-200m`; Moirai: `Salesforce/moirai-1.0-R-large`).
2. After deployment, copy the **REST endpoint** (scoring URI) and the
   **primary key** for each.
3. Paste them into `.env` as `AZURE_CHRONOS_ENDPOINT` / `AZURE_CHRONOS_KEY`
   (and the same pattern for TIMESFM, MOIRAI).
4. Run the eval normally — the time-series adapter handles request shaping
   per-variant and parses each model's response format.

The adapter slices the negative-time **pre-context window** (configurable
via `dataset.pre_context_seconds` in `config.yaml`, default 10s) from each
ground-truth trajectory and feeds it as history. Resampling is adaptive:
context length is capped at `max_context_length` and prediction length at
`max_prediction_length` per the model's tolerance.

If your AzureML deployment uses a custom scoring script with a non-standard
request body, edit `_build_<variant>_body` and `_parse_<variant>` in
`bench/models/timeseries.py`.

## Learned dynamics models (stubs)

- `bench/models/learned.py` — Neural ODE / HNN / LNN. Train these on the
  generated dataset, then point `checkpoint:` at the saved weights.

## Files

```
bench/
  simulator.py     # k-pendulum dynamics + Euler/RK4/leapfrog integrators
  rendering.py     # PIL renderer for VLM image prompts
  metrics.py       # all SPEC metrics
  prompts.py       # LLM/VLM prompt templates + response parser
  schema.py        # dataclasses (Trajectory, EvalCell, Prediction)
  export.py        # JSON/CSV trajectory writers
  runner.py        # async parallel runner with per-cell checkpoints
  models/
    base.py        # Predictor protocol
    numerical.py   # Euler / RK4 / leapfrog
    azure_llm.py   # Azure AI Foundry chat client (LLM + VLM)
    timeseries.py  # Azure ML real-time endpoint client (Chronos/TimesFM/Moirai)
    learned.py     # stub
scripts/
  generate_dataset.py
  run_eval.py
  aggregate.py
config.yaml
.env.example
DoublePendulum.py  # original interactive visualizer (kept for sanity checks)
SPEC.md            # research proposal
```

## Smoke test (no API key needed)

```bash
python scripts/generate_dataset.py --smoke
python scripts/run_eval.py --smoke --models rk4 euler symplectic
python scripts/aggregate.py
```

This validates the simulator, integrators, runner, and aggregation pipeline
without making any API calls.
