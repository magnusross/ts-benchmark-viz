# ts-benchmark-viz

A web app for exploring and comparing time series forecasts on the [FEV benchmark](https://github.com/autogluon/fev). Forecasts are pre-generated offline using the exact same code as the benchmark, then served statically — the app is just a viewer.

## Features

- All 100 tasks from `fev_bench` — loaded directly from the benchmark task definitions
- Navigate between evaluation windows and individual series within each task
- Multivariate tasks show one chart per target variable in a scrollable panel
- 80% prediction intervals shaded for all models
- Adjustable context window (2×–6× forecast horizon)
- Only models with pre-generated data are shown per task

## Architecture

```
generate_forecasts.py          # offline: runs models, writes Parquet files
    ↓
forecasts/<dataset>/<model>/   # static data on disk
    ↓
app.py (FastAPI)               # reads Parquet, serves to browser
    ↓
templates/index.html           # Chart.js visualisation
```

Forecast generation uses `predict_with_model` functions from locally stored copies of the `autogluon/fev` example scripts (see `examples/`), so the outputs closely match the published benchmark results.

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) for dependency management
- Internet access (datasets are downloaded from Hugging Face on first use and cached in `~/.cache/huggingface/`)
- GPU recommended for Chronos models (CPU works but is slow)

## Setup

```bash
git clone <repo-url>
cd ts-benchmark-viz
uv sync --no-install-project
```

## Generating forecasts

Forecasts must be generated before the app can display anything.

```bash
# Quick test — 3 tasks, two fast models (~2 min on CPU):
uv run python generate_forecasts.py --tasks 3 --models seasonal_naive auto_ets

# All 100 tasks, statistical models only (~several hours on CPU):
uv run python generate_forecasts.py --models naive seasonal_naive auto_ets auto_arima

# Add Chronos-Bolt-Tiny to existing data (GPU recommended):
uv run python generate_forecasts.py --models chronos_bolt_tiny

# Add Chronos-2 to existing data (GPU strongly recommended):
uv run python generate_forecasts.py --models chronos_2

# Full benchmark — everything at once (GPU strongly recommended):
uv run python generate_forecasts.py --models naive seasonal_naive auto_ets auto_arima chronos_bolt_tiny chronos_2 lightgbm catboost
```

Already-complete task/model combinations are skipped automatically, so runs can be interrupted and resumed freely, and models can be added incrementally.

### Output layout

```
forecasts/
  <dataset_config>/
    metadata.json                   # task parameters
    context_w<N>.parquet            # context + ground truth, one row per item
    <model_name>/
      window_<N>.parquet            # mean, lo80, hi80 forecasts, one row per item
      info.json                     # inference time and model config
```

## Running the app

```bash
uv run uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000). The app only serves tasks for which forecast data has been generated — tasks without data return a 404 with instructions.

For production:

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Multiple workers are fine since the app is stateless (reads only from disk).

## Running with Docker

The app can be run persistently in a Docker container. The `forecasts/` directory is mounted from the host at runtime, so no rebuild is needed when new forecasts are generated.

```bash
# Build the image (first time, or after code changes):
docker compose build

# Start in the background:
docker compose up -d

# View logs:
docker compose logs -f

# Stop:
docker compose down
```

Then open [http://localhost:8000](http://localhost:8000), or `http://<server-ip>:8000` from another machine on the network.

The Docker image installs only the lightweight serving dependencies. Forecast generation should still be done on the host using `uv run python generate_forecasts.py ...` (see above).

## Usage

1. **Pick a task** from the sidebar. Searchable by dataset name. Badges show frequency, forecast horizon (`h`), seasonality (`s`), number of variables, and number of evaluation windows.

2. **Navigate windows** with the `‹ ›` buttons next to *Window*. Windows are rolling evaluation periods; index 1 is the earliest.

3. **Navigate series** with the `‹ ›` buttons next to *Series*. The current series ID is shown to the right.

4. **Adjust context** with the slider. Controls how many multiples of the forecast horizon are shown as historical context.

5. **Toggle models** by clicking the model buttons. Only models with pre-generated data for the selected task appear.

6. **Click Load** to fetch and display the forecasts. Response time is typically < 5 ms.

## Project structure

```
ts-benchmark-viz/
├── app.py                          # FastAPI backend — reads from forecasts/
├── generate_forecasts.py           # Offline forecast generation script
├── examples/                       # Local copies of autogluon/fev example scripts
│   ├── statsforecast/              #   (minor modifications vs. upstream — see below)
│   ├── chronos/
│   ├── chronos-2/
│   └── mlforecast/
├── templates/
│   └── index.html                  # Single-page frontend (Chart.js)
├── benchmarks/
│   └── fev_bench/
│       └── tasks.yaml              # Benchmark task definitions (from autogluon/fev)
├── forecasts/                      # Generated data (not in git)
├── pyproject.toml
└── uv.lock
```

## Example scripts

The `examples/` directory contains local copies of the prediction scripts from [`autogluon/fev`](https://github.com/autogluon/fev/tree/main/examples). They are kept here so the codebase is self-contained and modifications persist across runs.

The copies differ from upstream only where necessary:

| Script | Change vs. upstream |
|---|---|
| `statsforecast/evaluate_model.py` | Drops exogenous columns before fitting — prevents AutoARIMA from fitting an ARIMAX model and crashing when `X_df` is not supplied at forecast time |
| `chronos/evaluate_model.py` | Renames `context=` → `inputs=` in `predict_quantiles()` to match the updated Chronos v2 API |
| `chronos-2/evaluate_model.py` | Removes `as_univariate` and `predict_batches_jointly` args from `predict_fev()`, which were dropped in a later API version |
| `mlforecast/evaluate_model.py` | Adds a thin `predict_with_model` wrapper to match the common interface; suppresses `FutureWarning` from pandas `groupby` |

## Models

| Model | Type | Notes |
|---|---|---|
| Naïve | Statistical | Repeats last observed value |
| Seasonal Naïve | Statistical | Repeats last seasonal cycle |
| AutoETS | Statistical | Auto-selects exponential smoothing model |
| AutoARIMA | Statistical | Auto-selects ARIMA order |
| Chronos-Bolt-Tiny | Foundation model | Zero-shot transformer, ~50M params |
| Chronos-2 | Foundation model | Zero-shot transformer, ~710M params (`autogluon/chronos-t5-large`) |
| LightGBM | ML | Recursive gradient-boosted trees with optional HPO via Optuna |
| CatBoost | ML | Recursive gradient-boosted trees with optional HPO via Optuna |

Statistical models use [StatsForecast](https://github.com/Nixtla/statsforecast). Chronos uses [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) v2. ML models use [MLForecast](https://github.com/Nixtla/mlforecast) with [autogluon.timeseries](https://auto.gluon.ai/stable/tutorials/timeseries/index.html) for frequency-aware lag/feature selection. All use the configurations from the `autogluon/fev` example scripts.
