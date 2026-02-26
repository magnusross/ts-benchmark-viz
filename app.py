"""
TS Benchmark Viz — FastAPI backend (static-forecast edition)
Serves pre-generated forecasts from the forecasts/ directory.
Run generate_forecasts.py first to populate it.
"""
import math
import time
from pathlib import Path

import fev
import pandas as pd
import pyarrow.parquet as pq
import yaml
from datasets import disable_progress_bars
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

disable_progress_bars()

app = FastAPI(title="TS Benchmark Viz")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Load tasks
# ---------------------------------------------------------------------------

TASKS_YAML = Path("benchmarks/fev_bench/tasks.yaml")
FORECASTS_DIR = Path("forecasts")

with open(TASKS_YAML) as f:
    _yaml_data = yaml.safe_load(f)

_benchmark = fev.Benchmark.from_list(_yaml_data["tasks"])
TASKS: list[fev.Task] = _benchmark.tasks


def _task_display_name(task: fev.Task) -> str:
    n = len(task.target_columns)
    return f"{task.dataset_config}" + (f" [{n} vars]" if n > 1 else "")


def _infer_freq(seasonality: int, dataset_config: str) -> str:
    cfg = dataset_config.lower()
    if "_10t" in cfg or "_10m" in cfg: return "10-min"
    if "_15t" in cfg or "_15m" in cfg: return "15-min"
    if "_30t" in cfg or "_30m" in cfg: return "30-min"
    if "_1h" in cfg: return "hourly"
    if "_1d" in cfg: return "daily"
    if "_1w" in cfg: return "weekly"
    if "_1m" in cfg: return "monthly"
    freq_map = {1: "annual", 4: "quarterly", 5: "daily(bus)", 7: "daily",
                12: "monthly", 24: "hourly", 52: "weekly", 144: "10-min",
                168: "hourly", 288: "5-min"}
    return freq_map.get(seasonality, f"s={seasonality}")


def _available_models(dataset_config: str) -> list[str]:
    """Scan forecasts/<dataset_config>/ for completed model directories."""
    task_dir = FORECASTS_DIR / dataset_config
    if not task_dir.exists():
        return []
    return sorted(
        d.name for d in task_dir.iterdir()
        if d.is_dir() and (d / "info.json").exists()
    )


def _tasks_metadata() -> list[dict]:
    out = []
    for i, task in enumerate(TASKS):
        out.append({
            "idx": i,
            "dataset_config": task.dataset_config,
            "display_name": _task_display_name(task),
            "horizon": task.horizon,
            "num_windows": task.num_windows,
            "seasonality": task.seasonality,
            "target_columns": task.target_columns,
            "is_multivariate": task.is_multivariate,
            "frequency": _infer_freq(task.seasonality, task.dataset_config),
            "has_known_dynamic": bool(task.known_dynamic_columns),
            "available_models": _available_models(task.dataset_config),
        })
    return out


TASKS_META = _tasks_metadata()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(values: list) -> list:
    return [
        None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v
        for v in values
    ]


def _parquet_row_count(path: Path) -> int:
    """Read row count from parquet file footer — no data scan needed."""
    return pq.read_metadata(path).num_rows


def _read_parquet_row(path: Path, item_idx: int) -> dict | None:
    """Read a single row by item_idx using predicate pushdown."""
    df = pd.read_parquet(path, filters=[("item_idx", "==", item_idx)])
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    task_idx: int
    window_idx: int = 0
    item_idx: int = 0
    models: list[str] = ["seasonal_naive", "auto_ets"]
    context_multiplier: int = 3


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "tasks_meta": TASKS_META},
    )


@app.get("/api/tasks")
async def get_tasks():
    return TASKS_META


@app.post("/api/forecast")
async def run_forecast(body: ForecastRequest):
    if body.task_idx < 0 or body.task_idx >= len(TASKS):
        raise HTTPException(status_code=404, detail=f"task_idx out of range (0–{len(TASKS)-1})")

    task = TASKS[body.task_idx]
    task_meta = TASKS_META[body.task_idx]

    if body.window_idx < 0 or body.window_idx >= task.num_windows:
        raise HTTPException(
            status_code=400,
            detail=f"window_idx out of range (0–{task.num_windows - 1})",
        )

    context_path = FORECASTS_DIR / task.dataset_config / f"context_w{body.window_idx}.parquet"
    if not context_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for '{task.dataset_config}' window {body.window_idx}. "
                f"Run: uv run python generate_forecasts.py"
            ),
        )

    # n_items from parquet footer — cheap, no data read
    n_items = _parquet_row_count(context_path)

    if body.item_idx < 0 or body.item_idx >= n_items:
        raise HTTPException(
            status_code=400,
            detail=f"item_idx out of range (0–{n_items - 1})",
        )

    # Read context / ground truth row
    ctx_row = _read_parquet_row(context_path, body.item_idx)
    if ctx_row is None:
        raise HTTPException(status_code=500, detail="item_idx not found in context file")

    item_id = ctx_row["item_id"]
    display_len = max(2, min(body.context_multiplier, 8)) * task.horizon

    def _col(row: dict, key: str) -> list:
        v = row.get(key)
        return list(v) if v is not None else []

    variables: dict[str, dict] = {}
    for col in task.target_columns:
        ctx_full = _col(ctx_row, f"{col}__context")
        actual = _col(ctx_row, f"{col}__actual")
        ctx_display = ctx_full[-display_len:] if len(ctx_full) > display_len else ctx_full
        variables[col] = {
            "context": _sanitize(ctx_display),
            "actual_future": _sanitize(actual),
            "forecasts": {},
        }

    # Read model predictions
    timing: dict = {}
    task_dir = FORECASTS_DIR / task.dataset_config

    for model_name in body.models:
        t0 = time.perf_counter()
        model_path = task_dir / model_name / f"window_{body.window_idx}.parquet"

        if not model_path.exists():
            for col in task.target_columns:
                variables[col]["forecasts"][model_name] = {
                    "error": f"No data for '{model_name}'. Run generate_forecasts.py --models {model_name}"
                }
            timing[model_name] = round(time.perf_counter() - t0, 4)
            continue

        row = _read_parquet_row(model_path, body.item_idx)
        if row is None:
            for col in task.target_columns:
                variables[col]["forecasts"][model_name] = {"error": "item not found in forecast file"}
            timing[model_name] = round(time.perf_counter() - t0, 4)
            continue

        for col in task.target_columns:
            variables[col]["forecasts"][model_name] = {
                "mean": _sanitize(_col(row, f"{col}__mean")),
                "lo80": _sanitize(_col(row, f"{col}__lo80")),
                "hi80": _sanitize(_col(row, f"{col}__hi80")),
            }
        timing[model_name] = round(time.perf_counter() - t0, 4)

    return {
        "task_idx": body.task_idx,
        "dataset_config": task.dataset_config,
        "frequency": task_meta["frequency"],
        "horizon": task.horizon,
        "num_windows": task.num_windows,
        "window_idx": body.window_idx,
        "n_items": n_items,
        "item_idx": body.item_idx,
        "item_id": str(item_id),
        "target_columns": task.target_columns,
        "is_multivariate": task.is_multivariate,
        "variables": variables,
        "timing": timing,
    }
