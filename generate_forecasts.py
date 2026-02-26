#!/usr/bin/env python3
"""
Generate and save fev_bench forecasts to disk for static visualization.

This script clones the autogluon/fev repo to use the official predict_with_model
functions, then saves forecasts, context, and ground truth as Parquet files.

Output layout
-------------
forecasts/
  <dataset_config>/
    metadata.json                         # task parameters
    context_w<N>.parquet                  # item_idx, item_id, <col>__context, <col>__actual
    <model_name>/
      window_<N>.parquet                  # item_idx, <col>__mean, <col>__lo80, <col>__hi80
      info.json                           # inference time, model config

Each Parquet row corresponds to one item (time series). Array columns store
variable-length Python lists — Parquet's LIST type handles these natively.

Usage
-----
# Quick test — 3 tasks, two fast models:
uv run python generate_forecasts.py --tasks 3 --models seasonal_naive auto_ets

# All 100 tasks, default models (seasonal_naive + auto_ets):
uv run python generate_forecasts.py

# Full benchmark — all stat models + Chronos-Bolt-Tiny + Chronos-2 (GPU recommended):
uv run python generate_forecasts.py --models naive seasonal_naive auto_ets auto_arima chronos_bolt_tiny chronos_2

# Add a model to already-generated data (done tasks/models are skipped automatically):
uv run python generate_forecasts.py --models chronos_2

# Specify a different output directory:
uv run python generate_forecasts.py --output-dir /data/forecasts
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import datasets as hf_datasets
import fev
import pandas as pd
import torch
import yaml

hf_datasets.disable_progress_bars()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCAL_EXAMPLES_DIR = Path("examples")
TASKS_YAML = Path("benchmarks/fev_bench/tasks.yaml")
FOUNDATION_FAILS_YAML = Path("benchmarks/fev_bench/foundation_fails.yaml")
FORECASTS_DIR = Path("forecasts")

BENCHMARK_YAMLS = {
    "fev_bench": TASKS_YAML,
    "foundation_fails": FOUNDATION_FAILS_YAML,
}

# Map model name → backend example script
AVAILABLE_MODELS = {
    "naive": "statsforecast",
    "seasonal_naive": "statsforecast",
    "auto_ets": "statsforecast",
    "auto_arima": "statsforecast",
    "chronos_bolt_tiny": "chronos",
    "chronos_2": "chronos_2",
    "lightgbm": "mlforecast",
    "catboost": "mlforecast",
}
DEFAULT_MODELS = ["seasonal_naive", "auto_ets"]


# ---------------------------------------------------------------------------
# Dynamic imports from the fev example scripts
# ---------------------------------------------------------------------------

_loaded_modules: dict[str, object] = {}


def _load_example_module(name: str, path: Path):
    if name in _loaded_modules:
        return _loaded_modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _loaded_modules[name] = module
    return module


def get_predict_fn(model_name: str):
    """Import predict_with_model from the appropriate fev example script."""
    backend = AVAILABLE_MODELS[model_name]
    if backend == "statsforecast":
        module = _load_example_module(
            "statsforecast_example",
            LOCAL_EXAMPLES_DIR / "statsforecast/evaluate_model.py",
        )
    elif backend == "chronos":
        module = _load_example_module(
            "chronos_example",
            LOCAL_EXAMPLES_DIR / "chronos/evaluate_model.py",
        )
    elif backend == "chronos_2":
        module = _load_example_module(
            "chronos_2_example",
            LOCAL_EXAMPLES_DIR / "chronos-2/evaluate_model.py",
        )
    elif backend == "mlforecast":
        module = _load_example_module(
            "mlforecast_example",
            LOCAL_EXAMPLES_DIR / "mlforecast/evaluate_model.py",
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return module.predict_with_model


def build_predict_kwargs(model_name: str) -> dict:
    """Build keyword arguments for the predict_with_model call."""
    backend = AVAILABLE_MODELS[model_name]

    if backend == "statsforecast":
        return {"model_name": model_name}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if backend == "chronos":
        return {
            "model_name": "amazon/chronos-bolt-tiny",
            "device_map": device,
            "torch_dtype": dtype,
        }

    if backend == "chronos_2":
        # Model is hosted on HuggingFace as autogluon/chronos-t5-large;
        # the fev example uses the S3 path s3://autogluon/chronos-2 which
        # requires AWS credentials. Override with --chronos2-model if needed.
        # batch_size=16 avoids OOM on tasks with many items; the model's
        # built-in max horizon is 64 so it autoregressively extends for
        # longer horizons — quality may degrade but predictions still run.
        return {
            "model_name": "autogluon/chronos-t5-large",
            "device_map": device,
            "torch_dtype": dtype,
            "batch_size": 16,
        }

    if backend == "mlforecast":
        return {"model_name": model_name}

    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize(values) -> list:
    """Replace NaN/Inf with None for safe JSON/Parquet serialisation."""
    return [
        None
        if (v is None or (isinstance(v, float) and not math.isfinite(v)))
        else float(v)
        for v in values
    ]


def needs_context(task_dir: Path, num_windows: int) -> bool:
    return any(
        not (task_dir / f"context_w{w}.parquet").exists() for w in range(num_windows)
    )


def needs_model(model_dir: Path, num_windows: int) -> bool:
    if not (model_dir / "info.json").exists():
        return True
    return any(
        not (model_dir / f"window_{w}.parquet").exists() for w in range(num_windows)
    )


# ---------------------------------------------------------------------------
# Core per-task logic
# ---------------------------------------------------------------------------


def save_context_and_ground_truth(task: fev.Task, task_dir: Path) -> None:
    """Save context and ground truth for each evaluation window."""
    print("  Saving context / ground truth ...")
    for w_idx in range(task.num_windows):
        out_path = task_dir / f"context_w{w_idx}.parquet"
        if out_path.exists():
            continue

        window = task.get_window(w_idx, trust_remote_code=True)
        past_data, _ = window.get_input_data()
        ground_truth = window.get_ground_truth()

        rows = []
        for i, (item, gt) in enumerate(zip(past_data, ground_truth)):
            row = {
                "item_idx": i,
                "item_id": str(item[window.id_column]),
            }
            for col in task.target_columns:
                row[f"{col}__context"] = sanitize(list(item[col]))
                row[f"{col}__actual"] = sanitize(list(gt[col]))
            rows.append(row)

        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"    context_w{w_idx}.parquet  ({len(rows)} items)")


def save_model_predictions(task: fev.Task, model_name: str, model_dir: Path) -> None:
    """Run predict_with_model and save per-window Parquet files."""
    predict_fn = get_predict_fn(model_name)
    kwargs = build_predict_kwargs(model_name)

    print(f"  [{model_name}] running predict_with_model ...")
    predictions_per_window, inference_time, extra_info = predict_fn(task, **kwargs)

    for w_idx, window_preds in enumerate(predictions_per_window):
        out_path = model_dir / f"window_{w_idx}.parquet"
        if out_path.exists():
            continue

        first_col = task.target_columns[0]
        n_items = len(window_preds[first_col])

        rows = []
        for i in range(n_items):
            row = {"item_idx": i}
            for col in task.target_columns:
                item_preds = window_preds[col][i]
                # "predictions" is the point forecast; quantiles are keyed by str(q)
                mean = item_preds.get("predictions") or item_preds.get("0.5")
                lo80 = item_preds.get("0.1", mean)
                hi80 = item_preds.get("0.9", mean)
                row[f"{col}__mean"] = sanitize(list(mean))
                row[f"{col}__lo80"] = sanitize(list(lo80))
                row[f"{col}__hi80"] = sanitize(list(hi80))
            rows.append(row)

        pd.DataFrame(rows).to_parquet(out_path, index=False)

    # Save timing / config
    info = {"inference_time_s": round(inference_time, 3), **extra_info}
    (model_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  [{model_name}] done — {inference_time:.1f}s")


def process_task(task: fev.Task, models: list[str], out_dir: Path) -> None:
    task_dir = out_dir / task.dataset_config
    task_dir.mkdir(parents=True, exist_ok=True)

    # Write task metadata once
    meta_path = task_dir / "metadata.json"
    if not meta_path.exists():
        meta = {
            "dataset_config": task.dataset_config,
            "horizon": task.horizon,
            "seasonality": task.seasonality,
            "num_windows": task.num_windows,
            "target_columns": task.target_columns,
            "is_multivariate": task.is_multivariate,
            "quantile_levels": task.quantile_levels,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    # Context / ground truth
    if needs_context(task_dir, task.num_windows):
        save_context_and_ground_truth(task, task_dir)
    else:
        print("  Context already saved.")

    # Model predictions
    for model_name in models:
        model_dir = task_dir / model_name
        if not needs_model(model_dir, task.num_windows):
            print(f"  [{model_name}] already complete, skipping.")
            continue
        model_dir.mkdir(exist_ok=True)
        try:
            save_model_predictions(task, model_name, model_dir)
        except Exception as e:
            print(f"  [{model_name}] ERROR: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fev_bench forecasts for static visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
models available:
  naive             Naïve (repeat last value)
  seasonal_naive    Seasonal Naïve
  auto_ets          AutoETS  (StatsForecast)
  auto_arima        AutoARIMA (StatsForecast)
  chronos_bolt_tiny Chronos-Bolt-Tiny (zero-shot transformer, ~50M params)
  chronos_2         Chronos-2 / Chronos-T5-Large (~710M params, GPU recommended)
                    Uses autogluon/chronos-t5-large from HuggingFace.
                    For the S3-hosted autogluon model, patch build_predict_kwargs
                    to use model_name="s3://autogluon/chronos-2".

examples:
  # Quick test — 3 tasks, two fast models:
  uv run python generate_forecasts.py --tasks 3 --models seasonal_naive auto_ets

  # All 100 tasks, default models:
  uv run python generate_forecasts.py

  # Full benchmark — all stat models + Chronos-Bolt-Tiny + Chronos-2 (GPU recommended):
  uv run python generate_forecasts.py --models naive seasonal_naive auto_ets auto_arima chronos_bolt_tiny chronos_2

  # Add a model to existing data (already-complete tasks/models are skipped):
  uv run python generate_forecasts.py --models chronos_2
        """,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(AVAILABLE_MODELS),
        metavar="MODEL",
        help=(
            f"One or more models to generate. "
            f"Choices: {', '.join(AVAILABLE_MODELS)}. "
            f"Default: {' '.join(DEFAULT_MODELS)}."
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_YAMLS),
        default="fev_bench",
        help=(
            "Which benchmark YAML to load tasks from. "
            "Choices: fev_bench (default, ~100 tasks), foundation_fails (17 tasks)."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N tasks (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FORECASTS_DIR,
        metavar="DIR",
        help=f"Directory to write forecast files (default: {FORECASTS_DIR}/).",
    )
    args = parser.parse_args()

    # 1. Load benchmark tasks from local YAML
    tasks_yaml = BENCHMARK_YAMLS[args.benchmark]
    with open(tasks_yaml) as f:
        yaml_data = yaml.safe_load(f)
    benchmark = fev.Benchmark.from_list(yaml_data["tasks"])
    tasks = benchmark.tasks[: args.tasks] if args.tasks else benchmark.tasks

    print(f"\nModels : {', '.join(args.models)}")
    print(f"Tasks  : {len(tasks)}")
    print(f"Output : {args.output_dir}/\n")

    # 2. Generate forecasts task by task
    for i, task in enumerate(tasks):
        print(f"[{i + 1}/{len(tasks)}] {task.dataset_config}")
        try:
            process_task(task, args.models, args.output_dir)
        except Exception as e:
            print(f"  FATAL ERROR for {task.dataset_config}: {e}", file=sys.stderr)

    print(f"\nDone. Output in {args.output_dir}/")


if __name__ == "__main__":
    main()
