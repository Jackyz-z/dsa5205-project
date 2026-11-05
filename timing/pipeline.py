"""Top-level workflows: build data, validate, run the one-shot test, and report."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timing.config import (
    ModelSpec,
    Paths,
    config_id,
    list_model_names,
    load_model_spec,
    read_json,
    stable_hash,
    write_json,
)
from timing.engine.rolling import run_rolling
from timing.engine.splits import development_rows, split_index
from timing.evaluation.metrics import forecast_metrics, trading_metrics
from timing.strategy.backtest import backtest_forecasts

LOG_FIELDS = ["timestamp_utc", "split", "family", "role", "config_id", "data_hash", "sharpe", "oos_r2_vs_prevailing_mean"]
BASELINES = ["prevailing_mean", "ar1"]


class GateError(RuntimeError):
    """Raised when a run would break the one-shot test rule."""


def _data_hash(paths: Paths) -> str:
    return read_json(paths.manifest)["data_hash"]


def _append_log(paths: Paths, row: dict[str, Any]) -> None:
    paths.experiment_log.parent.mkdir(parents=True, exist_ok=True)
    is_new = not paths.experiment_log.exists()
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    with open(paths.experiment_log, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _run_and_save(dataset: pd.DataFrame, spec: ModelSpec, params: dict[str, Any], cfg: dict[str, Any], out_root: Path) -> pd.DataFrame:
    result = run_rolling(
        dataset,
        spec,
        params,
        cfg["rolling"]["refit_every"],
        cfg["splits"]["prediction_start"],
        cfg["seed"],
    )
    out = out_root / spec.name / config_id(params)
    out.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(out / "predictions.csv", float_format="%.12g")
    if result.diagnostics is not None:
        result.diagnostics.to_csv(out / "diagnostics.csv", float_format="%.12g")
    write_json(out / "config.json", {"family": spec.name, "params": params, "feature_set": spec.feature_set, "preprocess": spec.preprocess, "seed": cfg["seed"]})
    return result.predictions


def _score(predictions: pd.DataFrame, benchmark: pd.Series, dataset: pd.DataFrame, dates: pd.DatetimeIndex, cfg: dict[str, Any]) -> dict[str, float]:
    backtest = backtest_forecasts(
        predictions["y_hat"], dataset[["spy_ret", "rf_ret"]], cfg["strategy"], cfg["costs"]["base_bps"]
    )
    missing = dates.difference(backtest.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} evaluation dates have no position, first {missing[0].date()}. "
            "The evaluation start must be after the forecast and position warm-up."
        )
    evaluation = cfg["evaluation"]
    metrics = trading_metrics(backtest.loc[dates], evaluation["periods_per_year"], evaluation["newey_west_lags"])
    forecasts = predictions.loc[dates]
    metrics.update(forecast_metrics(forecasts["y_true"], forecasts["y_hat"], benchmark.loc[dates]))
    return metrics


def _selection_record(spec: ModelSpec, params: dict[str, Any], sharpe: float) -> dict[str, Any]:
    return {
        "family": spec.name,
        "role": spec.role,
        "config_id": config_id(params),
        "params": params,
        "feature_set": spec.feature_set,
        "preprocess": spec.preprocess,
        "validation_sharpe": sharpe,
    }


def validate(cfg: dict[str, Any], paths: Paths, dataset: pd.DataFrame, model: str) -> dict[str, Any]:
    """Run each requested family's grid on development rows and record the selection.

    Baselines are always run because the out-of-sample R^2 is defined against them.
    """
    development = development_rows(dataset, cfg["splits"])
    dates = split_index(development, "validation", cfg["splits"])
    requested = list_model_names(paths) if model == "all" else [model]
    names = BASELINES + [n for n in requested if n not in BASELINES]
    specs = {name: load_model_spec(paths, name) for name in names}
    out_root = paths.results / "validation"
    data_hash = _data_hash(paths)

    windows = {params["window"] for spec in specs.values() for params in spec.configs()}
    prevailing_spec = specs["prevailing_mean"]
    benchmarks = {
        window: _run_and_save(development, prevailing_spec, {"window": window}, cfg, out_root)["y_hat"]
        for window in sorted(windows)
    }

    for name in names:
        spec = specs[name]
        rows = []
        for params in spec.configs():
            predictions = _run_and_save(development, spec, params, cfg, out_root)
            metrics = _score(predictions, benchmarks[params["window"]], development, dates, cfg)
            rows.append({"config_id": config_id(params), **{f"param_{k}": v for k, v in params.items()}, **metrics})
            _append_log(
                paths,
                {
                    "split": "validation",
                    "family": name,
                    "role": spec.role,
                    "config_id": config_id(params),
                    "data_hash": data_hash,
                    "sharpe": metrics["sharpe"],
                    "oos_r2_vs_prevailing_mean": metrics["oos_r2_vs_prevailing_mean"],
                },
            )
        grid = pd.DataFrame(rows)
        grid.to_csv(out_root / name / "grid_metrics.csv", index=False, float_format="%.6g")
        # The selection rule: highest validation net Sharpe. Sorting by id breaks exact ties.
        best = grid.sort_values(["sharpe", "config_id"], ascending=[False, True]).iloc[0]
        best_params = next(p for p in spec.configs() if config_id(p) == best["config_id"])
        write_json(out_root / name / "selection.json", _selection_record(spec, best_params, float(best["sharpe"])))

    return write_global_selection(paths)


def write_global_selection(paths: Paths) -> dict[str, Any]:
    """Combine family selections. Only candidates can become the final model."""
    root = paths.results / "validation"
    known = set(list_model_names(paths))
    families = {
        p.parent.name: read_json(p)
        for p in sorted(root.glob("*/selection.json"))
        if p.parent.name in known
    }
    candidates = [r for r in families.values() if r["role"] == "candidate"]
    if not candidates:
        raise ValueError("No candidate model has been validated yet.")
    final = max(sorted(candidates, key=lambda r: r["family"]), key=lambda r: r["validation_sharpe"])
    selection = {
        "rule": "Highest validation Sharpe net of base costs among candidate models",
        "final": final,
        "families": families,
    }
    write_json(root / "selection.json", selection)
    return selection


def _frozen_payload(cfg: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Everything that must not change once the test set has been opened."""
    keys = ("family", "config_id", "feature_set", "preprocess")
    return {
        "base_config": cfg,
        "final": {k: selection["final"][k] for k in keys},
        "families": {name: {k: record[k] for k in keys} for name, record in selection["families"].items()},
    }


def _run_test_models(
    cfg: dict[str, Any],
    paths: Paths,
    dataset: pd.DataFrame,
    families: dict[str, dict[str, Any]],
    out_root: Path,
    log_split: str,
) -> None:
    dates = split_index(dataset, "test", cfg["splits"])
    data_hash = _data_hash(paths)
    prevailing_spec = load_model_spec(paths, "prevailing_mean")
    ar1_spec = load_model_spec(paths, "ar1")
    windows = sorted({record["params"]["window"] for record in families.values()})

    prevailing = {
        window: _run_and_save(dataset, prevailing_spec, {"window": window}, cfg, out_root)
        for window in windows
    }
    runs = [(prevailing_spec, {"window": window}, prevailing[window]) for window in windows]
    jobs = [(ar1_spec, {"window": window}) for window in windows]
    jobs += [
        (load_model_spec(paths, name), record["params"])
        for name, record in families.items()
        if name not in BASELINES
    ]
    runs += [(spec, params, _run_and_save(dataset, spec, params, cfg, out_root)) for spec, params in jobs]

    for spec, params, predictions in runs:
        metrics = _score(predictions, prevailing[params["window"]]["y_hat"], dataset, dates, cfg)
        _append_log(
            paths,
            {
                "split": log_split,
                "family": spec.name,
                "role": spec.role,
                "config_id": config_id(params),
                "data_hash": data_hash,
                "sharpe": metrics["sharpe"],
                "oos_r2_vs_prevailing_mean": metrics["oos_r2_vs_prevailing_mean"],
            },
        )


def run_test(cfg: dict[str, Any], paths: Paths, dataset: pd.DataFrame, confirm_final: bool) -> dict[str, Any]:
    """The one-shot test. A later run must match the locked configuration exactly."""
    selection_path = paths.results / "validation" / "selection.json"
    if not selection_path.exists():
        raise GateError("Run validation before the test.")
    if not confirm_final:
        raise GateError(
            "The test set may be used only once. Freeze all models first, then pass --confirm-final."
        )
    selection = read_json(selection_path)
    payload_hash = stable_hash(_frozen_payload(cfg, selection))
    data_hash = _data_hash(paths)

    if paths.test_lock.exists():
        lock = read_json(paths.test_lock)
        if lock["config_hash"] != payload_hash:
            raise GateError(
                "The configuration or selection differs from the locked test run. "
                "Use --exploratory MODEL for models added after the lock."
            )
        if lock["data_hash"] != data_hash:
            print("Warning: the data differ from the locked run, so results may differ slightly.")
        print("Reproducing the locked test run.")
    else:
        write_json(
            paths.test_lock,
            {
                "config_hash": payload_hash,
                "data_hash": data_hash,
                "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "selection": selection,
            },
        )
    _run_test_models(cfg, paths, dataset, selection["families"], paths.results / "test", "test")
    return selection


def run_exploratory(cfg: dict[str, Any], paths: Paths, dataset: pd.DataFrame, model: str) -> None:
    """Evaluate a model added after the lock. It is kept apart and cannot become the final model."""
    if not paths.test_lock.exists():
        raise GateError("Exploratory runs are for models added after the locked test run.")
    selection_path = paths.results / "validation" / model / "selection.json"
    if not selection_path.exists():
        raise GateError(f"Validate {model!r} first.")
    record = read_json(selection_path)
    _run_test_models(cfg, paths, dataset, {model: record}, paths.results / "test_exploratory", "test_exploratory")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
