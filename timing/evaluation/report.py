"""Build tables and figures from saved predictions, without knowing how they were produced."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from timing.config import Paths, config_id, read_json
from timing.engine.splits import development_rows, split_index
from timing.evaluation import plots
from timing.evaluation.metrics import cumulative_oos_r2, forecast_metrics, trading_metrics
from timing.strategy.backtest import backtest_forecasts, buy_and_hold


@dataclass
class Entry:
    label: str
    family: str
    params: dict[str, Any]
    is_candidate: bool

    @property
    def window(self) -> int:
        return self.params["window"]


def load_predictions(split_dir: Path, family: str, params: dict[str, Any]) -> pd.DataFrame:
    path = split_dir / family / config_id(params) / "predictions.csv"
    return pd.read_csv(path, index_col="date", parse_dates=["date", "fit_date", "train_start", "train_end"])


def report_entries(selection: dict[str, Any]) -> list[Entry]:
    """The final model, baselines at its window, then other candidates (the palette's slot order)."""
    final = selection["final"]
    window = final["params"]["window"]
    entries = [
        Entry(f"{final['family']} (final)", final["family"], final["params"], True),
        Entry(f"Prevailing mean ({window}d)", "prevailing_mean", {"window": window}, False),
        Entry(f"AR(1) ({window}d)", "ar1", {"window": window}, False),
    ]
    for family, record in sorted(selection["families"].items()):
        if record["role"] == "candidate" and family != final["family"]:
            entries.append(Entry(family, family, record["params"], True))
    return entries


def build_split_report(
    cfg: dict[str, Any],
    paths: Paths,
    dataset: pd.DataFrame,
    split: str,
    split_dir_name: str,
    entries: list[Entry],
    output_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Read predictions from `split_dir_name` and write tables and figures under `output_name`."""
    evaluation = cfg["evaluation"]
    ppy, lags = evaluation["periods_per_year"], evaluation["newey_west_lags"]
    strategy = cfg["strategy"]
    base_bps = cfg["costs"]["base_bps"]

    split_dir = paths.results / split_dir_name
    table_dir = paths.results / "tables" / (output_name or split_dir_name)
    figure_dir = paths.results / "figures" / (output_name or split_dir_name)
    table_dir.mkdir(parents=True, exist_ok=True)

    dates = split_index(dataset, split, cfg["splits"])
    returns = dataset[["spy_ret", "rf_ret"]]
    candidate_order = [e.family for e in entries if e.is_candidate]

    colors = {e.label: plots.series_color(e.family, candidate_order) for e in entries}
    colors["SPY buy & hold"] = plots.SPY_COLOR

    spy_backtest = buy_and_hold(returns.loc[dates])
    predictions = {e.label: load_predictions(split_dir, e.family, e.params) for e in entries}
    backtests = {
        e.label: backtest_forecasts(predictions[e.label]["y_hat"], returns, strategy, base_bps).loc[dates]
        for e in entries
    }

    # Summary and forecast tables.
    summary_rows, r2_curves = [], []
    for e in entries:
        pred = predictions[e.label].loc[dates]
        benchmark = load_predictions(split_dir, "prevailing_mean", {"window": e.window}).loc[dates, "y_hat"]
        row = {"label": e.label, "config": config_id(e.params), "role": "candidate" if e.is_candidate else "baseline"}
        row.update(trading_metrics(backtests[e.label], ppy, lags))
        row.update(forecast_metrics(pred["y_true"], pred["y_hat"], benchmark))
        summary_rows.append(row)
        if e.family != "prevailing_mean":
            r2_curves.append(
                {
                    "label": e.label,
                    "color": colors[e.label],
                    "r2": cumulative_oos_r2(pred["y_true"], pred["y_hat"], benchmark),
                }
            )
    spy_row = {"label": "SPY buy & hold", "config": "", "role": "benchmark"}
    spy_row.update(trading_metrics(spy_backtest, ppy, lags))
    summary_rows.append(spy_row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "summary.csv", index=False, float_format="%.6g")

    # Subperiod, yearly and cost tables.
    # Display order follows the validated palette: final, SPY, prevailing mean, AR(1), others.
    first, *rest = entries
    display_labels = [first.label, "SPY buy & hold"] + [e.label for e in rest]
    all_backtests = {label: {**backtests, "SPY buy & hold": spy_backtest}[label] for label in display_labels}
    subperiod_rows, yearly_rows = [], []
    for label, frame in all_backtests.items():
        for start, end in evaluation["subperiods"][split]:
            part = frame.loc[start:end]
            if len(part):
                metrics = trading_metrics(part, ppy, lags)
                subperiod_rows.append(
                    {"label": label, "period": f"{start[:4]}–{end[:4]}", "start": start, "end": end, **metrics}
                )
        for year, part in frame.groupby(frame.index.year):
            yearly_rows.append({"label": label, "year": int(year), **trading_metrics(part, ppy, lags)})
    subperiods = pd.DataFrame(subperiod_rows)
    yearly = pd.DataFrame(yearly_rows)
    subperiods.to_csv(table_dir / "subperiods.csv", index=False, float_format="%.6g")
    yearly.to_csv(table_dir / "yearly.csv", index=False, float_format="%.6g")

    cost_rows = []
    for bps in cfg["costs"]["sensitivity_bps"]:
        for label in display_labels:
            if label == "SPY buy & hold":
                frame = spy_backtest
            else:
                frame = backtest_forecasts(predictions[label]["y_hat"], returns, strategy, bps).loc[dates]
            metrics = trading_metrics(frame, ppy, lags)
            cost_rows.append({"label": label, "cost_bps": bps, "ann_return": metrics["ann_return"], "sharpe": metrics["sharpe"]})
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(table_dir / "cost_sensitivity.csv", index=False, float_format="%.6g")

    # Figures.
    start_date = dates[0].date().isoformat()
    end_date = dates[-1].date().isoformat()
    period_text = f"{split} period, {start_date} to {end_date}"

    wealth_curves = [
        {"label": label, "color": colors[label], "wealth": (1 + frame["net_ret"]).cumprod()}
        for label, frame in all_backtests.items()
    ]
    plots.plot_cumulative_returns(
        wealth_curves, f"Growth of $1 net of {base_bps:g} bps costs, {period_text}", figure_dir / "cumulative_returns.png"
    )
    plots.plot_cumulative_r2(
        r2_curves, f"Cumulative out-of-sample R² vs prevailing mean, {period_text}", figure_dir / "cumulative_r2.png"
    )
    plots.plot_subperiods(
        subperiods, colors, f"Net return and Sharpe by period, {split} period", figure_dir / "subperiods.png"
    )
    plots.plot_cost_sensitivity(
        costs, colors, f"Net Sharpe vs trading cost, {period_text}", figure_dir / "cost_sensitivity.png"
    )
    final = entries[0]
    plots.plot_weights(
        backtests[final.label],
        colors[final.label],
        f"{final.label}: SPY weight, {period_text}",
        figure_dir / "weights.png",
    )
    diagnostics_path = split_dir / final.family / config_id(final.params) / "diagnostics.csv"
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path, index_col="fit_date", parse_dates=["fit_date"])
        diagnostics = diagnostics.loc[diagnostics.index.isin(dates)]
        if len(diagnostics) > 1:
            plots.plot_coefficients(
                diagnostics,
                f"{final.label}: coefficients at each refit, {period_text}",
                figure_dir / "coefficients.png",
            )
    return {"summary": summary, "subperiods": subperiods, "yearly": yearly, "costs": costs}


def plot_validation_grids(paths: Paths, families: list[str]) -> None:
    """Heatmap for any family whose grid has exactly one parameter besides the window."""
    for family in families:
        grid_path = paths.results / "validation" / family / "grid_metrics.csv"
        grid = pd.read_csv(grid_path)
        extra = [c for c in grid.columns if c.startswith("param_") and c != "param_window"]
        if len(extra) != 1:
            continue
        pivot = grid.pivot(index="param_window", columns=extra[0], values="sharpe")
        plots.plot_grid_heatmap(
            pivot,
            extra[0].removeprefix("param_"),
            f"{family}: validation net Sharpe across the grid",
            paths.results / "figures" / "validation" / f"grid_{family}.png",
        )


def build_reports(cfg: dict[str, Any], paths: Paths, dataset: pd.DataFrame) -> list[str]:
    """Report on whatever has been run. Validation never reads rows beyond its own end."""
    built = []
    selection_path = paths.results / "validation" / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("No validation results. Run: python -m timing.cli validate --model all")
    selection = read_json(selection_path)
    entries = report_entries(selection)

    build_split_report(cfg, paths, development_rows(dataset, cfg["splits"]), "validation", "validation", entries)
    plot_validation_grids(paths, list(selection["families"]))
    built.append("validation")

    if paths.test_lock.exists():
        lock = read_json(paths.test_lock)
        build_split_report(cfg, paths, dataset, "test", "test", report_entries(lock["selection"]))
        built.append("test")

    exploratory_dir = paths.results / "test_exploratory"
    if exploratory_dir.exists():
        for family_dir in sorted(p for p in exploratory_dir.iterdir() if p.is_dir()):
            if family_dir.name in {"prevailing_mean", "ar1"}:
                continue
            record = read_json(paths.results / "validation" / family_dir.name / "selection.json")
            window = record["params"]["window"]
            exploratory_entries = [
                Entry(f"{family_dir.name} (exploratory)", family_dir.name, record["params"], True),
                Entry(f"Prevailing mean ({window}d)", "prevailing_mean", {"window": window}, False),
                Entry(f"AR(1) ({window}d)", "ar1", {"window": window}, False),
            ]
            output_name = f"test_exploratory/{family_dir.name}"
            build_split_report(cfg, paths, dataset, "test", "test_exploratory", exploratory_entries, output_name)
            built.append(output_name)
    return built
