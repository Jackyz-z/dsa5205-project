"""Static report figures. Every figure has a matching CSV table in results/tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Colors follow the entity. Benchmarks keep fixed slots; candidates take the remainder in order.
SPY_COLOR = "#eb6834"
PREVAILING_COLOR = "#1baf7a"
AR1_COLOR = "#eda100"
CANDIDATE_COLORS = ["#2a78d6", "#e87ba4", "#008300", "#6250d6", "#e34948"]

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
    }
)


def series_color(family: str, candidate_order: list[str]) -> str:
    if family == "SPY":
        return SPY_COLOR
    if family == "prevailing_mean":
        return PREVAILING_COLOR
    if family == "ar1":
        return AR1_COLOR
    return CANDIDATE_COLORS[candidate_order.index(family) % len(CANDIDATE_COLORS)]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _legend_above(ax: plt.Axes, ncol: int) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.2, handlelength=1.6)


def _label_line_ends(ax: plt.Axes, ends: list[tuple[str, float, float]]) -> None:
    """Write 'value' labels at line ends, nudged apart so they cannot overlap."""
    min_gap_points = 12.0
    points_per_pixel = 72.0 / ax.figure.dpi
    positions = []
    for text, x, y in ends:
        numeric_x = float(ax.convert_xunits(x))
        positions.append([ax.transData.transform((numeric_x, y))[1] * points_per_pixel, text, x, y])
    positions.sort(key=lambda item: item[0])
    previous = -np.inf
    for item in positions:
        placed = max(item[0], previous + min_gap_points)
        previous = placed
        ax.annotate(
            item[1],
            xy=(item[2], item[3]),
            xytext=(6, placed - item[0]),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def plot_cumulative_returns(curves: list[dict], title: str, path: Path) -> None:
    """Each curve is a dict with label, color and wealth (a Series starting near 1)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
    ends = []
    for curve in curves:
        wealth = curve["wealth"]
        ax.plot(wealth.index, wealth.to_numpy(), color=curve["color"], linewidth=1.5, label=curve["label"])
        ends.append((f"{wealth.iloc[-1]:.2f}x", wealth.index[-1], float(wealth.iloc[-1])))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2, subs=(1.0, 1.5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title(title, pad=30)
    _legend_above(ax, ncol=len(curves))
    fig.canvas.draw()
    _label_line_ends(ax, ends)
    ax.margins(x=0.08)
    _save(fig, path)


def plot_cumulative_r2(curves: list[dict], title: str, path: Path) -> None:
    """The y-axis is scaled to data after the first year, when a few days no longer dominate."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8), layout="constrained")
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ends = []
    settled = []
    for curve in curves:
        series = curve["r2"] * 100
        ax.plot(series.index, series.to_numpy(), color=curve["color"], linewidth=1.5, label=curve["label"])
        ends.append((f"{series.iloc[-1]:.2f}%", series.index[-1], float(series.iloc[-1])))
        settled.append(series.iloc[252:] if len(series) > 252 else series)
    low, high = min(0.0, min(s.min() for s in settled)), max(0.0, max(s.max() for s in settled))
    padding = max((high - low) * 0.1, 0.01)
    ax.set_ylim(low - padding, high + padding)
    ax.set_ylabel("Cumulative out-of-sample R² (%)\ny-axis scaled from year 2")
    ax.set_title(title, pad=30)
    _legend_above(ax, ncol=len(curves))
    fig.canvas.draw()
    _label_line_ends(ax, ends)
    ax.margins(x=0.08)
    _save(fig, path)


def plot_subperiods(table: pd.DataFrame, colors: dict[str, str], title: str, path: Path) -> None:
    """Grouped bars of return and Sharpe by period, as two panels with their own scales."""
    periods = list(dict.fromkeys(table["period"]))
    labels = list(dict.fromkeys(table["label"]))
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True, layout="constrained")
    width = 0.8 / len(labels)
    x = np.arange(len(periods))
    for ax, column, ylabel, scale in (
        (axes[0], "ann_return", "Annualized return (%)", 100.0),
        (axes[1], "sharpe", "Sharpe ratio", 1.0),
    ):
        ax.axhline(0, color=AXIS, linewidth=1.0)
        ax.grid(axis="x", visible=False)
        for i, label in enumerate(labels):
            rows = table[table["label"] == label].set_index("period").reindex(periods)
            ax.bar(
                x + (i - (len(labels) - 1) / 2) * width,
                rows[column].to_numpy() * scale,
                width=width * 0.88,
                color=colors[label],
                label=label,
            )
        ax.set_ylabel(ylabel)
    axes[1].set_xticks(x, periods)
    axes[0].set_title(title, pad=30)
    _legend_above(axes[0], ncol=len(labels))
    _save(fig, path)


def plot_weights(backtest: pd.DataFrame, color: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.0), layout="constrained")
    ax.plot(backtest.index, backtest["weight"].to_numpy(), color=color, linewidth=0.8)
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    ax.set_ylabel("SPY weight")
    ax.set_ylim(-0.05, 1.65)
    ax.set_title(title)
    _save(fig, path)


def plot_cost_sensitivity(table: pd.DataFrame, colors: dict[str, str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.8), layout="constrained")
    ends = []
    for label, rows in table.groupby("label", sort=False):
        rows = rows.sort_values("cost_bps")
        ax.plot(rows["cost_bps"], rows["sharpe"], color=colors[label], linewidth=1.5, marker="o", markersize=5, label=label)
        ends.append((f"{rows['sharpe'].iloc[-1]:.2f}", rows["cost_bps"].iloc[-1], float(rows["sharpe"].iloc[-1])))
    ax.set_xlabel("One-way cost (bps of traded weight)")
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_title(title, pad=30)
    _legend_above(ax, ncol=table["label"].nunique())
    fig.canvas.draw()
    _label_line_ends(ax, ends)
    ax.margins(x=0.08)
    _save(fig, path)


def _diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    limit = max(float(np.nanmax(np.abs(values))), 1e-12)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def plot_grid_heatmap(pivot: pd.DataFrame, x_label: str, title: str, path: Path) -> None:
    """Validation Sharpe across a two-parameter grid; red is negative and blue is positive."""
    fig, ax = plt.subplots(figsize=(6.5, 1.2 + 0.7 * len(pivot)), layout="constrained")
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=DIVERGING, norm=_diverging_norm(values), aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [str(i) for i in pivot.index])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Window (trading days)")
    ax.grid(False)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color=INK, fontsize=9)
    fig.colorbar(image, ax=ax, label="Net Sharpe", shrink=0.85)
    ax.set_title(title)
    _save(fig, path)


def plot_coefficients(diagnostics: pd.DataFrame, title: str, path: Path) -> None:
    """Coefficient of each feature at each refit. The intercept is on a different scale."""
    frame = diagnostics.drop(columns=["intercept"], errors="ignore")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * frame.shape[1] + 1.6), layout="constrained")
    values = frame.to_numpy().T * 10_000
    extent = [matplotlib.dates.date2num(frame.index[0]), matplotlib.dates.date2num(frame.index[-1]), frame.shape[1] - 0.5, -0.5]
    image = ax.imshow(values, cmap=DIVERGING, norm=_diverging_norm(values), aspect="auto", extent=extent, interpolation="nearest")
    ax.xaxis_date()
    ax.set_yticks(range(frame.shape[1]), [c.removeprefix("f_") for c in frame.columns])
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Coefficient (bps per 1 std of feature)", shrink=0.9)
    ax.set_title(title)
    _save(fig, path)
