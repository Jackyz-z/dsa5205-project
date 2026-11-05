"""Walk-forward training and forecasting, independent of any specific model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from timing.config import ModelSpec
from timing.data.features import FEATURE_SETS
from timing.engine.preprocess import Preprocessor, has_missing
from timing.models.registry import build_model


@dataclass
class RollingResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame | None


def run_rolling(
    dataset: pd.DataFrame,
    spec: ModelSpec,
    params: dict[str, Any],
    refit_every: int,
    prediction_start: str,
    seed: int,
) -> RollingResult:
    """Refit every `refit_every` rows and forecast each row with the latest fit.

    At a refit on date t the model sees only rows with label_end <= t, so a label that
    is still unrealized at t cannot enter training. A fit is skipped until a full window
    of realized labels exists, which is why early dates may have no forecast.
    """
    if spec.feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set {spec.feature_set!r}")
    window = params["window"]
    model_params = {k: v for k, v in params.items() if k != "window"}

    X_all = dataset[FEATURE_SETS[spec.feature_set]]
    y_all = dataset["y"]
    dates = dataset.index
    label_end = pd.DatetimeIndex(dataset["label_end"])

    first_pos = int(dates.searchsorted(pd.Timestamp(prediction_start)))
    refit_positions = list(range(first_pos, len(dates), refit_every))

    prediction_blocks = []
    diagnostic_rows = []
    for block_start in refit_positions:
        fit_date = dates[block_start]
        n_known = int(label_end.searchsorted(fit_date, side="right"))
        if n_known < window:
            continue
        train_X = X_all.iloc[n_known - window : n_known]
        train_y = y_all.iloc[n_known - window : n_known]

        block = X_all.iloc[block_start : block_start + refit_every]
        preprocessor = Preprocessor(spec.preprocess).fit(train_X)
        model = build_model(spec.name, model_params, seed)
        model.fit(preprocessor.transform(train_X), train_y)
        block_input = preprocessor.transform(block)
        if has_missing(block_input):
            raise ValueError(f"Missing feature values in the block starting {fit_date.date()}")
        y_hat = np.asarray(model.predict(block_input), dtype=float)
        if y_hat.shape != (len(block),):
            raise ValueError(f"{spec.name}.predict returned shape {y_hat.shape}; expected ({len(block)},)")

        prediction_blocks.append(
            pd.DataFrame(
                {
                    "y_hat": y_hat,
                    "y_true": y_all.iloc[block_start : block_start + refit_every],
                    "fit_date": fit_date,
                    "train_start": train_X.index[0],
                    "train_end": train_X.index[-1],
                },
                index=block.index,
            )
        )
        diagnostics = model.diagnostics()
        if diagnostics is not None:
            diagnostic_rows.append({"fit_date": fit_date, **diagnostics})

    if not prediction_blocks:
        raise ValueError(f"{spec.name} {params}: no window had enough realized labels")
    predictions = pd.concat(prediction_blocks)
    predictions.index.name = "date"
    diagnostics_frame = pd.DataFrame(diagnostic_rows).set_index("fit_date") if diagnostic_rows else None
    return RollingResult(predictions, diagnostics_frame)
