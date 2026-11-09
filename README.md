# Daily SPY Market Timing

DSA5205 Project 2. Each day, a model forecasts SPY's next open-to-open excess return using
data known at today's close. A fixed rule turns the forecast into a SPY weight between 0% and
150%, with the rest in cash. This is a historical research exercise, not a live trading system.

The data, model, strategy, and evaluation layers are independent, so new models can be added
without changing the pipeline. The candidate models are Ridge, LASSO, Random Forest and XGBoost.

For XGBoost's design, leakage checks, run commands and feature-branch submission steps,
see [the XGBoost guide](docs/xgboost.md).

## Setup

Python 3.11 or newer is required. Tested on Python 3.12 under Ubuntu 24.04 (WSL2). The code
uses relative paths and has no OS-specific dependencies, so it is intended to work on macOS too.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3 -m venv` fails on Ubuntu because `ensurepip` is missing, either run
`sudo apt install python3-venv` or use [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

Run these from the project root, in order.

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Build data | `python -m timing.cli data` | About 20 s | `data/processed/dataset.csv`, `qa_report.json`, `manifest.json` |
| 2. Validate | `python -m timing.cli validate --model all` | About 90 s | `results/validation/`, `results/experiment_log.csv` |
| 3. Report on validation | `python -m timing.cli report` | About 15 s | `results/tables/validation/`, `results/figures/validation/` |
| 4. One-shot test | `python -m timing.cli test --confirm-final` | Under 1 min | `results/test/`, `results/test/lock.json` |
| 5. Report on test | `python -m timing.cli report` | About 30 s | `results/tables/test/`, `results/figures/test/` |

Automated checks use synthetic data and take about 5 minutes:

```bash
pytest
```

**Step 4 uses the test set. Run it only after all models are frozen.** Times are approximate:
steps 1 to 3 and `pytest` were measured under WSL2, and steps 4 and 5 are estimates because the
real test set has not been run.

### Expected output of steps 1 to 3

With the fixed data end date, the dataset has 8,243 rows with decision dates from 1994-01-27 to
2026-10-28. Validation selects `ridge lam=100.0__window=1260`, with a validation net Sharpe of
0.384. LASSO was added after the test lock; its best validation configuration is
`alpha=0.001__window=1260`, with a validation net Sharpe of 0.215. Its test-period output is
therefore kept under `results/test_exploratory/` and does not change the frozen selection.

Yahoo's adjusted SPY prices can differ between requests. Two downloads made on 2026-11-05 differed
by up to about 1.5e-6 in relative terms, which changed `data_hash` in `manifest.json` and moved
one reported Sharpe by 0.0001, but did not change the selected model. Yahoo may also revise its
history, so a much later download could differ by more.

## Add a model

1. Add `timing/models/my_model.py`:

   ```python
   from timing.models.base import BaseModel
   from timing.models.registry import register

   @register("my_model")
   class MyModel(BaseModel):
       def __init__(self, my_param, seed=0):
           super().__init__(seed)
           self.my_param = my_param

       def fit(self, X, y):
           ...
           return self

       def predict(self, X):
           ...  # Return a one-dimensional numpy array with one value per row of X.
   ```

2. Add `configs/models/my_model.yaml`:

   ```yaml
   name: my_model
   role: candidate
   feature_set: core
   preprocess: standard   # or none
   grid:
     window: [252, 1260]  # Required; used by the rolling engine.
     my_param: [0.1, 1.0]
   ```

3. Run `pytest tests/test_models.py`, then `python -m timing.cli validate --model my_model`.

No other file changes. The contract tests pick up the new model automatically.

## Project layout

```text
configs/             Shared settings and one YAML file per model family
timing/data/         Download, QA, features, and the dataset contract
timing/models/       BaseModel, registry, Ridge, LASSO, Random Forest, XGBoost, baselines
timing/engine/       Time splits, preprocessing, and the rolling walk-forward
timing/strategy/     Forecast-to-weight rule and the backtest
timing/evaluation/   Metrics, figures, and tables
timing/pipeline.py   Validate and test workflows, including the test lock
timing/cli.py        Command-line entry point
tests/               Leakage, accounting, contract, and end-to-end tests
```

## Design summary

| Item | Setting |
|---|---|
| Decision and fill | Decide after the close of day t; trade at the open of t+1; hold to the open of t+2 |
| Data | Yahoo Finance through `yfinance`: SPY, ^VIX, ^TNX, ^IRX |
| Label | SPY open-to-open return minus the cash return over the same period |
| Validation | 2000-03-01 to 2010-12-31 |
| Test | 2011-01-01 to 2026-10-30, used once |
| Rolling fit | Refit every 21 trading days, using only labels realized by the fit date |
| Position | `1 + 0.5 z`, clipped to 0-1.5, with a 10-point no-trade band |
| Cost | 5 bps per unit of traded weight (an assumption), with sensitivity at 0, 1, 2, 5, and 10 bps |
| Selection | Highest validation Sharpe net of base costs among candidate models |
| Seed | 42 (`configs/base.yaml`) |

## Data and reproducibility

- Market data are not included. `yfinance` data are for research and education and must not be
  redistributed, so step 1 downloads them again.
- No API keys are needed.
- Stooq is an optional cross-check. If the request fails, QA records it as skipped and continues.
