# Daily SPY Market Timing

DSA5205 Project 2. Each day, a model forecasts SPY's next open-to-open excess return using
data known at today's close. A fixed rule turns the forecast into a SPY weight between 0% and
150%, with the rest in cash. This is a historical research exercise, not a live trading system.

The data, model, strategy, and evaluation layers are independent, so new models can be added
without changing the pipeline. The candidate models are Ridge, LASSO, Random Forest and XGBoost.

For XGBoost's design and leakage checks, see [the XGBoost notes](docs/xgboost.md) (in Chinese).

## Setup

Python 3.11 or newer is required. Tested on Python 3.12 under Ubuntu 24.04 (WSL2) and on
Python 3.11 under Windows 11. The code uses relative paths and has no OS-specific dependencies,
so it is intended to work on macOS too.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows (PowerShell), create and activate the environment like this, then run the same
commands as below:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
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
| 1. Start clean | `python -m timing.cli clean` | 1 s | Removes old files under `results/` |
| 2. Build data | `python -m timing.cli data` | About 20 s | `data/processed/dataset.csv`, `qa_report.json`, `manifest.json` |
| 3. Validate | `python -m timing.cli validate --model all` | About 10 to 20 min | `results/validation/`, `results/experiment_log.csv` |
| 4. Report on validation | `python -m timing.cli report` | About 30 s | `results/tables/validation/`, `results/figures/validation/` |
| 5. Test | `python -m timing.cli test --confirm-final` | About 5 to 10 min | `results/test/` |
| 6. Report on test | `python -m timing.cli report` | About 1 min | `results/tables/test/`, `results/figures/test/` |
| 7. Random Forest figures (optional) | `python -m timing.evaluation.random_forest_report` | About 15 s | `results/figures/validation/random_forest/` |
| 8. Automated checks | `pytest` | About 5 to 15 min | `75 passed` |

Times are approximate. Step 3 is slow because the two tree models are refitted every 21 trading
days for 8 settings each; Ridge, LASSO and the baselines take about 2 minutes together.
`pytest` uses synthetic data and needs no network.

**Step 5 uses the test period.** It evaluates each family's validation-selected setting and the
baselines. The code does not limit how often it runs; `--confirm-final` is only a reminder. The
team's rule is to run it after all models and settings are frozen, and not to change anything
afterwards based on test results.

### Expected output

With the fixed data end date, the dataset has 8,243 rows with decision dates from 1994-01-27 to
2026-10-28.

Step 3 prints the selection. 34 candidate settings are tried (Ridge 10, LASSO 8, Random Forest 8,
XGBoost 8) plus 4 baseline settings, and each one is logged in `results/experiment_log.csv`.
The final model is the candidate with the highest validation Sharpe net of 5 bps. In our runs
Ridge `lam=100.0__window=1260` had the highest value among the four candidate families:

| Family | Best validation setting | Validation net Sharpe |
|---|---|---|
| Ridge | `lam=100.0__window=1260` | 0.384 |
| LASSO | `alpha=0.001__window=1260` | 0.215 |
| Random Forest | `max_depth=3`, `min_samples_leaf=20`, `window=1260` | 0.103 |
| XGBoost | `max_depth=2`, `learning_rate=0.03`, `window=1260` | 0.073 |

For reference, the prevailing-mean baseline (`window=1260`) reached a validation net Sharpe of
0.404. Baselines cannot be selected as the final model, so it is reported only for comparison.

Step 6 prints the summary tables. For the test period (2011-01-03 to 2026-10-28, net of 5 bps),
these rows should match:

| Strategy | Annual return | Annual volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Ridge `lam=100.0__window=1260` | 11.1% | 17.8% | 0.59 | -44.0% |
| Prevailing mean (1260d) | 9.2% | 16.5% | 0.52 | -43.1% |
| AR(1) (1260d) | 4.2% | 17.5% | 0.23 | -36.0% |
| SPY buy and hold | 14.0% | 16.4% | 0.78 | -32.0% |

Yahoo's adjusted SPY prices can differ between requests. Two downloads made on 2026-11-05 differed
by up to about 1.5e-6 in relative terms, which changed `data_hash` in `manifest.json` and moved
one reported Sharpe by 0.0001, but did not change the selected model. Yahoo may also revise its history, so a much later
download could differ by more.

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
timing/pipeline.py   Validate and test workflows
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
| Test | 2011-01-01 to 2026-10-30, run after all models are frozen |
| Rolling fit | Refit every 21 trading days, using only labels realized by the fit date |
| Position | `1 + 0.5 z`, clipped to 0-1.5, with a 10-point no-trade band |
| Cost | 5 bps per unit of traded weight (an assumption), with sensitivity at 0, 1, 2, 5, and 10 bps |
| Leverage | Weights above 100% borrow at the ^IRX rate (an optimistic assumption) |
| Selection | Highest validation Sharpe net of base costs among candidate models |
| Seed | 42 (`configs/base.yaml`) |

## Data and reproducibility

- Market data are not included. `yfinance` data are for research and education and must not be
  redistributed, so step 2 downloads them again.
- No API keys are needed.
- When packaging the code, leave out `data/` and any virtual environment folders
  (`.venv/`, `.venv-win/`); `.gitignore` already excludes them.
- Stooq is an optional cross-check. If the request fails, QA records it as skipped and continues.
