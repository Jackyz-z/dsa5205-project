"""End-to-end runs on synthetic data."""

import pandas as pd
import pytest

import timing.models  # noqa: F401
from timing.config import list_model_names, load_model_spec
from timing.evaluation.report import build_reports
from timing.pipeline import GateError, clean_results, run_test, validate


def test_validation_tries_the_whole_grid_and_applies_the_rule(sandbox):
    paths, cfg, dataset = sandbox
    selection = validate(cfg, paths, dataset, "ridge")
    grid = pd.read_csv(paths.results / "validation" / "ridge" / "grid_metrics.csv")
    assert len(grid) == 10
    assert selection["final"]["family"] == "ridge"
    assert selection["final"]["validation_sharpe"] == pytest.approx(grid["sharpe"].max(), abs=1e-5)


@pytest.mark.parametrize("model", ["ridge", "xgboost"])
def test_validation_never_reads_test_rows(sandbox, model):
    paths, cfg, dataset = sandbox
    validate(cfg, paths, dataset, model)
    first = pd.read_csv(paths.results / "validation" / model / "grid_metrics.csv")

    tampered = dataset.copy()
    tampered.loc[tampered["label_end"] > pd.Timestamp(cfg["splits"]["validation_end"]), ["y", "spy_ret"]] = 99.0
    validate(cfg, paths, tampered, model)
    second = pd.read_csv(paths.results / "validation" / model / "grid_metrics.csv")
    pd.testing.assert_frame_equal(first, second)


def test_every_validation_run_is_logged(sandbox):
    paths, cfg, dataset = sandbox
    validate(cfg, paths, dataset, "all")
    log = pd.read_csv(paths.experiment_log)
    assert len(log[log["family"] == "ridge"]) == 10
    assert len(log[log["family"] == "lasso"]) == 8
    assert set(log["family"]) == set(list_model_names(paths))
    for name in list_model_names(paths):
        assert len(log[log["family"] == name]) == len(load_model_spec(paths, name).configs())


def test_baselines_cannot_be_the_final_model(sandbox):
    paths, cfg, dataset = sandbox
    selection = validate(cfg, paths, dataset, "all")
    assert selection["final"]["role"] == "candidate"


def test_test_requires_validation_and_confirmation(sandbox):
    paths, cfg, dataset = sandbox
    with pytest.raises(GateError, match="validation"):
        run_test(cfg, paths, dataset, confirm_final=True)
    validate(cfg, paths, dataset, "all")
    with pytest.raises(GateError, match="confirm-final"):
        run_test(cfg, paths, dataset, confirm_final=False)
    assert not (paths.results / "test").exists()


def test_test_runs_every_selected_family_and_is_repeatable(sandbox):
    paths, cfg, dataset = sandbox
    selection = validate(cfg, paths, dataset, "all")
    run_test(cfg, paths, dataset, confirm_final=True)
    for name, record in selection["families"].items():
        assert (paths.results / "test" / name / record["config_id"] / "predictions.csv").exists()
    first = pd.read_csv(next((paths.results / "test" / "ridge").glob("*/predictions.csv")))

    run_test(cfg, paths, dataset, confirm_final=True)
    second = pd.read_csv(next((paths.results / "test" / "ridge").glob("*/predictions.csv")))
    pd.testing.assert_frame_equal(first, second)

    log = pd.read_csv(paths.experiment_log)
    assert set(log.loc[log["split"] == "test", "family"]) == set(list_model_names(paths))


def test_clean_removes_all_results(sandbox):
    paths, cfg, dataset = sandbox
    validate(cfg, paths, dataset, "ridge")
    assert clean_results(paths) > 0
    assert not [p for p in paths.results.rglob("*") if p.is_file()]


def test_reports_write_all_tables_and_figures(sandbox):
    paths, cfg, dataset = sandbox
    selection = validate(cfg, paths, dataset, "all")
    assert build_reports(cfg, paths, dataset) == ["validation"]
    run_test(cfg, paths, dataset, confirm_final=True)
    assert build_reports(cfg, paths, dataset) == ["validation", "test"]

    for split in ("validation", "test"):
        for table in ("summary", "subperiods", "yearly", "cost_sensitivity"):
            assert (paths.results / "tables" / split / f"{table}.csv").exists()
        for figure in ("cumulative_returns", "cumulative_r2", "subperiods", "cost_sensitivity", "weights", "coefficients"):
            assert (paths.results / "figures" / split / f"{figure}.png").exists()
    assert (paths.results / "figures" / "validation" / "grid_ridge.png").exists()

    summary = pd.read_csv(paths.results / "tables" / "test" / "summary.csv")
    assert summary["label"].tolist()[0] == f"{selection['final']['family']} (final)"
    assert "SPY buy & hold" in set(summary["label"])
