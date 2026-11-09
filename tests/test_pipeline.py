"""End-to-end runs on synthetic data, including the one-shot test rules."""

import pandas as pd
import pytest
import yaml

import timing.models  # noqa: F401
from timing.config import list_model_names, load_model_spec, read_json
from timing.evaluation.report import build_reports
from timing.pipeline import GateError, run_exploratory, run_test, validate


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
    assert not paths.test_lock.exists()


def test_locked_test_can_be_reproduced_but_not_changed(sandbox):
    paths, cfg, dataset = sandbox
    validate(cfg, paths, dataset, "all")
    run_test(cfg, paths, dataset, confirm_final=True)
    assert paths.test_lock.exists()
    first = pd.read_csv(next((paths.results / "test" / "ridge").glob("*/predictions.csv")))

    run_test(cfg, paths, dataset, confirm_final=True)
    second = pd.read_csv(next((paths.results / "test" / "ridge").glob("*/predictions.csv")))
    pd.testing.assert_frame_equal(first, second)

    changed = {**cfg, "costs": {**cfg["costs"], "base_bps": 0.0}}
    with pytest.raises(GateError, match="differs from the locked"):
        run_test(changed, paths, dataset, confirm_final=True)


def test_model_added_after_the_lock_stays_exploratory(sandbox):
    paths, cfg, dataset = sandbox
    validate(cfg, paths, dataset, "all")
    run_test(cfg, paths, dataset, confirm_final=True)
    locked_selection = read_json(paths.test_lock)["selection"]

    # A new family registered in YAML (reusing the ridge class under a new config is not
    # possible by name, so register a tiny subclass for this test).
    from timing.models.registry import _REGISTRY
    from timing.models.ridge import RidgeModel

    _REGISTRY.setdefault("ridge_variant", RidgeModel)
    try:
        with open(paths.configs / "models" / "ridge_variant.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "name": "ridge_variant",
                    "role": "candidate",
                    "feature_set": "core",
                    "preprocess": "standard",
                    "grid": {"window": [60], "lam": [5.0]},
                },
                f,
            )
        validate(cfg, paths, dataset, "ridge_variant")
        with pytest.raises(GateError, match="differs from the locked"):
            run_test(cfg, paths, dataset, confirm_final=True)

        run_exploratory(cfg, paths, dataset, "ridge_variant")
        assert (paths.results / "test_exploratory" / "ridge_variant").exists()
        assert not (paths.results / "test" / "ridge_variant").exists()
        assert read_json(paths.test_lock)["selection"] == locked_selection

        built = build_reports(cfg, paths, dataset)
        assert "test_exploratory/ridge_variant" in built
    finally:
        _REGISTRY.pop("ridge_variant", None)


def test_exploratory_requires_a_lock(sandbox):
    paths, cfg, dataset = sandbox
    validate(cfg, paths, dataset, "all")
    with pytest.raises(GateError, match="after the locked"):
        run_exploratory(cfg, paths, dataset, "ridge")


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
