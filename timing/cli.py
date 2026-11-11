"""Command-line entry point: python -m timing.cli <command>."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import timing.models  # noqa: F401  (registers every model)
from timing.config import PROJECT_ROOT, Paths, load_base_config
from timing.data.dataset import build_and_save, load_dataset
from timing.data.download import download_all
from timing.data.qa import run_qa
from timing.evaluation.report import build_reports
from timing.pipeline import GateError, clean_results, run_test, set_seed, validate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m timing.cli", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="download raw data, run QA, and build the dataset")
    data.add_argument("--refresh", action="store_true", help="download again even if files are cached")

    validate_cmd = commands.add_parser("validate", help="run a model grid on the validation period")
    validate_cmd.add_argument("--model", default="all", help="model name, or 'all' (default)")

    test = commands.add_parser("test", help="run the validation-selected models on the test period")
    test.add_argument("--confirm-final", action="store_true", help="confirm that all models are frozen")

    commands.add_parser("report", help="build tables and figures from saved predictions")
    commands.add_parser("clean", help="delete everything under results/")
    return parser


def _print_summary(path) -> None:
    columns = ["label", "ann_return", "ann_vol", "sharpe", "max_drawdown", "ann_turnover", "oos_r2_vs_prevailing_mean"]
    frame = pd.read_csv(path)
    print(frame[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = Paths(PROJECT_ROOT)
    cfg = load_base_config(paths)
    set_seed(cfg["seed"])

    if args.command == "data":
        download_all(cfg, paths.raw, refresh=args.refresh)
        run_qa(cfg, paths)
        table = build_and_save(cfg, paths)
        print(f"Dataset: {len(table)} rows, {table.index.min().date()} to {table.index.max().date()}")
        print(f"Wrote {paths.dataset.relative_to(paths.root)} and {paths.qa_report.relative_to(paths.root)}")
        return 0

    if args.command == "clean":
        removed = clean_results(paths)
        print(f"Removed {removed} generated files under results/.")
        return 0

    dataset = load_dataset(paths)

    if args.command == "validate":
        selection = validate(cfg, paths, dataset, args.model)
        final = selection["final"]
        print(f"Selected {final['family']} {final['config_id']}: validation net Sharpe {final['validation_sharpe']:.3f}")
        return 0

    if args.command == "test":
        try:
            run_test(cfg, paths, dataset, args.confirm_final)
        except GateError as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 2
        print("Test run complete. Build its tables and figures with: python -m timing.cli report")
        return 0

    if args.command == "report":
        built = build_reports(cfg, paths, dataset)
        for split in built:
            print(f"\n[{split}]")
            _print_summary(paths.results / "tables" / split / "summary.csv")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
