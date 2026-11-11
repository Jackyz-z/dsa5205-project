"""Validation-only RF figures from saved predictions; no fitting or test evaluation."""
from __future__ import annotations

import hashlib
import pandas as pd
import matplotlib.pyplot as plt

from timing.config import PROJECT_ROOT, Paths, config_id, load_base_config, read_json, write_json
from timing.data.dataset import load_dataset
from timing.engine.splits import development_rows
from timing.evaluation.report import Entry, build_split_report
from timing.evaluation import plots


def main():
    paths = Paths(PROJECT_ROOT)
    cfg = load_base_config(paths)
    source = paths.results / 'validation' / 'random_forest'
    record = read_json(source / 'selection.json')
    grid = pd.read_csv(source / 'grid_metrics.csv')
    window = record['params']['window']
    entries = [
        Entry('Random Forest', 'random_forest', record['params'], True),
        Entry(f'Prevailing mean ({window}d)', 'prevailing_mean', {'window': window}, False),
        Entry(f'AR(1) ({window}d)', 'ar1', {'window': window}, False),
    ]
    output = 'validation/random_forest'
    tables = build_split_report(cfg, paths, development_rows(load_dataset(paths), cfg['splits']),
                                'validation', 'validation', entries, output_name=output)
    actual = tables['summary'].iloc[0]['sharpe']
    if abs(actual - record['validation_sharpe']) > 1e-6:
        raise ValueError('Recomputed Sharpe differs from saved validation. Check data/config versions.')
    figure_dir = paths.results / 'figures' / output
    table_dir = paths.results / 'tables' / output
    ordered = grid.sort_values(['sharpe', 'config_id'], ascending=[False, True]).reset_index(drop=True)
    ordered.to_csv(table_dir / 'grid_metrics.csv', index=False)
    labels = [f"Window {int(r.param_window)} | depth {int(r.param_max_depth)} | leaf {int(r.param_min_samples_leaf)}" for r in ordered.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    colors = [plots.CANDIDATE_COLORS[0] if c == record['config_id'] else '#a7b9cb' for c in ordered.config_id]
    ax.barh(range(len(ordered)), ordered.sharpe, color=colors)
    ax.set_yticks(range(len(ordered)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=plots.AXIS)
    ax.set_xlabel('Validation net Sharpe (5 bps base cost)')
    ax.set_title('Random Forest: all 8 predeclared configurations\n200 trees; max_features=1.0; selected configuration in blue')
    ax.grid(axis='y', visible=False)
    for i, val in enumerate(ordered.sharpe):
        ax.annotate(f'{val:.4f}', (val, i), xytext=(4 if val >= 0 else -4, 0), textcoords='offset points', ha='left' if val >= 0 else 'right', va='center', fontsize=9)
    ax.margins(x=0.22)
    plots._save(fig, figure_dir / 'grid_random_forest.png')
    inputs = [source / 'selection.json', source / 'grid_metrics.csv', paths.dataset, paths.configs / 'base.yaml']
    for e in entries:
        inputs.append(paths.results / 'validation' / e.family / config_id(e.params) / 'predictions.csv')
    write_json(table_dir / 'figure_manifest.json', {
        'split': 'validation', 'selected_params': record['params'],
        'data_hash': read_json(paths.manifest)['data_hash'],
        'inputs_sha256': {str(p.relative_to(paths.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'note': 'Uses shared accounting; validation-selected RF is not the group final model. No test data evaluated.'})
    print('Saved six validation figures:', figure_dir)
    print('Saved tables and provenance:', table_dir)
    print(tables['summary'][['label', 'ann_return', 'sharpe', 'max_drawdown']].to_string(index=False))


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
