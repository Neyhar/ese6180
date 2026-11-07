"""Shared contracts for reading and comparing saved hard-example sweeps.

Configuration records define the expected deployment grid. A comparison must
reject missing/duplicate seeds and differing test states before aggregating.
"""
import csv
import json
from pathlib import Path
import numpy as np


def read_suite(folder):
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text())
    with (folder / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(row['seed']), int(row['context']), int(row['horizon']),
             int(row['execution'])) for row in rows]
    expected = {(s, r, h, e) for s in config['seeds'] for r in config['contexts']
                for h in config['horizons'] for e in config['executions'] if e <= h}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f'Missing or duplicate deployment records in {folder}')
    return config, rows, dict(zip(keys, rows))


def require_matched_tests(left, right):
    """Fail before comparison if test states or evaluation protocols differ."""
    left, right = Path(left), Path(right)
    if (left / 'test_states.csv').read_bytes() != (right / 'test_states.csv').read_bytes():
        raise ValueError('Comparison requires identical held-out test states')
    configs = [json.loads((path / 'config.json').read_text()) for path in (left, right)]
    for key in ['contexts', 'horizons', 'executions', 'seeds', 'steps',
                'long_steps', 'warmup', 'test_seed', 'test_pairs', 'mu', 'observation']:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Comparison protocols differ: {key}')


def require_single_action_agreement(index, contexts, seeds, horizon=16):
    """Check single-action training against a chunk predictor replanned each step."""
    for context in contexts:
        for seed in seeds:
            single = float(index[seed, context, 1, 1]['test_loss'])
            first = float(index[seed, context, horizon, 1]['test_loss'])
            if not np.isclose(single, first, rtol=1e-5, atol=1e-14):
                raise ValueError(f'First-action baseline differs: seed={seed}, r={context}')


def write_records(path, records):
    """Keep regime-specific diagnostics when concatenating result schemas."""
    if not records:
        raise ValueError('Cannot write an empty result table')
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
