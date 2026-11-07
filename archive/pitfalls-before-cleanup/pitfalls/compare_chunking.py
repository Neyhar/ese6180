"""Matched hard-example simulations: replan each step versus execute action chunks.

Uses chunk_sweep.py and the shared learner/plant/evaluation classes. H=1,ell=1
is a one-action learner; H=16,ell=1 isolates chunk prediction from commitment.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ese618-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_io import (read_suite, require_matched_tests,
                           require_single_action_agreement, write_records)

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'chunking_ablation')
    parser.add_argument('--summarize-only', action='store_true', help='Reuse saved runs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {}
    for training in ['narrow', 'recovery']:
        folder = args.output/training
        if not args.summarize_only:
            subprocess.run([sys.executable, str(HERE/'chunk_sweep.py'),
                '--observation', 'partial', '--training', training,
                '--contexts', '2', '8', '16', '--horizons', '1', '16',
                '--executions', '1', '4', '8', '16', '--seeds', '718', '719', '720',
                '--test-seed', '20261106', '--output', str(folder)], check=True)
        _, data[training], index = read_suite(folder)
        require_single_action_agreement(index, [2,8,16], [718,719,720])
    require_matched_tests(args.output/'narrow', args.output/'recovery')
    summary, pairs = [], []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex='col', constrained_layout=True)
    for col, (training, rows) in enumerate(data.items()):
        lookup = {(int(x['seed']), int(x['context']), int(x['horizon']), int(x['execution'])): x for x in rows}
        for r, color in zip([2, 8, 16], ['#40599a', '#189d8f', '#b08c16']):
            means, lows, highs, rates = [], [], [], []
            for ell in [1, 4, 8, 16]:
                selected = [lookup[seed, r, 16, ell] for seed in [718, 719, 720]]
                loss = np.array([float(x['test_loss']) for x in selected])
                rate = max(float(x['per_step_radius']) for x in selected)
                means.append(loss.mean()); lows.append(loss.min()); highs.append(loss.max()); rates.append(rate)
                summary.append(dict(training=training, history=r, prediction=16, execution=ell,
                    mean_test_mse=loss.mean(), min_seed_mse=loss.min(), max_seed_mse=loss.max(),
                    worst_rate=rate, stable_seeds=sum(x['stable']=='True' for x in selected)))
            axes[0,col].plot([1,4,8,16],means,'o-',color=color,label=f'History {r}')
            axes[0,col].fill_between([1,4,8,16],lows,highs,color=color,alpha=.15)
            axes[1,col].plot([1,4,8,16],rates,'o-',color=color)
            for seed in [718,719,720]:
                single = float(lookup[seed,r,1,1]['test_loss'])
                feedback = float(lookup[seed,r,16,1]['test_loss'])
                chunk = float(lookup[seed,r,16,16]['test_loss'])
                pairs.append(dict(training=training,history=r,seed=seed,
                    single_action_mse=single,predict16_execute1_mse=feedback,
                    predict16_execute16_mse=chunk,chunk_to_feedback_ratio=chunk/feedback))
        axes[0,col].set(title=f'Partial sensing / {training} training',yscale='log',ylabel='Mean test tracking MSE')
        axes[0,col].legend()
        axes[1,col].axhline(1,color='red',ls='--',label='Stability boundary')
        axes[1,col].set(xlabel='Execution length (1 = no open-loop commitment)',ylabel='Worst-seed spectral rate',xticks=[1,4,8,16])
        for ax in axes[:,col]: ax.grid(alpha=.2)
    fig.suptitle('Matched controllers: with and without action commitment\nH=16; 3 fresh training seeds; shading = seed range; 20 held-out perturbed rollouts')
    fig.savefig(args.output/'chunking_comparison.png',dpi=190)
    plt.close(fig)
    write_records(args.output/'summary.csv', summary)
    write_records(args.output/'paired_comparison.csv', pairs)
    lines=['# With and without action chunking','',
        'Fresh matched simulations use the existing hard-example learner and evaluation pipeline. '
        'All conditions use the same 20 perturbed held-out states, 200 scored steps, 15-step expert '
        'warm-up, no continuing noise, and training seeds 718–720. Test seed: 20261106. '
        'These differ from the earlier exploratory runs. Each seed trains 32 trajectories of 80 steps.', '',
        '## What is held fixed?', '',
        'For H=16 the same fitted policy is evaluated with execution lengths 1, 4, 8, and 16. '
        'Execution 1 replans every step; larger lengths execute an open-loop prefix. '
        'A separate H=1, execution-1 learner verifies the genuinely single-action baseline. '
        'Target times match across H and history, and all 18 paired baseline losses agree within numerical tolerance. '
        'Both use separable multi-output ridge regression, so extra prediction heads do not change the first-action fit.', '',
        '## Results', '', '| Training | History | No commitment MSE | Execute 16 MSE | Chunk / feedback | Stable seeds (1 / 16) |',
        '|---|---:|---:|---:|---:|---|']
    for training in data:
        for r in [2,8,16]:
            a=next(x for x in summary if x['training']==training and x['history']==r and x['execution']==1)
            b=next(x for x in summary if x['training']==training and x['history']==r and x['execution']==16)
            lines.append(f"| {training} | {r} | {a['mean_test_mse']:.3g} | {b['mean_test_mse']:.3g} | {b['mean_test_mse']/a['mean_test_mse']:.3g} | {a['stable_seeds']}/3 / {b['stable_seeds']}/3 |")
    lines += ['', 'A ratio below one favors chunking; above one favors replanning. These are ratios of seed-mean losses, not means of seed ratios.', '',
        '## Interpretation', '',
        'In these runs, narrow history-eight control is unstable without commitment and stable with execution eight or sixteen; execution sixteen lowers mean MSE by about 807 times. Recovery history-two control is stable throughout, but execution sixteen increases MSE by about 37 times. Narrow history-sixteen control is already stable without commitment, with only a modest chunking benefit. '
        'Narrow demonstrations omit corrective behavior. Committing to a decaying predicted chunk can suppress destabilizing feedback; longer history can also change the ridge-selected controller. '
        'With recovery data and sufficient history to infer hidden state, frequent feedback can correct departures sooner. '
        'Read the stability panel alongside loss: a low finite-horizon MSE does not establish stability. '
        'This comparison tests these competing mechanisms rather than presuming chunking always helps.', '',
        'The mean includes every seed; shading is a range, not a confidence interval. Three seeds, a stable two-dimensional plant, small initial departures, expert initialization, and noiseless deployment limit generalization. '
        'The reference expert runs separately for scoring; the deployed learner never queries it.', '',
        '## Reproduce', '', '```sh', 'python3 pitfalls/compare_chunking.py',
        '# Rebuild only the summary and figure:', 'python3 pitfalls/compare_chunking.py --summarize-only', '```', '',
        '`summary.csv` aggregates seed means; `paired_comparison.csv` preserves matched comparisons. '
        'The narrow/ and recovery/ folders retain configs, raw rollout losses, test states, and long-run diagnostics.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[12:23]))
    print('Saved comparison to',args.output)

if __name__=='__main__': main()
