"""Headless scientific figures; saved as PNG and vector PDF."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

LABELS = {'learned_y': 'Learned: observations', 'learned_yu': 'Learned: observations + actions',
          'exact_yu': 'Exact history map'}
COLORS = ['#2563eb', '#e87924', '#0d9488', '#9333ea']
FLOOR = 1e-24
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 170})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{extension}', bbox_inches='tight')
    plt.close(fig)


def experiment_plots(rows, config, traces, reference, directory):
    contexts, executions = config['contexts'], config['executions']
    names = list(LABELS)
    subtitle = (f"Prediction H={config['horizon']} | h={config['observability_index']} | "
                f"process noise SD={config['noise']:g} | {config['evaluation_trajectories']} evaluation episodes")
    grids = []
    for name in names:
        grid = np.full((len(contexts), len(executions)), np.nan)
        for row in rows:
            if row['policy'] == name:
                grid[contexts.index(row['context']), executions.index(row['execution'])] = row['trajectory_mse']
        grids.append(grid)
    values = np.concatenate([g[np.isfinite(g)] for g in grids])
    norm = LogNorm(vmin=max(FLOOR, values.min()), vmax=max(FLOOR*10, values.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
    for ax, name, grid in zip(axes, names, grids):
        im = ax.imshow(np.ma.masked_invalid(np.maximum(grid, FLOOR)), norm=norm, cmap='viridis', aspect='auto')
        ax.set(xticks=range(len(executions)), xticklabels=executions, yticks=range(len(contexts)),
               yticklabels=contexts, xlabel='Execution horizon ℓ', ylabel='Context r', title=LABELS[name])
        for i in range(len(contexts)):
            for j in range(len(executions)):
                v = grid[i,j]
                label = 'N/A' if np.isnan(v) else ('<1e−24' if v < FLOOR else f'{v:.1e}')
                color = 'black' if np.isnan(v) or norm(max(v, FLOOR)) > .6 else 'white'
                ax.text(j, i, label, ha='center', va='center', color=color, fontsize=9)
        ax.axhline(.5, color='white', linestyle='--', linewidth=1)
    fig.colorbar(im, ax=axes, label='Mean squared state-trajectory error (shared log scale)', shrink=.8)
    fig.suptitle('How context and execution horizon affect imitation\n'+subtitle, fontsize=13)
    fig.supxlabel('Lower is better. Dashed line: r reaches h=2. Values below 1e−24 are numerical zero; N/A = insufficient context.', fontsize=9)
    save(fig, directory, 'context_execution')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for name, marker in [('learned_y','o'), ('learned_yu','s')]:
        for r, color in zip(contexts, COLORS):
            subset = sorted([v for v in rows if v['policy']==name and v['context']==r], key=lambda v:v['execution'])
            axes[0].plot([v['execution'] for v in subset], [max(FLOOR,v['trajectory_mse']) for v in subset],
                         marker=marker, color=color, linestyle='-' if name=='learned_yu' else '--')
            axes[1].scatter([max(FLOOR,v['validation_mse']) for v in subset],
                            [max(FLOOR,v['trajectory_mse']) for v in subset], marker=marker, color=color, s=45,
                            label=f"r={r}, {'y,u' if name=='learned_yu' else 'y'}")
    axes[0].set(xlabel='Execution horizon ℓ', ylabel='Mean squared trajectory error', yscale='log', xticks=executions,
                title='Feedback frequency: same fitted policy')
    axes[1].set(xlabel='Held-out action-chunk MSE', ylabel='Mean squared trajectory error', xscale='log', yscale='log',
                title='Prediction accuracy versus rollout accuracy')
    for ax in axes: ax.grid(alpha=.2)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle('Learned-policy diagnostics\n'+subtitle, fontsize=13)
    fig.supxlabel('Each point is an evaluation mean, not a confidence interval. One training seed; smaller errors are better.', fontsize=9)
    save(fig, directory, 'learning_vs_control')

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, layout='constrained')
    for col, r in enumerate((1, 2)):
        for state in (0, 1):
            axes[state,col].plot(reference[:,state], color='black', linewidth=2, label='Full-state expert')
        expert_u = reference[:-1] @ np.asarray(config['K']).T
        axes[2,col].step(range(len(expert_u)), expert_u[:,0], where='post', color='black', linewidth=2)
        for ell, color in zip(executions, COLORS):
            tr = traces[('learned_yu',r,ell)]
            for state in (0,1):
                axes[state,col].plot(tr['x'][:,state], color=color, alpha=.85, label=f'ℓ={ell}')
            axes[2,col].step(range(len(tr['u'])), tr['u'][:,0], where='post', color=color, alpha=.85)
        axes[0,col].set_title(f'Context r={r} | learned observations + actions')
        for ax, label in zip(axes[:,col], ['Observed state x₁', 'Hidden state x₂', 'Action u']):
            ax.axvspan(0, config['warmup'], color='gray', alpha=.12)
            ax.set_ylabel(label); ax.grid(alpha=.2)
        axes[2,col].set_xlabel('Time step')
    axes[0,0].legend(ncol=3, fontsize=8)
    fig.suptitle('One rollout: what the controller actually does\n'+subtitle, fontsize=13)
    fig.supxlabel('First evaluation episode, selected before inspecting performance. Same initial state and disturbances; shaded region = expert warm-up.', fontsize=9)
    save(fig, directory, 'example_rollout')


def test_plots(reconstruction_errors, chunk_errors, directory):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    for ax, data, xlabel, title in zip(axes, [reconstruction_errors,chunk_errors],
                                     ['Context r','Execution horizon ℓ'],
                                     ['State recovery with arbitrary inputs','Exact chunks versus expert trajectory']):
        keys = sorted(data)
        vals = [max(data[k],1e-18) for k in keys]
        ax.bar([str(k) for k in keys], vals, color=COLORS[:len(keys)])
        ax.axhline(1e-12, color='#b91c1c', linestyle='--', label='Absolute error reference: 1e−12')
        ax.set(yscale='log', ylim=(1e-18,1e-11), xlabel=xlabel, ylabel='Maximum absolute error', title=title)
        ax.legend(fontsize=8)
        for i,k in enumerate(keys): ax.text(i, vals[i]*1.6, f'{data[k]:.1e}', ha='center', fontsize=9)
    fig.suptitle('Mathematical verification diagnostics', fontsize=14)
    fig.supxlabel('Recorded from the test calculations. Zero errors displayed at 1e−18. Test assertions determine pass/fail.', fontsize=9)
    save(fig, directory, 'mathematical_checks')
