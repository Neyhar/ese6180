"""Headless scientific figures; saved as PNG and vector PDF."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#2563eb', '#e87924', '#0d9488', '#9333ea', '#be123c', '#64748b']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 170})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{extension}', bbox_inches='tight')
    plt.close(fig)


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
