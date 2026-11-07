"""Shared display labels and PNG/SVG export for hard-example figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def training_label(training):
    return {"narrow":"Noiseless training data", "recovery":"Noisy training data"}[training]


def save_figure(fig, folder, name):
    folder.mkdir(parents=True,exist_ok=True)
    roots=[path for path in (folder,*folder.parents) if (path/"config.json").exists()]
    if roots:
        family=json.loads((roots[0]/"config.json").read_text()).get("family","coupled")
        title=fig._suptitle.get_text() if fig._suptitle is not None else ""
        if "Two different n=2" not in title and "reference alongside" not in title:
            prefix="Original Simchowitz n=2" if family=="original" else "Coupled family"
            fig.suptitle(prefix+(" | "+title if title else ""),fontsize=11)
    if any("training data" in ax.get_title() for ax in fig.axes):
        title = fig._suptitle.get_text() if fig._suptitle is not None else ""
        fig.suptitle(title+"\nNoise = action perturbations during data collection; all test rollouts are noiseless",fontsize=11)
    fig.savefig(folder/f"{name}.png",dpi=180)
    fig.savefig(folder/f"{name}.svg")
    plt.close(fig)


