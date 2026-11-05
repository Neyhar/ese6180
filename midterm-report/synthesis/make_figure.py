"""Regenerate the publication figure from this revision's frozen metrics."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ese618-matplotlib')
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'data/metrics.csv').open()))
summary=list(csv.DictReader((root/'data/context_summary.csv').open()))
contexts=[1,2,4,8,16]; executions=[1,2,4,8,16]
plt.rcParams.update({'font.size':18,'axes.titlesize':17,'axes.labelsize':17,'xtick.labelsize':15,'ytick.labelsize':15})
fig,ax=plt.subplots(1,3,figsize=(16,5.6))
for ell in executions:
 g=[[v for v in rows if int(v['context'])==r and int(v['execution'])==ell and v['horizon']=='16'] for r in contexts]
 ax[0].plot(contexts,[np.median([float(v['test_loss']) for v in group]) for group in g],'o-',label=f'ℓ={ell}')
 ax[1].plot(contexts,[max(float(v['per_step_radius']) for v in group) for group in g],'o-')
ax[0].axhline(1e-5,c='gray',ls=':');ax[0].set(yscale='log',xlabel='History r',ylabel='Mean test MSE',title='Tracking performance')
ax[0].legend(fontsize=13,ncol=2)
ax[1].axhline(1,c='red',ls=':');ax[1].set(xlabel='History r',ylabel='Worst per-step radius',title='Below 1: stable')
for row in summary:
 ell=int(row['shortest_execution']) if row['shortest_execution'] else None
 ax[2].scatter(int(row['context']),18 if ell is None else ell,marker='x' if ell is None else 'o',c='red' if ell is None else 'tab:blue',s=60)
ax[2].set_yticks([1,2,4,8,16,18],['1','2','4','8','16','None'])
ax[2].set(xlabel='History r',ylabel='Shortest passing ℓ',title='Need for chunking')
for a in ax:a.set_xticks(contexts);a.grid(alpha=.2)
fig.suptitle('H=16; passing requires all-seed stability and mean test MSE ≤ 10⁻⁵',fontsize=18)
fig.tight_layout();fig.savefig(root/'figures/context_hypothesis.png',dpi=220)
