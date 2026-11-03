from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
source=root/'data'/'metrics.csv'
rows=list(csv.DictReader(source.open()))
def values(n,h,r,e):
 return np.array([float(x['tail_mse']) for x in rows if x['policy']=='chunk' and tuple(int(x[k]) for k in ['dimension','observability_index','context','execution'])==(n,h,r,e)])
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(6.5,2.05),layout='constrained')
es=[1,2,4,8,16]
for n,h,c in [(2,2,'#2769aa'),(10,4,'#d17a13'),(100,8,'#148477')]:
 v=np.array([values(n,h,h,e) for e in es]); m=v.mean(axis=1)
 axs[0].plot(es,m,'o-',ms=3,color=c,label=f'n={n}, h=r={h}')
 axs[0].fill_between(es,v.min(axis=1),v.max(axis=1),alpha=.13,color=c)
axs[0].set_title('(a) Context at observability index',fontsize=10)
for r,c in [(3,'#d17a13'),(4,'#2769aa')]:
 v=np.array([values(10,4,r,e) for e in es]);m=v.mean(axis=1); ok=np.isfinite(m)
 axs[1].plot(np.array(es)[ok],m[ok],'o' if r==3 else 'o-',ms=4,color=c,label=f'r={r}')
 if r==4:axs[1].fill_between(es,v.min(axis=1),v.max(axis=1),alpha=.13,color=c)
axs[1].scatter([1,4,8],[.02]*3,marker='x',s=35,color='#b12a30',zorder=4,label='r=3: failure')
axs[1].set_ylim(4e-6,5e-2)
axs[1].set_title('(b) Fixed system: n=10, h=4',fontsize=10)
for ax in axs:
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(es,labels=es)
 ax.set_xlabel(r'Execution length $\ell$');ax.grid(alpha=.18);ax.legend(fontsize=7,frameon=False,loc='best')
axs[0].set_ylabel('Tail MSE / coordinate')
fig.savefig(root/'figures'/'history_execution.pdf',bbox_inches='tight')
fig.savefig(root/'figures'/'history_execution.png',dpi=220,bbox_inches='tight')
