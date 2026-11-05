"""Figures for the context/execution hypothesis; never clip raw losses."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(rows, baselines, paths, out, contexts, hs, steps, description="full observations / narrow training"):
    executions = sorted({v['execution'] for v in rows})
    fig,axes=plt.subplots(int(np.ceil(len(contexts)/2)),2,figsize=(12,4*int(np.ceil(len(contexts)/2))),sharex=True,sharey=True,squeeze=False)
    for ax in axes.flat[len(contexts):]: ax.set_visible(False)
    colors=plt.cm.viridis(np.linspace(.05,.9,len(hs)))
    for ax,r in zip(axes.flat,contexts):
        for H,c in zip(hs,colors):
            es=[e for e in executions if e<=H]; vals=[]
            for ell in es:
                g=[v for v in rows if v['context']==r and v['horizon']==H and v['execution']==ell]
                vals.append(np.median([v['test_loss'] for v in g]))
                if any(not v['stable'] for v in g): ax.scatter(ell,vals[-1],marker='x',c='red',zorder=5)
            ax.plot(es,vals,'o-',color=c,label=f'learned H={H}')
        for b,style in zip(baselines[2:],[':','-.','--',(0,(3,1,1,1))]): ax.axhline(b['test_loss'],color='gray',ls=style,label=b['controller'])
        ax.set(xscale='log',yscale='log',title=f'Context r={r}',xlabel='Execution length ℓ',ylabel='Mean squared tracking error')
        ax.set_xticks(hs,[str(h) for h in hs]); ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle(f'{description} | Hard example on A1: Held-out perturbed initial states; no continuing noise\nMean test loss per controller, median across training seeds; red × = unstable; K1 loss=0; wrong K2 (see baselines.csv) (both omitted for readability)')
    fig.tight_layout(rect=[0,.08,1,.93]); fig.savefig(out/'test_loss.png',dpi=170); plt.close(fig)
    # Use the largest prediction horizon to cover all execution lengths.
    # Earlier-output fits are invariant to H, so the other H curves overlap.
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (r, color) in enumerate(zip(contexts, plt.cm.viridis(np.linspace(.05,.85,len(contexts))))):
        shape = ['o','s','^','D','v'][i%5]
        medians, lows, highs = [], [], []
        for ell in executions:
            group = [v for v in rows if v['context']==r and v['horizon']==max(hs) and v['execution']==ell]
            values = [v['test_loss'] for v in group]
            medians.append(np.median(values)); lows.append(min(values)); highs.append(max(values))
            if any(not v['stable'] for v in group):
                ax.scatter(ell, medians[-1], marker='x', color='red', s=70, zorder=6)
        ax.plot(executions, medians, marker=shape, color=color, label=f'History r={r}')
        ax.fill_between(executions, lows, highs, color=color, alpha=.12)
    for b, style in zip(baselines[2:], [':','-.','--',(0,(3,1,1,1))]):
        ax.axhline(b['test_loss'],color='gray',ls=style,label=b['controller'])
    ax.set(xscale='log',yscale='log',xlabel='Execution horizon ℓ (steps before replanning)',
           ylabel='Mean squared tracking error on held-out test set',
           title=f'{description} | Hard example: history lengths compared on one plot\nH={max(hs)}; median over training seeds; shading = min–max; red × = unstable')
    ax.set_xticks(hs,[str(h) for h in hs]); ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.text(.5,.01,'Held-out perturbed test states; no continuing noise. K1 loss=0; wrong K2 (see baselines.csv) (omitted).',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1]); fig.savefig(out/'combined_test_loss.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    for name,xs in paths.items(): ax.semilogy(range(steps+1),np.maximum(abs(xs[:,0]),1e-16),label=name)
    ax.set(xlabel='Deployment step',ylabel='|x₁| (display floor 1e-16)',title=f'{description} | Hard-example sample rollout; no continuing noise')
    ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(out/'sample_rollouts.png',dpi=170); plt.close(fig)



def plot_hypothesis(rows, summary, out, contexts, executions, horizon, tolerance):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ell in executions:
        if ell > horizon: continue
        values = [np.median([v['test_loss'] for v in rows if v['context']==r
                  and v['horizon']==horizon and v['execution']==ell]) for r in contexts]
        axes[0].plot(contexts,values,'o-',label=f'ℓ={ell}')
    axes[0].axhline(tolerance,color='gray',ls=':',label='loss threshold')
    axes[0].set(yscale='log',xlabel='Observation history r',ylabel='Mean test tracking loss',title='Does history improve test performance?')
    axes[0].legend(fontsize=8)
    for ell in executions:
        if ell > horizon: continue
        radii = [max(v['per_step_radius'] for v in rows if v['context']==r
                 and v['horizon']==horizon and v['execution']==ell) for r in contexts]
        axes[1].plot(contexts,radii,'o-',label=f'ℓ={ell}')
    axes[1].axhline(1,color='red',ls=':')
    axes[1].set(xlabel='Observation history r',ylabel='Worst-seed per-step spectral radius',title='Below 1: asymptotically stable')
    none_level = max(executions)+2
    for cell in summary:
        selected = cell['shortest_execution']
        axes[2].scatter(cell['context'],none_level if selected is None else selected,
                        marker='x' if selected is None else 'o',color='red' if selected is None else 'tab:blue')
    axes[2].set_yticks(executions+[none_level],[str(v) for v in executions]+['None'])
    axes[2].set(xlabel='Observation history r',ylabel='Shortest passing execution ℓ',title='Does history reduce the need for chunking?')
    for ax in axes: ax.set_xticks(contexts); ax.grid(alpha=.2)
    fig.suptitle(f'H={horizon}; pass = every training seed stable and mean test MSE ≤ {tolerance:g}; ℓ=1 means no action chunking')
    fig.tight_layout(); fig.savefig(out/'context_hypothesis.png',dpi=180); plt.close(fig)
