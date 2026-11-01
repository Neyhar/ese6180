"""Dimension-independent synthesis of long-horizon experiments."""
from pathlib import Path
import numpy as np
from plots import plt, save, COLORS


def make_plots(rows,series,config,directory):
    dimensions=sorted({v['dimension'] for v in rows})
    contexts=sorted({v['context'] for v in rows}); maxH=max(config['horizons'])
    cols=min(4,len(dimensions)); nr=(len(dimensions)+cols-1)//cols
    fig,axes=plt.subplots(nr,cols,figsize=(max(9,4*cols),4.5*nr),squeeze=False,layout='constrained')
    for ax,n in zip(axes.flat,dimensions):
        for i,r in enumerate(contexts):
            color=COLORS[i%len(COLORS)]
            baseline=[v['tail_mse'] for v in rows if v['dimension']==n and v['context']==r and v['policy']=='one_step']
            subset=[v for v in rows if v['dimension']==n and v['context']==r and v['policy']=='chunk' and v['prediction']==maxH]
            es=sorted({v['execution'] for v in subset})
            means=[]; lows=[]; highs=[]
            for e in es:
                values=[v['tail_mse'] for v in subset if v['execution']==e]
                means.append(np.mean(values)); lows.append(np.min(values)); highs.append(np.max(values))
            ax.plot(es,np.where(np.isfinite(means),np.maximum(means,1e-24),np.nan),'o-',color=color,label=f'r={r}, chunk')
            if len(config['seeds'])>1:
                ax.fill_between(es,np.maximum(lows,1e-24),np.where(np.isfinite(highs),highs,np.nan),color=color,alpha=.12)
            if np.isfinite(np.mean(baseline)):
                ax.axhline(max(np.mean(baseline),1e-24),color=color,linestyle='--',label=f'r={r}, one-step')
            if not np.all(np.isfinite(means)) or not np.all(np.isfinite(baseline)):
                ax.text(.02,.98-i*.09,f'r={r}: divergent cases; see CSV',transform=ax.transAxes,va='top',fontsize=7,color=color)
        ax.set(title=f"n={n}, h={config.get('observability_index', '?')}",xscale='log',yscale='log',xlabel='Execution length ℓ',ylabel='Late-window MSE / state coordinate')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(dimensions):]: ax.set_visible(False)
    axes.flat[0].legend(fontsize=7,ncol=2)
    fig.suptitle(f"Long chunks versus a genuine one-step baseline | H={maxH}, noise SD={config['noise']:g}\n"
                 f"Last {config['tail_steps']} of {config['steps']} scored steps; {len(config['seeds'])} independent training seeds",fontsize=13)
    fig.supxlabel('Dashed: separately trained one-step policy, on identical windows. Shading: range of seed means, not a confidence interval. Divergent averages are flagged, not plotted as finite values.',fontsize=9)
    save(fig,directory,'late_error_vs_execution')

    # One context is selected by rule, not by observed performance.
    h=config.get('observability_index',2)
    r=h if h in contexts else min(contexts,key=lambda r:abs(r-h))
    es=sorted(set(config['executions']) & {200,500,1000})
    if not es: es=[max(config['executions'])]
    selections=[('one_step',1)]+[('chunk',e) for e in es]
    fig,axes=plt.subplots(nr,cols,figsize=(max(9,4*cols),4.5*nr),squeeze=False,layout='constrained')
    block=max(1,min(100,config['steps']//20))
    for ax,n in zip(axes.flat,dimensions):
        for i,(kind,e) in enumerate(selections):
            curves=[series[f'n{n}_s{s}_r{r}_{kind}_e{e}'] for s in config['seeds']]
            curve=np.mean(curves,axis=0)
            times=[]; values=[]
            for start in range(0,len(curve),block):
                end=min(start+block,len(curve)); times.append((start+end)/2); values.append(np.mean(curve[start:end]))
            ax.plot(times,np.maximum(values,1e-24),color=COLORS[i%len(COLORS)],label='One-step baseline' if kind=='one_step' else f'Chunk ℓ={e}')
        ax.axvspan(config['steps']-config['tail_steps'],config['steps'],color='gray',alpha=.1)
        ax.set(title=f'n={n}, context r={r}',yscale='log',xlabel='Steps after warm-up',ylabel='Block-mean trajectory MSE / coordinate')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(dimensions):]: ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'Transient versus late behavior | averages over {block}-step blocks',fontsize=14)
    fig.supxlabel('Gray region is the reported late window. Finite simulation does not prove convergence or stability; values below 1e−24 are display-clipped.',fontsize=9)
    save(fig,directory,'error_over_time')

    fig,axes=plt.subplots(nr,cols,figsize=(max(9,4*cols),4.5*nr),squeeze=False,layout='constrained')
    for ax,n in zip(axes.flat,dimensions):
        for i,(kind,e) in enumerate(selections):
            curve=np.mean([series[f'n{n}_s{s}_r{r}_{kind}_e{e}'] for s in config['seeds']],axis=0)
            ax.plot(np.arange(1,len(curve)+1),np.cumsum(curve),color=COLORS[i%len(COLORS)],
                    label='One-step baseline' if kind=='one_step' else f'Chunk ℓ={e}')
        ax.set(title=f'n={n}, context r={r}',xlabel='Steps after warm-up',ylabel='Cumulative squared error / coordinate')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(dimensions):]: ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Accumulated imitation error: persistent error continues to accumulate',fontsize=14)
    fig.supxlabel('All policies share initial states and disturbances within each seed. Noiseless decay can make full-episode averages misleading.',fontsize=9)
    save(fig,directory,'cumulative_error')

    failures=sum(v['failed_episodes'] for v in rows)
    Path(directory).mkdir(parents=True,exist_ok=True)
    Path(directory,'README.txt').write_text(
        'Use metrics.csv for per-seed means; episodes.csv retains paired comparisons.\n'
        'Prediction horizons reuse identical output blocks under separable ridge regression.\n'
        'Thus the displayed maximum-H prefix curves also describe smaller H wherever execution <= H.\n'
        f'Failed episodes counted across configurations (including reused comparisons): {failures}.\n'
        'Failure errors are infinity, never excluded; infinite points cannot appear on finite axes.\n'
        'Each case has a fixed n and measured h. See sweep_config.json for the sensor/index sweep.\n')


def observability_overview(rows,directory):
    """At sufficient context r=h, compare observability thresholds without pooling them."""
    dimensions=sorted({int(v['dimension']) for v in rows})
    cols=min(4,len(dimensions)); nr=(len(dimensions)+cols-1)//cols
    fig,axes=plt.subplots(nr,cols,figsize=(4*cols,4*nr),squeeze=False,layout='constrained')
    for ax,n in zip(axes.flat,dimensions):
        hs=sorted({int(v['observability_index']) for v in rows if int(v['dimension'])==n})
        drawn=False
        for i,h in enumerate(hs):
            subset=[v for v in rows if int(v['dimension'])==n and int(v['observability_index'])==h and int(v['context'])==h]
            if not subset: continue
            drawn=True
            H=max(int(v['prediction']) for v in subset)
            chunks=[v for v in subset if v['policy']=='chunk' and int(v['prediction'])==H]
            es=sorted({int(v['execution']) for v in chunks})
            values=[np.mean([float(v['tail_mse']) for v in chunks if int(v['execution'])==e]) for e in es]
            color=COLORS[i%len(COLORS)]
            ax.plot(es,np.where(np.isfinite(values),np.maximum(values,1e-24),np.nan),'o-',color=color,label=f'h={h}, r={h}')
            base=np.mean([float(v['tail_mse']) for v in subset if v['policy']=='one_step'])
            if np.isfinite(base): ax.axhline(max(base,1e-24),color=color,linestyle='--')
            if not np.all(np.isfinite(values)) or not np.isfinite(base):
                ax.text(.02,.98-i*.08,f'h={h}: divergence; see CSV',transform=ax.transAxes,va='top',fontsize=7)
        if not drawn: ax.text(.5,.5,'No r=h case requested',ha='center',transform=ax.transAxes)
        ax.set(title=f'n={n}',xlabel='Execution horizon ℓ',ylabel='Late-window MSE / coordinate',xscale='log',yscale='log')
        ax.grid(alpha=.2)
        if drawn: ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(dimensions):]: ax.set_visible(False)
    fig.suptitle('Observability-index comparison at context r=h',fontsize=14)
    fig.supxlabel('Solid: chunk policies at largest H. Dashed: matched one-step baseline. Sensor layouts and information change with h; finite-run results.',fontsize=9)
    save(fig,directory,'observability_comparison')
