"""Summarize an existing observability sweep without retraining."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .plotting_utils import plt, save, COLORS


def summarize(directory):
    directory = Path(directory)
    with (directory/'metrics.csv').open() as f:
        all_rows = list(csv.DictReader(f))
    rows = [v for v in all_rows if v['policy']=='chunk']
    if not rows:
        raise ValueError('No chunk evaluations found')
    horizon = max(int(v['prediction']) for v in rows)
    rows = [v for v in rows if int(v['prediction'])==horizon]
    cases = sorted({(int(v['dimension']),int(v['observability_index'])) for v in rows})
    summary = []
    for n,h in cases:
        case = [v for v in rows if int(v['dimension'])==n and int(v['observability_index'])==h]
        complete = [v for v in case if int(v['context'])==h]
        executions = sorted({int(v['execution']) for v in complete})
        if not executions or 1 not in executions:
            continue
        means = {e:float(np.mean([float(v['tail_mse']) for v in complete if int(v['execution'])==e])) for e in executions}
        best = min(means,key=means.get)
        summary.append(dict(dimension=n,observability_index=h,best_execution_at_r_equals_h=best,
                            one_step_tail_mse=means[1],best_tail_mse=means[best],
                            max_rate_at_sufficient_context=max(float(v['per_step_rate']) for v in case if int(v['context'])>=h)))
    with (directory/'case_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['dimension','observability_index','best_execution_at_r_equals_h','one_step_tail_mse','best_tail_mse','max_rate_at_sufficient_context']);writer.writeheader();writer.writerows(summary)
    unstable=[v for v in rows if float(v['per_step_rate'])>=1]
    failed=[v for v in rows if int(v['failed_episodes'])>0]
    sufficient=[v for v in rows if int(v['context'])>=int(v['observability_index'])]
    stats=dict(cases=len(cases),prediction_horizon=horizon,seed_configuration_rows=len(rows),
               unique_context_execution_settings=len({(v['dimension'],v['observability_index'],v['context'],v['execution']) for v in rows}),
               unstable_seed_settings=len(unstable),failed_seed_settings=len(failed),
               unstable_without_cutoff_failure=sum(int(v['failed_episodes'])==0 for v in unstable),
               unstable_at_or_above_observability_index=sum(float(v['per_step_rate'])>=1 for v in sufficient),
               cases_with_one_step_lowest_mean_tail_error=sum(v['best_execution_at_r_equals_h']==1 for v in summary))
    (directory/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    text=['# Saved observability results','',
          'Generated from metrics.csv; this summary does not retrain controllers.', '',
          f"- {stats['cases']} plant/sensor cases; {len(rows)} seed/configuration rows at H={horizon}.",
          f"- At history r=h, execution 1 has the lowest seed-mean tail error in {stats['cases_with_one_step_lowest_mean_tail_error']} of {len(summary)} cases.",
          f"- {len(unstable)} seed settings have spectral radius >=1; {len(failed)} reach the numerical failure cutoff.",
          f"- {stats['unstable_without_cutoff_failure']} unstable seed settings do not reach the cutoff within the finite rollout.",
          f"- {stats['unstable_at_or_above_observability_index']} unstable seed settings occur at r>=h.", '',
          'These are descriptive results for this plant family, not an observability or learning theorem.', '',
          '| State dimension n | Observability h | Best execution at r=h | One-step tail MSE |',
          '|---:|---:|---:|---:|']
    for v in summary:
        text.append(f"| {v['dimension']} | {v['observability_index']} | {v['best_execution_at_r_equals_h']} | {v['one_step_tail_mse']:.4g} |")
    example=(10,4) if (10,4) in cases else cases[0]
    selected=[v for v in rows if (int(v['dimension']),int(v['observability_index']))==example]
    text+=['','## Example to read first', '',f'See `figures/start_here.png`: n={example[0]}, h={example[1]}, H={horizon}.', '',
           '| History r | Execution ell | Mean tail MSE | Worst per-step rate | Failed episodes |',
           '|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for i,r in enumerate(sorted({int(v['context']) for v in selected})):
        subset=[v for v in selected if int(v['context'])==r]
        es=sorted({int(v['execution']) for v in subset}); means=[];rates=[]
        for e in es:
            group=[v for v in subset if int(v['execution'])==e]
            mean=float(np.mean([float(v['tail_mse']) for v in group]));means.append(mean)
            rate=max(float(v['per_step_rate']) for v in group);rates.append(rate)
            text.append(f"| {r} | {e} | {mean:.4g} | {rate:.6f} | {sum(int(v['failed_episodes']) for v in group)} |")
            if not np.isfinite(mean):
                axes[0].text(e,.95-i*.07,f'r={r}: ∞',transform=axes[0].get_xaxis_transform(),ha='center',color=COLORS[i%len(COLORS)],fontsize=9)
        axes[0].plot(es,[v if np.isfinite(v) else np.nan for v in means],'o-',color=COLORS[i%len(COLORS)],label=f'history r={r}')
        axes[1].plot(es,rates,'o-',color=COLORS[i%len(COLORS)],label=f'history r={r}')
    axes[0].set(yscale='log',ylabel='Mean late tracking MSE / coordinate',title='Tracking on held-out noisy rollouts')
    axes[1].axhline(1,color='red',ls=':');axes[1].set(ylabel='Worst-seed per-step spectral radius',title='Below 1 means stable')
    for ax in axes:
        ax.set(xscale='log',xlabel='Execution length ℓ');ax.set_xticks(es,[str(v) for v in es]);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle(f'Observability study: n={example[0]}, h={example[1]}, H={horizon}')
    fig.supxlabel('Infinite losses are labeled above the plot; they are not omitted from averages. Same evaluation episodes across policies.',fontsize=9)
    save(fig,directory/'figures','start_here')
    text+=['','## How this differs from pitfalls','',
           'This study uses partial observations, varied sensor layouts, and continuing process noise. '
           'Its loss is normalized per state coordinate and evaluated over the final window. '
           'Pitfalls uses full observations, narrow training states and an initial perturbation without continuing noise. '
           'Its main loss averages full-state tracking error over the whole deployment. Absolute scores cannot be compared directly.','']
    (directory/'RESULTS.md').write_text('\n'.join(text))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'results'/'latest')
    print(json.dumps(summarize(parser.parse_args().input),indent=2))

if __name__=='__main__':main()
