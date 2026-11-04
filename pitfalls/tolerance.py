"""Shortest tested execution meeting a tracking tolerance across every evaluated seed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def select_horizons(rows, tolerance):
    """Require complete paired seeds, finite RMS <= tolerance, and rho < 1 for all seeds.

    This is an empirical finite-horizon criterion, not a population error guarantee.
    Missing cells and duplicate seeds are rejected rather than silently accepted.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tracking tolerance must be finite and positive')
    if not rows:
        raise ValueError('Sweep is empty')
    groups = {}
    for row in rows:
        key = (row['observation'], row['data'], int(row['context']), int(row['execution']))
        seed = int(row['seed'])
        group = groups.setdefault(key, {})
        if seed in group:
            raise ValueError(f'Duplicate seed {seed} in {key}')
        group[seed] = row
    seeds = {int(row['seed']) for row in rows}
    contexts = sorted({int(row['context']) for row in rows})
    executions = sorted({int(row['execution']) for row in rows})
    settings = sorted({key[:2] for key in groups})
    candidates, selected = [], []
    for obs, data in settings:
        for r in contexts:
            eligible = []
            for ell in executions:
                group = groups.get((obs,data,r,ell), {})
                if set(group) != seeds:
                    raise ValueError(f'Incomplete paired seeds for {(obs,data,r,ell)}')
                errors = np.array([float(v['tracking_rms']) for v in group.values()])
                radii = np.array([float(v['boundary_radius']) for v in group.values()])
                stable = bool(np.all(np.isfinite(radii) & (radii >= 0) & (radii < 1)))
                accurate = bool(np.all(np.isfinite(errors) & (errors >= 0) & (errors <= tolerance)))
                passed = stable and accurate
                reason = 'pass' if passed else ('unstable_or_invalid_radius' if not stable else 'tracking_tolerance_failed')
                record = dict(observation=obs,data=data,context=r,execution=ell,
                              tolerance=tolerance,n_seeds=len(seeds),all_stable=stable,
                              median_tracking_rms=float(np.median(errors)),
                              worst_tracking_rms=float(np.max(errors)),
                              worst_boundary_radius=float(np.max(radii)),eligible=passed,reason=reason)
                candidates.append(record)
                if passed:
                    eligible.append(record)
            best = eligible[0] if eligible else None
            selected.append(dict(observation=obs,data=data,context=r,tolerance=tolerance,
                                 shortest_execution=best['execution'] if best else None,
                                 status='feasible' if best else 'no_tested_execution_meets_criterion',
                                 worst_tracking_rms=best['worst_tracking_rms'] if best else None,
                                 n_seeds=len(seeds)))
    return candidates, selected


def write_analysis(rows, tolerance, out):
    out = Path(out)
    candidates, selected = select_horizons(rows, tolerance)
    out.mkdir(parents=True,exist_ok=True)
    for name, records in [('tolerance_candidates.csv',candidates),('shortest_execution.csv',selected)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    observations=sorted({v['observation'] for v in selected})
    regimes=sorted({v['data'] for v in selected})
    contexts=sorted({v['context'] for v in selected})
    executions=sorted({v['execution'] for v in candidates})
    fig,axes=plt.subplots(len(observations),len(regimes),squeeze=False,
                          figsize=(4*len(regimes),3.2*len(observations)))
    missing_y=max(executions)*2
    for i,obs in enumerate(observations):
        for j,data in enumerate(regimes):
            ax=axes[i,j]
            group=[v for v in selected if v['observation']==obs and v['data']==data]
            if not group:
                ax.set_visible(False);continue
            xs=[v['context'] for v in group]
            ys=[v['shortest_execution'] if v['shortest_execution'] is not None else np.nan for v in group]
            ax.plot(xs,ys,'o-',color='tab:blue')
            ax.scatter([v['context'] for v in group if v['shortest_execution'] is None],
                       [missing_y for v in group if v['shortest_execution'] is None],marker='x',color='tab:red')
            ax.set_yscale('log',base=2)
            ax.set(title=f'{obs} / {data}',xlabel='Context length',ylabel='Shortest tested execution',
                   xticks=contexts,yticks=executions+[missing_y],
                   yticklabels=[str(v) for v in executions]+['None'],ylim=(.8,missing_y*1.35))
            ax.grid(alpha=.2)
    fig.suptitle(f'Tracking RMS ≤ {tolerance:g} and stable for every evaluated seed\nNone = no tested execution passes; finite-horizon empirical criterion')
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/'shortest_execution.png',dpi=160);plt.close(fig)
    lines=['# Context versus required execution length','',
           f'Fixed tracking RMS tolerance: **{tolerance:g}** (absolute state units).',
           'A cell passes only when every evaluated seed has finite tracking RMS at or below the tolerance and boundary spectral radius strictly below one.',
           'This is descriptive analysis of the supplied evaluation seeds, not a statistical guarantee. The minimum is over tested execution lengths only. No feasible setting is recorded as None, not as zero.', '',
           '| Observation | Data | Context | Shortest execution | Worst tracking RMS |',
           '|---|---|---:|---:|---:|']
    for v in selected:
        lines.append(f"| {v['observation']} | {v['data']} | {v['context']} | {v['shortest_execution'] if v['shortest_execution'] is not None else 'None'} | {format(v['worst_tracking_rms'], '.4g') if v['worst_tracking_rms'] is not None else '—'} |")
    lines += ['', 'A decreasing feasible execution length with increasing context supports the proposed relationship in that setting. Constant length one indicates no further reduction is possible. None does not prove infeasibility outside the tested grid.',
              'Choose the tolerance before a confirmatory run; exploring thresholds on existing results is exploratory. Confirm any selected setting with fresh training/evaluation seeds before making a generalization claim.']
    (out/'tolerance_report.md').write_text('\n'.join(lines)+'\n')
    (out/'tolerance_config.json').write_text(json.dumps(dict(tolerance=tolerance,
        criterion='all evaluated seeds stable and tracking_rms <= tolerance',
        seeds=sorted({int(v['seed']) for v in rows}), contexts=contexts, executions=executions),indent=2))
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'results/aligned/framework_sweep.csv')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--tracking-tolerance',type=float,default=.05)
    args=parser.parse_args()
    with args.input.open(newline='') as f:
        rows=list(csv.DictReader(f))
    try:
        selected=write_analysis(rows,args.tracking_tolerance,args.output or args.input.parent)
    except ValueError as e:
        parser.error(str(e))
    print(f'Analyzed {len(selected)} context/settings; outputs in {args.output or args.input.parent}')


if __name__=='__main__': main()
