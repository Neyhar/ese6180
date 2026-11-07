"""Read paired hard-example runs and compare sensing versus training coverage."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pitfalls import write_csv


def main():
    root=Path(__file__).parent/'results'/'partial_observation'
    suites={}
    for obs in ['full','partial']:
        for training in ['narrow','recovery']:
            path=root/f'{obs}_{training}'
            config=json.loads((path/'config.json').read_text())
            with (path/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
            suites[obs,training]=rows
            assert len(rows)==len(config['seeds'])*len(config['contexts'])*sum(e<=h for h in config['horizons'] for e in config['executions'])
    records=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    styles={('full','narrow'):('tab:blue','--','o'),('partial','narrow'):('tab:orange',':','s'),
            ('full','recovery'):('tab:green','--','^'),('partial','recovery'):('tab:purple','-','D')}
    for (obs,training),rows in suites.items():
        contexts=sorted({int(v['context']) for v in rows})
        vals=[];radii=[]
        for r in contexts:
            group=[v for v in rows if int(v['context'])==r and v['horizon']=='16' and v['execution']=='1']
            vals.append(float(np.median([float(v['test_loss']) for v in group])))
            radii.append(max(float(v['per_step_radius']) for v in group))
            records.append(dict(observation=obs,training=training,context=r,execution=1,
                                median_test_loss=vals[-1],worst_per_step_radius=radii[-1],
                                feature_rank=group[0]['feature_rank'],feature_dim=group[0]['feature_dim']))
        c,ls,marker=styles[obs,training]
        for ax,ys in zip(axes,[vals,radii]):ax.plot(contexts,ys,marker=marker,ls=ls,color=c,label=f'{obs} / {training}',markersize=6)
    axes[0].set(yscale='log',ylabel='Median mean test tracking MSE',title='Can history support one-step feedback?')
    axes[1].axhline(1,color='red',ls=':');axes[1].set(ylabel='Worst-seed per-step spectral radius',title='Below 1 means stable')
    for ax in axes:ax.set(xlabel='History r');ax.set_xticks(contexts);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Hard example: full state versus y=x₂; execution ℓ=1, prediction H=16\nNarrow demonstrations versus known-action exploration with expert labels',fontsize=13)
    fig.savefig(root/'observation_coverage_comparison.png',dpi=190);plt.close(fig)
    write_csv(records,root/'comparison.csv')
    # The removed x1 feature is identically zero in narrow training; fitted
    # deployed trajectories must therefore agree for paired full/partial runs.
    def index(rows):return {(v['seed'],v['context'],v['horizon'],v['execution']):v for v in rows}
    full=index(suites['full','narrow']);partial=index(suites['partial','narrow'])
    for key in full:
        np.testing.assert_allclose(float(full[key]['test_loss']),float(partial[key]['test_loss']),rtol=1e-5,atol=1e-15)
    text=['# Partial-observation hard example','',
          'All four runs share plant A1, evaluation initial states, three training seeds, contexts 1,2,4,8,16, H=1,4,16 and execution lengths up to H.',
          'There are 135 learned settings per run (540 total), each evaluated on 20 support and 20 perturbed 200-step rollouts, with no continuing noise.',
          'A separate 1000-step canonical rollout is a diagnostic. No test states enter training.', '',
          '## What does the learner see?','',
          'Full observation gives [x1,x2]; partial observation gives only y=x2. Both receive actual past actions.',
          'For partial observation, O2=[[0,1],[-0.1875,0.75]] has full rank: the observability index is two.',
          'An unannounced jump in x1 at deployment does not change any available y or past action immediately. Its effect appears in y one step later.',
          'Thus a partial learner cannot match the full-state expert on its first action for both paired initial states. After the next transition, known-model reconstruction is exact in these noiseless runs.', '',
          '## Coverage intervention','',
          'Narrow training stays on x1=0. Recovery training applies known probing actions u=Kx+0.05*Gaussian noise to the plant (no unknown process noise),',
          'then labels each visited history with the expert counterfactual action chunk K F^j x. Input histories contain applied actions, not expert target labels.',
          'The expert uses hidden state to generate labels; the learner receives only its assigned observations and applied-action history.', '',
          '## One-step results (H=16)','',
          '| Sensing | Training | History | Median test MSE | Worst per-step radius | Feature rank/dimension |',
          '|---|---|---:|---:|---:|---:|']
    for v in records:
        text.append(f"| {v['observation']} | {v['training']} | {v['context']} | {v['median_test_loss']:.4g} | {v['worst_per_step_radius']:.6f} | {v['feature_rank']}/{v['feature_dim']} |")
    text+=['','## Reading the files','',
           '- `observation_coverage_comparison.png`: separates observation loss from training coverage at one-step execution.',
           '- Each suite has `context_hypothesis.png` for the history/execution grid and `metrics.csv` for every configuration.',
           '- Partial suites also have `learner_view.png`, `learner_view.csv`, and `learner_inputs.csv` showing exactly which features are passed to the policy.',
           '- The reconstructed-state curves are privileged known-model diagnostics; they are not inputs to the learned ridge policies.',
           '- Full-state static and analytic-memory baselines are privileged references under partial observation. The separately labeled known-model observer uses only partial observations and actual past actions.', '',
           'The full and partial narrow fits agree numerically across all paired configurations: removing an always-zero training feature does not fix or worsen their learned extension.',
           'These results concern noiseless measurements and a single initial departure. They do not imply robustness to continuing process or measurement noise, or generalization to unknown plants.','']
    (root/'README.md').write_text('\n'.join(text))
    print('\n'.join(text[:8]))

if __name__=='__main__':main()
