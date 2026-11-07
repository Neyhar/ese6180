"""Audit saved pitfalls outputs and generate report statistics and figures.
No controllers are retrained. ROOT may be supplied as the first argument.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ese618-matplotlib')
from pathlib import Path
import sys,csv,json,hashlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def mean(rows,key='test_loss'):return float(np.mean([float(v[key]) for v in rows]))
def pick(rows,r=None,e=None,H=16):
    return [v for v in rows if (r is None or int(v['context'])==r) and (e is None or int(v['execution'])==e) and (H is None or int(v.get('horizon',16))==H)]
def cell(rows):
    vals=np.array([float(v['test_loss']) for v in rows]);rates=np.array([float(v['per_step_radius']) for v in rows])
    return dict(mean=float(vals.mean()),median=float(np.median(vals)),minimum=float(vals.min()),maximum=float(vals.max()),
                worst_rate=float(rates.max()),stable_seeds=int(sum(rates<1)),seeds=len(vals),
                mean_support=mean(rows,'support_test_loss') if 'support_test_loss' in rows[0] else None,
                worst_test_rollout=max(float(v['worst_test_loss']) for v in rows) if 'worst_test_loss' in rows[0] else None)
paths={'current':ROOT/'pitfalls/results/context_chunking'}
for name in ['full_narrow','partial_narrow','full_recovery','partial_recovery']:
    paths[name]=ROOT/'pitfalls/results/partial_observation'/name
for name in ['aligned','chunk_execution','hard_chunk_execution']:
    paths['archive_'+name]=ROOT/'archive/pitfalls-before-focus/results'/name
runs={};manifest=[]
for name,path in paths.items():
    file=path/('framework_sweep.csv' if name=='archive_aligned' else 'metrics.csv')
    runs[name]=read(file)
    for f in sorted(path.iterdir()):
        if f.is_file():
            raw=f.read_bytes();manifest.append(dict(path=str(f.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
stats={'inventory':[], 'current':{},'comparison':{},'audit':{},'historical':{},'baselines':{},'seed_effect':{},'learner_view':{}}
for name,path in paths.items():
    config=json.loads((path/'config.json').read_text())
    stats['inventory'].append(dict(name=name,path=str(path.relative_to(ROOT)),rows=len(runs[name]),config=config))

# Validate every modern per-controller mean against individual test rollouts.
checked=0;rollout_rows=0
for name in ['current','full_narrow','partial_narrow','full_recovery','partial_recovery','archive_hard_chunk_execution']:
    raw=read(paths[name]/'test_rollouts.csv');rollout_rows+=len(raw)
    groups=collections.defaultdict(list)
    for v in raw:
        groups[(v['controller'],v['seed'],v['context'],v['horizon'],v['execution'],v['suite'])].append(v)
    for row in runs[name]:
        key=tuple(row[k] for k in ['controller','seed','context','horizon','execution'])
        group=groups[key+('perturbed',)]
        assert len(group)==20 and len({v['test_id'] for v in group})==20
        np.testing.assert_allclose(mean(group),float(row['test_loss']),rtol=1e-12,atol=1e-24)
        np.testing.assert_allclose(mean(groups[key+('support',)]),float(row['support_test_loss']),rtol=1e-12,atol=1e-24)
        checked+=1
# Check the serialized learner-input vectors against observation/action blocks.
input_count=0
for name in ['partial_narrow','partial_recovery']:
    for row in read(paths[name]/'learner_inputs.csv'):
        obs=json.loads(row['observation_history']);actions=json.loads(row['applied_action_history'])
        feature=json.loads(row['input_feature']);r=int(row['context'])
        assert len(obs)==r and len(actions)==2*(r-1)
        np.testing.assert_allclose(feature,obs+actions,atol=0,rtol=0)
        input_count+=1
stats['audit']['verified_partial_input_vectors']=input_count
stats['audit'].update(verified_controller_means=checked,reviewed_rollout_rows=rollout_rows,
                      saved_learned_rows=sum(len(v) for v in runs.values()),source_files=len(manifest))
# Prefix identity and paired observation identity.
max_gap=0.
for name,rows in runs.items():
    if name=='archive_aligned':continue
    groups=collections.defaultdict(list)
    for v in rows:
        key=tuple(v.get(k,'') for k in ['seed','context','execution','observation','data'])
        groups[key].append(float(v['test_loss']))
    for vals in groups.values():
        denom=max(abs(vals[0]),1e-30);gap=max(abs(x-vals[0])/denom for x in vals)
        max_gap=max(max_gap,gap)
        np.testing.assert_allclose(vals,vals[0],rtol=1e-5,atol=1e-15)
stats['audit']['maximum_relative_prefix_gap']=max_gap
idx=lambda rows:{tuple(v[k] for k in ['seed','context','horizon','execution']):v for v in rows}
full=idx(runs['full_narrow']);part=idx(runs['partial_narrow'])
gaps=[]
for k in full:
    a,b=float(full[k]['test_loss']),float(part[k]['test_loss'])
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-15);gaps.append(abs(a-b)/max(abs(a),1e-30))
stats['audit']['maximum_full_partial_narrow_relative_gap']=max(gaps)
# All modern test states are exactly paired across suites and prior recovery runs.
state_bytes=(paths['current']/'test_states.csv').read_bytes()
stats['audit']['identical_active_test_states']=all((paths[n]/'test_states.csv').read_bytes()==state_bytes for n in ['full_narrow','partial_narrow','full_recovery','partial_recovery'])
assert stats['audit']['identical_active_test_states']
for r in [1,2,4,8,16]:
    stats['current'][str(r)]={str(e):cell(pick(runs['current'],r,e)) for e in [1,2,4,8,16]}
for name in ['full_narrow','partial_narrow','full_recovery','partial_recovery']:
    stats['comparison'][name]={str(r):{str(e):cell(pick(runs[name],r,e)) for e in [1,2,4,8,16]} for r in [1,2,4,8,16]}
    stats['baselines'][name]=read(paths[name]/'baselines.csv')
for name in ['current','archive_hard_chunk_execution']:
    stats['baselines'][name]=read(paths[name]/'baselines.csv')
for name in ['current','full_narrow','partial_narrow','full_recovery','partial_recovery']:
    stats.setdefault('criteria',{})[name]=read(paths[name]/'context_summary.csv')
for r,e in [(4,16),(8,1),(8,8),(16,1)]:
    g=pick(runs['current'],r,e)
    stats.setdefault('long_examples',[]).append(dict(r=r,e=e,test_mean=mean(g),
            long_canonical_mean=mean(g,'long_sample_loss'),worst_rate=max(float(v['per_step_radius']) for v in g)))
g=pick(runs['partial_recovery'],1,1)
stats['seed_effect']['partial_recovery_r1']=[dict(seed=v['seed'],loss=float(v['test_loss']),rate=float(v['per_step_radius'])) for v in g]
stats['seed_effect']['summary']=cell(g)
# Historical noisy regimes, at one-step execution; square each seed's RMS first.
for name in ['archive_aligned','archive_chunk_execution']:
    a=runs[name];output=[]
    for obs in ['full','partial']:
        for regime in ['narrow','initial','persistent']:
            contexts=sorted({int(v['context']) for v in a})
            for r in contexts:
                g=[v for v in a if v['observation']==obs and v['data']==regime and int(v['context'])==r and v['execution']=='1' and (name=='archive_aligned' or v['horizon']=='16')]
                if not g:continue
                output.append(dict(observation=obs,training=regime,context=r,
                    mean_mse=float(np.mean([float(v['tracking_rms'])**2 for v in g])),
                    median_rms=float(np.median([float(v['tracking_rms']) for v in g])),
                    worst_rate=max(float(v['per_step_radius']) for v in g),
                    mean_training_mse=float(np.mean([float(v['training_mse']) for v in g]))))
    stats['historical'][name]=output
stats['historical']['hard_example']=read(paths['archive_aligned']/'hard_example.csv')
stats['historical']['short_hard']={str(r):cell(pick(runs['archive_hard_chunk_execution'],r,1)) for r in [1,2,4,8]}
# Learner trace: early hidden-state recovery and observation delay.
trace=read(paths['partial_recovery']/'learner_view.csv')
for name in sorted({v['controller'] for v in trace}):
    g=[v for v in trace if v['controller']==name]
    hits=[int(v['step']) for v in g if abs(float(v['hidden_x1']))<1e-6]
    stats['learner_view'][name]=dict(first_below_1e_minus6=hits[0] if hits else None,
          first_three=[{k:v[k] for k in ['step','hidden_x1','observed_y','model_estimated_x1','actual_u1','actual_u2']} for v in g[:3]])

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
# Publication-scale charts use means across seeds, consistently with the report.
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for r,c in zip([1,2,4,8,16],plt.cm.viridis(np.linspace(.05,.9,5))):
    cs=stats['current'][str(r)]
    axes[0].plot([1,2,4,8,16],[cs[str(e)]['mean'] for e in [1,2,4,8,16]],'o-',color=c,label=f'r={r}')
    axes[1].plot([1,2,4,8,16],[cs[str(e)]['worst_rate'] for e in [1,2,4,8,16]],'o-',color=c)
axes[0].set(yscale='log',ylabel='Mean test MSE',title='Full observation, narrow training')
axes[1].axhline(1,c='red',ls=':');axes[1].set(ylabel='Worst-seed per-step radius',title='Stability is a separate test')
for ax in axes:ax.set(xscale='log',xlabel='Executed actions before replanning');ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.grid(alpha=.2)
axes[0].legend(ncol=2,fontsize=9);fig.savefig(FIG/'history_execution.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for name,c,ls in [('full_narrow','tab:blue','--'),('partial_narrow','tab:orange',':'),('full_recovery','tab:green','--'),('partial_recovery','tab:purple','-')]:
    cs=stats['comparison'][name];rs=[1,2,4,8,16]
    axes[0].plot(rs,[cs[str(r)]['1']['mean'] for r in rs],'o',ls=ls,color=c,label=name.replace('_',' / '))
    axes[1].plot(rs,[cs[str(r)]['1']['worst_rate'] for r in rs],'o',ls=ls,color=c)
axes[0].set(yscale='log',ylabel='Mean test MSE',title='History at one-step execution')
axes[1].axhline(1,c='red',ls=':');axes[1].set(ylabel='Worst-seed per-step radius',title='Coverage and sensing have different effects')
for ax in axes:ax.set(xlabel='History length r');ax.set_xticks(rs);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.savefig(FIG/'coverage_sensing.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for name,style in [('learned r=1','--'),('learned r=2','-'),('learned r=16','-.'),('known-model observer',':')]:
    g=[v for v in trace if v['controller']==name and int(v['step'])<=20]
    axes[0].semilogy([int(v['step']) for v in g],[max(abs(float(v['hidden_x1'])),1e-16) for v in g],style,label=name)
    axes[1].plot([int(v['step']) for v in g],[float(v['observed_y']) for v in g],style,label=name)
axes[0].set(title='Hidden departure',ylabel='|x1| (display floor 1e-16)')
axes[1].set(title='Actual observation',ylabel='y = x2')
for ax in axes:ax.set(xlabel='Deployment step');ax.grid(alpha=.2)
axes[0].legend(fontsize=9);fig.savefig(FIG/'learner_early.png');plt.close(fig)
(OUT/'analysis_stats.json').write_text(json.dumps(stats,indent=2))
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(stats['audit'],indent=2))
print('Partial recovery r1 seeds:',stats['seed_effect']['partial_recovery_r1'])
print('Partial recovery r2:',stats['comparison']['partial_recovery']['2']['1'])
