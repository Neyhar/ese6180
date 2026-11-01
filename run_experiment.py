"""Observability study with matched training and 600-step evaluation episodes."""
import os
# Avoid oversubscribing tiny batched linear algebra on desktop CPUs.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import argparse
import copy
import shutil
import csv
import json
from pathlib import Path
import numpy as np
from model import coupled_system, demonstrations, HistoryPolicy
from long_experiment import (window_ids, matched_arrays, fit_matched, batch_rollout,
                             expert_reference, trajectory_errors, summarize_errors, paired_tail_difference)


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prepare_output(root, overwrite=False):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise ValueError(f'{root} already contains results; use --overwrite or a new --output')
        allowed = Path(__file__).resolve().parent / 'results'
        resolved = root.resolve()
        if root.is_symlink() or allowed not in resolved.parents or not (root/'sweep_config.json').is_file():
            raise ValueError('Overwrite requires a generated sweep subfolder inside this repository/results')
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimensions', nargs='+', type=int, default=[2,3,4,5,10,50,100])
    p.add_argument('--horizons', nargs='+', type=int, default=[16])
    p.add_argument('--executions', nargs='+', type=int, default=[1,2,4,8,16])
    p.add_argument('--contexts', nargs='+', type=int, default=None, help='Default per case: 1,h-1,h,h+1 (positive, unique)')
    p.add_argument('--observability-indices', nargs='+', default=['2','4','8','n'], help='Exact requested indices; n means full state dimension; values > n skipped')
    p.add_argument('--steps', type=int, default=600, help='Scored steps AFTER expert warm-up')
    p.add_argument('--tail-steps', type=int, default=200)
    p.add_argument('--seeds', nargs='+', type=int, default=[618,619,620], help='Independent data/evaluation replicates')
    p.add_argument('--system-seed', type=int, default=618, help='Fixed plant across training seeds')
    p.add_argument('--decay', type=float, default=.995)
    p.add_argument('--sensors', type=int, default=None, help='Override index sweep with one fixed sensor count')
    p.add_argument('--noise', type=float, default=.02, help='Evaluation process-noise SD per coordinate; training noiseless')
    p.add_argument('--train-trajectories', type=int, default=32)
    p.add_argument('--eval-trajectories', type=int, default=8)
    p.add_argument('--validation-trajectories', type=int, default=8)
    p.add_argument('--windows-per-trajectory', type=int, default=24)
    p.add_argument('--training-prefix', type=int, default=128, help='Candidate training boundary times beyond maximum context')
    p.add_argument('--ridge', type=float, default=1e-6)
    p.add_argument('--output', default='results/latest')
    p.add_argument('--overwrite', action='store_true', help='Replace an existing generated run under results/')
    args=p.parse_args()
    if (min(args.dimensions)<2 or min((args.contexts or [1])+args.horizons+args.executions)<1
        or min(args.steps,args.tail_steps,args.train_trajectories,args.eval_trajectories,
               args.validation_trajectories,args.windows_per_trajectory,args.training_prefix)<1
        or args.noise<0 or args.ridge<0 or not .1<args.decay<1):
        p.error('Invalid dimensions, horizons, counts, decay, noise, or ridge')
    if args.tail_steps>args.steps or args.steps<2*max(args.executions):
        p.error('Tail must fit in scored steps; use at least two full longest execution chunks')
    if max(args.executions)>max(args.horizons): p.error('An execution exceeds every prediction horizon')
    if args.sensors is not None and not 1<=args.sensors<min(args.dimensions):
        p.error('Sensor count must be positive and smaller than every state dimension')
    try:
        plan=build_plan(args)
    except ValueError as exc:
        p.error(str(exc))
    root=Path(args.output)
    try:
        prepare_output(root, args.overwrite)
    except ValueError as exc:
        p.error(str(exc))
    (root/'sweep_config.json').write_text(json.dumps(vars(args)|{'cases':plan},indent=2)+'\n')
    combined={name:[] for name in ('metrics','episodes','fit_diagnostics','systems')}
    for item in plan:
        case=copy.deepcopy(args)
        case.dimensions=[item['dimension']]
        case.contexts=item['contexts']
        case.target_index=item['observability_index'] if args.sensors is None else None
        case.warmup_override=item['warmup']
        case.output=str(root/item['directory'])
        print(f"Starting n={item['dimension']}, h={item['observability_index']}, contexts={item['contexts']}",flush=True)
        run_case(case)
        for name in combined:
            with (Path(case.output)/f'{name}.csv').open() as f:
                combined[name].extend(csv.DictReader(f))
            write_csv(root/f'{name}.csv',combined[name])
    from long_plots import observability_overview
    observability_overview(combined['metrics'], root/'figures')
    print(f'Completed {len(plan)} dimension/observability cases. Combined data: {root}',flush=True)


def build_plan(args):
    tokens=args.observability_indices
    if any(v!='n' and (not v.isdigit() or int(v)<1) for v in tokens):
        raise ValueError('Observability indices must be positive integers or n')
    plan=[]
    for n in sorted(set(args.dimensions)):
        if args.sensors is not None:
            h=coupled_system(n,args.system_seed,args.decay,args.sensors, getattr(args, "target_index", None))[0].observability_index()
            indices=[h]
        else:
            indices=sorted({n if v=='n' else int(v) for v in tokens if v=='n' or int(v)<=n})
        if not indices:
            raise ValueError(f'No requested observability index is feasible for dimension {n}')
        contexts={h:sorted(set(args.contexts or [1,max(1,h-1),h,h+1])) for h in indices}
        warmup=max(max(v) for v in contexts.values())-1
        for h in indices:
            plan.append(dict(dimension=n,observability_index=h,contexts=contexts[h],warmup=warmup,
                             directory=f'n{n}/h{h}'))
    return plan


def run_case(args):
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config=vars(args)|{'training_noise':0, 'normalization':'state error averaged per coordinate',
                      'warmup':args.warmup_override, 'matching':'same trajectory/time IDs across all contexts and horizons',
                      'horizon_solver':'separable regression: shorter chunk fits are prefixes of maximum-H solution'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[]; episodes=[]; diagnostics=[]; series={}; system_rows=[]
    horizons=sorted(set(args.horizons)); contexts=sorted(set(args.contexts)); maxH=max(horizons)
    executions=sorted(set(args.executions)|{1})
    warmup=args.warmup_override; total=args.steps+warmup
    train_steps=maxH+warmup+args.training_prefix
    for n in sorted(set(args.dimensions)):
        system,K=coupled_system(n,args.system_seed,args.decay,args.sensors, getattr(args, "target_index", None))
        h=system.observability_index()
        if args.target_index is not None and h != args.target_index:
            raise AssertionError(f'Requested index {args.target_index}, measured {h}')
        config['observability_index']=h
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        O=system.observability(h)
        system_rows.append(dict(dimension=n,actions=K.shape[0],observations=system.C.shape[0],
                                observability_index=h,observability_condition=float(np.linalg.cond(O)),
                                open_loop_radius=float(np.max(np.abs(np.linalg.eigvals(system.A)))),
                                expert_radius=float(np.max(np.abs(np.linalg.eigvals(system.A+system.B@K))))))
        np.savez_compressed(out/f'system_n{n}.npz', A=system.A,B=system.B,C=system.C,K=K)
        for seed in args.seeds:
            rng=np.random.default_rng(seed+n*10000)
            train=demonstrations(system,K,rng.normal(size=(args.train_trajectories,n)),train_steps)
            val=demonstrations(system,K,rng.normal(size=(args.validation_trajectories,n)),train_steps)
            ids=window_ids(train,warmup+1,maxH,args.windows_per_trajectory)
            vids=window_ids(val,warmup+1,maxH,args.windows_per_trajectory)
            # Independent evaluation stream stays fixed if training-data count changes.
            erng=np.random.default_rng(seed+n*10000+100000000)
            initial=erng.normal(size=(args.eval_trajectories,n))
            disturbances=erng.normal(scale=args.noise,size=(args.eval_trajectories,total,n))
            reference=expert_reference(system,K,initial,disturbances)
            for r in contexts:
                X,Y=matched_arrays(train,ids,r,maxH)
                XV,YV=matched_arrays(val,vids,r,maxH)
                one,chunk,info=fit_matched(X,Y,r,K.shape[0],args.ridge)
                diagnostics.append(dict(dimension=n,observability_index=h,seed=seed,context=r,**info))
                base_x,base_u,base_fail=batch_rollout(system,K,one,1,initial,disturbances,warmup)
                base_errors=trajectory_errors(base_x,reference,warmup)
                cache={}
                val_cache={}
                choices=[('one_step',1,1)]+[('chunk',H,e) for H in horizons for e in executions if e<=H]
                for kind,H,e in choices:
                    if kind=='one_step': fail,errors=base_fail,base_errors
                    else:
                        if e not in cache:
                            # H does not affect these first e fitted rows. Evaluate once.
                            policy=HistoryPolicy(chunk.weights[:e*K.shape[0]],r,e,K.shape[0])
                            xx,uu,ff=batch_rollout(system,K,policy,e,initial,disturbances,warmup)
                            if e==1:
                                np.testing.assert_allclose(xx,base_x,atol=1e-8,rtol=1e-8)
                            cache[e]=(ff,trajectory_errors(xx,reference,warmup))
                        fail,errors=cache[e]
                    weights=one.weights if kind=='one_step' else chunk.weights[:H*K.shape[0]]
                    if (kind,H) not in val_cache:
                        val_cache[(kind,H)]=float(np.mean((XV@weights.T-YV[:,:H*K.shape[0]])**2))
                    validation=val_cache[(kind,H)]
                    first_val=float(np.mean((XV@weights[:K.shape[0]].T-YV[:,:K.shape[0]])**2))
                    summary=summarize_errors(errors,args.tail_steps)
                    row=dict(dimension=n,observability_index=h,seed=seed,context=r,policy=kind,prediction=H,execution=e,
                             **summary,validation_first_action_mse=first_val,validation_chunk_mse=validation,
                             failed_episodes=int(np.sum(fail>=0)),
                             paired_tail_difference=paired_tail_difference(errors,base_errors,args.tail_steps))
                    rows.append(row)
                    for ep in range(args.eval_trajectories):
                        episodes.append(dict(dimension=n,observability_index=h,seed=seed,context=r,policy=kind,prediction=H,execution=e,
                                             episode=ep,**summarize_errors(errors[ep:ep+1],args.tail_steps),
                                             first_failed_step=int(fail[ep]),
                                             paired_tail_difference=paired_tail_difference(errors[ep],base_errors[ep],args.tail_steps)))
                    if kind=='one_step' or H==maxH:
                        series[f'n{n}_s{seed}_r{r}_{kind}_e{e}']=np.mean(errors,axis=0)
                print(f'n={n}, seed={seed}, r={r}: rank {info["feature_rank"]}/{info["feature_dim"]}; first-action gap {info["first_action_weight_gap"]:.2e}',flush=True)
            # Save progress after every independent training/evaluation replicate.
            write_csv(out/'metrics.csv',rows); write_csv(out/'episodes.csv',episodes)
            write_csv(out/'fit_diagnostics.csv',diagnostics); write_csv(out/'systems.csv',system_rows)
            np.savez_compressed(out/'error_curves.npz',**series)
    from long_plots import make_plots
    make_plots(rows,series,config,out/'figures')
    print(f'Completed {len(rows)} configurations; data and PNG/PDF figures in {out}',flush=True)


if __name__=='__main__': main()
