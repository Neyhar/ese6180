"""Does longer history reduce the need to execute open-loop action chunks?"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ese618-matplotlib')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'pitfalls'), str(ROOT)]
from model import LinearSystem, HistoryPolicy, fit_policy, exact_policy
from pitfalls import challenging_pair, gains, train_set, write_csv
from stability import stability, memory_policy
from evaluation import (make_test_states, deploy, tracking_loss, evaluate_test_set,
                        summarize_hypothesis)
from plotting import plot_results, plot_hypothesis
from recovery_data import recovery_windows, fit_recovery


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', choices=['full','partial'], default='full')
    parser.add_argument('--training', choices=['narrow','recovery'], default='narrow')
    parser.add_argument('--probe-scale', type=float, default=.05)
    parser.add_argument('--contexts', type=int, nargs='+', default=[1,2,4,8,16])
    parser.add_argument('--horizons', type=int, nargs='+', default=[1,2,4,8,16])
    parser.add_argument('--executions', type=int, nargs='+', default=[1,2,4,8,16])
    parser.add_argument('--seeds', type=int, nargs='+', default=[618,619,620,621,622])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--long-steps', type=int, default=1000)
    parser.add_argument('--test-pairs', type=int, default=20)
    parser.add_argument('--test-seed', type=int, default=20261105)
    parser.add_argument('--loss-tolerance', type=float, default=1e-5,
                        help='Illustrative MSE threshold; fix before a confirmatory run')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        name = 'context_chunking' if args.observation=='full' and args.training=='narrow' else f'{args.observation}_{args.training}'
        args.output = Path(__file__).parent/'results'/name
    if not np.isfinite(args.probe_scale) or args.probe_scale<=0:
        parser.error('Probe scale must be finite and positive')
    for key in ['contexts','horizons','executions','seeds']:
        values = getattr(args,key)
        if len(set(values)) != len(values) or (key != 'seeds' and min(values)<1):
            parser.error(f'{key} must be unique, and lengths must be positive')
        setattr(args,key,sorted(values))
    if min(args.steps,args.long_steps,args.test_pairs)<1 or not np.isfinite(args.loss_tolerance) or args.loss_tolerance<=0:
        parser.error('Steps, test pairs and finite loss tolerance must be positive')
    if args.long_steps < args.steps or 1 not in args.executions or max(args.executions)>max(args.horizons):
        parser.error('Require long-steps >= steps, execution 1, and max execution <= max horizon')
    return args


def main():
    args = parse_args()
    out = args.output; out.mkdir(parents=True,exist_ok=True)
    contexts, hs, seeds = args.contexts, args.horizons, args.seeds
    mu, steps, warmup = .125, args.steps, max(1,max(contexts)-1)
    A, _, K, _ = challenging_pair(mu)
    full = LinearSystem(A,np.eye(2),np.eye(2))
    C = np.eye(2) if args.observation=='full' else np.array([[0.,1.]])
    system = LinearSystem(A,np.eye(2),C)
    expert = HistoryPolicy(K,1,1,2)
    states = make_test_states(args.test_pairs,args.test_seed)
    references = [deploy(full,K,expert,1,[t['x1'],t['x2']],steps,warmup) for t in states]
    long_reference = deploy(full,K,expert,1,[.001,1.],args.long_steps,warmup)
    rows, rollout_rows, baselines, paths = [], [], [], {}
    policies = [(name,HistoryPolicy(gain,1,1,2)) for name,gain in gains(mu).items()]
    policies.append(('analytic memory (model-informed)',memory_policy(mu)))
    policies.append(('zero control (stable plant reference)',HistoryPolicy(np.zeros((2,2)),1,1,2)))
    policies = [(name,policy,full) for name,policy in policies]
    if args.observation=='partial':
        policies.append(('known-model observer (partial)',exact_policy(system,K,2,1),system))
    for name, policy, baseline_system in policies:
        identity = dict(controller=name,observation='full' if baseline_system is full else 'partial',training='reference',seed='',context=policy.context,horizon=1,execution=1)
        metrics, records = evaluate_test_set(baseline_system,K,policy,1,states,references,steps,warmup,identity)
        baselines.append(dict(controller=name,**metrics,**stability(baseline_system,policy,1,0)))
        rollout_rows.extend(records)
        paths[name] = deploy(baseline_system,K,policy,1,[.001,1.],steps,warmup)
    sample_policies = []
    for seed in seeds:
        train_steps = max(80,max(contexts)+max(hs)+16)
        data = train_set(system,K,32,train_steps,0,np.random.default_rng(seed))
        if args.training=='recovery':
            recovery,ids,targets = recovery_windows(system,K,seed,max(contexts),max(hs),args.probe_scale)
        for r in contexts:
            for H in hs:
                # Same original target times across both context and prediction lengths.
                trimmed = [{key: value[max(contexts)-r:len(tr['u'])-max(hs)+H+(key!='u')]
                            for key,value in tr.items()} for tr in data]
                if args.training=='narrow':
                    policy, diagnostics = fit_policy(trimmed,r,H)
                else:
                    policy, diagnostics = fit_recovery(recovery,ids,targets,r,H,2)
                for ell in args.executions:
                    if ell>H: continue
                    identity = dict(controller='learned',observation=args.observation,training=args.training,seed=seed,context=r,horizon=H,execution=ell)
                    metrics, records = evaluate_test_set(system,K,policy,ell,states,references,steps,warmup,identity)
                    long_xs = deploy(system,K,policy,ell,[.001,1.],args.long_steps,warmup)
                    rows.append(dict(**identity,**metrics,**diagnostics,**stability(system,policy,ell,0),
                                     long_sample_loss=tracking_loss(long_xs,long_reference)))
                    rollout_rows.extend(records)
                    if seed==seeds[0] and H==max(hs) and ell==1:
                        sample_policies.append((f'learned r={r}',policy,ell))
                    if seed==seeds[0] and r==contexts[min(1,len(contexts)-1)] and H==max(hs):
                        paths[f'learned r={r}, H={H}, ℓ={ell}'] = deploy(system,K,policy,ell,[.001,1.],steps,warmup)
    candidates, summary = summarize_hypothesis(rows,contexts,args.executions,max(hs),seeds,args.loss_tolerance)
    for name, records in [('metrics',rows),('test_rollouts',rollout_rows),('baselines',baselines),
                          ('test_states',states),('execution_candidates',candidates),('context_summary',summary)]:
        write_csv(records,out/f'{name}.csv')
    write_csv([dict(controller=name,step=t,x1=float(x[0]),x2=float(x[1]))
               for name,xs in paths.items() for t,x in enumerate(xs)],out/'sample_trajectories.csv')
    config = vars(args).copy(); config['output']=str(out)
    config.update(mu=mu,warmup=warmup,sensor_matrix=C.tolist(),observability_index=system.observability_index(),training_description='narrow expert histories' if args.training=='narrow' else 'known-action exploration with expert counterfactual chunk labels',
                  test='paired support/perturbed initial states; no continuing noise',loss='mean squared full-state tracking error; exclude initial state',
                  long_sample='single canonical initial state [0.001,1]; diagnostic, not test-set mean')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    plot_results(rows,baselines,paths,out,contexts,hs,steps,description=f'{args.observation} observations / {args.training} training')
    if args.observation=='partial':
        from partial_view import save_learner_view
        save_learner_view(system,K,sample_policies,out,steps,warmup)
    plot_hypothesis(rows,summary,out,contexts,args.executions,max(hs),args.loss_tolerance)
    print(f'Saved {len(rows)} learned configurations and {len(rollout_rows)} test rollouts to {out}')
    for cell in summary:
        print(f"context={cell['context']}: shortest passing execution={cell['shortest_execution']}; no chunking={cell['no_chunking_passes']}")

if __name__ == '__main__':
    main()
