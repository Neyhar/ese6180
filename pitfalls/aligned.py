"""Definition 4.1 linear-core experiment; history/chunk extension, not Appendix J replication."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pitfalls import challenging_pair, gains, train_set, write_csv
from model import LinearSystem, HistoryPolicy, fit_policy, rollout


def memory_policy(mu):
    """Analytic full-state r=2 witness; model-informed, not learned from narrow data."""
    a1, a2 = 1 + mu, -(1 - mu / 4)
    W = np.zeros((2, 6))  # x_(t-1), x_t, u_(t-1)
    W[0, 0] = -((a1 - a2) / 4) ** 2
    W[0, 2] = -(a1 + a2) / 2
    W[0, 3] = -1.5 * mu
    return HistoryPolicy(W, 2, 1, 2)


def boundary_maps(system, policy, execution):
    """Exact maps at each phase of a committed chunk, including disturbance covariance.

    Boundary state z=[x_(t-r+1),...,x_t,u_(t-r+1),...,u_(t-1)].
    Return R_j, Q_j with z_(t+j)=R_j z_t + noise, unit iid plant noise.
    Pending actions are computed once from the boundary feature, never recomputed.
    """
    r, n, m, p = policy.context, system.A.shape[0], system.B.shape[1], system.C.shape[0]
    if not 1 <= execution <= policy.horizon:
        raise ValueError('Invalid execution horizon')
    d = n*r + m*(r-1)
    D = np.zeros((p*r + m*(r-1), d))
    D[:p*r, :n*r] = np.kron(np.eye(r), system.C)
    if r > 1:
        D[p*r:, n*r:] = np.eye(m*(r-1))
    S, U, E = np.zeros((d,d)), np.zeros((d,m)), np.zeros((d,n))
    if r > 1:
        S[:n*(r-1), n:n*r] = np.eye(n*(r-1))
        if r > 2:
            S[n*r:n*r+m*(r-2), n*r+m:] = np.eye(m*(r-2))
        U[-m:] = np.eye(m)
    current = slice(n*(r-1), n*r)
    S[current, current] = system.A
    U[current] = system.B
    E[current] = np.eye(n)
    R, Q = np.eye(d), np.zeros((d,d))
    phases = [(R.copy(), Q.copy())]
    for j in range(execution):
        G = policy.weights[j*m:(j+1)*m] @ D
        R = S @ R + U @ G
        Q = S @ Q @ S.T + E @ E.T
        phases.append((R.copy(), Q.copy()))
    return phases


def stability(system, policy, execution, noise):
    phases = boundary_maps(system, policy, execution)
    M, Q = phases[-1]
    radius = float(np.max(np.abs(np.linalg.eigvals(M))))
    result = dict(boundary_radius=radius, per_step_radius=radius**(1/execution),
                  stable=bool(radius < 1), stationary_state_rms=None)
    if radius < 1:
        P = solve_discrete_lyapunov(M, noise**2 * Q)
        n, r = system.A.shape[0], policy.context
        current = slice(n*(r-1), n*r)
        variances = [np.trace((R @ P @ R.T + noise**2 * V)[current,current])
                     for R,V in phases[:-1]]
        result['stationary_state_rms'] = float(np.sqrt(max(0, np.mean(variances))))
    return result


def demos(system, K, regime, seed, horizon):
    rng = np.random.default_rng(seed)
    steps = max(60, horizon+16)
    if regime != 'persistent':
        return train_set(system, K, 32, steps, .3 if regime == 'initial' else 0, rng)
    trajectories = []
    for _ in range(32):
        x = np.array([0., rng.uniform(-1,1)])
        xs, ys, us = [x], [system.observe(x)], []
        for _ in range(steps):
            u = K @ x
            x = system.step(x,u,.05*rng.normal(size=2))
            xs.append(x); ys.append(system.observe(x)); us.append(u)
        trajectories.append(dict(x=np.array(xs), y=np.array(ys), u=np.array(us)))
    return trajectories


def evaluate(system, K, policy, execution, steps, warmup, w):
    reference = HistoryPolicy(K,1,1,2)
    tr = rollout(system,K,policy,execution,np.array([0.,1.]),steps,warmup,w)
    # Full-state oracle must observe x even when the learner sees only x2.
    full = LinearSystem(system.A,system.B,np.eye(2))
    ref = rollout(full,K,reference,1,np.array([0.,1.]),steps,warmup,w)
    x, u = tr['x'][warmup+1:], tr['u'][warmup:]
    delta = u - tr['x'][warmup:-1] @ K.T
    error = x - ref['x'][warmup+1:]
    return dict(late_state_rms=float(np.sqrt(np.mean(np.sum(x[-50:]**2,axis=1)))),
                tracking_rms=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
                action_rms=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                peak_abs_x1=float(np.max(np.abs(x[:,0]))))


def hard_example(mu, out, steps=60, eps=1e-3):
    A1,A2,K1,K2 = challenging_pair(mu)
    F = np.diag([0.,1-2*mu])
    assert np.allclose(A1+K1,F) and np.allclose(A2+K2,F)
    assert np.allclose(K1[:,1],K2[:,1])
    assert max(abs(np.linalg.eigvals(A1))) < 1
    assert max(abs(np.linalg.eigvals(A2))) < 1
    rows=[]
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for ax, perturb in zip(axes,[0.,eps]):
        for name,K in gains(mu).items():
            x=np.array([perturb,1.]); xs=[x]
            for _ in range(steps):
                x=(A1+K)@x; xs.append(x)
            xs=np.array(xs)
            ax.semilogy(np.maximum(abs(xs[:,0]),1e-16),label=name)
            rows.append(dict(controller=name,initial_x1=perturb,
                             radius=float(max(abs(np.linalg.eigvals(A1+K)))),
                             final_abs_x1=float(abs(xs[-1,0]))))
        ax.set(title=f'Plant 1, initial x1={perturb:g}',xlabel='Step',ylabel='|x1| (display floor 1e-16)')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'hard_example.png',dpi=160);plt.close(fig)
    write_csv(rows,out/'hard_example.csv')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mu',type=float,default=.125)
    parser.add_argument('--noise',type=float,default=.01)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--horizon',type=int,default=16)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results'/'aligned')
    args=parser.parse_args()
    if args.steps < 50 or args.noise < 0 or args.horizon < 1:
        parser.error('Require steps >= 50, noise >= 0, horizon >= 1')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    hard_example(args.mu,out)
    A,_,K,_=challenging_pair(args.mu)
    contexts=[1,2,3,4,6,8];executions=[v for v in [1,2,4,8,16] if v<=args.horizon]
    seeds=[618,619,620,621,622];warmup=max(contexts)-1;total=args.steps+warmup
    rows=[]; baselines=[]
    for seed in seeds:
        # Independent evaluation stream, identical across data, observation, r and ell settings.
        w=args.noise*np.random.default_rng(seed+10000).normal(size=(total,2))
        w[:warmup]=0
        for obs,C in [('full',np.eye(2)),('partial',np.array([[0.,1.]]))]:
            system=LinearSystem(A,np.eye(2),C)
            for regime in ['narrow','initial','persistent']:
                data=demos(system,K,regime,seed,args.horizon)
                for r in contexts:
                    policy,diag=fit_policy(data,r,args.horizon)
                    for ell in executions:
                        rows.append(dict(observation=obs,data=regime,seed=seed,context=r,execution=ell,
                                         **diag,**stability(system,policy,ell,args.noise),
                                         **evaluate(system,K,policy,ell,total,warmup,w)))
        full=LinearSystem(A,np.eye(2),np.eye(2))
        for name,policy in [('expert',HistoryPolicy(K,1,1,2)),
                            ('zero',HistoryPolicy(np.zeros((2,2)),1,1,2)),
                            ('analytic_memory',memory_policy(args.mu))]:
            baselines.append(dict(controller=name,seed=seed,**stability(full,policy,1,args.noise),
                                  **evaluate(full,K,policy,1,total,warmup,w)))
    write_csv(rows,out/'framework_sweep.csv');write_csv(baselines,out/'baselines.csv')
    for obs in ['full','partial']:
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),sharey=True)
        for ax,regime in zip(axes,['narrow','initial','persistent']):
            matrix=np.zeros((len(contexts),len(executions)))
            for i,r in enumerate(contexts):
                for j,ell in enumerate(executions):
                    group=[v for v in rows if v['observation']==obs and v['data']==regime and v['context']==r and v['execution']==ell]
                    matrix[i,j]=np.median([v['tracking_rms'] for v in group])
                    label=f'{matrix[i,j]:.2g}'+(' *' if any(not v['stable'] for v in group) else '')
                    ax.text(j,i,label,ha='center',va='center',fontsize=6,color='white')
            im=ax.imshow(np.log10(np.maximum(matrix,1e-16)),aspect='auto',vmin=-3,vmax=12)
            ax.set(title=regime,xlabel='Execution length',xticks=range(len(executions)),xticklabels=executions,
                   yticks=range(len(contexts)),yticklabels=contexts)
        axes[0].set_ylabel('Context length')
        fig.suptitle(f'{obs}: median tracking RMS; * = unstable for at least one seed')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='log10 tracking RMS (color range capped)')
        fig.savefig(out/f'grid_{obs}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    config=vars(args).copy();config['output']=str(out);config.update(seeds=seeds,contexts=contexts,executions=executions,
        warmup=warmup,scope='Definition 4.1 linear core; extension, not Appendix J replication')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print(f'Wrote {len(rows)} paired sweep rows and {len(baselines)} baselines to {out}')


if __name__=='__main__':
    main()
