"""Noiseless exploratory histories with expert counterfactual chunk labels."""
import numpy as np
from observability.experiment import fit_matched, window_ids, matched_arrays


def recovery_windows(system, gain, seed, max_context, max_horizon, probe_scale=.05):
    """Collect known-action exploration; labels are nominal expert continuations.

    Applied actions fill the input history. They must not be replaced by expert
    labels: the state reconstruction requires the actions actually applied.
    No unknown process noise is used, so histories remain reconstructible.
    """
    rng=np.random.default_rng(seed)
    trajectories=[]
    steps=max(80,max_context+max_horizon+16)
    for _ in range(32):
        x=np.array([0.,rng.uniform(-1,1)])
        xs,ys,us=[x.copy()],[system.observe(x)],[]
        for _ in range(steps):
            u=gain@x+probe_scale*rng.normal(size=system.B.shape[1])
            x=system.step(x,u)
            xs.append(x.copy());ys.append(system.observe(x));us.append(u)
        trajectories.append(dict(x=np.array(xs),y=np.array(ys),u=np.array(us)))
    ids=window_ids(trajectories,max_context,max_horizon,steps)
    F=system.A+system.B@gain
    target_map=np.vstack([gain@np.linalg.matrix_power(F,j) for j in range(max_horizon)])
    targets=np.array([target_map@trajectories[i]['x'][t] for i,t in ids])
    return trajectories,ids,targets


def fit_recovery(data, ids, targets, context, horizon, action_dim):
    features,_=matched_arrays(data,ids,context,1)
    _,policy,info=fit_matched(features,targets[:,:horizon*action_dim],context,action_dim)
    error=features@policy.weights.T-targets[:,:horizon*action_dim]
    info['training_mse']=float(np.mean(error**2))
    return policy,info
