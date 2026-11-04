"""Exact history/chunk stability and stationary noise RMS (no finite-horizon proxy)."""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
from pitfalls import challenging_pair, train_set
from aligned import stability
from model import LinearSystem, fit_policy


def main():
    for mu in [.125,.2]:
        A,_,K,_=challenging_pair(mu)
        for tag,C,exc in [('full narrow',np.eye(2),0),('partial narrow',np.array([[0.,1.]]),0),('full initial',np.eye(2),.3)]:
            system=LinearSystem(A,np.eye(2),C)
            data=train_set(system,K,32,60,exc,np.random.default_rng(618))
            print(f'mu={mu}, {tag}')
            for r in [1,2,3,4,6,8]:
                policy,diag=fit_policy(data,r,16)
                for ell in [1,2,4,8,16]:
                    result=stability(system,policy,ell,.01)
                    rms=result['stationary_state_rms']
                    print(f" r={r} ell={ell} rank={diag['feature_rank']}/{diag['feature_dim']} "
                          f"per-step radius={result['per_step_radius']:.6f} "
                          f"stationary RMS={rms if rms is not None else 'undefined (unstable)'}")

if __name__=='__main__': main()
