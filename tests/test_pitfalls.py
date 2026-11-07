from pathlib import Path
import unittest
import numpy as np
from pitfalls import challenging_pair
from pitfalls.shared.stability import boundary_maps, stability, memory_policy
from model import LinearSystem, HistoryPolicy, exact_policy, rollout

class PitfallsTests(unittest.TestCase):
    def test_pair_and_indistinguishability(self):
        for mu in [.125,.2,.5]:
            A1,A2,K1,K2=challenging_pair(mu)
            for A,K in [(A1,K1),(A2,K2)]:
                self.assertLess(max(abs(np.linalg.eigvals(A))),1)
                np.testing.assert_allclose(A+K,np.diag([0,1-2*mu]))
            np.testing.assert_allclose(K1[:,1],K2[:,1])
            self.assertGreater(max(abs(np.linalg.eigvals(A1+K2))),1)

    def test_boundary_map_matches_direct_committed_actions(self):
        rng=np.random.default_rng(42)
        A,_,K,_=challenging_pair(.125)
        for C in [np.eye(2),np.array([[0.,1.]])]:
            s=LinearSystem(A,np.eye(2),C)
            for r in [1,2,4]:
                d=2*r+2*(r-1);W=.1*rng.normal(size=(8,r*C.shape[0]+2*(r-1)))
                p=HistoryPolicy(W,r,4,2)
                for ell in [1,2,4]:
                    z=rng.normal(size=d);xs=list(z[:2*r].reshape(r,2));us=list(z[2*r:].reshape(r-1,2))
                    feat=np.concatenate([np.array([C@x for x in xs]).ravel(),np.array(us).ravel()])
                    actions=(W@feat).reshape(4,2)
                    for u in actions[:ell]:
                        xs.append(A@xs[-1]+u);us.append(u)
                    actual=np.concatenate([np.array(xs[-r:]).ravel(),np.array(us[-(r-1):]).ravel() if r>1 else []])
                    np.testing.assert_allclose(boundary_maps(s,p,ell)[-1][0]@z,actual,atol=1e-12)

    def test_exact_chunks_and_oracle_metric(self):
        A,_,K,_=challenging_pair(.125);s=LinearSystem(A,np.eye(2),np.eye(2))
        for ell in [1,2,4]:
            p=exact_policy(s,K,2,4)
            tr=rollout(s,K,p,ell,np.array([.2,1.]),30,1)
            expected=np.array([np.linalg.matrix_power(A+K,t)@np.array([.2,1.]) for t in range(31)])
            np.testing.assert_allclose(tr['x'],expected,atol=1e-12)
        w=.01*np.random.default_rng(0).normal(size=(50,2))
        from pitfalls.original.evaluation import deploy, tracking_loss
        oracle = HistoryPolicy(K,1,1,2)
        xs = deploy(s,K,oracle,1,[.001,1.],50,7)
        expected = np.array([np.linalg.matrix_power(A+K,t) @ [.001,1.] for t in range(51)])
        np.testing.assert_allclose(xs,expected,atol=1e-12)
        self.assertLess(tracking_loss(xs,expected),1e-28)

    def test_memory_full_radius_and_stationary_covariance(self):
        A1,A2,K1,_=challenging_pair(.125)
        for A in [A1,A2]:
            s=LinearSystem(A,np.eye(2),np.eye(2))
            self.assertAlmostEqual(stability(s,memory_policy(.125),1,.01)['boundary_radius'],.75)
        s=LinearSystem(A1,np.eye(2),np.eye(2))
        oracle=HistoryPolicy(K1,1,1,2)
        expected=.01*np.sqrt(1+1/(1-.75**2))
        self.assertAlmostEqual(stability(s,oracle,1,.01)['stationary_state_rms'],expected)
        # Monte Carlo-independent covariance check: sum propagated impulse covariances.
        M,Q=boundary_maps(s,memory_policy(.125),1)[-1]
        P=np.zeros_like(M)
        for _ in range(300): P=M@P@M.T+.0001*Q
        self.assertAlmostEqual(np.sqrt(np.trace(P[2:4,2:4])),stability(s,memory_policy(.125),1,.01)['stationary_state_rms'])

if __name__=='__main__': unittest.main()
