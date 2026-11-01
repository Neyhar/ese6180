import unittest
import numpy as np
from model import coupled_system, demonstrations, HistoryPolicy, rollout
from long_experiment import (window_ids,matched_arrays,fit_matched,batch_rollout,
                             trajectory_errors,summarize_errors,paired_tail_difference)


class LongExperimentTests(unittest.TestCase):
    def test_systems_observable_and_expert_stable(self):
        for n in (2,3,4,5,10,50,100):
            s,K=coupled_system(n)
            self.assertEqual(s.observability_index(),2)
            self.assertLess(np.linalg.norm(s.A+s.B@K,2),1)
            self.assertEqual(s.A.shape,(n,n))
        s,_=coupled_system(10,sensors=1)
        self.assertEqual(s.observability_index(),10)

    def test_matched_first_actions_and_horizon_invariance(self):
        s,K=coupled_system(3)
        tr=demonstrations(s,K,np.random.default_rng(2).normal(size=(8,3)),30)
        ids=window_ids(tr,4,10,6)
        X,Y=matched_arrays(tr,ids,2,10)
        one,chunk,info=fit_matched(X,Y,2,1)
        # Independently solve the actual ridge objective, not just comparing slices.
        d=X.shape[1]
        expected=np.linalg.lstsq(np.vstack((X,1e-3*np.eye(d))),
                                np.concatenate((Y[:,0],np.zeros(d))),rcond=None)[0]
        np.testing.assert_allclose(one.weights[0],expected,atol=1e-10)
        np.testing.assert_allclose(one.weights,chunk.weights[:1],atol=1e-12)
        _,short,_=fit_matched(X,Y[:,:4],2,1)
        np.testing.assert_allclose(short.weights,chunk.weights[:4],atol=1e-12)
        self.assertEqual(len(ids),48)
        self.assertTrue(all(t>=3 and t+10<=30 for _,t in ids))
        XV,YV=matched_arrays(tr,ids,4,10)
        np.testing.assert_array_equal(Y,YV)

    def test_batched_rollout_matches_scalar_and_partial_last_chunk(self):
        s,K=coupled_system(4)
        rng=np.random.default_rng(3)
        # A non-expert arbitrary policy detects timing and history-order errors.
        policy=HistoryPolicy(rng.normal(scale=.03,size=(7,8)),3,7,1)
        initial=rng.normal(size=(2,4)); w=rng.normal(scale=.01,size=(2,19,4))
        xs,us,fail=batch_rollout(s,K,policy,7,initial,w,3)
        for i in range(2):
            tr=rollout(s,K,policy,7,initial[i],19,3,w[i])
            np.testing.assert_allclose(xs[i],tr['x'],atol=1e-12)
            np.testing.assert_allclose(us[i],tr['u'],atol=1e-12)
        self.assertTrue(np.all(fail==-1))

    def test_long_chunk_1000(self):
        s,K=coupled_system(2)
        policy=HistoryPolicy(np.zeros((1000,1)),1,1000,1)
        xs,us,fail=batch_rollout(s,K,policy,1000,np.ones((1,2)),np.zeros((1,2100,2)),0)
        self.assertEqual(us.shape,(1,2100,1))
        np.testing.assert_allclose(xs[0,-1],np.linalg.matrix_power(s.A,2100)@np.ones(2),atol=1e-12)

    def test_tail_normalization_and_failure(self):
        e=trajectory_errors(np.ones((1,5,2)),np.zeros((1,5,2)),1)
        self.assertEqual(e.shape,(1,3))
        self.assertEqual(summarize_errors(e,2)['cumulative_error'],3)
        s,K=coupled_system(2)
        p=HistoryPolicy(np.array([[1e9]]),1,1,1)
        xs,us,fail=batch_rollout(s,K,p,1,np.ones((1,2)),np.zeros((1,5,2)),0)
        self.assertEqual(fail[0],0)
        self.assertTrue(np.isinf(xs[0,1:]).all())
        self.assertTrue(np.isnan(paired_tail_difference(np.array([np.inf]),np.array([np.inf]),1)))
        self.assertEqual(paired_tail_difference(np.array([1.,3.]),np.array([2.,5.]),1),-2.)


if __name__=='__main__': unittest.main()
