from pathlib import Path
import unittest
import numpy as np
from pitfalls import challenging_pair
from model import LinearSystem, HistoryPolicy
from pitfalls.original.evaluation import make_test_states, deploy, summarize_hypothesis


class ExperimentTests(unittest.TestCase):
    def test_test_states_are_paired_and_deployment_preserves_perturbation(self):
        states=make_test_states(3,900)
        A,_,K,_=challenging_pair(.125)
        system=LinearSystem(A,np.eye(2),np.eye(2))
        for a,b in zip(states[::2],states[1::2]):
            self.assertEqual(a['x2'],b['x2'])
            self.assertEqual(a['x1'],0)
            self.assertTrue(.0005<=abs(b['x1'])<=.0015)
            xs=deploy(system,K,HistoryPolicy(K,1,1,2),1,[b['x1'],b['x2']],20,15)
            np.testing.assert_allclose(xs[0],[b['x1'],b['x2']],atol=1e-14)
            np.testing.assert_allclose(xs[1:],[[0,b['x2']*.75**t] for t in range(1,21)],atol=1e-14)

    def test_shortest_execution_requires_every_seed_stable_and_accurate(self):
        rows=[dict(context=r,horizon=4,execution=e,seed=s,test_loss=1e-6,
                   per_step_radius=.9) for r in [1,2] for e in [1,2,4] for s in [10,11]]
        rows[0]['per_step_radius']=1.001
        rows[2]['test_loss']=1.
        _,summary=summarize_hypothesis(rows,[1,2],[1,2,4],4,[10,11],1e-5)
        self.assertEqual([s['shortest_execution'] for s in summary],[4,1])
        self.assertFalse(summary[0]['no_chunking_passes'])
        self.assertTrue(summary[1]['no_chunking_passes'])
        with self.assertRaises(ValueError):
            summarize_hypothesis(rows[:-1],[1,2],[1,2,4],4,[10,11],1e-5)

if __name__=='__main__': unittest.main()
