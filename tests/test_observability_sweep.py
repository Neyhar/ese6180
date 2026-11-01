import unittest
from argparse import Namespace
import numpy as np
from model import coupled_system, demonstrations
from run_experiment import build_plan
from long_experiment import window_ids


class ObservabilitySweepTests(unittest.TestCase):
    def test_exact_requested_indices_without_changing_plant(self):
        for n in (2,3,4,5,10,50,100):
            reference,K0=coupled_system(n)
            for h in sorted({v for v in (1,2,4,8,n) if v<=n}):
                s,K=coupled_system(n,observability_index=h)
                self.assertEqual(s.observability_index(),h)
                np.testing.assert_array_equal(s.A,reference.A)
                np.testing.assert_array_equal(s.B,reference.B)
                np.testing.assert_array_equal(K,K0)

    def test_default_plan_and_matching(self):
        args=Namespace(dimensions=[2,3,4,5,10,50,100],observability_indices=['2','4','8','n'],
                       sensors=None,contexts=None,system_seed=618,decay=.995)
        plan=build_plan(args)
        self.assertEqual(len(plan),20)
        self.assertEqual([v['observability_index'] for v in plan if v['dimension']==100],[2,4,8,100])
        self.assertEqual([v['observability_index'] for v in plan if v['dimension']==2],[2])
        for n in args.dimensions:
            cases=[v for v in plan if v['dimension']==n]
            self.assertEqual(len({v['warmup'] for v in cases}),1)
            for v in cases:
                h=v['observability_index']
                self.assertIn(h,v['contexts']);self.assertIn(h+1,v['contexts'])
        # Same initial states and input trajectories imply identical training indices.
        trajectories=[]
        for h in (2,5):
            s,K=coupled_system(5,observability_index=h)
            trajectories.append(demonstrations(s,K,np.ones((2,5)),40))
        self.assertEqual(window_ids(trajectories[0],6,20,10),window_ids(trajectories[1],6,20,10))
        np.testing.assert_array_equal(trajectories[0][0]['x'],trajectories[1][0]['x'])

    def test_explicit_context_and_sensor_overrides(self):
        args=Namespace(dimensions=[5],observability_indices=['n'],sensors=None,contexts=[1,2],system_seed=618,decay=.995)
        self.assertEqual(build_plan(args)[0]['contexts'],[1,2])
        args.sensors=2
        self.assertEqual(build_plan(args)[0]['observability_index'],3)
        args.sensors=None;args.observability_indices=['9']
        with self.assertRaises(ValueError):build_plan(args)

if __name__=='__main__':unittest.main()
