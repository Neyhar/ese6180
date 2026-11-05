import sys
from pathlib import Path
import unittest
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'pitfalls'),str(Path(__file__).resolve().parents[1])]
from pitfalls import challenging_pair
from model import LinearSystem,exact_policy,history_feature
from evaluation import deploy_trace
from recovery_data import recovery_windows,fit_recovery

class PartialHardExampleTests(unittest.TestCase):
    def setUp(self):
        self.A,_,self.K,_=challenging_pair(.125)
        self.s=LinearSystem(self.A,np.eye(2),np.array([[0.,1.]]))

    def test_observability_and_initial_shock_visibility(self):
        self.assertEqual(self.s.observability_index(),2)
        p=exact_policy(self.s,self.K,2,1)
        traces=[deploy_trace(self.s,self.K,p,1,[d,1.],5,3) for d in [0.,.001]]
        a,b=traces
        for key in ['y','u']:
            # Before first decision, the entire available history is identical.
            np.testing.assert_allclose(a[key][:3+(key=='y')],b[key][:3+(key=='y')],atol=1e-14)
        self.assertNotEqual(a['y'][4,0],b['y'][4,0])
        for t in range(4,8):
            feature=history_feature(b['y'],b['u'],t,2)
            np.testing.assert_allclose(self.s.reconstruction(2)@feature,b['x'][t],atol=1e-12)

    def test_recovery_labels_use_applied_actions_and_expert_continuations(self):
        data,ids,Y=recovery_windows(self.s,self.K,900,2,4)
        i,t=ids[0]
        F=self.A+self.K
        for j in range(4):
            np.testing.assert_allclose(Y[0,2*j:2*j+2],self.K@np.linalg.matrix_power(F,j)@data[i]['x'][t])
        self.assertGreater(np.linalg.norm(data[i]['u'][t]-Y[0,:2]),0)
        p,info=fit_recovery(data,ids,Y,2,4,2)
        self.assertEqual(info['feature_rank'],4)
        self.assertLess(info['training_mse'],1e-12)
        trace=deploy_trace(self.s,self.K,p,1,[.001,1.],20,3)
        self.assertLess(abs(trace['x'][-1,0]),1e-6)

if __name__=='__main__':unittest.main()
