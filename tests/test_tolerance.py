import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pitfalls'))
from tolerance import select_horizons

class ToleranceTests(unittest.TestCase):
    def rows(self):
        return [dict(observation='full',data='narrow',context=r,execution=e,seed=s,
                     tracking_rms=.02,boundary_radius=.9)
                for r in [1,2] for e in [1,2,4] for s in [1,2,3]]

    def test_unstable_small_error_and_single_bad_seed_are_excluded(self):
        rows=self.rows()
        for v in rows:
            if v['context']==1 and v['execution']==1 and v['seed']==1:
                v['boundary_radius']=1.01
            if v['context']==1 and v['execution']==2 and v['seed']==3:
                v['tracking_rms']=.1
        candidates,selected=select_horizons(list(reversed(rows)),.05)
        self.assertEqual([v['shortest_execution'] for v in selected],[4,1])
        self.assertEqual(candidates[0]['reason'],'unstable_or_invalid_radius')
        self.assertEqual(candidates[1]['reason'],'tracking_tolerance_failed')

    def test_threshold_inclusive_stability_strict_and_no_feasible(self):
        rows=self.rows()
        for v in rows:
            v['tracking_rms']=.05
            if v['context']==1: v['boundary_radius']=1
        _,selected=select_horizons(rows,.05)
        self.assertIsNone(selected[0]['shortest_execution'])
        self.assertEqual(selected[1]['shortest_execution'],1)

    def test_invalid_numbers_never_pass(self):
        for field,value in [('tracking_rms',float('nan')),('tracking_rms',float('inf')),
                            ('boundary_radius',float('nan')),('tracking_rms',-1)]:
            rows=self.rows()
            for v in rows: v[field]=value
            _,selected=select_horizons(rows,.05)
            self.assertTrue(all(v['shortest_execution'] is None for v in selected))

    def test_missing_duplicates_and_invalid_tolerance_rejected(self):
        rows=self.rows()
        for bad in [rows[:-1],rows+[rows[0]],[]]:
            with self.assertRaises(ValueError): select_horizons(bad,.05)
        for limit in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): select_horizons(rows,limit)

if __name__=='__main__': unittest.main()
