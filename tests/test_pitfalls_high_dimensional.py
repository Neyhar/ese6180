"""Coupled hard pairs preserve expert ambiguity and require growing history."""
from pathlib import Path
import unittest
import numpy as np
from pitfalls.high_dimensional import higher_dimensional_pair, verify_pair, test_states, exact_schur_certificate
from model import LinearSystem, history_feature


class CoupledHigherDimensionalTests(unittest.TestCase):
    def test_whole_support_has_identical_expert_demonstrations(self):
        for n in [2,4,8,16]:
            with self.subTest(n=n):
                check=verify_pair(n)
                self.assertEqual(check['observability_index'],n)
                A1,A2,K1,K2,C=higher_dimensional_pair(n)
                F=np.zeros((n,n));F[-1,-1]=.75
                np.testing.assert_allclose(A1+K1,F,atol=1e-14)
                np.testing.assert_allclose(A2+K2,F,atol=1e-14)
                self.assertGreater(exact_schur_certificate(n),0)
                self.assertAlmostEqual(check['observability_condition'],1.)
                self.assertGreater(check['wrong_K1_on_A2_radius'],1)
                for amplitude in [-1.,.37,1.]:
                    x=np.zeros(n);x[-1]=amplitude;a=x.copy();b=x.copy()
                    for _ in range(25):
                        np.testing.assert_allclose(K1@a,K2@b,atol=1e-14)
                        a=A1@a+K1@a;b=A2@b+K2@b
                        np.testing.assert_allclose(a,b,atol=1e-14)

    def test_wrong_expert_grows_hidden_departures(self):
        for n in [2,4,8,16]:
            A1,A2,K1,K2,C=higher_dimensional_pair(n)
            x=np.zeros(n);x[0]=.001
            good=x.copy();bad=x.copy()
            for _ in range(50):good=(A1+K1)@good;bad=(A1+K2)@bad
            np.testing.assert_allclose(bad[0],.001*2.015**50,rtol=1e-12)
            self.assertLess(np.linalg.norm(good),1e-7)
            self.assertGreater(np.linalg.norm(bad),1.)
            self.assertLess(max(abs(np.linalg.eigvals(A1))),1)
            self.assertLess(max(abs(np.linalg.eigvals(A2))),1)

    def test_single_sensor_delay_and_full_reconstruction(self):
        rng=np.random.default_rng(42)
        for n in [2,4,8,16]:
            A,_,K,_,C=higher_dimensional_pair(n);system=LinearSystem(A,np.eye(n),C)
            self.assertEqual(C.shape,(1,n))
            self.assertEqual([np.linalg.matrix_rank(system.observability(r)) for r in range(1,n+1)],list(range(1,n+1)))
            d=np.eye(n)[:,0]
            for k in range(n-1):self.assertEqual(float((C@np.linalg.matrix_power(A,k)@d)[0]),0)
            self.assertGreater(abs(float((C@np.linalg.matrix_power(A,n-1)@d)[0])),0)
            # One n-state cycle reaches every coordinate; not independent 2D copies.
            for k in range(n):self.assertGreater(abs(np.linalg.matrix_power(A,k)[k,0]),0)
            self.assertNotEqual(A[0,-1],0)
            self.assertEqual(np.linalg.matrix_rank(K),n)
            x=rng.normal(size=n);xs=[x];ys=[C@x];us=[]
            for _ in range(n-1):
                u=rng.normal(size=n);x=A@x+u
                xs.append(x);ys.append(C@x);us.append(u)
            z=history_feature(np.array(ys),np.array(us),n-1,n)
            np.testing.assert_allclose(system.reconstruction(n)@z,x,atol=1e-9)

    def test_test_amplitudes_match_across_dimensions(self):
        previous=None
        for n in [2,4,8,16]:
            records,x=test_states(n,5,42)
            magnitudes=[]
            for i in range(5):
                a,b,c=x[3*i:3*i+3]
                self.assertEqual(a[-1],b[-1]);self.assertEqual(b[-1],c[-1])
                self.assertEqual(np.count_nonzero(b[:-1]),1)
                self.assertAlmostEqual(np.linalg.norm(b[:-1]),np.linalg.norm(c[:-1]))
                magnitudes.append(np.linalg.norm(b[:-1]))
            if previous is not None:np.testing.assert_array_equal(magnitudes,previous)
            previous=magnitudes

    def test_perfect_memoryless_demonstration_fit_is_unstable(self):
        for n in [2,4,8,16]:
            A1,_,K1,_,C=higher_dimensional_pair(n)
            L=K1@C.T
            # This is the unique linear y-only action rule matching all demonstrations.
            for amplitude in [-1.,.23,1.]:
                x=amplitude*C.T[:,0]
                np.testing.assert_allclose(L@(C@x),K1@x,atol=1e-14)
            radius=max(abs(np.linalg.eigvals(A1+L@C)))
            self.assertAlmostEqual(radius,1.02)

    def test_invalid_dimensions(self):
        for n in [0,1,4.5]:
            with self.assertRaises(ValueError):higher_dimensional_pair(n)


if __name__=='__main__':unittest.main()
