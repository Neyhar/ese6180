import unittest
import numpy as np
from model import LinearSystem, example_system, history_feature, exact_policy, rollout


class ModelTests(unittest.TestCase):
    def test_reconstruct_arbitrary_inputs(self):
        s, _ = example_system()
        rng = np.random.default_rng(9)
        for r in (2, 3, 5):
            x = rng.normal(size=2)
            ys, us = [s.observe(x)], []
            for _ in range(r-1):
                u = rng.normal(size=1)
                us.append(u); x = s.step(x, u); ys.append(s.observe(x))
            z = history_feature(ys, us, r-1, r)
            np.testing.assert_allclose(s.reconstruction(r) @ z, x, atol=1e-12)

    def test_exact_chunks_match_expert(self):
        s, K = example_system()
        F = s.A+s.B@K
        x0 = np.array([1.0, -0.3])
        expected = np.array([np.linalg.matrix_power(F, t)@x0 for t in range(31)])
        for execution in (1, 2, 4, 8):
            tr = rollout(s, K, exact_policy(s, K, 2, 8), execution, x0, 30, 3)
            np.testing.assert_allclose(tr['x'], expected, atol=1e-12)

    def test_one_observation_full_state(self):
        s = LinearSystem(np.eye(2)*0.8, np.eye(2), np.eye(2))
        self.assertEqual(s.observability_index(), 1)
        np.testing.assert_allclose(s.reconstruction(1), np.eye(2))

    def test_insufficient_context(self):
        s, _ = example_system()
        self.assertEqual(s.observability_index(), 2)
        with self.assertRaises(ValueError):
            s.reconstruction(1)


if __name__ == '__main__':
    unittest.main()
