"""Verify numerical evaluation, full-grid integrity, and reproducible re-analysis."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pitfalls.coupled.experiment import Settings, run, deploy, score_errors, squared_errors
from pitfalls.shared.audit import audit, read_table, summarize
from pitfalls.coupled.system import higher_dimensional_pair, experiment_pair, verify_experiment_pair
from model import LinearSystem, exact_policy, rollout


class CoupledPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name)
        cls.settings = Settings(dimensions=(4,), seeds=(53,), executions=(1,2), horizon=2,
                                test_pairs=2, steps=8, long_steps=12, trajectories=4,
                                training_steps=24)
        run(cls.settings, cls.output)
        cls.config = json.loads((cls.output/"config.json").read_text())
        cls.tables = [read_table(cls.output/f"{name}.csv") for name in
                      ("metrics","test_rollouts","baselines","test_states","temporal_profiles")]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_end_to_end_complete_grid_reconciles(self):
        result = audit(self.config, *self.tables)
        self.assertEqual(result["learned_deployments"], 42)
        self.assertEqual(result["prefix_checks"], 14)
        self.assertTrue(result["matched_initial_states"])

    def test_duplicate_seed_cannot_replace_missing_deployment(self):
        tables = copy.deepcopy(self.tables)
        tables[0][-1] = tables[0][0]
        with self.assertRaisesRegex(ValueError,"duplicate learned"):
            audit(self.config, *tables)

    def test_combined_mean_cannot_hide_swapped_perturbation_types(self):
        tables = copy.deepcopy(self.tables)
        row = next(r for r in tables[0] if r["farthest_hidden_mse"] != r["distributed_hidden_mse"])
        row["farthest_hidden_mse"], row["distributed_hidden_mse"] = row["distributed_hidden_mse"], row["farthest_hidden_mse"]
        with self.assertRaisesRegex(ValueError,"Suite aggregate"):
            audit(self.config, *tables)

    def test_missing_temporal_step_is_rejected(self):
        tables = copy.deepcopy(self.tables)
        tables[-1].pop()
        with self.assertRaisesRegex(ValueError,"temporal"):
            audit(self.config, *tables)

    def test_loss_excludes_initial_state_and_support_cases(self):
        paths = np.zeros((3,3,2)); reference = np.zeros_like(paths)
        paths[:,0] = 1000  # Initial error must not enter the score.
        paths[0,1:,0] = 10
        paths[1,1:,0] = 1
        paths[2,1:,0] = 3
        values, _ = score_errors(squared_errors(paths,reference),
            np.array(["support","farthest_hidden","distributed_hidden"]))
        self.assertEqual(values["test_mse"],5.)
        self.assertEqual(values["support_mse"],100.)

    def test_unstable_seed_stays_in_mean(self):
        row = self.tables[0][0]
        a, b = copy.deepcopy(row), copy.deepcopy(row)
        a.update(seed=1, test_mse=1., stable=True, per_step_radius=.9)
        b.update(seed=2, test_mse=100., stable=False, per_step_radius=1.1)
        cell = summarize([a,b])[0]
        self.assertEqual(cell["mean_test_mse"],50.5)
        self.assertEqual(cell["stable_seeds"],1)
        self.assertEqual(cell["max_seed_mse"],100.)

    def test_batched_deployment_matches_shared_scalar_rollout(self):
        A, _, K, _, C = higher_dimensional_pair(4)
        system = LinearSystem(A,np.eye(4),C)
        policy = exact_policy(system,K,4,2)
        initial = np.array([[.001,0,0,1.]])
        batched = deploy(system,K,policy,2,initial,10,4)
        noise = np.zeros((14,4)); noise[3,:-1] = initial[0,:-1]
        start = np.array([0,0,0,.75**(-4)])
        scalar = rollout(system,K,policy,2,start,14,4,noise)["x"][4:]
        np.testing.assert_allclose(batched[0],scalar,atol=1e-12)

    def test_original_n2_is_distinct_but_uses_the_matched_protocol(self):
        coupled=Settings()
        original=Settings(dimensions=(2,),family="original")
        original.validate()
        self.assertEqual(coupled.histories(2),original.histories(2))
        self.assertEqual(coupled.warmup,original.warmup)
        self.assertEqual(coupled.actual_training_steps,original.actual_training_steps)
        A_original,*_=experiment_pair(2,"original")
        A_coupled,*_=experiment_pair(2,"coupled")
        self.assertFalse(np.allclose(A_original,A_coupled))
        checks=verify_experiment_pair(2,"original")
        self.assertEqual(checks["observability_index"],2)
        self.assertGreater(checks["wrong_K2_radius"],1)

    def test_parameter_validation_and_comparable_history_grid(self):
        settings = Settings()
        for n in settings.dimensions:
            self.assertTrue({n-1,n,n+1,4,8,16}.issubset(settings.histories(n)))
        for change in [dict(ridge=np.nan), dict(probe_scale=np.inf), dict(long_steps=10),
                       dict(executions=(0,1)), dict(dimensions=(32,)), dict(seeds=(-1,))]:
            with self.assertRaises(ValueError):
                Settings(**change).validate()


if __name__ == "__main__":
    unittest.main()
