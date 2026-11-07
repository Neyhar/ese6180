"""Result contracts prevent invalid comparisons from producing plausible plots."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from pitfalls.shared.results import read_suite, require_matched_tests, write_records


class ResultContractTests(unittest.TestCase):
    def make_suite(self, path):
        path.mkdir()
        config = dict(seeds=[10, 11], contexts=[1], horizons=[1], executions=[1],
                      steps=200, long_steps=1000, warmup=15, test_seed=42,
                      test_pairs=20, mu=.125, observation='partial')
        (path/'config.json').write_text(json.dumps(config))
        (path/'test_states.csv').write_text('test_id,suite,x1,x2\n0,perturbed,.001,1\n')
        rows = [dict(seed=s, context=1, horizon=1, execution=1, test_loss=1.) for s in [10,11]]
        write_records(path/'metrics.csv', rows)
        return rows

    def test_missing_seed_cannot_be_hidden_by_duplicate_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run';rows=self.make_suite(path)
            self.assertEqual(len(read_suite(path)[1]),2)
            write_records(path/'metrics.csv',[rows[0],rows[0]])
            with self.assertRaisesRegex(ValueError,'Missing or duplicate'):
                read_suite(path)

    def test_equal_states_do_not_excuse_different_evaluation_horizons(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a',Path(directory)/'b'
            self.make_suite(a);self.make_suite(b)
            require_matched_tests(a,b)
            config=json.loads((b/'config.json').read_text());config['steps']=60
            (b/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'steps'):
                require_matched_tests(a,b)

    def test_combining_regimes_keeps_optional_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'combined.csv'
            write_records(path,[dict(seed=1,rank=1),dict(seed=2,samples=1600)])
            with path.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['rank'],'1')
            self.assertEqual(rows[1]['samples'],'1600')


if __name__=='__main__':unittest.main()
