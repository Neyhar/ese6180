"""Validate saved hard-example grids and aggregate seed/episode losses."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from pitfalls.coupled.system import verify_experiment_pair
from pitfalls.coupled.experiment import IDENTITY, SUITES

TRAINING = ("narrow", "recovery")
INTEGER_FIELDS = {"dimension", "seed", "context", "horizon", "execution", "test_id", "step",
                  "history", "rank", "feature_rank", "feature_dim", "samples", "seeds", "stable_seeds",
                  "sensors", "observability_index", "action_recovery_index", "demonstration_subspace_dimension"}
TEXT_FIELDS = {"family", "training", "controller", "suite", "state"}


def read_table(path):
    with Path(path).open() as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for record in raw:
        row = {}
        for key, value in record.items():
            if value == "": row[key] = None
            elif key in TEXT_FIELDS: row[key] = value
            elif key == "stable": row[key] = value == "True"
            elif key in INTEGER_FIELDS: row[key] = int(value)
            else: row[key] = float(value)
        rows.append(row)
    return rows


def identity(row):
    return tuple(row[key] for key in IDENTITY)


def audit(config, metrics, episodes, baselines, states, profiles):
    if config.get("schema_version") != 2:
        raise ValueError("This analyzer requires a schema-version-2 run; retain old snapshots separately")
    expected = set()
    for n in config["dimensions"]:
        verify_experiment_pair(n,config.get("family","coupled"))
        for training in TRAINING:
            for seed in config["seeds"]:
                for r in config["contexts_by_dimension"][str(n)]:
                    for horizon, execution in [(1,1)] + [(config["horizon"],e) for e in config["executions"]]:
                        expected.add((n,training,seed,r,horizon,execution,"learned"))
    index = {identity(row):row for row in metrics}
    if len(index) != len(metrics) or set(index) != expected:
        raise ValueError("Missing, unexpected, or duplicate learned deployments")
    reference_index = {identity(row):row for row in baselines}
    if len(reference_index) != 6*len(config["dimensions"]):
        raise ValueError("Missing or duplicate reference baselines")
    all_index = index.copy()
    all_index.update(reference_index)
    raw_groups = defaultdict(list)
    for row in episodes:
        raw_groups[identity(row)].append(row)
    if set(raw_groups) != set(all_index):
        raise ValueError("Rollout and metric controller identities differ")
    max_gap = 0.
    for key, row in all_index.items():
        raw = raw_groups[key]
        cases = {(x["test_id"],x["suite"]) for x in raw}
        wanted = {(i,suite) for i in range(config["test_pairs"]) for suite in SUITES}
        if len(raw) != len(wanted) or cases != wanted:
            raise ValueError(f"Missing or duplicate test cases for {key}")
        for suite in SUITES:
            suite_mean = np.mean([x["test_mse"] for x in raw if x["suite"] == suite])
            if not np.isclose(suite_mean,row[suite+"_mse"],rtol=1e-11,atol=1e-20):
                raise ValueError(f"Suite aggregate differs from raw episodes: {key}, {suite}")
        values = [x["test_mse"] for x in raw if x["suite"] != "support"]
        if any(np.isnan(x) or x < 0 for x in values):
            raise ValueError("Invalid loss in raw test data")
        actual = float(np.mean(values))
        if not np.isclose(actual,row["test_mse"],rtol=1e-11,atol=1e-20):
            raise ValueError(f"Aggregate does not match raw episodes: {key}")
        if np.isfinite(actual):
            max_gap = max(max_gap,abs(actual-row["test_mse"])/max(actual,1e-300))
        if row["stable"] != (row["per_step_radius"] < 1):
            raise ValueError("Stability flag disagrees with spectral radius")
    prefix_checks = 0
    for key, row in index.items():
        if row["horizon"] == 1:
            counterpart = index[key[:4] + (config["horizon"],1,"learned")]
            if not np.isclose(row["test_mse"],counterpart["test_mse"],rtol=1e-7,atol=1e-14):
                raise ValueError(f"First-action training control differs: {key}")
            prefix_checks += 1
    state_index = {(x["dimension"],x["test_id"],x["suite"]):x for x in states}
    if len(state_index) != len(states) or len(states) != len(config["dimensions"])*3*config["test_pairs"]:
        raise ValueError("Incomplete or duplicate initial-state records")
    first_n = config["dimensions"][0]
    for n in config["dimensions"]:
        for i in range(config["test_pairs"]):
            for suite in SUITES:
                row, other = state_index[n,i,suite], state_index[first_n,i,suite]
                a, b = json.loads(row["state"]), json.loads(other["state"])
                if len(a) != n or not np.isclose(a[-1],b[-1],atol=1e-14):
                    raise ValueError("Visible initial amplitudes are not matched across dimensions")
                if not np.isclose(row["perturbation_norm"],other["perturbation_norm"],atol=1e-14):
                    raise ValueError("Total perturbation magnitudes differ across dimensions")
    temporal_groups = defaultdict(list)
    temporal_steps = defaultdict(set)
    for row in profiles:
        key = identity(row)+(row["suite"],)
        temporal_groups[key].append(row["mean_squared_error"])
        temporal_steps[key].add(row["step"])
    for key, values in temporal_groups.items():
        if len(values) != config["steps"] or temporal_steps[key] != set(range(1,config["steps"]+1)):
            raise ValueError("Incomplete temporal mean profile")
        aggregate = all_index[key[:-1]][key[-1]+"_mse"]
        if not np.isclose(np.mean(values),aggregate,rtol=1e-10,atol=1e-20):
            raise ValueError("Temporal and episode means differ")
    return dict(learned_deployments=len(metrics),reference_deployments=len(baselines),
                raw_episode_records=len(episodes),perturbed_cases_per_deployment=2*config["test_pairs"],
                independent_training_seeds=len(config["seeds"]),prefix_checks=prefix_checks,
                temporal_profile_checks=len(temporal_groups),maximum_relative_aggregation_gap=max_gap,
                matched_initial_states=True,complete_grid=True)


def summarize(metrics):
    groups = defaultdict(list)
    for row in metrics:
        groups[tuple(row[k] for k in ("dimension","training","context","horizon","execution"))].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        values = np.array([x["test_mse"] for x in rows])
        result.append(dict(zip(("dimension","training","context","horizon","execution"),key),
            mean_test_mse=float(values.mean()),median_seed_mse=float(np.median(values)),
            min_seed_mse=float(values.min()),max_seed_mse=float(values.max()),
            mean_per_coordinate_mse=float(values.mean()/key[0]),
            farthest_hidden_mse=float(np.mean([x["farthest_hidden_mse"] for x in rows])),
            distributed_hidden_mse=float(np.mean([x["distributed_hidden_mse"] for x in rows])),
            support_mse=float(np.mean([x["support_mse"] for x in rows])),
            stable_seeds=sum(x["stable"] for x in rows),seeds=len(rows),
            worst_rate=max(x["per_step_radius"] for x in rows),
            min_rate=min(x["per_step_radius"] for x in rows)))
    return result


def key_index(summary):
    return {(r["dimension"],r["training"],r["context"],r["horizon"],r["execution"]):r for r in summary}


def paired_chunking(metrics, config):
    index = {identity(row):row for row in metrics}
    records = []
    for row in metrics:
        if row["horizon"] != config["horizon"] or row["execution"] == 1: continue
        baseline = index[identity(row)[:4]+(config["horizon"],1,"learned")]
        ratio = row["test_mse"]/baseline["test_mse"] if baseline["test_mse"] > 0 else np.inf
        records.append(dict(dimension=row["dimension"],training=row["training"],seed=row["seed"],
            context=row["context"],execution=row["execution"],chunk_to_feedback_ratio=ratio,
            chunk_mse=row["test_mse"],feedback_mse=baseline["test_mse"],
            chunk_stable=row["stable"],feedback_stable=baseline["stable"]))
    return records


