"""Matched training and evaluation for the coupled hard pair.

The shared model, matched-window ridge fitter, batched rollout, and exact
augmented-system stability routines implement the numerical work.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
from datetime import datetime, timezone

import numpy as np

from model import LinearSystem, HistoryPolicy, demonstrations, exact_policy
from observability.experiment import window_ids, matched_arrays, fit_matched, batch_rollout
from pitfalls.shared.stability import stability
from pitfalls.shared.results import write_records
from pitfalls.coupled.system import experiment_pair, verify_experiment_pair

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SUITES = ("support", "farthest_hidden", "distributed_hidden")
IDENTITY = ("dimension", "training", "seed", "context", "horizon", "execution", "controller")


@dataclass(frozen=True)
class Settings:
    dimensions: tuple = (2, 4, 8, 16)
    family: str = "coupled"
    seeds: tuple = (1118, 1119, 1120, 1121, 1122)
    executions: tuple = (1, 2, 4, 8, 16)
    horizon: int = 16
    test_seed: int = 20261110
    test_pairs: int = 40
    steps: int = 200
    long_steps: int = 1000
    trajectories: int = 128
    training_steps: int = 100
    ridge: float = 1e-6
    probe_scale: float = .05

    def histories(self, dimension):
        # Dense local grid plus common absolute contexts for dimension comparisons.
        return sorted(set(range(1, dimension + 2)) | {2, 4, 8, 16})

    @property
    def max_context(self):
        return max(17, max(max(self.histories(n)) for n in self.dimensions))

    @property
    def warmup(self):
        return self.max_context - 1

    @property
    def actual_training_steps(self):
        return max(self.training_steps, self.max_context + self.horizon + 16)

    def validate(self):
        for values in (self.dimensions, self.seeds, self.executions):
            if not values or len(values) != len(set(values)):
                raise ValueError("Grid entries must be nonempty and unique")
        if min(self.seeds) < 0 or self.test_seed < 0:
            raise ValueError("Random seeds must be nonnegative")
        if any(n not in (2, 4, 8, 16) for n in self.dimensions):
            raise ValueError("This verified family supports dimensions 2, 4, 8, and 16")
        if self.family not in ("coupled", "original") or (self.family == "original" and self.dimensions != (2,)):
            raise ValueError("The original family requires --dimensions 2")
        if self.horizon < 2 or 1 not in self.executions:
            raise ValueError("Require prediction horizon >= 2 and execution 1")
        if min(self.executions) < 1 or max(self.executions) > self.horizon:
            raise ValueError("Require 1 <= execution <= prediction horizon")
        if min(self.test_pairs, self.steps, self.trajectories, self.training_steps) < 1:
            raise ValueError("Sample counts and rollout lengths must be positive")
        if (self.long_steps < self.steps or not np.isfinite(self.ridge) or not np.isfinite(self.probe_scale)
                or self.ridge <= 0 or self.probe_scale <= 0):
            raise ValueError("Require long_steps >= steps and positive ridge/probe scale")


def test_states(n, count, seed):
    """Pair visible amplitudes and TOTAL hidden perturbation norms across n."""
    scalar_rng = np.random.default_rng(seed)
    direction_rng = np.random.default_rng(seed + n)
    states, records = [], []
    for test_id in range(count):
        visible = scalar_rng.uniform(-1, 1)
        epsilon = scalar_rng.uniform(.0005, .0015)
        farthest = np.zeros(n - 1)
        farthest[0] = scalar_rng.choice([-1, 1])
        direction = direction_rng.normal(size=n - 1)
        direction /= np.linalg.norm(direction)
        for suite, vector in zip(SUITES, (np.zeros(n - 1), farthest, direction)):
            state = np.r_[epsilon * vector, visible]
            states.append(state)
            records.append(dict(dimension=n, test_id=test_id, suite=suite,
                                perturbation_norm=float(np.linalg.norm(state[:-1])),
                                state=json.dumps(state.tolist())))
    return records, np.array(states)


def collect(system, gain, starts, steps, training, seed, probe_scale):
    if training == "narrow":
        return demonstrations(system, gain, starts, steps)
    rng = np.random.default_rng(seed)
    trajectories = []
    for initial in starts:
        state = initial.copy()
        xs, ys, us = [state.copy()], [system.observe(state)], []
        for _ in range(steps):
            action = gain @ state + probe_scale * rng.normal(size=len(state))
            state = system.step(state, action)
            xs.append(state.copy())
            ys.append(system.observe(state))
            us.append(action)
        trajectories.append(dict(x=np.array(xs), y=np.array(ys), u=np.array(us)))
    return trajectories


def deploy(system, expert_gain, policy, execution, initial, steps, warmup):
    """Expert fills the buffer, then a hidden jump occurs before learner control."""
    start = np.zeros_like(initial)
    start[:, -1] = initial[:, -1] * .75 ** (-warmup)
    disturbances = np.zeros((len(initial), steps + warmup, initial.shape[1]))
    disturbances[:, warmup - 1, :-1] = initial[:, :-1]
    paths, _, _ = batch_rollout(system, expert_gain, policy, execution, start,
                               disturbances, warmup, divergence_limit=np.inf)
    np.testing.assert_allclose(paths[:, warmup], initial, atol=1e-11)
    return paths[:, warmup:]


def squared_errors(paths, reference):
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.sum((paths[:, 1:] - reference[:, 1:]) ** 2, axis=2)
    return np.where(np.isfinite(errors), errors, np.inf)


def score_errors(errors, suite_names):
    """Full-state distance; omit initial state; give both perturbation types equal weight."""
    episode_means = errors.mean(axis=1)
    by_suite = {suite: float(episode_means[suite_names == suite].mean()) for suite in SUITES}
    total = (by_suite["farthest_hidden"] + by_suite["distributed_hidden"]) / 2
    return dict(test_mse=total, **{f"{suite}_mse": value for suite, value in by_suite.items()}), episode_means


def evaluate(system, expert_gain, policy, execution, initial, records, reference,
             identity, settings, keep_profile=False):
    paths = deploy(system, expert_gain, policy, execution, initial, settings.steps, settings.warmup)
    errors = squared_errors(paths, reference)
    names = np.array([row["suite"] for row in records])
    scores, means = score_errors(errors, names)
    scores["per_coordinate_mse"] = scores["test_mse"] / system.A.shape[0]
    scores["worst_episode_mse"] = float(max(means[names != "support"]))
    episodes = [dict(**identity, test_id=row["test_id"], suite=row["suite"], test_mse=float(value))
                for row, value in zip(records, means)]
    profiles, samples = [], []
    if keep_profile:
        for suite in SUITES:
            values = errors[names == suite].mean(axis=0)
            profiles.extend(dict(**identity, suite=suite, step=t+1, mean_squared_error=float(value))
                            for t, value in enumerate(values))
        # Same test_id=0 farthest-hidden example at every dimension; sample, not test mean.
        for t, state in enumerate(paths[1]):
            samples.append(dict(**identity, step=t, x1=float(state[0]), observed=float(state[-1]),
                                hidden_norm=float(np.linalg.norm(state[:-1])),
                                squared_tracking_error=float(np.sum((state-reference[1,t])**2))))
    return scores, episodes, profiles, samples


def wants_diagnostic(n, r, horizon, execution, settings):
    return ((horizon == 1 and r in {1, n-1, n}) or
            (horizon == settings.horizon and r == n and execution == max(settings.executions)))


def run(settings, output):
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    metrics, episodes, profiles, samples = [], [], [], []
    checks, states, baselines, ranks = [], [], [], []
    for n in settings.dimensions:
        checks.append(verify_experiment_pair(n, settings.family))
        A1, A2, K1, K2, C = experiment_pair(n, settings.family)
        partial = LinearSystem(A1, np.eye(n), C)
        full1 = LinearSystem(A1, np.eye(n), np.eye(n))
        full2 = LinearSystem(A2, np.eye(n), np.eye(n))
        np.savez(output/f"pair_n{n}.npz", A1=A1, A2=A2, K1=K1, K2=K2, B=np.eye(n), C=C)
        records, initial = test_states(n, settings.test_pairs, settings.test_seed)
        states.extend(records)
        expert = HistoryPolicy(K1, 1, 1, n)
        reference = deploy(full1, K1, expert, 1, initial, settings.steps, settings.warmup)
        for r in range(1, n+1):
            O = partial.observability(r)
            ranks.append(dict(dimension=n, history=r, rank=int(np.linalg.matrix_rank(O)),
                              smallest_singular_value=float(np.linalg.svd(O, compute_uv=False)[-1])))
        references = [
            ("expert_K1", full1, K1, expert),
            ("expert_K2_on_A2", full2, K2, HistoryPolicy(K2,1,1,n)),
            ("wrong_K2", full1, K1, HistoryPolicy(K2,1,1,n)),
            ("wrong_K1_on_A2", full2, K2, HistoryPolicy(K1,1,1,n)),
            ("zero", full1, K1, HistoryPolicy(np.zeros((n,n)),1,1,n)),
            ("known_model_observer", partial, K1, exact_policy(partial,K1,n,1)),
        ]
        for name, plant, gain, policy in references:
            identity = dict(dimension=n, training="reference", seed="", context=policy.context,
                            horizon=1, execution=1, controller=name)
            values, raw, temporal, example = evaluate(plant, gain, policy, 1, initial, records,
                                                     reference, identity, settings, True)
            baselines.append(dict(**identity, **values, **stability(plant,policy,1,0)))
            episodes.extend(raw); profiles.extend(temporal); samples.extend(example)
        for training in ("narrow", "recovery"):
            for seed in settings.seeds:
                starts = np.zeros((settings.trajectories, n))
                starts[:,-1] = np.random.default_rng(seed).uniform(-1,1,settings.trajectories)
                data = collect(partial,K1,starts,settings.actual_training_steps,training,
                               seed+100000,settings.probe_scale)
                ids = window_ids(data,settings.max_context,settings.horizon,settings.actual_training_steps)
                target_map = np.vstack([K1 @ np.linalg.matrix_power(A1+K1,j) for j in range(settings.horizon)])
                targets = np.array([target_map @ data[i]["x"][t] for i,t in ids])
                for r in settings.histories(n):
                    features, _ = matched_arrays(data,ids,r,1)
                    one, chunk, info = fit_matched(features,targets,r,n,ridge=settings.ridge)
                    for policy, ell in [(one,1)] + [(chunk,e) for e in settings.executions]:
                        identity = dict(dimension=n,training=training,seed=seed,context=r,
                                        horizon=policy.horizon,execution=ell,controller="learned")
                        keep = wants_diagnostic(n,r,policy.horizon,ell,settings)
                        values, raw, temporal, example = evaluate(partial,K1,policy,ell,initial,
                            records,reference,identity,settings,keep)
                        row = dict(**identity,**values,**info,**stability(partial,policy,ell,0),
                                   first_action_training_mse=float(np.mean((features@one.weights.T-targets[:,:n])**2)),
                                   long_sample_mse="")
                        if keep:
                            far = initial[1:2]
                            long_x = deploy(partial,K1,policy,ell,far,settings.long_steps,settings.warmup)
                            long_ref = deploy(full1,K1,expert,1,far,settings.long_steps,settings.warmup)
                            row["long_sample_mse"] = float(squared_errors(long_x,long_ref).mean())
                        metrics.append(row)
                        episodes.extend(raw); profiles.extend(temporal); samples.extend(example)
            print(f"Completed n={n}, {training}; histories={settings.histories(n)}", flush=True)
    tables = dict(metrics=metrics,test_rollouts=episodes,baselines=baselines,pair_checks=checks,
                  test_states=states,temporal_profiles=profiles,sample_trajectories=samples,
                  observability_ranks=ranks)
    for name, rows in tables.items():
        write_records(output/f"{name}.csv", rows)
    config = asdict(settings)
    config.update(schema_version=2,common_history_lengths=[2,4,8,16],
                  training_labels={"narrow":"Noiseless training data","recovery":"Noisy training data"},
                  training_noise="Known Gaussian action perturbations with std .05; exact expert labels",
                  test_noise="None after the initial hidden-state departure",construction="single coupled delay feedback loop" if settings.family == "coupled" else "original Simchowitz 2D pair at mu=1/8",warmup=settings.warmup,
                  actual_training_steps=settings.actual_training_steps,
                  contexts_by_dimension={n:settings.histories(n) for n in settings.dimensions},
                  creation_utc=datetime.now(timezone.utc).isoformat(),
                  loss="time-mean squared full-state distance, equal-weight farthest and distributed cases",
                  uncertainty="range of independent training-seed means; not a confidence interval",
                  python=platform.python_version(),numpy=np.__version__)
    (output/"config.json").write_text(json.dumps(config,indent=2))
    source_files = [HERE/"high_dimensional.py",HERE/"__init__.py",HERE/"__main__.py",
                    *sorted((HERE/"coupled").glob("*.py")),
                    *sorted((HERE/"original").glob("*.py")),
                    *sorted((HERE/"shared").glob("*.py")),
                    ROOT/"model.py",ROOT/"observability/__init__.py",ROOT/"observability/experiment.py"]
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    (output/"source_manifest.json").write_text(json.dumps(manifest,indent=2))
    for source in source_files:
        destination = output/"source_snapshot"/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    print(f"Saved {len(metrics)} learned deployments, {len(episodes)} episode records",flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions",type=int,nargs="+",default=[2,4,8,16])
    parser.add_argument("--family",choices=["coupled","original"],default="coupled")
    parser.add_argument("--seeds",type=int,nargs="+",default=[1118,1119,1120,1121,1122])
    parser.add_argument("--executions",type=int,nargs="+",default=[1,2,4,8,16])
    parser.add_argument("--horizon",type=int,default=16)
    parser.add_argument("--test-seed",type=int,default=20261110)
    parser.add_argument("--test-pairs",type=int,default=40)
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--long-steps",type=int,default=1000)
    parser.add_argument("--trajectories",type=int,default=128)
    parser.add_argument("--training-steps",type=int,default=100)
    parser.add_argument("--ridge",type=float,default=1e-6)
    parser.add_argument("--probe-scale",type=float,default=.05)
    parser.add_argument("--output",type=Path,default=HERE/"results/coupled_higher_dimensions")
    parser.add_argument("--exploratory",action="store_true",help="Also regenerate the broader diagnostic plots")
    parser.add_argument("--original-reference",action="store_true",help="Also rerun the distinct original paper n=2 reference")
    parser.add_argument("--plot-only",action="store_true",help="Audit saved tables and rebuild plots; no fitting")
    parser.add_argument("--overwrite",action="store_true",help="Explicitly replace an existing run")
    args = parser.parse_args()
    keys = Settings.__dataclass_fields__
    values = {key:getattr(args,key) for key in keys}
    for key in ("dimensions","seeds","executions"):
        values[key] = tuple(sorted(values[key]))
    settings = Settings(**values)
    try:
        settings.validate()
    except ValueError as error:
        parser.error(str(error))
    if not args.plot_only and (args.output/"config.json").exists() and not args.overwrite:
        parser.error("Output already contains a run; use --plot-only, --output NEW_PATH, or --overwrite")
    return args, settings


def main():
    args, settings = parse_args()
    if not args.plot_only:
        run(settings,args.output)
    original_output = args.output/"original_n2"
    saved_config = json.loads((args.output/"config.json").read_text())
    include_original = (args.original_reference and saved_config.get("family","coupled") == "coupled"
                        and 2 in saved_config["dimensions"])
    if include_original and not args.plot_only:
        run(replace(settings,dimensions=(2,),family="original"),original_output)
    if args.exploratory:
        from pitfalls.coupled.diagnostics import analyze, compare_original_reference
        analyze(args.output)
        if include_original and (original_output/"config.json").exists():
            analyze(original_output)
            compare_original_reference(args.output,original_output)
        shutil.copy2(args.output/"README.md",args.output/"EXPLORATORY_RESULTS.md")
    from pitfalls.coupled.plots import analyze as focused_overview
    focused_overview(args.output)
