"""A focused view: change x1 once; compare history and executed action count.

Uses only the existing farthest_hidden cases. No new dynamics, training, or
mixture of disturbance types. Broader diagnostics remain optional.
"""
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pitfalls.shared.audit import read_table, audit
from pitfalls.shared.plotting import save_figure, training_label
from pitfalls.shared.results import write_records


def analyze(output):
    output=Path(output)
    config=json.loads((output/"config.json").read_text())
    metrics=read_table(output/"metrics.csv")
    episodes=read_table(output/"test_rollouts.csv")
    references=read_table(output/"baselines.csv")
    states=read_table(output/"test_states.csv")
    profiles=read_table(output/"temporal_profiles.csv")
    checks=audit(config,metrics,episodes,references,states,profiles)
    identity=lambda x:tuple(x[k] for k in ("dimension","training","seed","context","horizon","execution","controller"))
    losses=defaultdict(list)
    for row in episodes:
        if row["suite"]=="farthest_hidden":losses[identity(row)].append(row["test_mse"])
    seed_rows=[]
    for row in metrics:
        values=losses[identity(row)]
        if len(values)!=config["test_pairs"]:raise ValueError("Incomplete x1-only test set")
        seed_rows.append(dict(dimension=row["dimension"],training=row["training"],seed=row["seed"],
            context=row["context"],horizon=row["horizon"],execution=row["execution"],
            test_mse=float(np.mean(values)),stable=row["stable"],rate=row["per_step_radius"]))
    groups=defaultdict(list)
    for row in seed_rows:
        groups[tuple(row[k] for k in ("dimension","training","context","horizon","execution"))].append(row)
    summary=[];index={}
    for key,rows in sorted(groups.items()):
        values=[r["test_mse"] for r in rows]
        cell=dict(zip(("dimension","training","context","horizon","execution"),key),
            mean_test_mse=float(np.mean(values)),min_seed_mse=min(values),max_seed_mse=max(values),
            stable_seeds=sum(r["stable"] for r in rows),seeds=len(rows))
        summary.append(cell);index[key]=cell
    write_records(output/"focused_seed_metrics.csv",seed_rows)
    write_records(output/"focused_summary.csv",summary)
    colors={"narrow":"#2878b5","recovery":"#d67a17"}
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none"})
    dimensions=config["dimensions"]
    columns=2;rows=(len(dimensions)+1)//2
    fig,axes=plt.subplots(rows,columns,figsize=(12,4*rows),squeeze=False,layout="constrained")
    for ax,n in zip(axes.flat,dimensions):
        rs=[r for r in config["contexts_by_dimension"][str(n)] if r<=n+1]
        for training in ("narrow","recovery"):
            cells=[index[n,training,r,1,1] for r in rs]
            means=[c["mean_test_mse"] for c in cells]
            ax.plot(rs,means,"o-",color=colors[training],label=training_label(training))
            for r,c in zip(rs,cells):
                if c["stable_seeds"]<c["seeds"]:ax.scatter(r,c["mean_test_mse"],marker="x",color="red",s=55,zorder=5)
        ax.axvline(n,color="black",ls=":",label=f"{n} observations identify the state")
        ax.set(title=f"{n} states; only x{n} is observed",xlabel="Number of observations available",ylabel="Mean squared tracking error",yscale="log",
               xticks=rs if n<=8 else [1,4,8,12,16,17])
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(dimensions):]:ax.set_visible(False)
    fig.suptitle("1. Does more history help when we choose one action at a time?\nNoiseless testing; change only x1 once; mean across five training seeds; red ×: at least one unstable fit")
    save_figure(fig,output/"figures","overview_history")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    palette=["#8c4d9e","#2878b5","#d67a17","#268a64"]
    for ax,training in zip(axes,("narrow","recovery")):
        for n,color in zip(dimensions,palette):
            cells=[index[n,training,n,config["horizon"],e] for e in config["executions"]]
            ax.plot(config["executions"],[c["mean_test_mse"] for c in cells],"o-",color=color,label=f"{n} states, {n} observations")
            for e,c in zip(config["executions"],cells):
                if c["stable_seeds"]<c["seeds"]:ax.scatter(e,c["mean_test_mse"],marker="x",color="red",s=55,zorder=5)
        ax.set(title=training_label(training),xlabel="Actions executed before checking again",ylabel="Mean squared tracking error",yscale="log",xticks=config["executions"])
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("2. With enough history, does executing more actions help?\nOne change in x1; the same fitted predictor is reused across execution lengths")
    save_figure(fig,output/"figures","overview_execution")

    lines=["# Start here: the focused higher-dimensional experiment","",
        "## The question","",
        "Can remembering observations let the learner choose one action at a time instead of executing an action sequence before checking again?","",
        "## The setup","",
        "Use one coupled system at each dimension: 2, 4, 8, and 16. The learner sees only the last state coordinate. "
        "At deployment change x1 once, leaving all other hidden coordinates zero. The change takes n-1 steps to reach the sensor. "
        "The expert sees every state. K1 and K2 agree on demonstrations, but the wrong expert destabilizes the other system.","",
        "Noiseless training follows the expert without action changes. Noisy training adds known action perturbations and supplies exact expert actions at the visited states. "
        "All testing is noiseless after the initial change. Noise here describes training-data collection, not noisy expert answers or continuing test noise.","",
        f"Use {config['test_pairs']} held-out starting conditions and {len(config['seeds'])} training seeds. "
        f"Each learner runs for {config['steps']} steps. Score the mean squared distance from an expert rollout. "
        "An unstable fit stays in the mean; red crosses identify instability. This focused view uses only the saved x1-change cases, not a mixture of disturbance types.","",
        "## Two plots","",
        "1. [History versus one-action control](figures/overview_history.png): vary the number of observations and choose one action every step. The dotted line marks enough information to reconstruct the state.",
        "2. [Execution versus error at sufficient history](figures/overview_execution.png): give each n-state system n observations, then vary how many actions it executes before checking again.","",
        "Both plots have SVG versions for export. Lines show mean error across training seeds; there is no shading.","",
        "## Results with enough history","",
        "| States | Observations | Noiseless training, one action | Noisy training, one action | Noisy training, longest execution | Stable noisy one-action fits |",
        "|---:|---:|---:|---:|---:|---|"]
    for n in dimensions:
        a=index[n,"narrow",n,1,1];b=index[n,"recovery",n,1,1]
        c=index[n,"recovery",n,config["horizon"],max(config["executions"])]
        lines.append(f"| {n} | {n} | {a['mean_test_mse']:.4g} | {b['mean_test_mse']:.4g} | {c['mean_test_mse']:.4g} | {b['stable_seeds']}/{b['seeds']} |")
    lines += ["","## What to conclude","",
        "With noisy training and n observations, one-action control is stable for all five seeds at every dimension. Executing sixteen actions gives higher tracking error in all four systems. "
        "Before enough history is available, noisy-data fits can be unstable. Noiseless-data fits can become stable without learning the expert correction. "
        "Read the first plot for the effect of history and the second for the effect of action execution. "
        "Stable control does not by itself mean the learner reconstructed the state or learned the expert correction. "
        "Noiseless demonstrations omit corrections; noisy collection supplies them. This is one constructed family, not a universal claim about history or action chunking.","",
        "## Reproduce and inspect","", "```sh","python3 -m pitfalls --plot-only", "```","",
        "`focused_summary.csv` and `focused_seed_metrics.csv` contain exactly the x1-only scores shown here. "
        "The underlying saved experiment contains extra exploratory cases; they are excluded from these two plots. "
        "Original Simchowitz n=2 results remain separately available in [original_n2/](original_n2/README.md). "
        "They are a reference, not another point mixed into this coupled-family curve. "
        "The broader [exploratory results](EXPLORATORY_RESULTS.md) retain the additional plots and disturbance types; they are not needed for this comparison."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    checks.update(focused_test_suite="x1_only",focused_cases_per_seed=config["test_pairs"],
                  focused_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/"focused_audit.json").write_text(json.dumps(checks,indent=2))
    print(f"Focused overview: {len(dimensions)} dimensions; only x1 perturbations; two main plots",flush=True)
