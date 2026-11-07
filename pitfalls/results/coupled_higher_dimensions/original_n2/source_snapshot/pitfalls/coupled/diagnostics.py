"""Audit saved coupled experiments and create reproducible scientific figures.

Means include all seeds and unstable runs. Shading is a seed range, never a
confidence interval. Fixed-total departures use total MSE across dimensions.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pitfalls.shared.results import write_records
from pitfalls.shared.audit import read_table, identity, audit, summarize, key_index, paired_chunking
from pitfalls.coupled.experiment import IDENTITY
from pitfalls.shared.plotting import save_figure, training_label

DIM_COLORS = {2:"#8c4d9e", 4:"#2878b5", 8:"#d67a17", 16:"#268a64"}
TRAINING = ("narrow", "recovery")


ROLES = ("No history", "One below h", "At h", "One above h")


def log_values(values):
    # Zero is omitted on a logarithmic axis, never replaced in the raw data.
    return np.array([x if np.isfinite(x) and x > 0 else np.nan for x in values])


def line_with_range(ax, x, cells, label, color, mark_instability=True):
    mean = log_values([c["mean_test_mse"] for c in cells])
    ax.plot(x,mean,"o-",color=color,label=label,markersize=4)
    ax.fill_between(x,log_values([c["min_seed_mse"] for c in cells]),
                    log_values([c["max_seed_mse"] for c in cells]),color=color,alpha=.15)
    if mark_instability:
        bad = [i for i,c in enumerate(cells) if c["stable_seeds"] < c["seeds"]]
        ax.scatter(np.asarray(x)[bad],mean[bad],marker="x",c="red",s=45,zorder=5)
    finite = [c["mean_test_mse"] for c in cells if np.isfinite(c["mean_test_mse"]) and c["mean_test_mse"]>0]
    top, bottom = (max(finite)*10, min(finite)/10) if finite else (1.,.1)
    for position,cell in zip(x,cells):
        value=cell["mean_test_mse"]
        if not np.isfinite(value) or value==0:
            proxy=top if not np.isfinite(value) else bottom
            ax.scatter(position,proxy,marker="^" if not np.isfinite(value) else "v",color=color)
            ax.annotate("∞" if not np.isfinite(value) else "0",(position,proxy),xytext=(3,3),textcoords="offset points",fontsize=8)
    ax.set_yscale("log")
    ax.grid(alpha=.2)


def per_dimension(summary, config, output):
    index = key_index(summary)
    H = config["horizon"]; executions = config["executions"]
    colors = plt.get_cmap("viridis")(np.linspace(.05,.9,len(executions)))
    for n in config["dimensions"]:
        folder = output/"figures"/f"n{n}"
        histories = config["contexts_by_dimension"][str(n)]
        fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
        for row,training in enumerate(TRAINING):
            for ell,color in zip(executions,colors):
                cells = [index[n,training,r,H,ell] for r in histories]
                line_with_range(axes[row,0],histories,cells,f"execution {ell}",color)
                axes[row,1].plot(histories,[c["worst_rate"] for c in cells],"o-",color=color,label=f"execution {ell}",markersize=4)
            axes[row,0].set(title=f"{training_label(training)}: tracking error",ylabel="Mean total test MSE")
            axes[row,1].axhline(1,color="red",ls="--",label="stability boundary")
            axes[row,1].set(title=f"{training_label(training)}: stability",ylabel="Worst-seed per-step spectral rate")
            for ax in axes[row]:
                ax.axvline(n,color="black",ls=":",lw=1)
                ax.set(xlabel="Observed history r",xticks=histories)
                ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
        fig.suptitle(f"n={n}: history, execution, and stability (H={H})\nDotted line: full-state reconstruction possible; shading: seed range; red ×: any unstable seed")
        save_figure(fig,folder,"history_and_stability")

        fig,axes = plt.subplots(2,2,figsize=(11,max(7,.48*len(histories)+2)),layout="constrained")
        for row,training in enumerate(TRAINING):
            global_losses = [x["mean_test_mse"] for x in summary if x["training"]==training and x["horizon"]==H and np.isfinite(x["mean_test_mse"]) and x["mean_test_mse"]>0]
            grid = np.array([[index[n,training,r,H,e]["mean_test_mse"] for e in executions] for r in histories])
            stable = np.array([[index[n,training,r,H,e]["stable_seeds"] for e in executions] for r in histories])
            norm=LogNorm(min(global_losses),max(global_losses))
            image=axes[row,0].imshow(np.ma.masked_where(~np.isfinite(grid)|(grid<=0),grid),norm=norm,cmap="viridis",aspect="auto")
            fig.colorbar(image,ax=axes[row,0],label="Mean total MSE (shared scale across n)")
            image2=axes[row,1].imshow(stable,vmin=0,vmax=len(config["seeds"]),cmap="RdYlGn",aspect="auto")
            fig.colorbar(image2,ax=axes[row,1],label="Number of stable training seeds")
            for i,r in enumerate(histories):
                for j,e in enumerate(executions):
                    value=grid[i,j]
                    axes[row,0].text(j,i,f"{value:.1e}" if np.isfinite(value) else "∞",ha="center",va="center",fontsize=7,color="white" if np.isfinite(value) and norm(value)<.6 else "black")
                    axes[row,1].text(j,i,f"{stable[i,j]}/{len(config['seeds'])}",ha="center",va="center",fontsize=8)
            for col,ax in enumerate(axes[row]):
                ax.set(title=f"{training_label(training)}: {'loss' if col==0 else 'stability'}",xlabel="Execution horizon",ylabel="History r",
                       xticks=range(len(executions)),xticklabels=executions,yticks=range(len(histories)),yticklabels=histories)
                ax.axhline(histories.index(n)-.5,color="black",ls=":")
        fig.suptitle(f"n={n}: complete tested grid, H={H}\nBlack boundary marks history n; finite loss and asymptotic stability are separate")
        save_figure(fig,folder,"grid")


def cross_history(summary,config,output):
    index=key_index(summary)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for row,training in enumerate(TRAINING):
        for n in config["dimensions"]:
            rs=config["contexts_by_dimension"][str(n)]
            cells=[index[n,training,r,1,1] for r in rs]
            for col,x in enumerate([rs,np.array(rs)/n]):
                line_with_range(axes[row,col],x,cells,f"n={n}",DIM_COLORS[n])
        for col,ax in enumerate(axes[row]):
            ax.set(title=f"{training_label(training)}: single-action learner",ylabel="Mean total MSE",
                   xlabel="Absolute history r" if col==0 else "History / observability index (r/h)")
            if col==1:
                ax.axvline(1,color="black",ls=":",label="enough information")
                ax.set_xlim(0,1.3)
            ax.legend(fontsize=9)
    fig.suptitle("How much history is needed across dimensions?\nH=execution=1; right panels zoom near h; shading: seed range; red ×: any unstable seed")
    save_figure(fig,output/"figures/cross_dimension","one_step_history")


def observable_comparison(summary,baselines,config,output):
    index=key_index(summary)
    ref={(x["dimension"],x["controller"]):x for x in baselines}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    comparisons=[("r=h, one action",lambda n:(n,1,1),"#2878b5"),
                 ("r=h, longest execution",lambda n:(n,config["horizon"],max(config["executions"])),"#d67a17"),
                 ("r=16, one action",lambda n:(16,1,1),"#8c4d9e")]
    for row,training in enumerate(TRAINING):
        for label,choose,color in comparisons:
            ns=[];cells=[]
            for n in config["dimensions"]:
                r,H,e=choose(n)
                if (n,training,r,H,e) in index:
                    ns.append(n);cells.append(index[n,training,r,H,e])
            if not ns:
                continue
            line_with_range(axes[row,0],ns,cells,label,color)
            normalized=[]
            for n,cell in zip(ns,cells):
                denominator=ref[n,"known_model_observer"]["test_mse"]
                normalized.append({**cell,**{k:cell[k]/denominator for k in ("mean_test_mse","min_seed_mse","max_seed_mse")}})
            line_with_range(axes[row,1],ns,normalized,label,color)
        ns=config["dimensions"]
        axes[row,0].plot(ns,[ref[n,"known_model_observer"]["test_mse"] for n in ns],"k--",label="known-model observer")
        axes[row,1].axhline(1,color="black",ls="--",label="observer reference")
        for col,ax in enumerate(axes[row]):
            ax.set(title=f"{training_label(training)} data",xlabel="State dimension n",xticks=ns,
                   ylabel="Mean total test MSE" if col==0 else "MSE / known-model observer MSE")
            ax.legend(fontsize=8)
    fig.suptitle("Cross-dimension comparison with enough history\nFixed total perturbation norm; observer reference accounts for the growing initial information delay")
    save_figure(fig,output/"figures/cross_dimension","observable_context")


def chunking_comparison(pairs,config,output):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    role_colors=plt.get_cmap("tab10")(range(4))
    for ax,training in zip(axes,TRAINING):
        for role,color in zip(ROLES,role_colors):
            ns=[];means=[];low=[];high=[];bad=[]
            for n in config["dimensions"]:
                r={ROLES[0]:1,ROLES[1]:n-1,ROLES[2]:n,ROLES[3]:n+1}[role]
                rows=[x for x in pairs if x["dimension"]==n and x["training"]==training and x["context"]==r and x["execution"]==max(config["executions"])]
                values=[x["chunk_to_feedback_ratio"] for x in rows]
                ns.append(n);means.append(np.mean(values));low.append(min(values));high.append(max(values))
                bad.append(any(not x["chunk_stable"] or not x["feedback_stable"] for x in rows))
            ax.plot(ns,log_values(means),"o-",color=color,label=role)
            ax.fill_between(ns,log_values(low),log_values(high),color=color,alpha=.15)
            for n,mean,unstable in zip(ns,means,bad):
                if unstable:ax.scatter(n,mean,marker="x",color="red",s=55,zorder=5)
        ax.axhline(1,color="black",ls="--")
        ax.set(yscale="log",title=f"{training_label(training)} data",xlabel="State dimension n",xticks=config["dimensions"],ylabel="Paired MSE ratio: chunk / feedback")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f"Does executing {max(config['executions'])} actions help?\nMean and range of within-seed ratios; below 1 favors chunking; red ×: either compared controller unstable")
    save_figure(fig,output/"figures/cross_dimension","chunking_effect")


def perturbation_comparison(summary,config,output):
    index=key_index(summary)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout="constrained")
    for row,training in enumerate(TRAINING):
        for col,suite in enumerate(("farthest_hidden","distributed_hidden")):
            for e,color in [(1,"#2878b5"),(max(config["executions"]),"#d67a17")]:
                ns=config["dimensions"]
                cells=[index[n,training,n,config["horizon"],e] for n in ns]
                axes[row,col].plot(ns,[x[suite+"_mse"] for x in cells],"o-",color=color,label=f"execution {e}")
            axes[row,col].set(yscale="log",title=f"{training_label(training)}: {suite.replace('_',' ')}",xlabel="State dimension n",xticks=config["dimensions"],ylabel="Mean total MSE at history h=n")
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=9)
    fig.suptitle("Separate perturbation types, rather than pooling away their differences\nBoth have the same total initial hidden-state norm")
    save_figure(fig,output/"figures/cross_dimension","perturbation_families")


def pair_diagnostics(checks,ranks,samples,config,output):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    for n in config["dimensions"]:
        rows=[x for x in ranks if x["dimension"]==n]
        axes[0].plot([x["history"] for x in rows],[x["rank"] for x in rows],"o-",color=DIM_COLORS[n],label=f"n={n}")
    axes[0].set(title="One sensor: history must grow with n",xlabel="Number of observations r",ylabel="Observability rank")
    for field,label,style in [("A1_radius","plant A1","o-"),("A2_radius","plant A2","s-"),
                              ("expert_K1_radius","correct expert","^-"),("wrong_K2_radius","wrong expert","x-")]:
        axes[1].plot([x["dimension"] for x in checks],[x[field] for x in checks],style,label=label)
    axes[1].axhline(1,color="red",ls="--")
    axes[1].set(title="Both plants stable; wrong experts unstable",xlabel="State dimension n",ylabel="Spectral radius",xticks=config["dimensions"])
    n=max(config["dimensions"])
    for controller,label in [("expert_K1","K1 on A1"),("wrong_K2","K2 on A1"),("wrong_K1_on_A2","K1 on A2")]:
        rows=[x for x in samples if x["dimension"]==n and x["controller"]==controller and x["step"]<=40]
        axes[2].plot([x["step"] for x in rows],[x["x1"] for x in rows],label=label)
    axes[2].set(yscale="symlog",title=f"Hidden departure, n={n} (one example)",xlabel="Step after deployment",ylabel="Signed hidden state x1")
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig,output/"figures","pair_diagnostics")


def rollout_figures(profiles,config,output):
    grouped=defaultdict(list)
    for row in profiles:
        grouped[(row["dimension"],row["training"],row["context"],row["horizon"],row["execution"],row["controller"],row["suite"],row["step"])].append(row["mean_squared_error"])
    for n in config["dimensions"]:
        fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout="constrained")
        selections=[(1,1,1,"no history, one action","#7f7f7f"),
                    (n-1,1,1,"history h-1, one action","#d67a17"),
                    (n,1,1,"history h, one action","#2878b5"),
                    (n,config["horizon"],max(config["executions"]),"history h, longest execution","#8c4d9e")]
        for row,training in enumerate(TRAINING):
            for col,suite in enumerate(("farthest_hidden","distributed_hidden")):
                ax=axes[row,col]
                for r,H,e,label,color in selections:
                    time=list(range(1,min(config["steps"],max(40,3*n))+1))
                    values=[grouped[n,training,r,H,e,"learned",suite,t] for t in time]
                    ax.plot(time,[np.mean(v) for v in values],color=color,label=label)
                    ax.fill_between(time,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.12)
                time=list(range(1,min(config["steps"],max(40,3*n))+1))
                observer=[grouped[n,"reference",n,1,1,"known_model_observer",suite,t][0] for t in time]
                ax.plot(time,observer,"k--",label="known-model observer")
                ax.axvline(n-1,color="black",ls=":",lw=1)
                ax.set_yscale("symlog",linthresh=1e-12)
                ax.set_ylim(bottom=0)
                ax.set(title=f"{training_label(training)}: {suite.replace('_',' ')}",xlabel="Step after hidden departure",ylabel="Mean instantaneous squared tracking error")
                ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f"n={n}: when can the learner correct the departure?\nTest-case means; seed-range shading; dotted line: first x1 effect reaches sensor; linear scale below 1e-12")
        save_figure(fig,output/"figures"/f"n{n}","rollouts")


def horizon_comparison(metrics, samples, episodes, config, output):
    """Compare 200/1000-step means on the SAME canonical initial state."""
    short_groups = defaultdict(list)
    for row in samples:
        if row["controller"] == "learned" and row["step"] > 0:
            short_groups[identity(row)].append(row["squared_tracking_error"])
    raw = {identity(row):row["test_mse"] for row in episodes
           if row["controller"] == "learned" and row["suite"] == "farthest_hidden" and row["test_id"] == 0}
    records=[]
    for row in metrics:
        if row["long_sample_mse"] is None:
            continue
        values=short_groups[identity(row)]
        if len(values)!=config["steps"]:
            raise ValueError("Missing canonical sample steps")
        short=float(np.mean(values))
        if not np.isclose(short,raw[identity(row)],rtol=1e-11,atol=1e-20):
            raise ValueError("Canonical sample does not match its held-out episode")
        records.append(dict(**{key:row[key] for key in IDENTITY},short_canonical_mse=short,
                            long_canonical_mse=row["long_sample_mse"],stable=row["stable"]))
    write_records(output/"paired_test_horizons.csv",records)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    selections=[("No history, one action",lambda n:(1,1,1),"#7f7f7f"),
                ("History h-1, one action",lambda n:(n-1,1,1),"#d67a17"),
                ("History h, one action",lambda n:(n,1,1),"#2878b5"),
                ("History h, longest execution",lambda n:(n,config["horizon"],max(config["executions"])),"#8c4d9e")]
    for row,training in enumerate(TRAINING):
        for label,choose,color in selections:
            for col,field in enumerate(("short_canonical_mse","long_canonical_mse")):
                cells=[]
                for n in config["dimensions"]:
                    r,H,e=choose(n)
                    group=[x for x in records if (x["dimension"],x["training"],x["context"],x["horizon"],x["execution"])==(n,training,r,H,e)]
                    values=[x[field] for x in group]
                    cells.append(dict(mean_test_mse=float(np.mean(values)),min_seed_mse=min(values),max_seed_mse=max(values),
                                      stable_seeds=sum(x["stable"] for x in group),seeds=len(group)))
                line_with_range(axes[row,col],config["dimensions"],cells,label,color)
        for col,ax in enumerate(axes[row]):
            horizon=config["steps"] if col==0 else config["long_steps"]
            ax.set(title=f"{training_label(training)}: {horizon}-step mean",xlabel="State dimension n",xticks=config["dimensions"],
                   ylabel="MSE on the SAME single farthest-hidden test state")
            ax.legend(fontsize=7)
    fig.suptitle("Long-run check on matched initial conditions\nShading: seed range; red ×: any unstable seed; longer averaging can dilute stable transient errors")
    save_figure(fig,output/"figures/cross_dimension","test_horizon_check")
    return len(records)


def write_report(summary,baselines,checks,audit_result,pairs,config,output):
    index=key_index(summary)
    reference={(x["dimension"],x["controller"]):x for x in baselines}
    lines=["# Coupled hard-example results", "",
        "Each system has one coupled n-state feedback loop and ONE scalar sensor. This is not a collection of independent 2D copies. "
        "K1 and K2 produce identical demonstrations, but each destabilizes the other plant. The history required for exact action recovery is n.","",
        "## Start with the plots", "",
        "- [Construction, observation ranks, and wrong-expert failure](figures/pair_diagnostics.png)",
        "- [One-step error versus absolute and relative history](figures/cross_dimension/one_step_history.png)",
        "- [Cross-dimension error with enough history, including observer ratios](figures/cross_dimension/observable_context.png)",
        "- [Paired effect of longer execution](figures/cross_dimension/chunking_effect.png)",
        "- [Farthest versus distributed hidden departures](figures/cross_dimension/perturbation_families.png)",
        "- [Matched short/long test-horizon check](figures/cross_dimension/test_horizon_check.png)","",
        "Every figure also has an SVG for clean export. PNGs are previews. Shading is the range of independent training-seed means, not a confidence interval. "
        "If zero or infinite means occur, labeled boundary markers replace unplottable logarithmic points; raw scores are never floored or clipped. "
        "Red crosses flag comparisons involving unstable controllers; a smaller finite-rollout MSE is not a stability guarantee.","",
        "## Per-dimension results", "",
        "| Dimension | History / stability | Complete execution grid | Time-resolved test error |",
        "|---:|---|---|---|"]
    for n in config["dimensions"]:
        lines.append(f"| {n} | [profiles](figures/n{n}/history_and_stability.png) | [heatmaps](figures/n{n}/grid.png) | [rollouts](figures/n{n}/rollouts.png) |")
    lines += ["","## What was run","",
        f"{len(config['seeds'])} independent training seeds ({config['seeds']}); dimensions {config['dimensions']}; "
        f"{config['trajectories']} training trajectories of {config['actual_training_steps']} steps per seed, training regime, and dimension. "
        "Noiseless training trajectories stay on the demonstrated line. Noisy training data use known Gaussian action perturbations and exact expert continuation labels. "
        "Both retain the actual applied actions as history inputs.","",
        f"Histories cover every length from 1 through n+1, plus common absolute lengths {config.get('common_history_lengths',config['dimensions'])}. "
        f"Execution horizons are {config['executions']}, prediction horizons are 1 and {config['horizon']}. "
        "At fixed history the same fitted multi-action predictor is reused for every execution length. "
        "Single-action H=1 and multi-action H>1 executed one step at a time are paired training controls.","",
        f"Each policy is tested on {config['test_pairs']} shared visible amplitudes, each with an on-support, farthest-hidden, and distributed-hidden initial state. "
        f"The {2*config['test_pairs']} perturbed cases have fixed TOTAL hidden Euclidean norm in [.0005,.0015]. "
        f"Scores average full-state squared distance over {config['steps']} post-action steps and perturbed cases, then training seeds. "
        "Total MSE is primary; per-coordinate MSE divides the same error by n and is not the primary scaling comparison.","",
        f"Expert history initialization lasts {config['warmup']} steps before a hidden jump is injected. "
        "There are no continuing disturbances, observation noise, or deployed expert queries. Training target times and test perturbation amplitudes are matched across dimensions. "
        "Farthest-hidden departures act on x1; distributed departures allocate the same total norm over all hidden coordinates.","",
        "## One action at the reconstruction threshold", "",
        "| n | Data | History | Mean total MSE | Seed range | Stable seeds | Observer MSE | Ratio to observer |",
        "|---:|---|---:|---:|---|---|---:|---:|"]
    for n in config["dimensions"]:
        for training in TRAINING:
            cell=index[n,training,n,1,1];observer=reference[n,"known_model_observer"]["test_mse"]
            lines.append(f"| {n} | {training_label(training)} | {n} | {cell['mean_test_mse']:.4g} | {cell['min_seed_mse']:.4g}–{cell['max_seed_mse']:.4g} | {cell['stable_seeds']}/{cell['seeds']} | {observer:.4g} | {cell['mean_test_mse']/observer:.3g} |")
    lines += ["","## One observation below the threshold", "",
        "| n | Data | History | One-step MSE | Stable seeds | Longest-execution MSE | Stable seeds |",
        "|---:|---|---:|---:|---|---:|---|"]
    for n in config["dimensions"]:
        for training in TRAINING:
            a=index[n,training,n-1,config["horizon"],1]
            b=index[n,training,n-1,config["horizon"],max(config["executions"])]
            lines.append(f"| {n} | {training_label(training)} | {n-1} | {a['mean_test_mse']:.4g} | {a['stable_seeds']}/{a['seeds']} | {b['mean_test_mse']:.4g} | {b['stable_seeds']}/{b['seeds']} |")
    recovery_at = [index[n,"recovery",n,1,1] for n in config["dimensions"]]
    recovery_below = [index[n,"recovery",n-1,1,1] for n in config["dimensions"]]
    lines += ["", "## Main observations in this run", "",
        f"At history h=n, one-step controllers trained on noisy data are stable in {sum(x['stable_seeds'] for x in recovery_at)}/{sum(x['seeds'] for x in recovery_at)} seed/dimension cases. "
        f"At h-1, that count is {sum(x['stable_seeds'] for x in recovery_below)}/{sum(x['seeds'] for x in recovery_below)}. "
        "Their exact loss and seed ranges are listed above; the denser history plot shows the full transition, including non-monotone failures below h.", ""]
    for n in config["dimensions"]:
        cell=index[n,"recovery",n,1,1]
        observer=reference[n,"known_model_observer"]["test_mse"]
        long_cell=index[n,"recovery",n,config["horizon"],max(config["executions"])]
        lines.append(f"- n={n}: history-h one-step MSE is {cell['mean_test_mse']:.4g}, "
                     f"{cell['mean_test_mse']/observer:.4g} times the observer reference. "
                     f"Executing {max(config['executions'])} actions gives {long_cell['mean_test_mse']:.4g} "
                     f"({long_cell['mean_test_mse']/cell['mean_test_mse']:.4g} times the one-step seed-mean loss).")
    lines += ["","## Analysis and limits","",
        "The comparison at r=h separates representability from identification: enough history permits exact expert recovery from known equations, "
        "but noiseless training does not identify the missing corrections. Compare the two training-data regimes at identical history and execution, not different selected optima.","",
        "The observer provides a reference for the initial information delay. Its tracking loss can grow with n even though its eventual spectral rate remains .75. "
        "Ratios to this reference help distinguish that unavoidable observation delay from additional learned-controller error. The observer is not claimed optimal.","",
        "Longer execution may stabilize an otherwise bad feedback rule, or may postpone a useful correction. The paired-ratio plot fixes the fitted predictor "
        "and shows the mean and range of within-seed ratios; it is NOT the ratio of two separately averaged losses. "
        "Consult stability counts before interpreting a favorable finite-horizon ratio. All unstable seeds remain in the means.","",
        "More raw context is not guaranteed to improve loss or stability. Coupled-family reconstruction O_n has condition number one, but raw history features "
        "and ridge-selected feedback can still be ill-conditioned or underidentified. The original pair has a different reconstruction condition, recorded in pair_checks.csv. Training data counts are fixed across n. Changing n changes "
        "information delay, unexcited directions, and policy parameter count simultaneously; this is not dimension-only scaling. "
        "Noiseless training target amplitudes differ from test amplitudes because targets start after a common history window.","",
        "## Audit", ""]
    for key,value in audit_result.items():lines.append(f"- {key}: {value}")
    lines += ["","Each metric row is a deployment, not an independent fit. The first-action control is deliberately redundant. "
        "Raw test episodes, seed-level losses, per-step diagnostic profiles, matrices, full settings, exact Schur checks, and source hashes are retained. CSV codes narrow/recovery remain as stable internal identifiers for noiseless/noisy training respectively.","",
        "## Files and reproduction", "",
        "`metrics.csv`: learned seed-level deployments. `summary.csv`: all grid cells. `paired_chunking.csv`: within-seed effects. "
        "`test_rollouts.csv`: all individual test cases. `temporal_profiles.csv`: selected test-case mean time profiles. "
        "`sample_trajectories.csv`: one matched farthest-direction example per diagnostic deployment. `paired_test_horizons.csv` compares short and long means on that same initial condition. "
        "`baselines.csv`, `pair_checks.csv`, `observability_ranks.csv`, `test_states.csv`, `audit.json`, and `source_manifest.json` document references and checks. `source_snapshot/` preserves the exact original run code; `analysis_manifest.json` records the current figure code and input hashes.","",
        "```sh", "# Audit and regenerate figures from these saved results:",
        "python3 -m pitfalls --plot-only --exploratory", "# Run into a separate directory:",
        "python3 pitfalls/high_dimensional.py --output /tmp/coupled-new-run", "```", "",
        "Use `--overwrite` only to intentionally replace a saved run. The mathematical construction is explained in `../../docs/coupled-hard-example.md`. "
        "The previous three-seed snapshot is preserved separately; do not pool it with these fresh-seed results."]
    if config.get("family") == "original":
        lines[0] = "# Original Simchowitz n=2: matched reference run"
        lines[2] = "These are the paper's original 2D matrices at mu=1/8, not the new coupled-family matrices. The same training and evaluation protocol is used for comparison."
    else:
        lines += ["", "For n=2, no history and one observation below h are the same setting; the two perturbation families also occupy the same hidden axis. They are not independent spatial directions."]
    (output/"README.md").write_text("\n".join(lines)+"\n")


def analyze(output):
    output=Path(output)
    config=json.loads((output/"config.json").read_text())
    metrics=read_table(output/"metrics.csv")
    episodes=read_table(output/"test_rollouts.csv")
    baselines=read_table(output/"baselines.csv")
    states=read_table(output/"test_states.csv")
    profiles=read_table(output/"temporal_profiles.csv")
    samples=read_table(output/"sample_trajectories.csv")
    checks=read_table(output/"pair_checks.csv")
    ranks=read_table(output/"observability_ranks.csv")
    audit_result=audit(config,metrics,episodes,baselines,states,profiles)
    summary=summarize(metrics)
    pairs=paired_chunking(metrics,config)
    write_records(output/"summary.csv",summary)
    if pairs:write_records(output/"paired_chunking.csv",pairs)
    (output/"audit.json").write_text(json.dumps(audit_result,indent=2))
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    per_dimension(summary,config,output)
    cross_history(summary,config,output)
    observable_comparison(summary,baselines,config,output)
    if pairs:chunking_comparison(pairs,config,output)
    perturbation_comparison(summary,config,output)
    pair_diagnostics(checks,ranks,samples,config,output)
    rollout_figures(profiles,config,output)
    audit_result["matched_horizon_checks"] = horizon_comparison(metrics,samples,episodes,config,output)
    (output/"audit.json").write_text(json.dumps(audit_result,indent=2))
    write_report(summary,baselines,checks,audit_result,pairs,config,output)
    analysis_manifest = dict(analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in
                      ("config.json","metrics.csv","test_rollouts.csv","baselines.csv","test_states.csv",
                       "temporal_profiles.csv","sample_trajectories.csv","pair_checks.csv","observability_ranks.csv")})
    (output/"analysis_manifest.json").write_text(json.dumps(analysis_manifest,indent=2))
    print(f"Audited {len(metrics)} learned deployments; figures and report saved to {output}",flush=True)


def compare_original_reference(output, original_output):
    """Compare actual paper n=2 with coupled-family n=2; never pool their data."""
    output, original_output = Path(output), Path(original_output)
    configs = [json.loads((path/"config.json").read_text()) for path in (output,original_output)]
    for field in ("seeds","executions","horizon","test_seed","test_pairs","steps","long_steps",
                  "trajectories","actual_training_steps","ridge","probe_scale","warmup"):
        if configs[0][field] != configs[1][field]:
            raise ValueError(f"Original/coupled reference protocols differ: {field}")
    if configs[0]["contexts_by_dimension"]["2"] != configs[1]["contexts_by_dimension"]["2"]:
        raise ValueError("Original/coupled reference histories differ")
    state_sets = [[(x["test_id"],x["suite"],x["state"]) for x in read_table(path/"test_states.csv")
                   if x["dimension"]==2] for path in (output,original_output)]
    if state_sets[0] != state_sets[1]:
        raise ValueError("Original/coupled n=2 test states differ")
    summaries = [read_table(path/"summary.csv") for path in (output,original_output)]
    indexes = [key_index(rows) for rows in summaries]
    config=configs[0]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for row,training in enumerate(TRAINING):
        for index,label,color in zip(indexes,("Coupled-family n=2","Original Simchowitz n=2"),(DIM_COLORS[2],"#555555")):
            histories=config["contexts_by_dimension"]["2"]
            cells=[index[2,training,r,1,1] for r in histories]
            line_with_range(axes[row,0],histories,cells,label,color)
            executions=config["executions"]
            cells=[index[2,training,2,config["horizon"],e] for e in executions]
            line_with_range(axes[row,1],executions,cells,label,color)
        axes[row,0].axvline(2,color="black",ls=":")
        axes[row,0].set(title=f"{training_label(training)}: one action at a time",xlabel="History r",ylabel="Mean total MSE")
        axes[row,1].set(title=f"{training_label(training)}: history r=2, H={config['horizon']}",xlabel="Execution horizon",ylabel="Mean total MSE",xticks=config["executions"])
        for ax in axes[row]:ax.legend(fontsize=8)
    fig.suptitle("Two different n=2 systems, compared under the same protocol\nSame test states, training seeds, data counts, and timing; seed-range shading; red ×: any unstable seed")
    save_figure(fig,output/"figures/cross_dimension","original_vs_coupled_n2")
    # Show the actual paper reference alongside every coupled dimension without
    # pretending it is the n=2 member of the same scaling family.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for row,training in enumerate(TRAINING):
        for col,(H,e) in enumerate([(1,1),(config["horizon"],max(config["executions"]))]):
            cells=[indexes[0][n,training,n,H,e] for n in config["dimensions"]]
            line_with_range(axes[row,col],config["dimensions"],cells,"Coupled family, r=n",DIM_COLORS[2])
            original=indexes[1][2,training,2,H,e]
            axes[row,col].plot(2,original["mean_test_mse"],marker="*",color="black",ms=13,ls="none",label="Original Simchowitz n=2, r=2")
            axes[row,col].vlines(2,original["min_seed_mse"],original["max_seed_mse"],color="black")
            if original["stable_seeds"]<original["seeds"]:
                axes[row,col].scatter(2,original["mean_test_mse"],marker="x",color="red",s=65,zorder=6)
            axes[row,col].set(title=f"{training_label(training)}: H={H}, execution={e}",
                              xlabel="State dimension n",ylabel="Mean total MSE",xticks=config["dimensions"])
            axes[row,col].legend(fontsize=8)
    fig.suptitle("Original n=2 reference alongside all coupled dimensions\nStars are a different system; lines connect only the coupled family; r equals each system's observability index")
    save_figure(fig,output/"figures/cross_dimension","original_reference_scaling")
    combined=[]
    for family,rows in zip(("coupled","original"),summaries):
        combined.extend(dict(family=family,**row) for row in rows if row["dimension"]==2)
    write_records(output/"original_vs_coupled_n2.csv",combined)
    report=output/"README.md"
    lines=["", "## Original Simchowitz n=2 comparison", "",
           "The n=2 point in the dimension-scaling plots uses the coupled family, so all four points share one construction. "
           "The paper's original n=2 matrices are different and are kept as a separately labeled matched reference. "
           "They use exactly the same test states, five training seeds, history grid, trajectory counts, target times, and warm-up.", "",
           "[Original versus coupled n=2 graph](figures/cross_dimension/original_vs_coupled_n2.png) · "
           "[Original n=2 alongside all dimensions](figures/cross_dimension/original_reference_scaling.png) · "
           "[Original n=2 raw results and plots](original_n2/README.md)", "",
           "| Family | Training | History | Prediction | Execution | Mean MSE | Stable seeds |",
           "|---|---|---:|---:|---:|---:|---|"]
    for family,index in zip(("Coupled n=2","Original n=2"),indexes):
        for training in TRAINING:
            for H,e in [(1,1),(config["horizon"],max(config["executions"]))]:
                cell=index[2,training,2,H,e]
                lines.append(f"| {family} | {training_label(training)} | 2 | {H} | {e} | {cell['mean_test_mse']:.4g} | {cell['stable_seeds']}/{cell['seeds']} |")
    report.write_text(report.read_text()+"\n".join(lines)+"\n")
