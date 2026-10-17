"""Aggregate statistics, model comparisons, the graph ablation table and plots, computed only from the run files.

Usage: python -m experiments.analyze
Reads  results/per_seed_results.csv, results/per_class_results.csv, results/graph_stats.csv, results/predictions.npz
Writes results/summary.csv, results/per_class_summary.csv, results/comparisons.csv, results/ablation.csv,
       results/graph_stats_summary.csv and results/plots/*.png

Uncertainty conventions
  * mean +/- SD: sample standard deviation (ddof=1) over the 5 evaluation seeds. It reflects training randomness
    (initialisation, dropout, and for rewired/drop50 the random graph) only - not split or test-sample variation.
  * Seed-level 95% CI of a difference: Welch t-interval for different models (independent runs); paired
    t-interval for the same model on two graph conditions (same seed => same initial weights).
  * Test-sample 95% CI: paired bootstrap over the 1,000 test nodes (10,000 resamples) of the difference in
    seed-averaged accuracy. It reflects how much the result could move with a different test sample.
  * With n = 5 seeds, p-values are coarse; they are Holm-corrected within each table and are reported alongside
    effect sizes, never instead of them.
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiments.common import RESULTS_DIR

METRICS = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
ORDER = [("mlp", "original"), ("gcn", "original"), ("gat", "original"),
         ("gcn", "no_edges"), ("gat", "no_edges"), ("gcn", "drop50"), ("gat", "drop50"),
         ("gcn", "rewired"), ("gat", "rewired")]
LABEL = {"mlp": "MLP", "gcn": "GCN", "gat": "GAT"}
COND = {"original": "original graph", "no_edges": "no edges", "drop50": "50% edges dropped",
        "rewired": "degree-preserving rewired"}


def read_csv(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with open(os.path.join(RESULTS_DIR, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def by_group(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["graph_condition"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: int(r["seed"]))
    return groups


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def summary_table(groups):
    rows = []
    for key in ORDER:
        if key not in groups:
            continue
        g = groups[key]
        row = {"model": key[0], "graph_condition": key[1], "n_runs": len(g),
               "seeds": " ".join(r["seed"] for r in g)}
        for metric in METRICS + ["best_val_accuracy", "best_epoch"]:
            m, s = mean_sd([r[metric] for r in g])
            row[f"{metric}_mean"], row[f"{metric}_sd"] = round(m, 4), round(s, 4)
        infer = [r["infer_no_edges_test_accuracy"] for r in g if r["infer_no_edges_test_accuracy"]]
        if infer:
            m, s = mean_sd(infer)
            row["infer_no_edges_test_accuracy_mean"], row["infer_no_edges_test_accuracy_sd"] = round(m, 4), round(s, 4)
        else:
            row["infer_no_edges_test_accuracy_mean"] = row["infer_no_edges_test_accuracy_sd"] = ""
        rows.append(row)
    return rows


def bootstrap_diff(pred, y, ids_a, ids_b, n_boot=10000, seed=12345):
    """95% percentile CI of mean-over-seeds test accuracy(A) - accuracy(B), resampling test nodes."""
    correct_a = np.mean([pred[i] == y for i in ids_a], axis=0)   # per-node fraction of seeds correct
    correct_b = np.mean([pred[i] == y for i in ids_b], axis=0)
    diff = correct_a - correct_b
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compare(groups, pred, y, a, b, paired):
    ga, gb = groups[a], groups[b]
    row = {"A": f"{LABEL[a[0]]} ({COND[a[1]]})", "B": f"{LABEL[b[0]]} ({COND[b[1]]})",
           "pairing": "paired by seed" if paired else "independent (Welch)"}
    for metric in ("test_accuracy", "test_macro_f1"):
        xa = np.array([float(r[metric]) for r in ga])
        xb = np.array([float(r[metric]) for r in gb])
        d = xa - xb
        if paired:
            t = stats.ttest_rel(xa, xb)
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = d.mean() - half, d.mean() + half
        else:
            t = stats.ttest_ind(xa, xb, equal_var=False)
            ci = t.confidence_interval(0.95)
            lo, hi = ci.low, ci.high
        short = metric.replace("test_", "")
        row[f"{short}_A_mean"], row[f"{short}_B_mean"] = round(xa.mean(), 4), round(xb.mean(), 4)
        row[f"{short}_diff"] = round(d.mean(), 4)
        row[f"{short}_seed_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        row[f"{short}_p"] = float(t.pvalue)
        row[f"{short}_A_better_in_seeds"] = f"{int((xa > xb).sum())}/{len(xa)}"
    _, lo, hi = bootstrap_diff(pred, y, [r["run_id"] for r in ga], [r["run_id"] for r in gb])
    row["accuracy_test_sample_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
    return row


def finish_p(rows):
    for metric in ("accuracy", "macro_f1"):
        adj = holm(np.array([r[f"{metric}_p"] for r in rows]))
        for r, p in zip(rows, adj):
            r[f"{metric}_p"] = f"{r[f'{metric}_p']:.2g}"
            r[f"{metric}_p_holm"] = f"{float(p):.2g}"
    return rows


def plot_per_seed(groups, path):
    keys = [k for k in ORDER if k in groups]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    colours = {"mlp": "#7f7f7f", "gcn": "#1f77b4", "gat": "#d62728"}
    for ax, metric, title in zip(axes, ("test_accuracy", "test_macro_f1"), ("Test accuracy", "Test macro-F1")):
        for i, k in enumerate(keys):
            vals = np.array([float(r[metric]) for r in groups[k]])
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(i + jitter, vals, s=22, color=colours[k[0]], alpha=0.75, zorder=3,
                       label="individual seed" if i == 0 else None)
            m, s = vals.mean(), vals.std(ddof=1)
            ax.errorbar(i + 0.3, m, yerr=s, fmt="D", color="black", ms=5, capsize=4, zorder=4,
                        label="mean ± SD (5 seeds)" if i == 0 else None)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{LABEL[m]}\n{COND[c]}" for m, c in keys], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{title} per seed (Cora public split, n = 1,000 test nodes)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(groups, path):
    conds = ["original", "drop50", "rewired", "no_edges"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.36
    for j, (model, colour) in enumerate((("gcn", "#1f77b4"), ("gat", "#d62728"))):
        for i, c in enumerate(conds):
            vals = np.array([float(r["test_accuracy"]) for r in groups[(model, c)]])
            x = i + (j - 0.5) * width
            ax.bar(x, vals.mean(), width, yerr=vals.std(ddof=1), capsize=4, color=colour, alpha=0.75,
                   label=LABEL[model] if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, s=10, color="black", zorder=3)
    mlp = np.array([float(r["test_accuracy"]) for r in groups[("mlp", "original")]])
    ax.axhspan(mlp.mean() - mlp.std(ddof=1), mlp.mean() + mlp.std(ddof=1), color="grey", alpha=0.2)
    ax.axhline(mlp.mean(), color="grey", ls="--", label="MLP (features only), mean ± SD band")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND[c] for c in conds])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 0.9)
    ax.set_title("Graph-structure ablation: models retrained on each graph (bars = mean ± SD over 5 seeds, dots = seeds)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(path):
    import json
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, model in zip(axes, ("mlp", "gcn", "gat")):
        with open(os.path.join(RESULTS_DIR, "logs", f"{model}__original__seed0.json")) as f:
            log = json.load(f)
        h = log["history"]
        ep = [e["epoch"] for e in h]
        ax.plot(ep, [e["train_accuracy"] for e in h], label="train accuracy", color="#2ca02c")
        ax.plot(ep, [e["val_accuracy"] for e in h], label="validation accuracy", color="#ff7f0e")
        ax.axvline(int(log["best_epoch"]), color="black", ls=":", label=f"selected epoch {log['best_epoch']}")
        ax.set_title(f"{LABEL[model]}, seed 0 (selection uses validation only)", fontsize=9)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rows = read_csv("per_seed_results.csv")
    groups = by_group(rows)
    preds = np.load(os.path.join(RESULTS_DIR, "predictions.npz"))
    test_mask = preds["test_mask"]
    y = preds["y"][test_mask]
    pred = {k: preds[k][test_mask] for k in preds.files if "__" in k}
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

    write_csv("summary.csv", summary_table(groups))

    per_class = defaultdict(list)
    for r in read_csv("per_class_results.csv"):
        per_class[(r["model"], r["graph_condition"], int(r["class_id"]), r["class_name"], r["support"])].append(r)
    write_csv("per_class_summary.csv", [
        {"model": m, "graph_condition": c, "class_id": cid, "class_name": name, "support": sup,
         **{f"{k}_{s}": round(v, 4) for k in ("precision", "recall", "f1")
            for s, v in zip(("mean", "sd"), mean_sd([float(r[k]) for r in rs]))}}
        for (m, c, cid, name, sup), rs in sorted(per_class.items(), key=lambda kv: (ORDER.index(kv[0][:2]), kv[0][2]))])

    comparisons = finish_p([
        compare(groups, pred, y, ("gcn", "original"), ("mlp", "original"), paired=False),
        compare(groups, pred, y, ("gat", "original"), ("mlp", "original"), paired=False),
        compare(groups, pred, y, ("gcn", "original"), ("gat", "original"), paired=False),
    ])
    write_csv("comparisons.csv", comparisons)

    ablation = []
    for model in ("gcn", "gat"):
        for cond in ("drop50", "rewired", "no_edges"):
            ablation.append(compare(groups, pred, y, (model, cond), (model, "original"), paired=True))
    for model in ("gcn", "gat"):
        for cond in ("no_edges", "rewired"):
            ablation.append(compare(groups, pred, y, (model, cond), ("mlp", "original"), paired=False))
    write_csv("ablation.csv", finish_p(ablation))

    stat_groups = defaultdict(list)
    for r in read_csv("graph_stats.csv"):
        stat_groups[r["graph_condition"]].append(r)
    stat_rows = []
    for cond, rs in stat_groups.items():
        row = {"graph_condition": cond, "n_graphs": len(rs)}
        for k in rs[0]:
            if k in ("graph_condition", "seed"):
                continue
            vals = [float(r[k]) for r in rs]
            if np.isnan(vals).any():
                row[k] = "undefined (no edges)"
                continue
            row[k] = round(float(np.mean(vals)), 4)
            if len(set(vals)) > 1:
                row[k] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
        stat_rows.append(row)
    write_csv("graph_stats_summary.csv", stat_rows)

    supp_path = os.path.join(RESULTS_DIR, "supplementary_retuned_results.csv")
    if os.path.exists(supp_path):
        supp_groups = by_group(read_csv("supplementary_retuned_results.csv"))
        mlp = [float(r["test_accuracy"]) for r in groups[("mlp", "original")]]
        supp_rows = []
        for key, g in sorted(supp_groups.items()):
            acc = [float(r["test_accuracy"]) for r in g]
            main = [float(r["test_accuracy"]) for r in groups[key]]
            welch = stats.ttest_ind(acc, mlp, equal_var=False)
            ci = welch.confidence_interval(0.95)
            row = {"model": key[0], "graph_condition": key[1], "n_runs": len(g),
                   "hparams_retuned_for_condition": " ".join(f"{k[3:]}={g[0][k]}" for k in g[0] if k.startswith("hp_") and g[0][k]),
                   "test_accuracy_mean": round(float(np.mean(acc)), 4), "test_accuracy_sd": round(float(np.std(acc, ddof=1)), 4)}
            m, s = mean_sd([r["test_macro_f1"] for r in g])
            row.update(test_macro_f1_mean=round(m, 4), test_macro_f1_sd=round(s, 4),
                       main_experiment_test_accuracy_mean=round(float(np.mean(main)), 4),
                       diff_vs_mlp=round(float(np.mean(acc) - np.mean(mlp)), 4),
                       diff_vs_mlp_seed_ci95=f"[{ci.low:.4f}, {ci.high:.4f}]", diff_vs_mlp_p=f"{welch.pvalue:.2g}")
            supp_rows.append(row)
        write_csv("supplementary_retuned_summary.csv", supp_rows)

    plot_per_seed(groups, os.path.join(RESULTS_DIR, "plots", "per_seed_results.png"))
    plot_ablation(groups, os.path.join(RESULTS_DIR, "plots", "graph_ablation.png"))
    plot_curves(os.path.join(RESULTS_DIR, "plots", "training_curves_seed0.png"))
    print("wrote summary, comparisons, ablation, graph statistics and plots")


if __name__ == "__main__":
    main()
