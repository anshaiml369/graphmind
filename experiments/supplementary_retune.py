"""Supplementary check: are the ablation penalties caused by hyperparameters tuned on the original graph?

The main experiment reuses each model's hyperparameters (tuned on the original graph) for every graph condition,
so only the graph changes. That can disadvantage the ablated graphs - e.g. GAT's tuned attention dropout of 0.8
randomly removes a node's only message when the graph has no edges. Here GCN and GAT are re-tuned separately on
the `no_edges` and `rewired` conditions (same grid, same tuning seeds, validation only; for `rewired` each tuning
seed uses its own rewired graph) and then evaluated on the 5 evaluation seeds.

Usage: python -m experiments.supplementary_retune
Writes results/supplementary_retuned_results.csv and results/supplementary_retuned_hparams.json.
"""
import csv
import json
import os

import numpy as np

from experiments.common import RESULTS_DIR, load_config, load_cora, make_graph
from experiments.train import run_once
from experiments.tune import grid


def main():
    config = load_config()
    data = load_cora(config)
    tuning, rows, chosen = config["tuning"], [], {}
    for model_name in ("gcn", "gat"):
        for condition in ("no_edges", "rewired"):
            candidates = []
            for hparams in grid(tuning["grids"][model_name]):
                accs, losses = [], []
                for seed in tuning["seeds"]:
                    ei = make_graph(data.edge_index, data.num_nodes, condition, seed)
                    record, _, _ = run_once(data, ei, model_name, hparams, seed, config["training"],
                                            evaluate_test=False)
                    accs.append(record["best_val_accuracy"])
                    losses.append(record["best_val_loss"])
                candidates.append((float(np.mean(accs)), -float(np.mean(losses)), hparams))
            acc, _, hparams = max(candidates, key=lambda c: (c[0], c[1]))
            chosen[f"{model_name}__{condition}"] = {"hparams": hparams, "mean_val_accuracy": acc}
            print(f"==> {model_name} {condition}: {hparams} (mean val acc {acc:.4f})", flush=True)
            for seed in config["evaluation"]["seeds"]:
                ei = make_graph(data.edge_index, data.num_nodes, condition, seed)
                record, _, _ = run_once(data, ei, model_name, hparams, seed, config["training"])
                rows.append({"run_id": f"{model_name}__{condition}__retuned__seed{seed}", "model": model_name,
                             "graph_condition": condition, "seed": seed,
                             **{f"hp_{k}": v for k, v in hparams.items()},
                             "best_epoch": record["best_epoch"], "best_val_accuracy": record["best_val_accuracy"],
                             "test_accuracy": record["test"]["accuracy"], "test_macro_f1": record["test"]["macro_f1"],
                             "test_weighted_f1": record["test"]["weighted_f1"]})
                print(rows[-1]["run_id"], f"test acc {record['test']['accuracy']:.3f}", flush=True)
    fields = ["run_id", "model", "graph_condition", "seed", "hp_lr", "hp_weight_decay", "hp_dropout", "hp_hidden",
              "hp_heads", "best_epoch", "best_val_accuracy", "test_accuracy", "test_macro_f1", "test_weighted_f1"]
    with open(os.path.join(RESULTS_DIR, "supplementary_retuned_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "supplementary_retuned_hparams.json"), "w") as f:
        json.dump(chosen, f, indent=2)


if __name__ == "__main__":
    main()
