"""Main experiment: every model x graph condition x evaluation seed, using the tuned hyperparameters.

For GCN/GAT trained on the original graph it also records the original project's inference-time ablation
(the same trained weights evaluated with every citation edge removed), which is how the earlier 0.678 figure
was produced.

Usage (from the repository root, after `python -m experiments.tune`):
    python -m experiments.run_experiment
Writes:
    results/per_seed_results.csv     one row per run (validation selection details + test metrics)
    results/per_class_results.csv    per-class precision / recall / F1 on the test set, per run
    results/graph_stats.csv          statistics of every graph actually used
    results/predictions.npz          node-level predictions per run (for bootstrap analysis)
    results/logs/<run_id>.json       per-epoch training history and selected checkpoint details
    results/checkpoints/<run_id>.pt  selected weights (not committed; regenerate with this script)
"""
import csv
import hashlib
import json
import os

import numpy as np
import torch

from experiments.common import RESULTS_DIR, classification_metrics, graph_stats, load_config, load_cora, make_graph
from experiments.train import predict, run_once

PER_SEED_FIELDS = [
    "run_id", "model", "graph_condition", "seed", "hp_lr", "hp_weight_decay", "hp_dropout", "hp_hidden", "hp_heads",
    "best_epoch", "stopped_epoch", "best_val_accuracy", "best_val_loss",
    "test_accuracy", "test_macro_f1", "test_weighted_f1",
    "infer_no_edges_test_accuracy", "infer_no_edges_test_macro_f1",
    "num_undirected_edges", "train_seconds", "config_sha256",
]


def main():
    config = load_config()
    with open(os.path.join(RESULTS_DIR, "best_hparams.json")) as f:
        best_hparams = json.load(f)
    config_sha = hashlib.sha256(json.dumps([config, best_hparams], sort_keys=True).encode()).hexdigest()[:12]
    data = load_cora(config)
    y = data.y.numpy()
    for sub in ("logs", "checkpoints"):
        os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)

    graphs, stats_rows = {}, []
    for condition in ("original", "no_edges", "rewired", "drop50"):
        for seed in config["evaluation"]["seeds"]:
            ei = make_graph(data.edge_index, data.num_nodes, condition, seed)
            graphs[(condition, seed)] = ei
            stats_rows.append({"graph_condition": condition, "seed": seed, **graph_stats(ei, y, data.num_nodes)})

    per_seed, per_class, predictions = [], [], {}
    empty = torch.empty((2, 0), dtype=torch.long)
    for model_name, conditions in config["evaluation"]["conditions"].items():
        hparams = best_hparams[model_name]["hparams"]
        for condition in conditions:
            for seed in config["evaluation"]["seeds"]:
                run_id = f"{model_name}__{condition}__seed{seed}"
                ei = graphs[(condition, seed)]
                record, model, state = run_once(data, ei, model_name, hparams, seed, config["training"])
                row = {"run_id": run_id, "model": model_name, "graph_condition": condition, "seed": seed,
                       **{f"hp_{k}": v for k, v in hparams.items()},
                       "best_epoch": record["best_epoch"], "stopped_epoch": record["stopped_epoch"],
                       "best_val_accuracy": record["best_val_accuracy"], "best_val_loss": record["best_val_loss"],
                       "test_accuracy": record["test"]["accuracy"], "test_macro_f1": record["test"]["macro_f1"],
                       "test_weighted_f1": record["test"]["weighted_f1"],
                       "num_undirected_edges": int(ei.shape[1] // 2) if model.uses_graph else 0,
                       "train_seconds": round(record["train_seconds"], 2), "config_sha256": config_sha}
                if model.uses_graph and condition == "original":
                    pred_ne = predict(model, data.x, empty)
                    m = classification_metrics(y[data.test_mask.numpy()], pred_ne.numpy()[data.test_mask.numpy()])
                    row["infer_no_edges_test_accuracy"] = m["accuracy"]
                    row["infer_no_edges_test_macro_f1"] = m["macro_f1"]
                per_seed.append(row)
                for c in record["test"]["per_class"]:
                    per_class.append({"run_id": run_id, "model": model_name, "graph_condition": condition,
                                      "seed": seed, **c})
                predictions[run_id] = record["predictions"]
                torch.save(state, os.path.join(RESULTS_DIR, "checkpoints", f"{run_id}.pt"))
                with open(os.path.join(RESULTS_DIR, "logs", f"{run_id}.json"), "w") as f:
                    json.dump({k: v for k, v in row.items()} | {"hparams": hparams, "history": record["history"]},
                              f, indent=1)
                print(f"{run_id}: best epoch {record['best_epoch']}, val {record['best_val_accuracy']:.3f}, "
                      f"test acc {record['test']['accuracy']:.3f}, macro-F1 {record['test']['macro_f1']:.3f}",
                      flush=True)

    def write(name, rows, fields):
        with open(os.path.join(RESULTS_DIR, name), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)

    write("per_seed_results.csv", per_seed, PER_SEED_FIELDS)
    write("per_class_results.csv", per_class, list(per_class[0]))
    write("graph_stats.csv", stats_rows, list(stats_rows[0]))
    np.savez_compressed(os.path.join(RESULTS_DIR, "predictions.npz"), y=y,
                        test_mask=data.test_mask.numpy(), **predictions)


if __name__ == "__main__":
    main()
