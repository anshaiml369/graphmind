"""Hyperparameter search on the original graph using training and validation data only.

Every grid configuration is trained with each tuning seed (disjoint from the evaluation seeds); the configuration
with the highest mean validation accuracy wins (ties: lower mean validation loss). Test data is never evaluated.

Usage (from the repository root):
    python -m experiments.tune                # all models
    python -m experiments.tune --models gcn   # one model
Writes results/tuning_results.csv and results/best_hparams.json.
"""
import argparse
import csv
import itertools
import json
import os

import numpy as np

from experiments.common import RESULTS_DIR, load_config, load_cora
from experiments.train import run_once


def grid(spec):
    keys = list(spec)
    for values in itertools.product(*(spec[k] for k in keys)):
        yield dict(zip(keys, values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mlp", "gcn", "gat"])
    args = parser.parse_args()

    config = load_config()
    data = load_cora(config)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tuning = config["tuning"]
    rows, best = [], {}
    best_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)

    for model_name in args.models:
        candidates = []
        for hparams in grid(tuning["grids"][model_name]):
            val_accs, val_losses, epochs = [], [], []
            for seed in tuning["seeds"]:
                record, _, _ = run_once(data, data.edge_index, model_name, hparams, seed,
                                        config["training"], evaluate_test=False)
                val_accs.append(record["best_val_accuracy"])
                val_losses.append(record["best_val_loss"])
                epochs.append(record["best_epoch"])
                rows.append({"model": model_name, **{f"hp_{k}": v for k, v in hparams.items()}, "seed": seed,
                             "best_epoch": record["best_epoch"], "val_accuracy": record["best_val_accuracy"],
                             "val_loss": record["best_val_loss"]})
            candidates.append((float(np.mean(val_accs)), -float(np.mean(val_losses)), hparams,
                               float(np.mean(epochs))))
            print(f"{model_name} {hparams} mean val acc {np.mean(val_accs):.4f}", flush=True)
        acc, neg_loss, hparams, mean_epoch = max(candidates, key=lambda c: (c[0], c[1]))
        best[model_name] = {"hparams": hparams, "mean_val_accuracy": acc, "mean_val_loss": -neg_loss,
                            "mean_best_epoch": mean_epoch, "tuning_seeds": tuning["seeds"],
                            "configurations_tried": len(candidates)}
        print(f"==> {model_name}: {hparams} (mean val acc {acc:.4f})", flush=True)

    tuning_csv = os.path.join(RESULTS_DIR, "tuning_results.csv")
    existing = []
    if os.path.exists(tuning_csv):
        with open(tuning_csv) as f:
            existing = [r for r in csv.DictReader(f) if r["model"] not in args.models]
    fields = ["model", "hp_lr", "hp_weight_decay", "hp_dropout", "hp_hidden", "hp_heads", "seed",
              "best_epoch", "val_accuracy", "val_loss"]
    with open(tuning_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(existing + rows)
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
