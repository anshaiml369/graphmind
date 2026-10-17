"""Checks for the experiment's methodological guarantees. Run from the repository root:

    python -m unittest discover tests -v

The results-consistency tests are skipped until `python -m experiments.run_experiment` and
`python -m experiments.analyze` have produced the files in results/.
"""
import csv
import inspect
import json
import os
import unittest
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from experiments.common import RESULTS_DIR, classification_metrics, load_config, load_cora, make_graph
from experiments.models import MLP, build_model
from experiments.train import fit, run_once

SMALL = {"max_epochs": 40, "patience": 10, "num_threads": 1}
HP = {"lr": 0.01, "weight_decay": 5e-4, "dropout": 0.5, "hidden": 16, "heads": 2}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_cora(load_config())


class TestNoLeakage(Base):
    def test_mlp_cannot_receive_edges(self):
        self.assertFalse(MLP.uses_graph)
        self.assertEqual(list(inspect.signature(MLP.forward).parameters), ["self", "x"])
        random_edges = torch.randint(0, self.data.num_nodes, (2, 5000))
        a, _, _ = run_once(self.data, self.data.edge_index, "mlp", HP, 3, SMALL)
        b, _, _ = run_once(self.data, random_edges, "mlp", HP, 3, SMALL)
        self.assertTrue((a["predictions"] == b["predictions"]).all())
        self.assertEqual(a["history"], b["history"])

    def test_fit_signature_has_no_test_data(self):
        params = list(inspect.signature(fit).parameters)
        self.assertNotIn("test_mask", params)
        self.assertIn("val_mask", params)

    def test_selection_ignores_test_labels(self):
        """Scrambling test labels must not change training, early stopping or the selected checkpoint."""
        for model in ("mlp", "gcn"):
            a, _, sa = run_once(self.data, self.data.edge_index, model, HP, 5, SMALL)
            scrambled = self.data.clone()
            test = scrambled.test_mask
            scrambled.y = scrambled.y.clone()
            scrambled.y[test] = scrambled.y[test][torch.randperm(int(test.sum()), generator=torch.Generator().manual_seed(0))]
            scrambled.y[test] = (scrambled.y[test] + 1) % 7
            b, _, sb = run_once(scrambled, scrambled.edge_index, model, HP, 5, SMALL)
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            for k in sa:
                self.assertTrue(torch.equal(sa[k], sb[k]))
            self.assertNotEqual(a["test"]["accuracy"], b["test"]["accuracy"])  # test labels really changed

    def test_checkpoint_is_best_validation_epoch(self):
        record, model, state = run_once(self.data, self.data.edge_index, "gcn", HP, 2, SMALL)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]))
        best = record["history"][record["best_epoch"] - 1]
        self.assertAlmostEqual(best["val_accuracy"], record["best_val_accuracy"])
        self.assertAlmostEqual(record["val"]["accuracy"], record["best_val_accuracy"], places=6)
        self.assertEqual(record["best_val_accuracy"], max(h["val_accuracy"] for h in record["history"]))

    def test_seeded_runs_are_identical(self):
        a, _, _ = run_once(self.data, self.data.edge_index, "gat", HP, 11, SMALL)
        b, _, _ = run_once(self.data, self.data.edge_index, "gat", HP, 11, SMALL)
        self.assertEqual(a["history"], b["history"])
        self.assertTrue((a["predictions"] == b["predictions"]).all())


class TestGraphConditions(Base):
    def degrees(self, ei):
        return np.bincount(ei[0].numpy(), minlength=self.data.num_nodes)

    def check_simple_undirected(self, ei):
        e = ei.numpy()
        self.assertTrue((e[0] != e[1]).all(), "no self-loops")
        pairs = set(map(tuple, e.T))
        self.assertEqual(len(pairs), e.shape[1], "no duplicate edges")
        self.assertTrue(all((v, u) in pairs for u, v in pairs), "symmetric")

    def test_original_unchanged(self):
        self.assertTrue(torch.equal(make_graph(self.data.edge_index, self.data.num_nodes, "original", 0),
                                    self.data.edge_index))

    def test_no_edges(self):
        self.assertEqual(make_graph(self.data.edge_index, self.data.num_nodes, "no_edges", 0).shape, (2, 0))

    def test_drop50(self):
        ei = make_graph(self.data.edge_index, self.data.num_nodes, "drop50", 0)
        self.check_simple_undirected(ei)
        self.assertEqual(ei.shape[1], 2 * (self.data.edge_index.shape[1] // 4))
        original = set(map(tuple, self.data.edge_index.numpy().T))
        self.assertTrue(set(map(tuple, ei.numpy().T)) <= original)

    def test_rewired_keeps_degrees_changes_edges(self):
        ei = make_graph(self.data.edge_index, self.data.num_nodes, "rewired", 0)
        self.check_simple_undirected(ei)
        self.assertTrue((self.degrees(ei) == self.degrees(self.data.edge_index)).all())
        kept = len(set(map(tuple, ei.numpy().T)) & set(map(tuple, self.data.edge_index.numpy().T)))
        self.assertLess(kept / ei.shape[1], 0.05)

    def test_interventions_are_seeded(self):
        for cond in ("drop50", "rewired"):
            a = make_graph(self.data.edge_index, self.data.num_nodes, cond, 3)
            b = make_graph(self.data.edge_index, self.data.num_nodes, cond, 3)
            self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(os.path.exists(os.path.join(RESULTS_DIR, "summary.csv")), "run the experiment first")
class TestResultsConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(RESULTS_DIR, "per_seed_results.csv")) as f:
            cls.rows = list(csv.DictReader(f))
        with open(os.path.join(RESULTS_DIR, "summary.csv")) as f:
            cls.summary = list(csv.DictReader(f))

    def test_five_seeds_per_group(self):
        config = load_config()
        for model, conds in config["evaluation"]["conditions"].items():
            for cond in conds:
                seeds = sorted(int(r["seed"]) for r in self.rows if r["model"] == model and r["graph_condition"] == cond)
                self.assertEqual(seeds, sorted(config["evaluation"]["seeds"]))

    def test_summary_recomputes_from_per_seed(self):
        for s in self.summary:
            g = [r for r in self.rows if r["model"] == s["model"] and r["graph_condition"] == s["graph_condition"]]
            for metric in ("test_accuracy", "test_macro_f1", "test_weighted_f1"):
                vals = np.array([float(r[metric]) for r in g])
                self.assertAlmostEqual(float(s[f"{metric}_mean"]), vals.mean(), places=4)
                self.assertAlmostEqual(float(s[f"{metric}_sd"]), vals.std(ddof=1), places=4)

    def test_csv_matches_logs_and_predictions(self):
        preds = np.load(os.path.join(RESULTS_DIR, "predictions.npz"))
        mask, y = preds["test_mask"], preds["y"]
        for r in self.rows:
            with open(os.path.join(RESULTS_DIR, "logs", f"{r['run_id']}.json")) as f:
                log = json.load(f)
            self.assertAlmostEqual(float(log["test_accuracy"]), float(r["test_accuracy"]), places=9)
            self.assertEqual(int(log["best_epoch"]), int(r["best_epoch"]))
            self.assertAlmostEqual(log["history"][int(r["best_epoch"]) - 1]["val_accuracy"],
                                   float(r["best_val_accuracy"]), places=9)
            m = classification_metrics(y[mask], preds[r["run_id"]][mask])
            self.assertAlmostEqual(m["accuracy"], float(r["test_accuracy"]), places=9)
            self.assertAlmostEqual(m["macro_f1"], float(r["test_macro_f1"]), places=9)

    def test_hyperparameters_selected_without_test(self):
        with open(os.path.join(RESULTS_DIR, "tuning_results.csv")) as f:
            header = next(csv.reader(f))
        self.assertFalse(any("test" in h for h in header))
        config = load_config()
        self.assertFalse(set(config["tuning"]["seeds"]) & set(config["evaluation"]["seeds"]))


if __name__ == "__main__":
    unittest.main()
