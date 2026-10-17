"""Verify the original GraphMind figures and measure the effect of the checkpointing bug.

Part 1 re-evaluates the shipped ONNX model (simple_gcn_cora.onnx) with and without citation edges, which is
how the 0.792 / 0.678 figures were produced (ui_api.py, inference-time edge removal).

Part 2 re-runs the notebook's training procedure (SimpleGCN: GCNConv 1433->32->7, ReLU, dropout 0.5 after layer 1,
Adam lr 0.01, no weight decay, max 200 epochs, early stopping on validation accuracy with patience 20) on the
evaluation seeds, twice per seed:
  * "reference" checkpoint - best_state = model.state_dict() (the notebook's code; tensors are later
    overwritten in place by the optimizer, so the final-epoch weights are restored);
  * "deep copy" checkpoint - the corrected version.

Usage: python -m experiments.reproduce_original
Writes results/original_onnx_check.json and results/original_protocol_results.csv.
"""
import csv
import json
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from experiments.common import REPO_ROOT, RESULTS_DIR, classification_metrics, load_config, load_cora, set_seed


class SimpleGCN(nn.Module):  # copied from the notebook, cell 22
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def onnx_check(data):
    session = ort.InferenceSession(os.path.join(REPO_ROOT, "simple_gcn_cora.onnx"),
                                   providers=["CPUExecutionProvider"])
    x = data.x.numpy().astype(np.float32)
    n = x.shape[0]
    y = data.y.numpy()
    out = {}
    for name, ei in {"original_graph": data.edge_index.numpy(),
                     "edges_replaced_by_self_loops": np.vstack([np.arange(n), np.arange(n)])}.items():
        pred = session.run(["logits"], {"node_features": x, "edge_indices": ei.astype(np.int64)})[0].argmax(1)
        out[name] = {split: {k: v for k, v in classification_metrics(y[m], pred[m]).items() if k != "per_class"}
                     for split, m in (("val", data.val_mask.numpy()), ("test", data.test_mask.numpy()))}
    return out


def notebook_protocol(data, seed, deep_copy):
    set_seed(seed)
    model = SimpleGCN(data.num_features, 32, 7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def evaluate(mask):
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
        return (pred[mask] == data.y[mask]).float().mean().item(), pred

    best_val, best_state, best_epoch, waited = 0.0, None, 0, 0
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        val_acc, _ = evaluate(data.val_mask)
        if val_acc > best_val:
            best_val, best_epoch, waited = val_acc, epoch, 0
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                          if deep_copy else model.state_dict())
        else:
            waited += 1
        if waited >= 20:
            break
    model.load_state_dict(best_state)
    restored_val, pred = evaluate(data.val_mask)
    test = classification_metrics(data.y[data.test_mask].numpy(), pred[data.test_mask].numpy())
    return {"seed": seed, "checkpoint": "deep_copy" if deep_copy else "reference (original bug)",
            "logged_best_epoch": best_epoch, "stopped_epoch": epoch, "logged_best_val_accuracy": best_val,
            "restored_model_val_accuracy": restored_val, "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"]}


def main():
    config = load_config()
    data = load_cora(config)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "original_onnx_check.json"), "w") as f:
        json.dump(onnx_check(data), f, indent=2)
    rows = [notebook_protocol(data, seed, deep_copy)
            for seed in config["evaluation"]["seeds"] for deep_copy in (False, True)]
    with open(os.path.join(RESULTS_DIR, "original_protocol_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
