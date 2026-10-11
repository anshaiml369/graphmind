"""
UI adapter for GraphMind.

This module does NOT change the model, the inference code, or the existing API.
It imports the existing FastAPI app from main.py and adds read-only endpoints
under /api/ui/* that the HTML frontend needs (graph neighbourhoods, dataset
statistics and an evaluation of the shipped ONNX model). All predictions are
produced by main.run_model() — the same function behind POST /predict and
POST /predict/cora_node.

Run:  uvicorn ui_api:app --reload
(`uvicorn main:app` still works; the frontend then shows prediction only.)
"""
import json
import os
import threading
from collections import deque

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from starlette.routing import Mount

from main import (  # existing, unmodified
    BASE_DIR,
    CORA_CLASSES,
    DATA_DIR,
    MODEL_PATH,
    app,
    model_session,
    run_model,
)

RESULTS_PATH = os.path.join(BASE_DIR, "results", "notebook_results.json")

router = APIRouter(prefix="/api/ui", tags=["ui"])

_cache = {}
_lock = threading.Lock()


def _load():
    """Load Cora once and run the shipped ONNX model over the full graph."""
    with _lock:
        if _cache:
            return _cache
        from torch_geometric.datasets import Planetoid

        data = Planetoid(root=DATA_DIR, name="Cora")[0]
        x = data.x.numpy()
        edge_index = data.edge_index.numpy()
        y = data.y.numpy()
        n = x.shape[0]

        full = run_model(x, edge_index, range(n))
        preds = np.array([p["predicted_class_id"] for p in full["predictions"]])
        probs = np.array([p["probabilites"] for p in full["predictions"]])

        # Same request shape as POST /predict without edge_indices: self-loops only.
        self_loops = np.vstack([np.arange(n), np.arange(n)])
        no_edges = run_model(x, self_loops, range(n))
        preds_no_edges = np.array([p["predicted_class_id"] for p in no_edges["predictions"]])

        adjacency = [[] for _ in range(n)]
        for s, t in edge_index.T:
            adjacency[int(s)].append(int(t))
        adjacency = [sorted(set(a)) for a in adjacency]

        split = np.full(n, "unlabelled", dtype=object)
        split[data.train_mask.numpy()] = "train"
        split[data.val_mask.numpy()] = "val"
        split[data.test_mask.numpy()] = "test"

        _cache.update(
            x=x, y=y, n=n, edge_index=edge_index, adjacency=adjacency, split=split,
            preds=preds, probs=probs, preds_no_edges=preds_no_edges,
            masks={
                "train": data.train_mask.numpy(),
                "val": data.val_mask.numpy(),
                "test": data.test_mask.numpy(),
            },
        )
        return _cache


def _metrics(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = []
    f1s = []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class.append({
            "class_id": c, "class_name": CORA_CLASSES[c],
            "precision": precision, "recall": recall, "f1": f1, "support": int(cm[c, :].sum()),
        })
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(f1s)),
        "n": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


@router.get("/overview")
def overview():
    d = _load()
    degrees = np.array([len(a) for a in d["adjacency"]])
    class_counts = np.bincount(d["y"], minlength=len(CORA_CLASSES)).tolist()
    return {
        "dataset": "Cora (PyTorch Geometric Planetoid, public split)",
        "num_nodes": int(d["n"]),
        "num_directed_edges": int(d["edge_index"].shape[1]),
        "num_undirected_edges": int(d["edge_index"].shape[1] // 2),
        "num_features": int(d["x"].shape[1]),
        "num_classes": len(CORA_CLASSES),
        "classes": CORA_CLASSES,
        "class_counts": class_counts,
        "split_sizes": {k: int(v.sum()) for k, v in d["masks"].items()},
        "degree": {
            "mean": float(degrees.mean()), "median": float(np.median(degrees)),
            "max": int(degrees.max()), "min": int(degrees.min()),
        },
        "features_per_node": {
            "mean": float(d["x"].sum(axis=1).mean()),
            "min": int(d["x"].sum(axis=1).min()),
            "max": int(d["x"].sum(axis=1).max()),
        },
        "model": {
            "file": os.path.basename(MODEL_PATH),
            "size_bytes": os.path.getsize(MODEL_PATH)
            + (os.path.getsize(MODEL_PATH + ".data") if os.path.exists(MODEL_PATH + ".data") else 0),
            "runtime": "onnxruntime",
            "providers": model_session.get_providers(),
            "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in model_session.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in model_session.get_outputs()],
        },
    }


@router.get("/evaluation")
def evaluation():
    d = _load()
    k = len(CORA_CLASSES)
    live = {split: _metrics(d["y"][m], d["preds"][m], k) for split, m in d["masks"].items()}
    test = d["masks"]["test"]
    ablation = _metrics(d["y"][test], d["preds_no_edges"][test], k)
    recorded = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            recorded = json.load(f)
    return {
        "live": live,
        "edge_ablation_test": {
            "description": "Same ONNX model and features, but every citation edge replaced by a self-loop "
                           "(identical to POST /predict without edge_indices).",
            "accuracy": ablation["accuracy"], "macro_f1": ablation["macro_f1"], "n": ablation["n"],
        },
        "recorded": recorded,
    }


@router.get("/samples")
def samples():
    """Real Cora nodes for quick selection: first test node of each class, plus test errors."""
    d = _load()
    test_ids = np.where(d["masks"]["test"])[0]
    per_class = []
    for c in range(len(CORA_CLASSES)):
        ids = [int(i) for i in test_ids if d["y"][i] == c]
        if ids:
            per_class.append({"node": ids[0], "true_class": CORA_CLASSES[c]})
    errors = [int(i) for i in test_ids if d["preds"][i] != d["y"][i]][:8]
    return {
        "per_class": per_class,
        "misclassified": [
            {"node": i, "true_class": CORA_CLASSES[int(d["y"][i])], "predicted_class": CORA_CLASSES[int(d["preds"][i])]}
            for i in errors
        ],
    }


@router.get("/node/{node_id}")
def node_neighbourhood(
    node_id: int,
    hops: int = Query(1, ge=1, le=2),
    max_nodes: int = Query(60, ge=5, le=200),
):
    d = _load()
    if node_id < 0 or node_id >= d["n"]:
        raise HTTPException(400, f"node_id out of range (must be 0 to {d['n'] - 1})")

    hop_of = {node_id: 0}
    queue = deque([node_id])
    truncated = False
    while queue:
        u = queue.popleft()
        if hop_of[u] >= hops:
            continue
        for v in d["adjacency"][u]:
            if v in hop_of:
                continue
            if len(hop_of) >= max_nodes:
                truncated = True
                break
            hop_of[v] = hop_of[u] + 1
            queue.append(v)

    ids = sorted(hop_of, key=lambda i: (hop_of[i], i))
    included = set(ids)
    edges = []
    for u in ids:
        for v in d["adjacency"][u]:
            if v in included and u < v:
                edges.append([u, v])

    total_in_hops = {1: len(d["adjacency"][node_id])}
    if hops == 2:
        two = set()
        for v in d["adjacency"][node_id]:
            two.update(d["adjacency"][v])
        two.discard(node_id)
        two -= set(d["adjacency"][node_id])
        total_in_hops[2] = len(two)

    def describe(i):
        return {
            "id": int(i),
            "hop": hop_of[i],
            "true_class_id": int(d["y"][i]),
            "pred_class_id": int(d["preds"][i]),
            "confidence": float(d["probs"][i][d["preds"][i]]),
            "split": d["split"][i],
            "degree": len(d["adjacency"][i]),
        }

    center = describe(node_id)
    center["active_feature_indices"] = np.flatnonzero(d["x"][node_id]).astype(int).tolist()
    center["pred_without_edges_class_id"] = int(d["preds_no_edges"][node_id])

    return {
        "center": center,
        "nodes": [describe(i) for i in ids],
        "edges": edges,
        "hops": hops,
        "truncated": truncated,
        "neighbour_counts": total_in_hops,
    }


app.include_router(router)

# main.py mounts StaticFiles at "/" as a catch-all; Starlette matches routes in
# order, so move that mount behind the routes registered above.
for route in [r for r in app.router.routes if isinstance(r, Mount) and r.path in ("", "/")]:
    app.router.routes.remove(route)
    app.router.routes.append(route)
