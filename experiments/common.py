"""Shared helpers: configuration, seeding, data loading, graph interventions, graph statistics and metrics.

All paths are relative to the repository root, so the experiment runs from any checkout.
"""
import json
import os
import random

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "experiments", "configs", "experiment.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Planetoid label ids -> Cora topics. Inferred from class sizes: PyG's counts [351, 217, 418, 818, 426, 298, 180]
# match Cora's published topic sizes only in this order (Neural_Networks is the largest topic, 818 papers).
# main.py's CORA_CLASSES used an alphabetical order until commit 8042c6a (RESEARCH_REPORT.md, B5).
CLASS_NAMES = ["Theory", "Reinforcement_Learning", "Genetic_Algorithms", "Neural_Networks",
               "Probabilistic_Methods", "Case_Based", "Rule_Learning"]
GRAPH_CONDITIONS = ("original", "no_edges", "rewired", "drop50")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def set_seed(seed, num_threads=1):
    """Seed Python, NumPy and PyTorch (PyG draws from the PyTorch generator) and force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


def load_cora(config=None):
    """Cora with the Planetoid public split (140 train / 500 val / 1000 test), unchanged from the original project."""
    from torch_geometric.datasets import Planetoid

    config = config or load_config()
    root = os.path.join(REPO_ROOT, config["dataset"]["root"])
    return Planetoid(root=root, name=config["dataset"]["name"], split=config["dataset"]["split"])[0]


def _undirected_pairs(edge_index):
    """Unique undirected edges (u < v) from a symmetric edge_index."""
    ei = edge_index.numpy()
    keep = ei[0] < ei[1]
    return np.stack([ei[0][keep], ei[1][keep]], axis=1)


def _to_edge_index(pairs):
    if len(pairs) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    pairs = np.asarray(pairs, dtype=np.int64)
    both = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    order = np.lexsort((both[:, 1], both[:, 0]))
    return torch.from_numpy(both[order].T.copy())


def make_graph(edge_index, num_nodes, condition, seed):
    """Return the edge_index for a graph condition. Nodes, features, labels and splits are never touched."""
    if condition == "original":
        return edge_index.clone()
    if condition == "no_edges":
        return torch.empty((2, 0), dtype=torch.long)
    pairs = _undirected_pairs(edge_index)
    if condition == "drop50":
        rng = np.random.default_rng(seed)
        keep = rng.permutation(len(pairs))[: len(pairs) // 2]
        return _to_edge_index(pairs[np.sort(keep)])
    if condition == "rewired":
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        g.add_edges_from(map(tuple, pairs))
        nx.double_edge_swap(g, nswap=10 * g.number_of_edges(), max_tries=1000 * g.number_of_edges(), seed=seed)
        return _to_edge_index(sorted((min(u, v), max(u, v)) for u, v in g.edges()))
    raise ValueError(f"unknown graph condition: {condition}")


def graph_stats(edge_index, y, num_nodes):
    """Descriptive statistics of an undirected graph given as a symmetric edge_index."""
    pairs = _undirected_pairs(edge_index)
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(map(tuple, pairs))
    degrees = np.array([d for _, d in g.degree()])
    components = list(nx.connected_components(g))
    y = np.asarray(y)
    homophily = float((y[pairs[:, 0]] == y[pairs[:, 1]]).mean()) if len(pairs) else float("nan")
    return {
        "nodes": num_nodes,
        "undirected_edges": int(len(pairs)),
        "directed_edges": int(edge_index.shape[1]),
        "density": float(nx.density(g)),
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "isolated_nodes": int((degrees == 0).sum()),
        "connected_components": len(components),
        "largest_component_nodes": max(len(c) for c in components),
        "edge_homophily": homophily,
        "average_clustering": float(nx.average_clustering(g)),
    }


def classification_metrics(y_true, y_pred, num_classes=7):
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": [
            {"class_id": c, "class_name": CLASS_NAMES[c], "precision": float(precision[c]),
             "recall": float(recall[c]), "f1": float(f1[c]), "support": int(support[c])}
            for c in labels
        ],
    }
