"""Training with validation-only model selection, then a single test evaluation.

`fit` is given the training and validation masks only; it never receives test labels or a test mask, so
early stopping and checkpoint selection cannot depend on the test set. The selected checkpoint is a deep copy
of the weights (the original notebook stored references from `state_dict()`, which later optimizer steps
overwrote - see RESEARCH_REPORT.md, bug B1).
"""
import time

import torch
import torch.nn.functional as F

from experiments.common import classification_metrics, set_seed
from experiments.models import build_model


def _logits(model, x, edge_index):
    return model(x, edge_index) if model.uses_graph else model(x)


def fit(model, x, edge_index, y, train_mask, val_mask, lr, weight_decay, max_epochs, patience):
    """Full-batch training. Selection: highest validation accuracy, ties broken by lower validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"val_accuracy": -1.0, "val_loss": float("inf"), "epoch": 0}
    best_state, history, since_best = None, [], 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(_logits(model, x, edge_index)[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = _logits(model, x, edge_index)
            pred = out.argmax(1)
            train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
            val_loss = F.cross_entropy(out[val_mask], y[val_mask]).item()
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})

        if val_acc > best["val_accuracy"] or (val_acc == best["val_accuracy"] and val_loss < best["val_loss"]):
            best = {"val_accuracy": val_acc, "val_loss": val_loss, "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return best, history, best_state


@torch.no_grad()
def predict(model, x, edge_index):
    model.eval()
    return _logits(model, x, edge_index).argmax(1)


def run_once(data, edge_index, model_name, hparams, seed, train_cfg, evaluate_test=True):
    """One seeded training run. Returns a record with selection details and (optionally) test results."""
    set_seed(seed, train_cfg.get("num_threads", 1))
    model = build_model(model_name, data.num_features, int(data.y.max()) + 1, hparams)
    graph = edge_index if model.uses_graph else None
    start = time.perf_counter()
    best, history, state = fit(model, data.x, graph, data.y, data.train_mask, data.val_mask,
                               hparams["lr"], hparams["weight_decay"],
                               train_cfg["max_epochs"], train_cfg["patience"])
    record = {
        "best_epoch": best["epoch"], "stopped_epoch": history[-1]["epoch"],
        "best_val_accuracy": best["val_accuracy"], "best_val_loss": best["val_loss"],
        "train_seconds": time.perf_counter() - start, "history": history,
    }
    if not evaluate_test:
        return record, model, state

    # Model selection is finished; the test mask is used here for the first and only time.
    pred = predict(model, data.x, graph)
    test = data.test_mask
    metrics = classification_metrics(data.y[test].numpy(), pred[test].numpy())
    val_metrics = classification_metrics(data.y[data.val_mask].numpy(), pred[data.val_mask].numpy())
    record.update(test=metrics, val=val_metrics, predictions=pred.numpy())
    return record, model, state
