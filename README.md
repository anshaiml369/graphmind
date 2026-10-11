# GraphMind — citation-graph topic classification

A portfolio project that began from a YouTube tutorial and was extended with a graph-exploration research interface, live re-evaluation of the exported ONNX model, an edge-ablation experiment and documented limitations. In short: a two-layer graph convolutional network (GCN) that classifies papers in the
Cora citation network into seven topics, served with ONNX Runtime behind FastAPI, with a research-oriented
HTML/CSS/JavaScript interface.

## What the interface does

| Section | What it shows | Data source |
|---|---|---|
| Graph explorer | 1- or 2-hop citation neighbourhood of any of the 2,708 papers, prediction, class probabilities, ground truth, data split, and an edge ablation for that paper | `POST /predict/cora_node` (existing) + `GET /api/ui/node/{id}` |
| Edge experiment | Build a small graph from real Cora feature vectors and compare predictions with your edges vs. self-loops only | `POST /predict` (existing), called twice |
| Evaluation | Test/val/train accuracy, macro-F1, confusion matrix, per-class metrics of the shipped ONNX model; Random Forest baseline; edge-ablation result | Live: `GET /api/ui/evaluation`. Recorded: `results/notebook_results.json` (copied from notebook outputs) |
| Method | Architecture, dataset statistics, ONNX I/O signature, limitations, reproduction steps | `GET /api/ui/overview` + notebook |

Live results reproduced by this server from `simple_gcn_cora.onnx`: **test accuracy 0.792, macro-F1 0.782**
(identical to the notebook), and **0.678** accuracy when every citation edge is replaced by a self-loop.

## Architecture

```
static/                 HTML/CSS/vanilla JS frontend (served by FastAPI)
  index.html
  css/core.css          shared design system (synced from ../../_design_system)
  css/app.css           project-specific styling
  js/core.js            shared helpers (fetch, tabs, charts, markdown)
  js/app.js             explorer, experiment and evaluation logic
main.py                 EXISTING API + ONNX inference (unchanged)
ui_api.py               NEW thin adapter: imports main.app, adds read-only /api/ui/* endpoints
results/notebook_results.json   recorded metrics, with cell references
simple_gcn_cora.onnx(.data)     model weights (unchanged)
data/Planetoid/Cora     dataset (unchanged)
cora_citation_network_classification_(1).ipynb   training notebook (unchanged)
```

`ui_api.py` never re-implements inference: every prediction it reports comes from `main.run_model()`.

## Run locally

```bash
cd DL/GNN_RESEARCH
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn ui_api:app --port 8000
```

Open <http://localhost:8000>. API docs: <http://localhost:8000/docs>.

`uvicorn main:app` still works exactly as before; the UI then shows predictions but explains that the graph and
evaluation panels need `ui_api:app`. The first request takes a few seconds while Cora is loaded and the model is
evaluated once over the full graph (results are cached in memory).

## API

Existing (unchanged): `GET /health`, `GET /info`, `POST /predict`, `POST /predict/cora_node`.
Note: the existing response field is spelled `probabilites`; the frontend reads it as-is.

Added by `ui_api.py` (read-only): `GET /api/ui/overview`, `GET /api/ui/evaluation`, `GET /api/ui/samples`,
`GET /api/ui/node/{id}?hops=1|2&max_nodes=5..200`.

## Limitations

- Single training run without a fixed PyTorch seed; no variance across seeds is reported.
- `best_state = model.state_dict()` in the notebook stores references, so the exported weights are the
  final-epoch (35) weights, not the best-validation weights (validation accuracy 0.792 vs. logged best 0.812).
- Training accuracy is 1.000 on 140 labelled nodes (overfitting).
- Transductive: test nodes' features and edges are present at inference time.
- Planetoid Cora has no raw text or vocabulary, so the UI cannot show words or accept free text.
- The notebook text says the Random Forest also uses degree features; the code uses TF-IDF word features only.

## Deployment notes

- No secrets or API keys are required.
- `torch` and `torch-geometric` are only needed to load the Planetoid dataset (`/predict/cora_node` and `/api/ui/*`).
- The ONNX weights (~250 KB) and processed Cora data (~15 MB) are small enough to commit, but check the Cora
  licence/terms for redistribution; alternatively let PyTorch Geometric download it on first start.

References: McCallum et al. (2000); Yang, Cohen & Salakhutdinov (2016); Kipf & Welling (2017); Fey & Lenssen (2019).
