# How much does citation structure contribute to topic classification on Cora?

A controlled, seed-replicated comparison of GCN, GAT and a feature-only MLP, with retrained graph ablations.

Every number in this report is copied from files in `results/`, which were produced by the scripts in
`experiments/` (commands in §12). Labels used below: **Verified** = measured in this study;
**Interpretation** = our reading of verified results; **Hypothesis / future work** = not tested here.

---

## 1. Research question and motivation

> How much does graph structure contribute to node-classification performance compared with feature-only
> learning, and how sensitive are the results to model architecture and random initialisation?

GraphMind's earlier evidence for "the graph helps" was one training run of one GCN (test accuracy 0.792) and an
ablation in which the same trained model was evaluated with its edges removed (0.678). That design cannot separate
the value of the graph from a train/test mismatch, says nothing about architecture or seed variation, and — as
shown in §4 — was affected by a checkpointing bug. This study replaces it with a controlled comparison.

## 2. Dataset and provenance

* **Cora** citation network, loaded with PyTorch Geometric's `Planetoid` (files committed in `data/Planetoid/Cora`,
  originally from Yang, Cohen & Salakhutdinov, 2016, derived from McCallum et al., 2000).
* 2,708 papers (nodes), 5,278 undirected citation links (10,556 directed edges, no self-loops, symmetric),
  1,433 binary bag-of-words features per paper, 7 topics.
* Task: **transductive semi-supervised node classification** with the Planetoid **public split** — 140 labelled
  training nodes (20 per class), 500 validation, 1,000 test; the remaining 1,068 nodes are unlabelled but their
  features and links are visible. Splits are disjoint (verified).
* Class sizes (all nodes): 351 / 217 / 418 / 818 / 426 / 298 / 180. Test set: 130 / 91 / 144 / 319 / 149 / 103 / 64.
  Training is balanced; validation and test are not.
* Edge homophily (fraction of links joining same-topic papers): **0.810**; chance level for these class
  proportions (Σ p_c²) is 0.180.

## 3. Original implementation (audit)

| Component | Original GraphMind | Location |
|---|---|---|
| Model | 2-layer GCN, 1433→32→7, ReLU, dropout 0.5 after layer 1 | notebook cell 22 |
| Training | Adam lr 0.01, no weight decay, ≤200 epochs, full batch | cell 26 |
| Early stopping | validation accuracy, patience 20 | cell 26 |
| Test evaluation | once, after training | cell 26 |
| Seeds | none set for PyTorch | — |
| Baseline | Random Forest on TF-IDF-reweighted word indicators (train nodes only) | cells 18–19 |
| Edge ablation | shipped ONNX model run with every edge replaced by a self-loop | `ui_api.py`, `_load()` |
| Reported | test acc 0.792, macro-F1 0.782; edge ablation 0.678; RF 0.581 | `results/notebook_results.json` |

**Verified** (`results/original_onnx_check.json`): the shipped `simple_gcn_cora.onnx` gives test accuracy
**0.792**, macro-F1 **0.782**, and **0.678** with edges replaced by self-loops — the earlier figures reproduce
exactly. Its validation accuracy is **0.792**, although the notebook logged a best validation accuracy of 0.812.

## 4. Identified problems and corrections

| ID | Problem | Evidence | Impact | Correction |
|---|---|---|---|---|
| **B1** | Checkpoint stores references: `best_state = model.state_dict()`; later optimizer steps overwrite those tensors in place, so `load_state_dict(best_state)` restores the **final-epoch** weights. | Notebook protocol re-run on 5 seeds (`results/original_protocol_results.csv`): restored model's validation accuracy 0.764–0.778 vs. logged best 0.788–0.794 in **every** seed; shipped model val 0.792 vs. logged 0.812. | Early stopping did not select what it claimed. Test effect here was modest: mean test accuracy 0.790 (bug) vs. 0.799 (fixed); fixed ≥ bug in 5/5 seeds. | Deep copy `{k: v.detach().clone()}` (`experiments/train.py`); unit test `test_checkpoint_is_best_validation_epoch`. |
| **B2** | The 0.678 "no citation links" figure is an **inference-time** intervention on a model trained with edges. | `ui_api.py` `_load()`. | It measures how a graph-trained model degrades under a distribution shift, not what can be learned without the graph. It **under**states the graph's contribution: a GCN trained with edges and evaluated without them scores 0.696 ± 0.013 (our runs), while models trained without edges score 0.571–0.583 (§9). | Ablations are **retrained** on each graph; the inference-time number is kept as a separate column for comparison. |
| **B3** | No random seeds; one run. | Notebook. | No variance estimate; not reproducible. | Python/NumPy/PyTorch seeded, deterministic kernels, 1 CPU thread; 5 evaluation seeds per condition; reruns verified bit-identical (§10). |
| **B4** | Hyperparameters were not tuned, and the comparison baseline (RF) was not tuned either. | Notebook. | Comparison may reflect tuning effort, not model class. | Same grid size (16 configs) for every model, tuned on validation only with separate tuning seeds. |
| **B5** | `CORA_CLASSES` in `main.py` maps label ids to topic names alphabetically. PyG's class sizes [351, 217, 418, 818, 426, 298, 180] match Cora's published topic sizes only in the order Theory, Reinforcement_Learning, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Case_Based, Rule_Learning (Neural_Networks is the 818-paper class). | Class counts. | The web app showed **wrong topic names** (e.g. label 3 shown as "Probabilistic_Methods"). Metrics are unaffected (they use ids). | Correct names used in `experiments/common.py`; **fixed in the app** in commit 8042c6a (`main.py`, `static/js/app.js`). |
| B6 | Notebook text says the Random Forest also uses degree features; the code uses word features only. | Cells 15–19. | Documentation error. | Noted; RF is not part of this study. |

**Leakage checks (verified).**
* Training updates use `train_mask` only. Early stopping, checkpoint selection and hyperparameter selection use
  validation accuracy (ties: validation loss). `fit()` never receives a test mask; the test set is evaluated once,
  after selection. Unit test `test_selection_ignores_test_labels` scrambles every test label and confirms that
  training curves, selected epoch and selected weights are bit-identical.
* Tuning seeds (1000–1002) and evaluation seeds (0–4) are disjoint; `tuning_results.csv` contains no test columns.
* The MLP's `forward(x)` has no edge argument, and `run_once` passes no graph to it; a unit test confirms the MLP
  produces identical results with the real graph or 5,000 random edges.
* **Duplicates:** 11 groups of papers share identical feature vectors (27 nodes). One test node (2144) duplicates a
  training node (137), same label; three validation–test groups exist. All duplicates within a group share a label.
  Effect on a 1,000-node test set is ≤ 0.1–0.4 percentage points; the split was kept unchanged and this is reported.
* **Graph–label leakage:** in the transductive protocol, test nodes' features and links are visible during training
  but their labels are not. This is the standard Planetoid protocol, applies equally to GCN and GAT, and is the
  quantity under study; the MLP sees test features only at prediction time. Results do **not** transfer to an
  inductive setting (§11).
* Feature preprocessing: none (raw binary indicators, as in the original notebook), so no statistics are fitted on
  test data. (The original RF's TF-IDF used all nodes' features — label-free, transductive; not used here.)

## 5. Experimental design and baselines

| Model | Layers (all: dropout → layer 1 → activation → dropout → layer 2) | Uses graph | Model-specific settings |
|---|---|---|---|
| **MLP** | Linear → ReLU → Linear | No | — |
| **GCN** (Kipf & Welling, 2017) | GCNConv → ReLU → GCNConv | Yes | symmetric normalisation, self-loops added by the layer |
| **GAT** (Veličković et al., 2018) | GATConv (8 heads, concatenated) → ELU → GATConv (1 head) | Yes | ELU as in the GAT paper; dropout also applied to attention coefficients |

Graph conditions (GCN and GAT retrained on each; nodes, features, labels and splits never change):

| Condition | Definition | Seeding |
|---|---|---|
| `original` | Cora graph as distributed | — |
| `no_edges` | every citation link removed (GNN layers still add self-loops, so GCN/GAT reduce to per-node transforms) | — |
| `drop50` | 50% of undirected links removed uniformly at random, both directions | graph seed = run seed |
| `rewired` | degree-preserving random rewiring (`networkx.double_edge_swap`, 10×|E| swaps): every paper keeps its number of links, but its neighbours become random | graph seed = run seed |

`rewired` is the key control: it keeps the amount and degree distribution of structure and destroys only *which*
papers are linked. Graph statistics (`results/graph_stats_summary.csv`, mean over the 5 seeded graphs):

| Condition | Undirected edges | Density | Mean degree | Isolated nodes | Components | Largest comp. | Edge homophily | Avg. clustering |
|---|---|---|---|---|---|---|---|---|
| original | 5,278 | 0.00144 | 3.90 | 0 | 78 | 2,485 | 0.810 | 0.241 |
| drop50 | 2,639 | 0.00072 | 1.95 | 493 ± 15 | 648 ± 6 | 1,737 ± 39 | 0.806 ± 0.006 | 0.066 |
| rewired | 5,278 | 0.00144 | 3.90 | 0 | 13 ± 4 | 2,683 ± 9 | 0.169 ± 0.004 | 0.008 |
| no_edges | 0 | 0 | 0 | 2,708 | 2,708 | 1 | undefined | 0 |

Rewired homophily (0.169) is at the chance level (0.180), confirming that the label–structure link is destroyed;
`drop50` keeps homophily but halves the degree and isolates 18% of papers.

## 6. Train / validation / test protocol

1. **Split:** Planetoid public split, unchanged (140 / 500 / 1,000).
2. **Tuning** (`experiments/tune.py`): for each model, a 16-configuration grid (lr ∈ {0.01, 0.005},
   weight decay ∈ {5e-4, 5e-3}, dropout ∈ {0.5, 0.8}, hidden ∈ {16, 64} for MLP/GCN or {4, 8} per head × 8 heads for
   GAT), each trained with 3 tuning seeds on the **original graph**. Selection: highest mean validation accuracy.
   The same tuned hyperparameters are then reused on every graph condition, so that only the graph changes.
   (A supplementary analysis re-tunes per condition, §9.3.)
3. **Training:** full batch, Adam, cross-entropy on training nodes, ≤ 500 epochs, early stopping with patience 100
   on validation accuracy; the best-validation checkpoint is deep-copied and restored. No run reached the 500-epoch
   cap (latest stop: epoch 476).
4. **Testing:** the restored checkpoint is evaluated once on the 1,000 test nodes. Metrics: accuracy, macro-F1,
   weighted-F1, per-class precision/recall/F1.
5. **Seeds:** evaluation seeds 0, 1, 2, 3, 4 for every model × condition (45 runs). A run with seed *s* uses the
   same initial weights for the same architecture across graph conditions, so ablation comparisons are paired.

## 7. Hyperparameters and compute environment

Selected hyperparameters (`results/best_hparams.json`; mean validation accuracy over tuning seeds):

| Model | lr | weight decay | dropout | hidden | heads | mean val acc (tuning) |
|---|---|---|---|---|---|---|
| MLP | 0.005 | 0.005 | 0.8 | 64 | — | 0.598 |
| GCN | 0.01 | 0.005 | 0.5 | 64 | — | 0.809 |
| GAT | 0.005 | 0.005 | 0.8 | 8 per head | 8 | 0.814 |

Several selected values lie on the edge of the grid (weight decay 0.005 for all models; dropout 0.8 for MLP and
GAT), so better settings may exist outside it (§11).

Environment: macOS (Darwin 25.6), Apple M2, CPU only, 1 PyTorch thread; Python 3.11.14, torch 2.14.1,
torch-geometric 2.8.0.post1, scikit-learn 1.7.2, SciPy 1.17.1, NetworkX 3.6.1, Matplotlib 3.11.2.
Compute: 144 tuning runs, 45 main runs (361 s of training in total), 3 × 10 original-protocol runs, and
192 + 20 supplementary runs.

## 8. Results (all completed runs)

**Verified.** All 45 planned main runs completed; none were skipped. Per-run values: `results/per_seed_results.csv`
(selected epoch, validation and test metrics), per-class: `results/per_class_results.csv`, training curves:
`results/logs/<run_id>.json`.

### 8.1 Individual runs on the original graph (test set, n = 1,000)

| Model | Seed | Selected epoch | Val acc | Test acc | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|---|
| MLP | 0 | 67 | 0.604 | 0.572 | 0.559 | 0.574 |
| MLP | 1 | 39 | 0.618 | 0.599 | 0.574 | 0.602 |
| MLP | 2 | 70 | 0.598 | 0.586 | 0.573 | 0.586 |
| MLP | 3 | 66 | 0.590 | 0.573 | 0.562 | 0.576 |
| MLP | 4 | 41 | 0.602 | 0.584 | 0.567 | 0.586 |
| GCN | 0 | 97 | 0.800 | 0.824 | 0.815 | 0.824 |
| GCN | 1 | 285 | 0.802 | 0.831 | 0.820 | 0.832 |
| GCN | 2 | 70 | 0.806 | 0.826 | 0.817 | 0.826 |
| GCN | 3 | 62 | 0.798 | 0.818 | 0.809 | 0.819 |
| GCN | 4 | 97 | 0.804 | 0.814 | 0.806 | 0.815 |
| GAT | 0 | 94 | 0.806 | 0.806 | 0.801 | 0.807 |
| GAT | 1 | 245 | 0.810 | 0.812 | 0.808 | 0.813 |
| GAT | 2 | 276 | 0.824 | 0.834 | 0.827 | 0.835 |
| GAT | 3 | 332 | 0.814 | 0.810 | 0.802 | 0.811 |
| GAT | 4 | 126 | 0.806 | 0.825 | 0.817 | 0.826 |

### 8.2 Mean ± SD over 5 seeds (`results/summary.csv`)

SD is the sample standard deviation over seeds and reflects initialisation/dropout randomness only.

| Model | Graph | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|
| MLP | (none) | 0.583 ± 0.011 | 0.567 ± 0.007 | 0.585 ± 0.011 |
| GCN | original | **0.823 ± 0.007** | **0.814 ± 0.006** | 0.823 ± 0.007 |
| GAT | original | **0.817 ± 0.012** | **0.811 ± 0.011** | 0.819 ± 0.012 |

### 8.3 Model comparisons (`results/comparisons.csv`)

Seed-level CI: Welch t-interval over the 5 + 5 runs. Test-sample CI: paired bootstrap over the 1,000 test nodes of
the seed-averaged accuracy difference (10,000 resamples). p-values: Welch t-test, Holm-corrected over the three
comparisons.

| Comparison | Δ accuracy | 95% CI (seeds) | 95% CI (test sample) | Holm p | Seeds where A > B | Δ macro-F1 |
|---|---|---|---|---|---|---|
| GCN − MLP | **+0.240** | [0.226, 0.254] | [0.212, 0.268] | 6.3e-09 | 5/5 | +0.247 |
| GAT − MLP | **+0.235** | [0.218, 0.251] | [0.206, 0.263] | 2.7e-09 | 5/5 | +0.244 |
| GCN − GAT | +0.005 | [−0.009, 0.020] | [−0.009, 0.019] | 0.42 | 3/5 | +0.002 |

The Welch test assumes approximately normal seed-to-seed variation; with five runs per group this cannot be checked
and the p-values are coarse. The conclusions rest on the size and consistency of the effects (every GNN run beats
every MLP run by more than 0.2), not on p-values.

### 8.4 Per-class F1 on the original graph (mean ± SD over seeds, `results/per_class_summary.csv`)

| Topic (test n) | MLP | GCN | GAT |
|---|---|---|---|
| Theory (130) | 0.487 ± 0.022 | 0.718 ± 0.006 | 0.723 ± 0.017 |
| Reinforcement_Learning (91) | 0.601 ± 0.021 | 0.848 ± 0.005 | 0.830 ± 0.013 |
| Genetic_Algorithms (144) | 0.644 ± 0.020 | 0.894 ± 0.009 | 0.916 ± 0.011 |
| Neural_Networks (319) | 0.623 ± 0.023 | 0.846 ± 0.008 | 0.831 ± 0.014 |
| Probabilistic_Methods (149) | 0.605 ± 0.013 | 0.824 ± 0.024 | 0.808 ± 0.018 |
| Case_Based (103) | 0.531 ± 0.015 | 0.798 ± 0.015 | 0.801 ± 0.009 |
| Rule_Learning (64) | 0.480 ± 0.026 | 0.767 ± 0.014 | 0.770 ± 0.015 |

The graph improves every topic. Theory remains the hardest class for all models. For Neural_Networks, the largest
class, the GNNs have high precision (0.89–0.92) and lower recall (0.76–0.81): these papers are under-predicted,
plausibly because training is class-balanced (20 per class) while the test set is not.

Plots: `results/plots/per_seed_results.png` (every run, with mean ± SD), `results/plots/graph_ablation.png`,
`results/plots/training_curves_seed0.png` (train/validation curves with the selected epoch marked).

## 9. Graph-structure ablation

### 9.1 Retrained ablation (`results/summary.csv`, `results/ablation.csv`)

| Graph condition | GCN test acc | GAT test acc | GCN macro-F1 | GAT macro-F1 |
|---|---|---|---|---|
| original | 0.823 ± 0.007 | 0.817 ± 0.012 | 0.814 ± 0.006 | 0.811 ± 0.011 |
| drop50 (50% links removed) | 0.762 ± 0.013 | 0.760 ± 0.019 | 0.751 ± 0.013 | 0.746 ± 0.020 |
| rewired (same degrees, random neighbours) | 0.301 ± 0.008 | 0.271 ± 0.024 | 0.287 ± 0.005 | 0.228 ± 0.042 |
| no_edges | 0.571 ± 0.021 | 0.496 ± 0.019 | 0.557 ± 0.016 | 0.471 ± 0.021 |
| *MLP, no graph (reference)* | *0.583 ± 0.011* | | *0.567 ± 0.007* | |

Paired differences from the original graph (same seed ⇒ same initial weights; paired t-interval; Holm over the
ablation table):

| Model | Condition | Δ accuracy | 95% CI (seeds) | 95% CI (test sample) | Holm p | Seeds lower |
|---|---|---|---|---|---|---|
| GCN | drop50 | −0.060 | [−0.073, −0.048] | [−0.075, −0.046] | 5.7e-04 | 5/5 |
| GCN | rewired | −0.522 | [−0.540, −0.504] | [−0.546, −0.497] | 1.2e-06 | 5/5 |
| GCN | no_edges | −0.251 | [−0.281, −0.222] | [−0.280, −0.223] | 8.9e-05 | 5/5 |
| GAT | drop50 | −0.058 | [−0.083, −0.033] | [−0.072, −0.043] | 6.2e-03 | 5/5 |
| GAT | rewired | −0.546 | [−0.575, −0.517] | [−0.570, −0.521] | 5.4e-06 | 5/5 |
| GAT | no_edges | −0.321 | [−0.353, −0.289] | [−0.347, −0.295] | 5.8e-05 | 5/5 |

### 9.2 Inference-time removal (the original protocol) vs. retraining

| | GCN | GAT |
|---|---|---|
| Trained with edges, evaluated with edges | 0.823 ± 0.007 | 0.817 ± 0.012 |
| Trained with edges, **evaluated without** edges (original GraphMind protocol) | 0.696 ± 0.013 | 0.735 ± 0.007 |
| **Retrained** without edges | 0.571 ± 0.021 | 0.496 ± 0.019 (0.576 ± 0.009 re-tuned, §9.3) |

The original single-run version of the first two rows (0.792 → 0.678) reproduces the same pattern. The inference-time
number is higher than a model that never saw the graph, so the original ablation understated the graph's
contribution (≈ 0.11 vs. ≈ 0.24 accuracy).

### 9.3 Supplementary: hyperparameters re-tuned for each ablated graph

Because §9.1 reuses hyperparameters tuned on the original graph, the ablated conditions could be disadvantaged.
GCN and GAT were re-tuned on `no_edges` and `rewired` with the same grid and protocol
(`experiments/supplementary_retune.py`, `results/supplementary_retuned_summary.csv`):

| Model | Condition | Re-tuned hyperparameters | Test acc (re-tuned) | Test acc (§9.1) | Δ vs. MLP (95% CI, seeds) |
|---|---|---|---|---|---|
| GCN | no_edges | lr 0.005, wd 0.005, dropout 0.5, hidden 64 | 0.576 ± 0.013 | 0.571 | −0.007 [−0.024, 0.011] |
| GAT | no_edges | lr 0.005, wd 0.005, dropout 0.5, hidden 4×8 | 0.576 ± 0.009 | 0.496 | −0.006 [−0.021, 0.008] |
| GCN | rewired | unchanged (same as original) | 0.301 ± 0.008 | 0.301 | −0.282 [−0.297, −0.268] |
| GAT | rewired | lr 0.01, wd 0.005, dropout 0.8, hidden 8×8 | 0.282 ± 0.030 | 0.271 | −0.301 [−0.338, −0.264] |

**Interpretation.** GAT's large `no_edges` penalty in §9.1 was largely a hyperparameter artefact: with only
self-loops, an attention dropout of 0.8 deletes a node's single message 80% of the time during training. Re-tuned,
both GNNs without edges match the MLP — as expected, since a GCN/GAT layer with only self-loops is a per-node linear
transform. The `rewired` collapse is **not** a tuning artefact.

### 9.4 What the ablation establishes — and what it does not

**Supported by these results (Interpretation):**
1. On this split, adding Cora's citation links to a feature-based model raises test accuracy by about 0.24 (≈ 41%
   relative) for both GCN and GAT, consistently across all seeds and well beyond test-sample uncertainty.
2. The gain depends on **who is linked to whom**, not on having links or a particular degree distribution: keeping
   every paper's degree but randomising its neighbours (homophily 0.81 → 0.17) makes the GNNs far *worse* than using
   no graph at all (0.27–0.30 vs. 0.58). Aggregating random neighbours mixes in other topics' features.
3. Performance degrades gradually with fewer true links: removing half of them costs ≈ 0.06 accuracy, while
   homophily of the remaining links is unchanged — consistent with fewer informative neighbours (and 18% isolated
   nodes) rather than lower-quality links.
4. Architecture matters little here: GCN and GAT are indistinguishable (Δ = 0.005, CI spans zero, 3/5 seeds).
5. Seed sensitivity is small relative to the effects (SD ≤ 0.012 on the original graph). The test-sample CIs are
   wider than the seed CIs: evaluation on one fixed test set is a larger source of uncertainty than initialisation.

**Not established (threats; see §11):**
* *Causality in general.* These are interventions on one dataset's edges under one split; they show that this model
  family exploits homophilous citation links on Cora, not that "graph structure helps" for other graphs or tasks.
* *Rewiring changes more than homophily.* It also removes clustering (0.24 → 0.01) and merges components (78 → 13),
  so the collapse is attributed to label-correlated structure as a whole, not to homophily alone.
* *Dropping edges changes several things at once* (degree, isolation, components).

## 10. Verification performed

| Check | Result |
|---|---|
| Unit and consistency tests (`python -m unittest discover tests -v`) | 14/14 pass |
| MLP never receives edges | signature has no edge argument; identical output with the real graph or random edges |
| Early stopping / checkpoint read validation only | `fit()` has no test argument; scrambling all test labels leaves curves, selected epoch and weights bit-identical |
| Test data unused during selection | tuning file has no test columns; tuning and evaluation seeds disjoint |
| Checkpoint = best validation epoch | restored model's validation accuracy equals logged best |
| Determinism | full experiment re-run from scratch: all 45 runs identical in every metric and every node-level prediction |
| Summary statistics | recomputed from `per_seed_results.csv` (test `test_summary_recomputes_from_per_seed`) |
| Reported results match logs | CSV ↔ per-run JSON logs ↔ stored predictions agree to 1e-9 for every run |
| Graph interventions | `drop50` ⊂ original, symmetric, no duplicates; `rewired` preserves every node's degree, < 5% of original edges kept |
| Original web app | still serves predictions and reproduces 0.792 / 0.678 (`uvicorn ui_api:app`) |
| Plots | inspected: axes labelled, error bars labelled as mean ± SD over 5 seeds, individual seeds shown |

## 11. Limitations and threats to validity

* **One dataset, one split.** Seeds vary initialisation and dropout only. The public split's 140 training nodes are
  fixed; results may differ on random splits (Shchur et al., 2018, show split choice can reorder GNN rankings).
* **Transductive only.** Test nodes' features and links are present during training. Inductive conclusions (new
  papers arriving without labels) are not supported.
* **Small grid; values at grid edges.** Weight decay 0.005 was chosen for all models and dropout 0.8 for MLP and
  GAT, so better settings may lie outside the grid. Hyperparameters were tuned for the original graph (addressed for
  `no_edges`/`rewired` in §9.3 but not for `drop50`).
* **n = 5 seeds.** Enough to show that large effects are consistent; too few to resolve small differences such as
  GCN vs. GAT, or to check normality assumptions behind t-intervals.
* **Selection noise.** Validation curves plateau with ±1–2 point fluctuations (see training-curve plot); choosing
  the maximum of a noisy curve on 500 nodes makes the selected validation accuracy optimistically biased (test
  accuracy is unaffected because it is measured separately).
* **Duplicate feature vectors** across splits (one train–test pair, three val–test groups) — negligible effect,
  not removed so as not to alter the split.
* **Class imbalance:** training is balanced (20 per class) but test is not; Neural_Networks is under-predicted.

## 12. Future experiments and reproduction

**Hypotheses / future work (not tested here):**
1. **Random splits** (e.g. 20 random 20-per-class splits × seeds) to separate split variance from seed variance —
   the most important next experiment, because the test-sample CIs show a single test set is the dominant source of
   uncertainty.
2. **Controlled homophily sweep:** rewire a fraction *p* ∈ {0, 0.1, …, 1} of edges to trace accuracy as a function
   of edge homophily, separating homophily from clustering and component structure.
3. **Label-propagation and feature-propagation baselines** (no learned weights) to measure how much of the gain is
   label smoothing along links.
4. Other citation graphs (CiteSeer, PubMed) and a heterophilous graph to test whether conclusions generalise.

**Reproduction** (from the repository root; about 35 minutes on one CPU core):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-experiments.txt
python -m experiments.reproduce_original      # verify 0.792 / 0.678 and measure bug B1
python -m experiments.tune                    # validation-only tuning -> results/best_hparams.json
python -m experiments.run_experiment          # 45 runs -> results/per_seed_results.csv
python -m experiments.supplementary_retune    # §9.3
python -m experiments.analyze                 # summary, comparisons, ablation, plots
python -m unittest discover tests -v          # 14 checks
```

## References

Fey & Lenssen (2019), *Fast graph representation learning with PyTorch Geometric*. Kipf & Welling (2017),
*Semi-supervised classification with graph convolutional networks*, ICLR. McCallum et al. (2000), *Automating the
construction of internet portals with machine learning*. Shchur et al. (2018), *Pitfalls of graph neural network
evaluation*. Veličković et al. (2018), *Graph attention networks*, ICLR. Yang, Cohen & Salakhutdinov (2016),
*Revisiting semi-supervised learning with graph embeddings*, ICML.
