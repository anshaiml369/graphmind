/* GraphMind — frontend logic. Talks to main.py (predictions) and ui_api.py (graph + evaluation). */
(function () {
  "use strict";
  const { $, $$, el, svg, api, fmt, renderBars, renderMatrix, plotXY, legend, setBusy, setStatus, callout, toast, copyText } = window.RUI;

  const FEATURE_DIM = 1433;
  const MAX_EXP_NODES = 12;
  // Colour-blind-aware categorical palette (Paul Tol "muted" + Okabe–Ito orange), one per Cora class id.
  const COLORS = ["#CC6677", "#999933", "#332288", "#44AA99", "#E69F00", "#88CCEE", "#882255"];
  let CLASSES = {
    0: "Theory", 1: "Reinforcement_Learning", 2: "Genetic_Algorithms", 3: "Neural_Networks",
    4: "Probabilistic_Methods", 5: "Case_Based", 6: "Rule_Learning",
  };
  const name = (id) => fmt.label(CLASSES[id] ?? `class ${id}`);

  const state = {
    adapter: null, // true when /api/ui/* is available
    numNodes: 2708,
    hops: 2,
    colorBy: "true",
    nbr: null,
    lastPrediction: null,
    exp: { nodes: [], edges: [] },
  };

  /* --------------------------------------------------------------------- */
  /* Boot                                                                  */
  /* --------------------------------------------------------------------- */
  async function boot() {
    renderLegend();
    try {
      const health = await api("/health", { timeout: 8000 });
      setStatus($("#api-status"), "ok", `API online · ${(health.providers || []).join(", ") || "ready"}`);
    } catch (err) {
      setStatus($("#api-status"), "err", "API unreachable");
      $("#result-body").replaceChildren(callout("err", "Backend not reachable", [err.message, "Start it with: uvicorn ui_api:app --port 8000"]));
      return;
    }
    try {
      const overview = await api("/api/ui/overview", { timeout: 60000 });
      state.adapter = true;
      CLASSES = overview.classes;
      state.numNodes = overview.num_nodes;
      renderOverview(overview);
      loadSamples();
      loadEvaluation();
      classify(parseInt($("#node-id").value, 10));
    } catch (err) {
      state.adapter = false;
      const msg = err.status === 404
        ? ["The server is running main.py only, so graph structure and evaluation endpoints are unavailable.", "Restart with: uvicorn ui_api:app --port 8000 — prediction still works."]
        : [err.message];
      for (const id of ["#eval-root", "#dataset-body", "#io-body"]) $(id).replaceChildren(callout("warn", "UI adapter not available", msg));
      $("#graph-empty").replaceChildren(el("strong", { text: "Graph view unavailable" }), el("span", { text: msg[0] }));
      $("#samples-class").replaceChildren(el("span", { class: "hint", text: "Requires ui_api.py" }));
    }
  }

  function renderLegend() {
    $("#graph-legend").replaceChildren(
      ...Object.keys(CLASSES).map((id) =>
        el("span", { class: "legend-item" }, [el("span", { class: "legend-swatch dot", style: { background: COLORS[id] }, "aria-hidden": "true" }), name(id)])
      )
    );
  }

  function renderOverview(o) {
    $("#meta-graph").innerHTML = `Graph <strong>${fmt.num(o.num_nodes)} papers · ${fmt.num(o.num_undirected_edges)} citation links</strong>`;
    const body = $("#dataset-body");
    const maxCount = Math.max(...o.class_counts);
    const bars = el("div");
    renderBars(bars, o.class_counts.map((c, i) => ({ label: name(i), value: c / maxCount, color: COLORS[i], count: c })),
      { valueFormat: (v) => fmt.num(Math.round(v * maxCount)), ariaLabel: "Papers per topic" });
    body.replaceChildren(
      el("div", { class: "grid-3" }, [
        stat("Papers (nodes)", fmt.num(o.num_nodes)),
        stat("Citation links", fmt.num(o.num_undirected_edges), `${fmt.num(o.num_directed_edges)} directed entries`),
        stat("Word indicators", fmt.num(o.num_features), `${o.features_per_node.mean.toFixed(1)} active per paper on average`),
      ]),
      el("dl", { class: "kv" }, [
        el("dt", { text: "Split (labelled)" }), el("dd", { text: `${o.split_sizes.train} train · ${o.split_sizes.val} validation · ${o.split_sizes.test} test` }),
        el("dt", { text: "Node degree" }), el("dd", { text: `mean ${o.degree.mean.toFixed(2)} · median ${o.degree.median} · max ${o.degree.max}` }),
      ]),
      el("p", { class: "subhead", text: "Papers per topic (all 2,708 nodes)" }),
      bars
    );
    const ioRows = [...o.model.inputs.map((i) => ["input", i]), ...o.model.outputs.map((i) => ["output", i])];
    $("#io-body").replaceChildren(
      el("div", { class: "table-wrap" }, [
        el("table", { class: "table" }, [
          el("thead", {}, [el("tr", {}, ["Role", "Name", "Shape", "Type"].map((h) => el("th", { text: h })))]),
          el("tbody", {}, ioRows.map(([role, io]) => el("tr", {}, [
            el("td", { text: role }), el("td", { class: "mono", text: io.name }),
            el("td", { class: "mono", text: `[${io.shape.join(", ")}]` }), el("td", { class: "mono", text: io.type }),
          ]))),
        ]),
      ]),
      el("p", { class: "hint", style: { marginTop: "12px" }, text: `${o.model.file} · ${(o.model.size_bytes / 1024).toFixed(0)} KB on disk · providers: ${o.model.providers.join(", ")}` })
    );
  }

  function stat(label, value, sub, prov) {
    return el("div", { class: "stat" }, [
      el("div", { class: "stat-label", text: label }),
      el("div", { class: "stat-value", text: value }),
      sub ? el("div", { class: "stat-sub", text: sub }) : null,
      prov ? el("div", { class: "provenance", text: prov }) : null,
    ]);
  }

  async function loadSamples() {
    try {
      const s = await api("/api/ui/samples", { timeout: 60000 });
      const chip = (node, cls, extra) => el("button", { type: "button", class: "chip", onclick: () => selectNode(node), title: extra || `Node ${node}, true topic ${name(cls)}` }, [
        el("span", { class: "bar-swatch", style: { background: COLORS[cls] }, "aria-hidden": "true" }),
        el("span", { class: "chip-id", text: `#${node}` }),
        el("span", { text: extra ? extra : name(cls) }),
      ]);
      const idOf = (n) => Number(Object.keys(CLASSES).find((k) => CLASSES[k] === n));
      $("#samples-class").replaceChildren(...s.per_class.map((p) => chip(p.node, idOf(p.true_class))));
      $("#samples-errors").replaceChildren(...s.misclassified.slice(0, 6).map((p) =>
        chip(p.node, idOf(p.true_class), `${fmt.label(p.true_class)} → ${fmt.label(p.predicted_class)}`)));
    } catch (err) {
      $("#samples-class").replaceChildren(el("span", { class: "field-error", text: err.message }));
    }
  }

  /* --------------------------------------------------------------------- */
  /* Explorer                                                              */
  /* --------------------------------------------------------------------- */
  function validateNodeId(raw, errorEl, input) {
    const v = String(raw).trim();
    let msg = "";
    if (v === "") msg = "Enter a node index.";
    else if (!/^\d+$/.test(v)) msg = "Node index must be a whole number.";
    else if (+v < 0 || +v > state.numNodes - 1) msg = `Must be between 0 and ${state.numNodes - 1}.`;
    errorEl.textContent = msg;
    if (input) input.setAttribute("aria-invalid", msg ? "true" : "false");
    return msg ? null : +v;
  }

  function selectNode(id) {
    $("#node-id").value = id;
    classify(id);
    $("#explorer").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  let classifySeq = 0;
  async function classify(nodeId) {
    const id = validateNodeId(nodeId, $("#node-id-error"), $("#node-id"));
    if (id === null) return;
    const seq = ++classifySeq;
    const btn = $("#classify-btn");
    setBusy(btn, true, "Running…");
    $("#graph-stage").classList.add("is-loading");
    $("#result-node-badge").textContent = `node #${id}`;
    try {
      const predP = api("/predict/cora_node", { method: "POST", body: { node_indices: [id] }, timeout: 60000 });
      const nbrP = state.adapter ? api(`/api/ui/node/${id}?hops=${state.hops}&max_nodes=60`, { timeout: 60000 }) : Promise.resolve(null);
      const [pred, nbr] = await Promise.all([predP, nbrP]);
      if (seq !== classifySeq) return;
      state.nbr = nbr;
      state.lastPrediction = pred;
      renderPrediction(pred, nbr);
      if (nbr) { renderGraph(nbr); renderNeighbourTable(nbr); }
      $("#exp-add-nbrs").disabled = !nbr;
    } catch (err) {
      if (seq !== classifySeq) return;
      $("#result-body").replaceChildren(callout("err", "Prediction failed", [err.message]));
    } finally {
      if (seq === classifySeq) {
        setBusy(btn, false);
        $("#graph-stage").classList.remove("is-loading");
      }
    }
  }

  function renderPrediction(resp, nbr) {
    const p = resp.predictions[0];
    const probs = p.probabilites; // field name as returned by main.py
    const top = p.predicted_class_id;
    const body = $("#result-body");
    const center = nbr ? nbr.center : null;
    const bars = el("div");
    renderBars(bars, probs.map((v, i) => ({ label: name(i), value: v, color: COLORS[i], top: i === top })), { sort: true, ariaLabel: "Class probabilities", valueFormat: (v) => (v < 0.001 ? "<0.1%" : fmt.pct(v, 1)) });

    const children = [
      el("div", { class: "pred-hero" }, [
        el("div", {}, [
          el("p", { class: "subhead", text: "Predicted topic" }),
          el("div", { class: "pred-label" }, [el("span", { class: "bar-swatch", style: { background: COLORS[top] }, "aria-hidden": "true" }), name(top)]),
          el("p", { class: "pred-conf", text: `softmax p = ${probs[top].toFixed(3)}` }),
        ]),
      ]),
    ];

    if (center) {
      const correct = center.true_class_id === top;
      const splitText = {
        train: "Training node — its label was used to fit the model",
        val: "Validation node — used for early stopping",
        test: "Test node — held out from training",
        unlabelled: "Outside the train/val/test masks (label not used)",
      }[center.split];
      children.push(el("div", { class: "truth-row" }, [
        el("div", { class: "truth-cell" }, [
          el("div", { class: "stat-label", text: "Ground-truth label (Cora)" }),
          el("strong", { text: name(center.true_class_id) }), " ",
          el("span", { class: `badge ${correct ? "badge-ok" : "badge-err"}`, text: correct ? "match" : "mismatch" }),
        ]),
        el("div", { class: "truth-cell" }, [el("div", { class: "stat-label", text: "Data split" }), el("strong", { text: center.split }), el("div", { class: "hint", text: splitText })]),
      ]));
    }
    children.push(el("p", { class: "subhead", text: "Class probabilities" }), bars);

    if (center) {
      const noEdge = center.pred_without_edges_class_id;
      const changed = noEdge !== top;
      const verdict = (cls) => (cls === center.true_class_id ? "correct" : "incorrect");
      children.push(callout("info", "Edge ablation for this node", [
        el("p", {}, [
          "With every citation edge replaced by a self-loop, the same model predicts ",
          el("strong", { text: name(noEdge) }),
          ` (${verdict(noEdge)}). `,
          changed
            ? `The citation context changes this paper's prediction (with edges: ${verdict(top)}).`
            : "The prediction does not depend on the citation edges for this paper.",
        ]),
      ]));
      children.push(el("p", { class: "hint", text: `${center.active_feature_indices.length} of ${FEATURE_DIM} word indicators are active for this paper. Degree: ${center.degree} citation link${center.degree === 1 ? "" : "s"}.` }));
    }

    children.push(
      el("details", { class: "disclosure" }, [
        el("summary", { text: "Raw response (logits, indices)" }),
        el("div", { class: "disclosure-body stack-sm" }, [
          el("pre", { class: "code", text: JSON.stringify({ node_index: p.node_index, predicted_class_id: top, logits: p.logits.map((v) => +v.toFixed(4)), active_feature_indices: center ? center.active_feature_indices : undefined }, null, 2) }),
          el("button", { class: "btn btn-sm", type: "button", text: "Copy full JSON", onclick: () => copyText(JSON.stringify(resp, null, 2)) }),
        ]),
      ])
    );
    body.replaceChildren(el("div", { class: "stack" }, children));
  }

  /* Deterministic force-directed layout (Fruchterman–Reingold) on the real subgraph. */
  function layout(nodes, edges, W, H) {
    const index = new Map(nodes.map((n, i) => [n.id, i]));
    const rand = (id, salt) => ((Math.imul(id + 1, 2654435761) ^ Math.imul(salt, 40503)) >>> 0) / 4294967296;
    const P = nodes.map((n) => {
      if (n.hop === 0) return { x: W / 2, y: H / 2 };
      const a = rand(n.id, 1) * Math.PI * 2;
      const r = (n.hop * 0.22 + rand(n.id, 2) * 0.08) * Math.min(W, H);
      return { x: W / 2 + r * Math.cos(a), y: H / 2 + r * Math.sin(a) };
    });
    const E = edges.map(([u, v]) => [index.get(u), index.get(v)]).filter(([a, b]) => a !== undefined && b !== undefined);
    const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.6;
    const iters = 320;
    for (let it = 0; it < iters; it++) {
      const temp = (1 - it / iters) * W * 0.06 + 0.5;
      const D = P.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < P.length; i++) {
        for (let j = i + 1; j < P.length; j++) {
          let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y;
          let d = Math.hypot(dx, dy) || 0.01;
          const f = (k * k) / d;
          dx /= d; dy /= d;
          D[i].x += dx * f; D[i].y += dy * f; D[j].x -= dx * f; D[j].y -= dy * f;
        }
      }
      for (const [a, b] of E) {
        let dx = P[a].x - P[b].x, dy = P[a].y - P[b].y;
        const d = Math.hypot(dx, dy) || 0.01;
        const f = (d * d) / k;
        dx /= d; dy /= d;
        D[a].x -= dx * f; D[a].y -= dy * f; D[b].x += dx * f; D[b].y += dy * f;
      }
      for (let i = 0; i < P.length; i++) {
        if (nodes[i].hop === 0) continue;
        D[i].x += (W / 2 - P[i].x) * 0.02 * k / 10;
        D[i].y += (H / 2 - P[i].y) * 0.02 * k / 10;
        const d = Math.hypot(D[i].x, D[i].y) || 1;
        P[i].x += (D[i].x / d) * Math.min(d, temp);
        P[i].y += (D[i].y / d) * Math.min(d, temp);
      }
    }
    // Fit into the viewport, keeping the selected node centred.
    const pad = 36;
    let maxDx = 1, maxDy = 1;
    for (const p of P) { maxDx = Math.max(maxDx, Math.abs(p.x - W / 2)); maxDy = Math.max(maxDy, Math.abs(p.y - H / 2)); }
    const s = Math.min((W / 2 - pad) / maxDx, (H / 2 - pad) / maxDy, 2.2);
    return P.map((p) => ({ x: W / 2 + (p.x - W / 2) * s, y: H / 2 + (p.y - H / 2) * s }));
  }

  function renderGraph(nbr) {
    const W = 800, H = 600;
    const nodes = nbr.nodes;
    const pos = layout(nodes, nbr.edges, W, H);
    const posById = new Map(nodes.map((n, i) => [n.id, pos[i]]));
    const centerId = nbr.center.id;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": `Citation neighbourhood of node ${centerId}: ${nodes.length} nodes, ${nbr.edges.length} edges` });
    const gEdges = svg("g");
    for (const [u, v] of nbr.edges) {
      const a = posById.get(u), b = posById.get(v);
      gEdges.append(svg("line", { class: `graph-edge${u === centerId || v === centerId ? " is-center" : ""}`, x1: a.x, y1: a.y, x2: b.x, y2: b.y }));
    }
    root.append(gEdges);
    const gNodes = svg("g");
    const labels = svg("g");
    nodes.forEach((n, i) => {
      const p = pos[i];
      const isCenter = n.id === centerId;
      const r = isCenter ? 12 : n.hop === 1 ? 7.5 : 5.5;
      const cls = state.colorBy === "true" ? n.true_class_id : n.pred_class_id;
      const g = svg("g", {
        class: `graph-node${isCenter ? " is-center" : ""}`, tabindex: "0", role: "button",
        "aria-label": `Node ${n.id}, hop ${n.hop}, true ${name(n.true_class_id)}, predicted ${name(n.pred_class_id)}, ${n.split} split`,
        transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`,
      });
      g.append(svg("circle", { class: "focus-ring", r: r + 5, fill: "none", stroke: "none" }));
      if (n.pred_class_id !== n.true_class_id) g.append(svg("circle", { class: "wrong", r: r + 3.5 }));
      g.append(n.split === "train"
        ? svg("rect", { class: "shape", x: -r * 0.9, y: -r * 0.9, width: r * 1.8, height: r * 1.8, rx: 1.5, fill: COLORS[cls] })
        : svg("circle", { class: "shape", r, fill: COLORS[cls] }));
      const show = () => showTooltip(n, p, W, H);
      g.addEventListener("mouseenter", show);
      g.addEventListener("focus", show);
      g.addEventListener("mouseleave", hideTooltip);
      g.addEventListener("blur", hideTooltip);
      const pick = () => { if (!isCenter) selectNode(n.id); };
      g.addEventListener("click", pick);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      gNodes.append(g);
      if (isCenter || (n.hop === 1 && nodes.length <= 40)) {
        const t = svg("text", { class: `graph-label${isCenter ? " is-center" : ""}`, x: p.x, y: p.y - r - 6, "text-anchor": "middle" });
        t.textContent = `#${n.id}`;
        labels.append(t);
      }
    });
    root.append(gNodes, labels);
    const stage = $("#graph-stage");
    stage.replaceChildren(root);
    const shownHop = (h) => nodes.filter((n) => n.hop === h).length;
    let cap = `${nodes.length} nodes · ${nbr.edges.length} edges. Coloured by ${state.colorBy === "true" ? "ground-truth topic" : "GCN prediction"}. Click a node to recentre.`;
    if (nbr.truncated) cap = `Showing ${shownHop(1)} of ${nbr.neighbour_counts[1]} direct and ${shownHop(2)} of ${nbr.neighbour_counts[2] ?? 0} second-hop neighbours (capped at 60 nodes). ` + cap;
    $("#graph-caption").textContent = cap;
  }

  function showTooltip(n, p, W, H) {
    const tip = $("#graph-tooltip");
    const stage = $("#graph-stage").getBoundingClientRect();
    const panel = $(".graph-panel").getBoundingClientRect();
    const scale = Math.min(stage.width / W, stage.height / H);
    const ox = (stage.width - W * scale) / 2, oy = (stage.height - H * scale) / 2;
    const x = stage.left - panel.left + ox + p.x * scale;
    const y = stage.top - panel.top + oy + p.y * scale;
    tip.innerHTML = "";
    tip.append(
      el("div", {}, [el("b", { text: `Node #${n.id}` }), ` · hop ${n.hop} · degree ${n.degree}`]),
      el("div", { text: `True: ${name(n.true_class_id)}` }),
      el("div", { text: `Predicted: ${name(n.pred_class_id)} (p = ${n.confidence.toFixed(2)})` }),
      el("div", { text: `Split: ${n.split}` })
    );
    tip.hidden = false;
    const left = Math.min(Math.max(8, x + 14), panel.width - tip.offsetWidth - 8);
    const top = y + 14 + tip.offsetHeight > panel.height ? y - tip.offsetHeight - 10 : y + 14;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }
  function hideTooltip() { $("#graph-tooltip").hidden = true; }

  function renderNeighbourTable(nbr) {
    $("#nbr-count").textContent = `(${nbr.nodes.length})`;
    const rows = nbr.nodes.map((n) => el("tr", { class: n.id === nbr.center.id ? "row-highlight" : "" }, [
      el("td", {}, [n.id === nbr.center.id ? el("span", { class: "mono", text: `#${n.id}` }) : el("button", { class: "btn btn-sm btn-ghost mono", type: "button", text: `#${n.id}`, onclick: () => selectNode(n.id) })]),
      el("td", { class: "num", text: n.hop }),
      el("td", { class: "num", text: n.degree }),
      el("td", { text: n.split }),
      el("td", {}, [cellClass(n.true_class_id)]),
      el("td", {}, [cellClass(n.pred_class_id)]),
      el("td", { class: "num", text: n.confidence.toFixed(2) }),
    ]));
    $("#nbr-table").replaceChildren(el("table", { class: "table" }, [
      el("thead", {}, [el("tr", {}, ["Node", "Hop", "Degree", "Split", "True label", "Prediction", "p"].map((h, i) => el("th", { class: [1, 2, 6].includes(i) ? "num" : "", text: h })))]),
      el("tbody", {}, rows),
    ]));
  }

  function cellClass(id) {
    return el("span", { class: "cell-pred" }, [el("span", { class: "bar-swatch", style: { background: COLORS[id] }, "aria-hidden": "true" }), name(id)]);
  }

  /* --------------------------------------------------------------------- */
  /* Edge experiment                                                       */
  /* --------------------------------------------------------------------- */
  async function fetchCoraNode(id) {
    const d = await api(`/api/ui/node/${id}?hops=1&max_nodes=5`, { timeout: 30000 });
    const features = new Array(FEATURE_DIM).fill(0);
    for (const i of d.center.active_feature_indices) features[i] = 1;
    return { label: `Cora #${id}`, source: "cora", coraId: id, trueClass: d.center.true_class_id, features, active: d.center.active_feature_indices.length };
  }

  function addExpNode(node) {
    if (state.exp.nodes.length >= MAX_EXP_NODES) { toast(`At most ${MAX_EXP_NODES} nodes in an experiment`, "err"); return false; }
    state.exp.nodes.push(node);
    return true;
  }

  function addEdge(a, b) {
    if (a === b) return;
    const has = (x, y) => state.exp.edges.some(([s, t]) => s === x && t === y);
    if (!has(a, b)) state.exp.edges.push([a, b]);
    if ($("#exp-undirected").checked && !has(b, a)) state.exp.edges.push([b, a]);
  }

  function renderExperiment() {
    const { nodes, edges } = state.exp;
    $("#exp-node-count").textContent = `${nodes.length} / ${MAX_EXP_NODES}`;
    $("#exp-edge-count").textContent = `${edges.length} directed edge${edges.length === 1 ? "" : "s"}`;
    const list = $("#exp-node-list");
    list.replaceChildren(...nodes.map((n, i) => el("li", {}, [
      el("span", { class: "idx", text: i }),
      el("span", {}, [n.source === "cora" ? el("span", { class: "mono", text: n.label }) : el("span", { text: n.label }), n.source === "synthetic" ? el("span", { class: "badge badge-warn", style: { marginLeft: "6px" }, text: "synthetic" }) : null]),
      el("span", { class: "meta", text: `${n.active} active${n.trueClass !== undefined ? ` · ${name(n.trueClass)}` : ""}` }),
      el("button", { class: "btn btn-sm btn-ghost", type: "button", "aria-label": `Remove node ${i}`, text: "×", onclick: () => removeExpNode(i) }),
    ])));
    if (!nodes.length) list.append(el("li", { class: "muted" }, ["No nodes yet — add a Cora node or the selected node's neighbourhood."]));
    const opts = nodes.map((n, i) => el("option", { value: i, text: `${i} · ${n.label}` }));
    for (const sel of [$("#exp-src"), $("#exp-dst")]) {
      const prev = sel.value;
      sel.replaceChildren(...opts.map((o) => o.cloneNode(true)));
      if (prev && +prev < nodes.length) sel.value = prev;
    }
    if (nodes.length > 1 && $("#exp-dst").value === $("#exp-src").value) $("#exp-dst").value = "1";
    $("#exp-edge-list").replaceChildren(...edges.map(([s, t], i) => el("li", {}, [
      `${s} → ${t}`,
      el("button", { class: "btn btn-sm btn-ghost", type: "button", "aria-label": `Remove edge ${s} to ${t}`, text: "×", onclick: () => { edges.splice(i, 1); renderExperiment(); } }),
    ])));
    $("#exp-run").disabled = nodes.length === 0;
    $("#exp-payload").textContent = payloadPreview();
  }

  function removeExpNode(i) {
    state.exp.nodes.splice(i, 1);
    state.exp.edges = state.exp.edges.filter(([s, t]) => s !== i && t !== i).map(([s, t]) => [s > i ? s - 1 : s, t > i ? t - 1 : t]);
    renderExperiment();
  }

  function buildPayload(withEdges) {
    const payload = { node_features: state.exp.nodes.map((n) => n.features) };
    if (withEdges && state.exp.edges.length) {
      payload.edge_indices = [state.exp.edges.map((e) => e[0]), state.exp.edges.map((e) => e[1])];
    }
    return payload;
  }

  function payloadPreview() {
    const p = buildPayload(true);
    return JSON.stringify({
      node_features: p.node_features.map((f) => `<${f.length} values, ${f.reduce((a, b) => a + b, 0)} ones>`),
      edge_indices: p.edge_indices,
    }, null, 2) + "\n// feature vectors abbreviated for display; the full 1433-value vectors are sent";
  }

  async function runExperiment() {
    const btn = $("#exp-run");
    setBusy(btn, true, "Running…");
    const out = $("#exp-results");
    try {
      const [withE, without] = await Promise.all([
        api("/predict", { method: "POST", body: buildPayload(true), timeout: 60000 }),
        api("/predict", { method: "POST", body: buildPayload(false), timeout: 60000 }),
      ]);
      const changed = withE.predictions.filter((p, i) => p.predicted_class_id !== without.predictions[i].predicted_class_id).length;
      const rows = withE.predictions.map((p, i) => {
        const q = without.predictions[i];
        const n = state.exp.nodes[i];
        const diff = p.predicted_class_id !== q.predicted_class_id;
        return el("tr", {}, [
          el("td", { class: "mono", text: i }),
          el("td", { text: n.label }),
          el("td", {}, [n.trueClass !== undefined ? cellClass(n.trueClass) : el("span", { class: "faint", text: "n/a" })]),
          el("td", {}, [cellClass(p.predicted_class_id), el("span", { class: "minibar" }, [el("i", { style: { width: `${p.probabilites[p.predicted_class_id] * 100}%` } })])]),
          el("td", {}, [cellClass(q.predicted_class_id), el("span", { class: "minibar" }, [el("i", { style: { width: `${q.probabilites[q.predicted_class_id] * 100}%` } })])]),
          el("td", {}, [diff ? el("span", { class: "badge badge-info", text: "changed" }) : el("span", { class: "faint", text: "same" })]),
        ]);
      });
      out.replaceChildren(el("div", { class: "panel" }, [
        el("div", { class: "panel-head" }, [
          el("h3", { class: "panel-title", text: "Results" }),
          el("span", { class: "panel-sub", text: `${withE.num_nodes} nodes · ${withE.num_edges} edges sent · ${changed} prediction${changed === 1 ? "" : "s"} changed by the edges` }),
        ]),
        el("div", { class: "table-wrap" }, [el("table", { class: "table" }, [
          el("thead", {}, [el("tr", {}, ["#", "Node", "True label", "With your edges", "Self-loops only", "Effect"].map((h) => el("th", { text: h })))]),
          el("tbody", {}, rows),
        ])]),
        el("div", { class: "panel-foot", text: "Bars show the softmax probability of the predicted class. Both runs use the same feature vectors and the same ONNX model via POST /predict." }),
      ]));
    } catch (err) {
      out.replaceChildren(callout("err", "Inference request failed", [err.message]));
    } finally {
      setBusy(btn, false);
    }
  }

  /* --------------------------------------------------------------------- */
  /* Evaluation                                                            */
  /* --------------------------------------------------------------------- */
  async function loadEvaluation() {
    const root = $("#eval-root");
    try {
      const e = await api("/api/ui/evaluation", { timeout: 120000 });
      const live = e.live;
      $("#meta-acc").innerHTML = `Test accuracy <strong>${live.test.accuracy.toFixed(3)}</strong> <span class="faint">(live, n=${live.test.n})</span>`;
      const labels = Object.keys(CLASSES).map((k) => name(k));

      const tiles = el("div", { class: "grid-4" }, [
        stat("Test accuracy", live.test.accuracy.toFixed(3), `${Math.round(live.test.accuracy * live.test.n)} / ${live.test.n} test nodes`, "live · shipped ONNX"),
        stat("Test macro-F1", live.test.macro_f1.toFixed(3), "Unweighted mean over 7 topics", "live · shipped ONNX"),
        stat("Validation accuracy", live.val.accuracy.toFixed(3), `n = ${live.val.n}`, "live · shipped ONNX"),
        stat("Training accuracy", live.train.accuracy.toFixed(3), `n = ${live.train.n} — fitted labels`, "live · shipped ONNX"),
      ]);

      const rec = e.recorded;
      const compareRows = [];
      if (rec) {
        for (const m of rec.models) compareRows.push({ name: m.name, source: "recorded · notebook", acc: m.test_accuracy, f1: m.test_macro_f1, desc: m.description });
      }
      compareRows.push({ name: "GCN — shipped ONNX model", source: "live · this server", acc: live.test.accuracy, f1: live.test.macro_f1, desc: "Re-evaluated on the 1,000 Planetoid test nodes with all citation edges." });
      compareRows.push({ name: "GCN — edges removed (ablation)", source: "live · this server", acc: e.edge_ablation_test.accuracy, f1: e.edge_ablation_test.macro_f1, desc: e.edge_ablation_test.description });

      const cmpBars = el("div", { class: "cmp-bars stack" });
      const accBars = el("div"), f1Bars = el("div");
      renderBars(accBars, compareRows.map((r) => ({ label: r.name, value: r.acc, color: r.source.startsWith("live") ? "var(--accent)" : "#8a909b" })), { valueFormat: (v) => v.toFixed(3), ariaLabel: "Test accuracy by model" });
      renderBars(f1Bars, compareRows.map((r) => ({ label: r.name, value: r.f1, color: r.source.startsWith("live") ? "var(--accent)" : "#8a909b" })), { valueFormat: (v) => v.toFixed(3), ariaLabel: "Test macro-F1 by model" });
      cmpBars.append(el("p", { class: "subhead", text: "Test accuracy" }), accBars, el("p", { class: "subhead", text: "Test macro-F1" }), f1Bars,
        legend([{ label: "Live (computed by this server)", color: "var(--accent)", dot: true }, { label: "Recorded (notebook output)", color: "#8a909b", dot: true }]));

      const cmpTable = el("div", { class: "table-wrap" }, [el("table", { class: "table" }, [
        el("thead", {}, [el("tr", {}, [el("th", { text: "Model" }), el("th", { text: "Source" }), el("th", { class: "num", text: "Accuracy" }), el("th", { class: "num", text: "Macro-F1" })])]),
        el("tbody", {}, compareRows.map((r) => el("tr", {}, [
          el("td", {}, [el("div", { text: r.name }), el("div", { class: "hint", text: r.desc })]),
          el("td", {}, [el("span", { class: `badge ${r.source.startsWith("live") ? "badge-accent" : ""}`, text: r.source })]),
          el("td", { class: "num", text: r.acc.toFixed(3) }),
          el("td", { class: "num", text: r.f1.toFixed(3) }),
        ]))),
      ])]);

      const matrixBox = el("div");
      let normalize = false;
      const drawMatrix = () => renderMatrix(matrixBox, live.test.confusion_matrix, labels.map((l) => l.replace("Reinforcement Learning", "Reinforcement L.").replace("Probabilistic Methods", "Probabilistic M.").replace("Genetic Algorithms", "Genetic Alg.")), { normalize, color: "#1f3a5f", caption: "Test-set confusion matrix" });
      const toggle = el("div", { class: "segmented", role: "group", "aria-label": "Confusion matrix values" }, [
        el("button", { type: "button", "aria-pressed": "true", text: "Counts" }),
        el("button", { type: "button", "aria-pressed": "false", text: "Row %" }),
      ]);
      $$("button", toggle).forEach((b, i) => b.addEventListener("click", () => {
        normalize = i === 1;
        $$("button", toggle).forEach((x, j) => x.setAttribute("aria-pressed", String(j === i)));
        drawMatrix();
      }));
      drawMatrix();

      const perClass = el("div", { class: "table-wrap" }, [el("table", { class: "table" }, [
        el("thead", {}, [el("tr", {}, [el("th", { text: "Topic" }), el("th", { class: "num", text: "Precision" }), el("th", { class: "num", text: "Recall" }), el("th", { class: "num", text: "F1" }), el("th", { class: "num", text: "Support" })])]),
        el("tbody", {}, live.test.per_class.map((c) => el("tr", {}, [
          el("td", {}, [cellClass(c.class_id)]),
          el("td", { class: "num", text: c.precision.toFixed(3) }),
          el("td", { class: "num", text: c.recall.toFixed(3) }),
          el("td", { class: "num", text: c.f1.toFixed(3) }),
          el("td", { class: "num", text: c.support }),
        ]))),
      ])]);

      let curve = null;
      if (rec && rec.training_log) {
        const box = el("div");
        plotXY(box, [
          { name: "train", color: "#8a909b", points: rec.training_log.map((r) => [r.epoch, r.train_acc]), markers: true, dashed: true },
          { name: "val", color: "#1f3a5f", points: rec.training_log.map((r) => [r.epoch, r.val_acc]), markers: true },
        ], { xLabel: "Epoch", yLabel: "Accuracy", xDomain: [0, 40], yDomain: [0.4, 1], ticks: 4, height: 260, ariaLabel: "Training and validation accuracy at logged epochs" });
        curve = el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Training curve (logged epochs only)" }), el("span", { class: "provenance", text: "recorded · notebook cell 26" })]),
          el("div", { class: "panel-body stack-sm" }, [box, legend([{ label: "Training accuracy", color: "#8a909b" }, { label: "Validation accuracy", color: "#1f3a5f" }]),
            el("p", { class: "hint", text: "The notebook printed metrics at epochs 1, 10, 20 and 30 only; early stopping triggered at epoch 35. Intermediate epochs are not shown because they were not recorded." })]),
        ]);
      }

      root.replaceChildren(
        tiles,
        el("div", { class: "eval-grid" }, [
          el("div", { class: "panel span-all" }, [
            el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Baseline vs. GCN vs. edge ablation" }), el("span", { class: "panel-sub", text: "1,000 Planetoid test nodes" })]),
            el("div", { class: "panel-body cmp-layout" }, [cmpBars, cmpTable]),
          ]),
          el("div", { class: "panel" }, [
            el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Test confusion matrix" }), toggle]),
            el("div", { class: "panel-body stack-sm" }, [matrixBox, el("p", { class: "hint", text: "Rows: ground-truth topic. Columns: GCN prediction. Outlined cells are correct predictions." })]),
          ]),
          el("div", { class: "panel" }, [
            el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Per-class metrics (test)" }), el("span", { class: "provenance", text: "live · shipped ONNX" })]),
            el("div", { class: "panel-body" }, [perClass]),
          ]),
          curve,
        ])
      );
    } catch (err) {
      root.replaceChildren(callout("err", "Could not compute evaluation", [err.message]));
    }
  }

  /* --------------------------------------------------------------------- */
  /* Wiring                                                                */
  /* --------------------------------------------------------------------- */
  function wire() {
    $("#node-form").addEventListener("submit", (e) => { e.preventDefault(); classify($("#node-id").value); });
    $("#node-id").addEventListener("input", () => validateNodeId($("#node-id").value || "0", $("#node-id-error"), $("#node-id")));
    $("#random-btn").addEventListener("click", () => selectNode(Math.floor(Math.random() * state.numNodes)));
    $$("[data-hops]").forEach((b) => b.addEventListener("click", () => {
      state.hops = +b.dataset.hops;
      $$("[data-hops]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      if (state.nbr) classify(state.nbr.center.id);
    }));
    $$("[data-color]").forEach((b) => b.addEventListener("click", () => {
      state.colorBy = b.dataset.color;
      $$("[data-color]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      if (state.nbr) renderGraph(state.nbr);
    }));

    $("#exp-add-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const errEl = $("#exp-add-error");
      const id = validateNodeId($("#exp-node-id").value, errEl, $("#exp-node-id"));
      if (id === null) return;
      if (!state.adapter) { errEl.textContent = "Copying Cora features requires ui_api.py."; return; }
      const btn = e.submitter || $("#exp-add-form button");
      setBusy(btn, true);
      try {
        if (addExpNode(await fetchCoraNode(id))) { $("#exp-node-id").value = ""; renderExperiment(); }
      } catch (err) { errEl.textContent = err.message; }
      finally { setBusy(btn, false); }
    });
    $("#exp-add-nbrs").addEventListener("click", async () => {
      if (!state.nbr) return;
      const btn = $("#exp-add-nbrs");
      const ids = state.nbr.nodes.filter((n) => n.hop <= 1).map((n) => n.id).slice(0, MAX_EXP_NODES);
      setBusy(btn, true);
      try {
        const fetched = await Promise.all(ids.map(fetchCoraNode));
        state.exp = { nodes: fetched, edges: [] };
        const pos = new Map(ids.map((id, i) => [id, i]));
        for (const [u, v] of state.nbr.edges) if (pos.has(u) && pos.has(v)) { state.exp.edges.push([pos.get(u), pos.get(v)], [pos.get(v), pos.get(u)]); }
        renderExperiment();
        if (state.nbr.nodes.filter((n) => n.hop === 1).length + 1 > MAX_EXP_NODES) toast(`Neighbourhood truncated to ${MAX_EXP_NODES} nodes`, "info");
      } catch (err) { $("#exp-add-error").textContent = err.message; }
      finally { setBusy(btn, false); }
    });
    $("#exp-add-random").addEventListener("click", () => {
      const f = new Array(FEATURE_DIM).fill(0);
      const picked = new Set();
      while (picked.size < 18) picked.add(Math.floor(Math.random() * FEATURE_DIM));
      for (const i of picked) f[i] = 1;
      if (addExpNode({ label: "Random vector", source: "synthetic", features: f, active: 18 })) renderExperiment();
    });
    $("#exp-edge-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const a = +$("#exp-src").value, b = +$("#exp-dst").value;
      if (state.exp.nodes.length < 2) return toast("Add at least two nodes first", "err");
      if (a === b) return toast("Choose two different nodes (self-loops are added by the model)", "err");
      addEdge(a, b); renderExperiment();
    });
    $("#exp-chain").addEventListener("click", () => { for (let i = 1; i < state.exp.nodes.length; i++) addEdge(i - 1, i); renderExperiment(); });
    $("#exp-star").addEventListener("click", () => { for (let i = 1; i < state.exp.nodes.length; i++) addEdge(0, i); renderExperiment(); });
    $("#exp-clear-edges").addEventListener("click", () => { state.exp.edges = []; renderExperiment(); });
    $("#exp-reset").addEventListener("click", () => { state.exp = { nodes: [], edges: [] }; $("#exp-results").replaceChildren(); renderExperiment(); });
    $("#exp-run").addEventListener("click", runExperiment);
    renderExperiment();
  }

  wire();
  boot();
})();
