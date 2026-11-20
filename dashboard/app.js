/* Capstone Prototype Dashboard Client */
(function () {
  "use strict";

  const reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let metricsData = null;
  let vocabData = null;
  const charts = {};
  let cmState = null;

  // ------------------------------------------------------------------ Tab Switching
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      tabBtns.forEach((b) => b.classList.toggle("active", b === btn));
      tabPanels.forEach((p) => p.classList.toggle("active", p.id === `tab-${target}`));
      if (target === "confusion" && cmState) drawConfusionMatrix();
    });
  });

  // ------------------------------------------------------------------ Live Console
  const pillLink = document.getElementById("pill-link");
  const pillFps = document.getElementById("pill-fps");
  const pillLat = document.getElementById("pill-lat");
  const pillHand = document.getElementById("pill-hand");
  const liveText = document.getElementById("live-text");
  const liveAr = document.getElementById("live-ar");
  const probBars = document.getElementById("prob-bars");
  const liveLogs = document.getElementById("live-logs");
  const liveGuessBadge = document.getElementById("live-guess-badge");
  const videoImg = document.getElementById("live-video");
  const videoOffline = document.getElementById("video-offline");

  videoImg.addEventListener("load", () => videoOffline.classList.add("hidden"));
  videoImg.addEventListener("error", () => videoOffline.classList.remove("hidden"));

  async function sendCmd(cmd) {
    try {
      await fetch("/api/cmd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd }),
      });
    } catch (e) {
      console.error("cmd error:", e);
    }
  }

  document.querySelectorAll("[data-cmd]").forEach((btn) => {
    btn.addEventListener("click", () => sendCmd(btn.dataset.cmd));
  });

  function renderLiveState(s) {
    if (!s) return;
    if (s.connected) {
      pillLink.className = "pill ok";
      pillLink.innerHTML = `<span class="dot"></span>Pi Online (${s.pi_host || "LAN"})`;
      videoOffline.classList.add("hidden");
    } else {
      pillLink.className = "pill warn";
      pillLink.innerHTML = `<span class="dot"></span>Standby (Offline Demo)`;
    }

    pillFps.textContent = `${Number(s.fps || 0).toFixed(1)} FPS`;
    pillLat.textContent = `${Number(s.latency_ms || 0).toFixed(1)} ms`;
    pillHand.textContent = s.hand ? `${s.num_hands || 1} Hand(s) Tracked` : "No Hand";
    pillHand.className = s.hand ? "pill ok" : "pill";

    const isArsl = s.sign_lang === "arsl";
    document.querySelectorAll("[data-sign]").forEach((b) => {
      b.classList.toggle("active", !isArsl && b.dataset.sign === "asl");
      b.classList.toggle("arsl-active", isArsl && b.dataset.sign === "arsl");
    });
    document.querySelectorAll("[data-lang]").forEach((b) => {
      b.classList.toggle("active", b.dataset.lang === s.out_lang);
    });
    document.querySelectorAll("[data-mode]").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === s.mode);
    });

    const primaryText = (s.text || "").trim();
    const arText = (s.translated_text || "").trim();
    liveText.textContent = primaryText || "Start signing...";
    liveAr.textContent = arText || (s.out_lang === "ar" ? "ابدأ بلغة الإشارة..." : "Switch Output to AR for live Arabic translation");
    liveGuessBadge.textContent = s.letter || s.top2 || "-";

    const top3 = Array.isArray(s.top3) ? s.top3 : [];
    if (top3.length > 0) {
      probBars.innerHTML = top3.map((item) => {
        const pct = Math.max(0, Math.min(100, Math.round((item.prob || 0) * 100)));
        return `
          <div class="prob-row">
            <div class="prob-name" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div>
            <div class="prob-track"><div class="prob-fill ${isArsl ? "arsl" : ""}" style="width:${pct}%"></div></div>
            <div class="prob-pct tabular">${pct}%</div>
          </div>
        `;
      }).join("");
    }

    if (Array.isArray(s.logs) && s.logs.length > 0) {
      liveLogs.textContent = s.logs.slice(-18).join("\n");
      liveLogs.scrollTop = liveLogs.scrollHeight;
    }
  }

  function startTelemetry() {
    fetch("/api/state").then((r) => r.json()).then(renderLiveState).catch(() => {});
    if (window.EventSource) {
      const es = new EventSource("/api/stream");
      es.onmessage = (ev) => {
        try {
          renderLiveState(JSON.parse(ev.data));
        } catch (_) {}
      };
    } else {
      setInterval(() => fetch("/api/state").then((r) => r.json()).then(renderLiveState).catch(() => {}), 800);
    }
  }

  // ------------------------------------------------------------------ Benchmarks & Charts
  async function loadMetrics() {
    const res = await fetch("/api/metrics");
    metricsData = await res.json();
    const models = metricsData.models || [];
    if (!models.length) {
      document.getElementById("summary-table-body").innerHTML =
        `<tr><td colspan="9">No benchmark file found yet. Run <code>evaluate_all_models.py</code>.</td></tr>`;
      return;
    }

    const byId = Object.fromEntries(models.map((m) => [m.model_id, m]));
    const aslA = byId.asl_alpha;
    const arslA = byId.arsl_alpha;
    const aslW = byId.asl_words_30;
    const arslW = byId.arsl_words_30;

    if (aslA) {
      document.getElementById("kpi-asl-alpha").textContent = `${aslA.test_top1_acc.toFixed(1)}%`;
      document.getElementById("kpi-asl-alpha-sub").textContent = `${aslA.num_classes} letters | Macro-F1 ${aslA.test_macro_f1.toFixed(1)}%`;
    }
    if (arslA) {
      document.getElementById("kpi-arsl-alpha").textContent = `${arslA.test_top1_acc.toFixed(1)}%`;
      document.getElementById("kpi-arsl-alpha-sub").textContent = `${arslA.num_classes} Arabic letters | Top-3 ${arslA.test_top3_acc.toFixed(1)}%`;
    }
    if (aslW || arslW) {
      const a1 = aslW ? `${aslW.test_top1_acc.toFixed(1)}%` : "--%";
      const b1 = arslW ? `${arslW.test_top1_acc.toFixed(1)}%` : "--%";
      document.getElementById("kpi-words-pair").textContent = `${a1} / ${b1}`;
      document.getElementById("kpi-words-sub").textContent = `Top-3: ${aslW ? aslW.test_top3_acc.toFixed(1) : "--"}% (ASL) / ${arslW ? arslW.test_top3_acc.toFixed(1) : "--"}% (ArSL)`;
    }
    const lats = models.map((m) => m.inference_ms).filter((v) => typeof v === "number" && v > 0);
    if (lats.length) {
      const avgLat = lats.reduce((a, b) => a + b, 0) / lats.length;
      document.getElementById("kpi-latency").textContent = `${avgLat.toFixed(2)} ms`;
      document.getElementById("kpi-latency-sub").textContent = `Real-time CPU inference across ${lats.length} models`;
    }

    renderSummaryTable(models);
    renderCharts(models, metricsData.scaling || []);
    initConfusionUI(models);
    renderExportSnippets(models);
  }

  function renderSummaryTable(models) {
    const tbody = document.getElementById("summary-table-body");
    tbody.innerHTML = models.map((m) => {
      const isArsl = m.language === "ArSL";
      return `
        <tr>
          <td><strong>${escapeHtml(m.title)}</strong></td>
          <td><span class="badge ${isArsl ? "arsl" : "asl"}">${escapeHtml(m.language)}</span></td>
          <td>${escapeHtml(m.best_model)}</td>
          <td class="tabular">${m.num_classes}</td>
          <td class="tabular">${Number(m.train_samples || 0).toLocaleString()} / ${Number(m.test_samples || 0).toLocaleString()}</td>
          <td class="tabular"><strong>${m.test_top1_acc.toFixed(2)}%</strong></td>
          <td class="tabular">${m.test_top3_acc !== null && m.test_top3_acc !== undefined ? m.test_top3_acc.toFixed(2) + "%" : "—"}</td>
          <td class="tabular">${m.test_macro_f1.toFixed(2)}%</td>
          <td class="tabular">${m.inference_ms !== undefined ? m.inference_ms.toFixed(3) : "—"}</td>
        </tr>
      `;
    }).join("");
  }

  function chartBaseOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: reducedMotion ? false : { duration: 220 },
      plugins: {
        legend: { labels: { usePointStyle: true, boxWidth: 8, color: "#0F172A", font: { family: "Inter", size: 12 } } },
        tooltip: { backgroundColor: "#0F172A", titleColor: "#FFFFFF", bodyColor: "#E2E8F0", padding: 10, cornerRadius: 6 },
      },
      scales: {
        x: { grid: { display: false }, border: { display: false }, ticks: { color: "#475569", font: { family: "Inter", size: 11 } } },
        y: {
          beginAtZero: true,
          max: 100,
          border: { display: false },
          grid: { color: "#E2E8F0", tickLength: 0 },
          ticks: { color: "#475569", callback: (v) => `${v}%`, font: { family: "Inter", size: 11 } },
        },
      },
    };
  }

  function renderCharts(models, scaling) {
    if (!window.Chart) return;
    Object.values(charts).forEach((c) => c.destroy());

    // 1. Overview Top-1, Top-3, Macro-F1
    const labels = models.map((m) => m.short_title || m.title);
    const barColors = models.map((m) => (m.language === "ArSL" ? "#B45309" : "#0F766E"));
    charts["chart-overview"] = new Chart(document.getElementById("chart-overview"), {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Top-1 Accuracy (%)",
            data: models.map((m) => m.test_top1_acc),
            backgroundColor: barColors,
            borderRadius: 4,
          },
          {
            label: "Macro-F1 (%)",
            data: models.map((m) => m.test_macro_f1),
            backgroundColor: "#64748B",
            borderRadius: 4,
          },
          {
            label: "Top-3 Accuracy (%)",
            data: models.map((m) => m.test_top3_acc ?? null),
            backgroundColor: "#CBD5E1",
            borderRadius: 4,
          },
        ],
      },
      options: chartBaseOptions(),
    });

    // 2. WLASL Scaling Curve
    const sortedSc = [...scaling].sort((a, b) => a.num_classes - b.num_classes);
    charts["chart-scaling"] = new Chart(document.getElementById("chart-scaling"), {
      type: "line",
      data: {
        labels: sortedSc.map((r) => `${r.num_classes}w`),
        datasets: [
          {
            label: "Top-1 Accuracy (%)",
            data: sortedSc.map((r) => r.top1_acc),
            borderColor: "#0F766E",
            backgroundColor: "rgba(15, 118, 110, 0.12)",
            fill: true,
            tension: 0.25,
            pointRadius: 2.5,
            borderWidth: 2.5,
          },
          {
            label: "Top-3 Accuracy (%)",
            data: sortedSc.map((r) => r.top3_acc),
            borderColor: "#B45309",
            borderDash: [5, 4],
            fill: false,
            tension: 0.25,
            pointRadius: 2.5,
            borderWidth: 2,
          },
        ],
      },
      options: chartBaseOptions(),
    });

    // 3. Architecture Comparison across tasks with comparison arrays
    const archRows = [];
    models.forEach((m) => {
      if (Array.isArray(m.comparison)) {
        m.comparison.forEach((c) => {
          archRows.push({
            label: `${m.short_title || m.title} · ${c.model}`,
            top1: c.top1_acc,
            f1: c.macro_f1 ?? c.top1_acc,
            isArsl: m.language === "ArSL",
          });
        });
      }
    });
    charts["chart-arch"] = new Chart(document.getElementById("chart-arch"), {
      type: "bar",
      data: {
        labels: archRows.map((r) => r.label),
        datasets: [
          {
            label: "Held-Out Top-1 Accuracy (%)",
            data: archRows.map((r) => r.top1),
            backgroundColor: archRows.map((r) => (r.isArsl ? "#B45309" : "#0F766E")),
            borderRadius: 4,
          },
          {
            label: "Macro-F1 (%)",
            data: archRows.map((r) => r.f1),
            backgroundColor: "#94A3B8",
            borderRadius: 4,
          },
        ],
      },
      options: chartBaseOptions(),
    });
  }

  document.querySelectorAll("[data-export-chart]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.exportChart;
      const src = document.getElementById(id);
      if (!src) return;
      const tmp = document.createElement("canvas");
      tmp.width = src.width;
      tmp.height = src.height;
      const ctx = tmp.getContext("2d");
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, tmp.width, tmp.height);
      ctx.drawImage(src, 0, 0);
      const a = document.createElement("a");
      a.download = `${id}.png`;
      a.href = tmp.toDataURL("image/png");
      a.click();
    });
  });

  // ------------------------------------------------------------------ Confusion Matrix
  const cmSelect = document.getElementById("cm-model-select");
  const cmCanvas = document.getElementById("cm-canvas");
  const cmTooltip = document.getElementById("cm-tooltip");
  const classFilter = document.getElementById("class-filter");

  function initConfusionUI(models) {
    const withCm = models.filter((m) => Array.isArray(m.confusion_matrix) && m.confusion_matrix.length > 0);
    cmSelect.innerHTML = withCm.map((m) => `<option value="${m.model_id}">${escapeHtml(m.title)} (${m.test_top1_acc.toFixed(1)}%)</option>`).join("");
    if (withCm.length) {
      selectCmModel(withCm[withCm.length - 1].model_id);
    }
    cmSelect.addEventListener("change", () => selectCmModel(cmSelect.value));
    classFilter.addEventListener("input", () => renderPerClassTable());
  }

  function selectCmModel(modelId) {
    const m = (metricsData?.models || []).find((x) => x.model_id === modelId);
    if (!m) return;
    cmSelect.value = modelId;
    cmState = m;
    drawConfusionMatrix();
    renderConfusedPairs(m.top_confused || []);
    renderPerClassTable();
  }

  function drawConfusionMatrix() {
    if (!cmState || !cmState.confusion_matrix) return;
    const cm = cmState.confusion_matrix;
    const labels = cmState.classes || cm.map((_, i) => String(i));
    const n = cm.length;
    const padLeft = 86;
    const padTop = 76;
    const cell = Math.max(15, Math.min(34, Math.floor(540 / Math.max(1, n))));
    const W = padLeft + n * cell + 20;
    const H = padTop + n * cell + 20;
    cmCanvas.width = W;
    cmCanvas.height = H;
    cmState._geom = { padLeft, padTop, cell, n, labels, cm };

    const ctx = cmCanvas.getContext("2d");
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, W, H);
    const isArsl = cmState.language === "ArSL";
    const rowSums = cm.map((r) => r.reduce((a, b) => a + b, 0) || 1);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const val = cm[i][j];
        const ratio = val / rowSums[i];
        if (val === 0) {
          ctx.fillStyle = "#F8FAFC";
        } else if (isArsl) {
          const alpha = (0.12 + 0.88 * ratio).toFixed(2);
          ctx.fillStyle = `rgba(180, 83, 9, ${alpha})`;
        } else {
          const alpha = (0.12 + 0.88 * ratio).toFixed(2);
          ctx.fillStyle = `rgba(15, 118, 110, ${alpha})`;
        }
        const x = padLeft + j * cell;
        const y = padTop + i * cell;
        ctx.fillRect(x, y, cell - 1, cell - 1);
        if (cell >= 22 && val > 0) {
          ctx.fillStyle = ratio > 0.5 ? "#FFFFFF" : "#0F172A";
          ctx.font = "600 10px Inter, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(val), x + cell / 2, y + cell / 2);
        }
      }
    }

    ctx.fillStyle = "#475569";
    ctx.font = "600 10px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
      ctx.fillText(String(labels[i]).slice(0, 12), padLeft - 6, padTop + i * cell + cell / 2);
    }
    ctx.save();
    for (let j = 0; j < n; j++) {
      ctx.save();
      ctx.translate(padLeft + j * cell + cell / 2, padTop - 6);
      ctx.rotate(-Math.PI / 3);
      ctx.textAlign = "left";
      ctx.fillText(String(labels[j]).slice(0, 12), 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  cmCanvas.addEventListener("mousemove", (ev) => {
    if (!cmState || !cmState._geom) return;
    const rect = cmCanvas.getBoundingClientRect();
    const scaleX = cmCanvas.width / rect.width;
    const scaleY = cmCanvas.height / rect.height;
    const x = (ev.clientX - rect.left) * scaleX;
    const y = (ev.clientY - rect.top) * scaleY;
    const { padLeft, padTop, cell, n, labels, cm } = cmState._geom;
    const j = Math.floor((x - padLeft) / cell);
    const i = Math.floor((y - padTop) / cell);
    if (i >= 0 && i < n && j >= 0 && j < n) {
      const tot = cm[i].reduce((a, b) => a + b, 0) || 1;
      const val = cm[i][j];
      cmTooltip.style.display = "block";
      cmTooltip.style.left = `${ev.clientX + 12}px`;
      cmTooltip.style.top = `${ev.clientY + 12}px`;
      cmTooltip.textContent = `True: ${labels[i]}  →  Pred: ${labels[j]} | ${val} / ${tot} (${((val / tot) * 100).toFixed(1)}%)`;
    } else {
      cmTooltip.style.display = "none";
    }
  });
  cmCanvas.addEventListener("mouseleave", () => { cmTooltip.style.display = "none"; });

  document.getElementById("btn-export-cm").addEventListener("click", () => {
    const a = document.createElement("a");
    a.download = `confusion_${cmState ? cmState.model_id : "matrix"}.png`;
    a.href = cmCanvas.toDataURL("image/png");
    a.click();
  });

  function renderConfusedPairs(pairs) {
    const tbody = document.getElementById("confused-pairs-body");
    if (!pairs.length) {
      tbody.innerHTML = `<tr><td colspan="3">Zero misclassifications on held-out split.</td></tr>`;
      return;
    }
    tbody.innerHTML = pairs.map((p) => `
      <tr>
        <td>${escapeHtml(p.true)}</td>
        <td>${escapeHtml(p.pred)}</td>
        <td class="tabular"><strong>${p.count}</strong></td>
      </tr>
    `).join("");
  }

  function renderPerClassTable() {
    if (!cmState) return;
    const q = (classFilter.value || "").trim().toLowerCase();
    const rows = (cmState.per_class || []).filter((r) => {
      if (!q) return true;
      return `${r.label} ${r.ar || ""} ${r.en || ""}`.toLowerCase().includes(q);
    });
    const tbody = document.getElementById("per-class-body");
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td><strong>${escapeHtml(r.label)}</strong>${r.ar ? ` <span class="rtl-ar" style="color:#475569;">(${escapeHtml(r.ar)})</span>` : ""}</td>
        <td class="tabular">${r.precision.toFixed(1)}%</td>
        <td class="tabular">${r.recall.toFixed(1)}%</td>
        <td class="tabular"><strong>${r.f1.toFixed(1)}%</strong></td>
        <td class="tabular">${r.support}</td>
      </tr>
    `).join("");
  }

  // ------------------------------------------------------------------ Export & Vocabulary
  function renderExportSnippets(models) {
    const mdLines = [
      "| Model | Track | Algorithm | Classes | Train / Test | Top-1 Acc | Top-3 Acc | Macro-F1 | Infer (ms) |",
      "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
      ...models.map((m) => `| ${m.title} | ${m.language} | ${m.best_model} | ${m.num_classes} | ${m.train_samples} / ${m.test_samples} | ${m.test_top1_acc.toFixed(2)}% | ${m.test_top3_acc != null ? m.test_top3_acc.toFixed(2) + "%" : "—"} | ${m.test_macro_f1.toFixed(2)}% | ${m.inference_ms != null ? m.inference_ms.toFixed(3) : "—"} |`),
    ];
    document.getElementById("export-md").textContent = mdLines.join("\n");

    const texLines = [
      "\\begin{table}[h]",
      "\\centering",
      "\\caption{Capstone Bilingual Sign Language Benchmark Results}",
      "\\begin{tabular}{llcccc}",
      "\\hline",
      "Model & Algorithm & Classes & Top-1 (\\%) & Top-3 (\\%) & Macro-F1 (\\%) \\\\",
      "\\hline",
      ...models.map((m) => `${m.title.replace(/&/g, "\\&")} & ${m.best_model} & ${m.num_classes} & ${m.test_top1_acc.toFixed(2)} & ${m.test_top3_acc != null ? m.test_top3_acc.toFixed(2) : "--"} & ${m.test_macro_f1.toFixed(2)} \\\\`),
      "\\hline",
      "\\end{tabular}",
      "\\end{table}",
    ];
    document.getElementById("export-tex").textContent = texLines.join("\n");
  }

  function wireCopy(btnId, srcId) {
    const btn = document.getElementById(btnId);
    btn.addEventListener("click", async () => {
      const txt = document.getElementById(srcId).textContent;
      await navigator.clipboard.writeText(txt).catch(() => {});
      const prev = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = prev; }, 1200);
    });
  }
  wireCopy("btn-copy-md", "export-md");
  wireCopy("btn-copy-tex", "export-tex");

  document.getElementById("btn-download-json").addEventListener("click", () => {
    if (!metricsData) return;
    const blob = new Blob([JSON.stringify(metricsData, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.download = "capstone_metrics.json";
    a.href = URL.createObjectURL(blob);
    a.click();
  });

  const vocabTrack = document.getElementById("vocab-track");
  const vocabSearch = document.getElementById("vocab-search");

  async function loadVocab() {
    const res = await fetch("/api/vocab");
    vocabData = await res.json();
    renderVocab();
    vocabTrack.addEventListener("change", renderVocab);
    vocabSearch.addEventListener("input", renderVocab);
  }

  function renderVocab() {
    if (!vocabData) return;
    const track = vocabTrack.value;
    const q = (vocabSearch.value || "").trim().toLowerCase();
    let rows = [];
    if (track === "arsl_words") {
      rows = (vocabData.arsl_words || []).map((x) => ({ k: x.id, en: x.en, ar: x.ar, note: `${x.samples || "-"} KArSL samples` }));
    } else if (track === "arsl_letters") {
      rows = (vocabData.arsl_letters || []).map((x) => ({ k: x.key, en: x.en, ar: `${x.ar} (${x.name_ar})`, note: "ArSL21L Class" }));
    } else {
      rows = (vocabData.asl_words || []).map((x, idx) => ({ k: String(idx + 1).padStart(2, "0"), en: x.en, ar: x.ar, note: "Curated ASL + Translation" }));
    }
    const filtered = rows.filter((r) => !q || `${r.k} ${r.en} ${r.ar}`.toLowerCase().includes(q));
    document.getElementById("vocab-body").innerHTML = filtered.map((r) => `
      <tr>
        <td class="mono">${escapeHtml(r.k)}</td>
        <td><strong>${escapeHtml(r.en)}</strong></td>
        <td class="rtl-ar" style="font-size:1.05rem;">${escapeHtml(r.ar)}</td>
        <td style="color:#475569;">${escapeHtml(r.note)}</td>
      </tr>
    `).join("") || `<tr><td colspan="4">No matching vocabulary entries.</td></tr>`;
  }

  function escapeHtml(s) {
    return String(s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  startTelemetry();
  loadMetrics();
  loadVocab();
})();
