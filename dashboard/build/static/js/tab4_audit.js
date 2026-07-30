(function () {
  let loaded = false;
  let heatMap;

  function renderKPIs(data) {
    const bm = {};
    data.benchmark.forEach((r) => { bm[r.Metric] = r; });
    const row = document.getElementById("kpi-row");
    row.innerHTML = `
      <div class="kpi-card">
        <div class="kpi-label">MAPE — SpatialGAT (GNN)</div>
        <div class="kpi-main">${fmtPct(data.gnn_eval.mape, 2)}</div>
        <div class="kpi-sub">vs XGBoost ${fmtPct(bm.MAPE["XGBoost Baseline"], 2)}</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Improvement vs Baseline</div>
        <div class="kpi-main">${fmtPct(bm.MAPE["Improvement (%)"], 1)}</div>
        <div class="kpi-sub">MAPE reduction</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">MAE</div>
        <div class="kpi-main">${fmtMoney(data.gnn_eval.mae)}</div>
        <div class="kpi-sub">vs XGBoost ${fmtMoney(bm.MAE["XGBoost Baseline"])}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">RMSE</div>
        <div class="kpi-main">${fmtMoney(data.gnn_eval.rmse)}</div>
        <div class="kpi-sub">vs XGBoost ${fmtMoney(bm.RMSE["XGBoost Baseline"])}</div>
      </div>
    `;
  }

  function renderCalibration(calibration) {
    const ctx = document.getElementById("chart-calibration").getContext("2d");
    const maxV = Math.max(...calibration.map((c) => Math.max(c.actual, c.predicted)));
    new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Actual vs Predicted",
            data: calibration.map((c) => ({ x: c.actual, y: c.predicted })),
            backgroundColor: "rgba(0,242,195,0.5)",
            pointRadius: 2.5,
          },
          {
            label: "Perfect calibration",
            type: "line",
            data: [{ x: 0, y: 0 }, { x: maxV, y: maxV }],
            borderColor: THEME.magenta,
            borderWidth: 1.5,
            pointRadius: 0,
          },
        ],
      },
      options: {
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
        scales: {
          x: { title: { display: true, text: "Actual Price ($)" }, ticks: { callback: (v) => (v / 1000) + "k" }, grid: { color: THEME.grid } },
          y: { title: { display: true, text: "Predicted Price ($)" }, ticks: { callback: (v) => (v / 1000) + "k" }, grid: { color: THEME.grid } },
        },
      },
    });
  }

  function mapeColor(mape) {
    // green -> orange -> red scale
    if (mape < 10) return THEME.green;
    if (mape < 16) return THEME.orange;
    return THEME.red;
  }

  function renderHeatmap(cells) {
    if (!heatMap) {
      heatMap = L.map("error-heatmap-map", { attributionControl: false }).setView([47.55, -122.2], 9);
      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { maxZoom: 18 }).addTo(heatMap);
    }
    cells.forEach((c) => {
      L.circleMarker([c.lat, c.long], {
        radius: 5 + Math.min(c.count / 20, 10),
        color: mapeColor(c.mape),
        fillColor: mapeColor(c.mape),
        fillOpacity: 0.55,
        weight: 1,
      }).bindTooltip(`MAPE ${c.mape.toFixed(1)}% (n=${c.count})`).addTo(heatMap);
    });
  }

  function renderDecay(decay) {
    const ctx = document.getElementById("chart-decay").getContext("2d");
    new Chart(ctx, {
      type: "line",
      data: {
        labels: decay.map((d) => d.distance_mi.toFixed(2)),
        datasets: [{
          data: decay.map((d) => d.attention_weight),
          borderColor: THEME.purple, backgroundColor: "rgba(168,85,247,0.15)",
          fill: true, pointRadius: 0, tension: 0.3,
        }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "Distance (miles)" }, grid: { display: false }, ticks: { maxTicksLimit: 8 } },
          y: { title: { display: true, text: "Attention Weight" }, grid: { color: THEME.grid } },
        },
      },
    });
  }

  function renderFolds(cv) {
    const ctx = document.getElementById("chart-folds").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: cv.fold_results.map((f) => "Fold " + f.fold),
        datasets: [
          { label: "MAPE %", data: cv.fold_results.map((f) => f.mape), backgroundColor: THEME.orange, borderRadius: 4 },
        ],
      },
      options: {
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
        scales: {
          y: { title: { display: true, text: "MAPE %" }, grid: { color: THEME.grid } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function renderImportance(items) {
    const ctx = document.getElementById("chart-importance").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: items.map((f) => f.feature),
        datasets: [{ data: items.map((f) => f.importance_pct), backgroundColor: THEME.magenta, borderRadius: 4 }],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "Share of Total Gain (%)" }, grid: { color: THEME.grid } },
          y: { grid: { display: false }, ticks: { font: { size: 9 } } },
        },
      },
    });
  }

  function renderErrorDist(dist) {
    const ctx = document.getElementById("chart-error-dist").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: dist.map((b) => b.bin_start.toFixed(0) + "-" + b.bin_end.toFixed(0) + "%"),
        datasets: [{ data: dist.map((b) => b.count), backgroundColor: THEME.blue, borderRadius: 3 }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "Absolute % Error" }, grid: { display: false }, ticks: { maxTicksLimit: 8, font: { size: 8 } } },
          y: { title: { display: true, text: "# Properties" }, grid: { color: THEME.grid } },
        },
      },
    });
  }

  async function load() {
    if (loaded) { if (heatMap) heatMap.invalidateSize(); return; }
    loaded = true;
    const data = await apiGet("/api/model-performance");
    renderKPIs(data);
    renderCalibration(data.calibration);
    renderDecay(data.attention_decay);
    renderFolds(data.cv_results);
    renderImportance(data.feature_importance);
    renderErrorDist(data.error_distribution);
    setTimeout(() => renderHeatmap(data.spatial_heatmap), 100);
  }

  window.addEventListener("tabshown", (e) => { if (e.detail === "tab4") load(); });
  document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("tab4").classList.contains("active")) load();
  });
})();
