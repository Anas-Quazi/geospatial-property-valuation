(function () {
  let distChart, gradeChart, bedChart, badgeChart;
  const BADGE_COLOR = { Undervalued: THEME.green, Fair: THEME.orange, Overpriced: THEME.red };

  async function populateZips() {
    const sel = document.getElementById("n-zip");
    if (sel.options.length) return;
    const res = await apiGet("/api/zipcodes");
    res.zipcodes.forEach((z) => {
      const opt = document.createElement("option");
      opt.value = z; opt.textContent = z;
      sel.appendChild(opt);
    });
  }

  function renderKPIs(d) {
    const row = document.getElementById("n-kpi-row");
    row.innerHTML = `
      <div class="kpi-card">
        <div class="kpi-label">Avg Price</div>
        <div class="kpi-main">${fmtMoney(d.avg_price)}</div>
        <div class="kpi-sub">Median ${fmtMoney(d.median_price)}</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Avg Price / Sqft</div>
        <div class="kpi-main">$${d.avg_price_per_sqft.toFixed(0)}</div>
        <div class="kpi-sub">${d.count} properties</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Price Range</div>
        <div class="kpi-main" style="font-size:16px">${fmtMoney(d.price_min)} – ${fmtMoney(d.price_max)}</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Local Model Accuracy</div>
        <div class="kpi-main">${fmtPct(d.local_mape, 1)}</div>
        <div class="kpi-sub">Mean absolute % error here</div>
      </div>
    `;
  }

  function renderDist(dist) {
    const ctx = document.getElementById("chart-n-dist").getContext("2d");
    if (distChart) distChart.destroy();
    distChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: dist.map((b) => "$" + Math.round(b.bin_start / 1000) + "k"),
        datasets: [{ data: dist.map((b) => b.count), backgroundColor: THEME.teal, borderRadius: 3 }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { y: { grid: { color: THEME.grid } }, x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } } },
      },
    });
  }

  function renderGradeMix(items) {
    const ctx = document.getElementById("chart-n-grade").getContext("2d");
    if (gradeChart) gradeChart.destroy();
    gradeChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: items.map((g) => "Grade " + g.grade),
        datasets: [{ data: items.map((g) => g.count), backgroundColor: THEME.purple, borderRadius: 3 }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { y: { grid: { color: THEME.grid } }, x: { grid: { display: false }, ticks: { font: { size: 8 } } } },
      },
    });
  }

  function renderBedMix(items) {
    const ctx = document.getElementById("chart-n-beds").getContext("2d");
    if (bedChart) bedChart.destroy();
    bedChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: items.map((b) => b.bedrooms + " bd"),
        datasets: [{ data: items.map((b) => b.count), backgroundColor: THEME.blue, borderRadius: 3 }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { y: { grid: { color: THEME.grid } }, x: { grid: { display: false } } },
      },
    });
  }

  function renderBadges(counts) {
    const ctx = document.getElementById("chart-n-badges").getContext("2d");
    if (badgeChart) badgeChart.destroy();
    badgeChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: Object.keys(counts),
        datasets: [{ data: Object.values(counts), backgroundColor: Object.keys(counts).map((k) => BADGE_COLOR[k]) }],
      },
      options: {
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 9 } } } },
        cutout: "60%",
      },
    });
  }

  function renderTable(props) {
    const tbody = document.querySelector("#n-table tbody");
    tbody.innerHTML = "";
    props.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${p.id}</td><td>${fmtMoney(p.price)}</td><td>${p.bedrooms}bd/${p.bathrooms}ba</td><td>${p.sqft_living}</td><td>${dealBadgeHtml(p.deal_badge)}</td>`;
      tbody.appendChild(tr);
    });
  }

  async function loadNeighborhood() {
    const zip = document.getElementById("n-zip").value;
    if (!zip) return;
    const btn = document.getElementById("btn-load-neighborhood");
    btn.textContent = "Loading...";
    btn.disabled = true;
    try {
      const d = await apiGet(`/api/neighborhood?zipcode=${zip}`);
      document.getElementById("n-result").classList.remove("hidden");
      renderKPIs(d);
      renderDist(d.distribution);
      renderTable(d.top_properties);
      renderGradeMix(d.grade_breakdown);
      renderBedMix(d.bed_breakdown);
      renderBadges(d.badge_counts);
    } catch (e) {
      alert("Failed to load neighborhood: " + e.message);
    } finally {
      btn.textContent = "Load Neighborhood";
      btn.disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    populateZips().then(() => {
      document.getElementById("n-zip").value = "98178";
    });
    document.getElementById("btn-load-neighborhood").addEventListener("click", loadNeighborhood);
  });
  window.addEventListener("tabshown", (e) => { if (e.detail === "tab5") populateZips(); });
})();
