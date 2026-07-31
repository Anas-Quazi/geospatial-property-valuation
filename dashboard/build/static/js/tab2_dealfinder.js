(function () {
  let scatterChart, badgeChart;
  const BADGE_COLOR = { Undervalued: THEME.green, Fair: THEME.orange, Overpriced: THEME.red };

  function renderScatter(results) {
    const ctx = document.getElementById("chart-deal-scatter").getContext("2d");
    const maxV = Math.max(1, ...results.map((r) => Math.max(r.price, r.predicted_price)));
    const groups = ["Undervalued", "Fair", "Overpriced"].map((badge) => ({
      label: badge,
      data: results.filter((r) => r.deal_badge === badge).map((r) => ({ x: r.price, y: r.predicted_price })),
      backgroundColor: BADGE_COLOR[badge],
      pointRadius: 3,
    }));
    if (scatterChart) scatterChart.destroy();
    scatterChart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          ...groups,
          { label: "Fair-value line", type: "line", data: [{ x: 0, y: 0 }, { x: maxV, y: maxV }],
            borderColor: THEME.text, borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
        ],
      },
      options: {
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
        scales: {
          x: { title: { display: true, text: "Actual Price ($)" }, ticks: { callback: (v) => (v / 1000) + "k" }, grid: { color: THEME.grid } },
          y: { title: { display: true, text: "Model-Predicted Price ($)" }, ticks: { callback: (v) => (v / 1000) + "k" }, grid: { color: THEME.grid } },
        },
      },
    });
  }

  function renderBadgeBreakdown(results) {
    const counts = { Undervalued: 0, Fair: 0, Overpriced: 0 };
    results.forEach((r) => { counts[r.deal_badge] = (counts[r.deal_badge] || 0) + 1; });
    const ctx = document.getElementById("chart-deal-badges").getContext("2d");
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

  function collectFilters() {
    const body = {};
    const map = {
      "d-min-sqft": "min_sqft", "d-max-sqft": "max_sqft",
      "d-min-beds": "min_beds", "d-max-beds": "max_beds",
      "d-min-price": "min_price", "d-max-price": "max_price",
      "d-zip": "zipcode",
    };
    for (const [id, key] of Object.entries(map)) {
      const el = document.getElementById(id);
      if (el.value !== "") body[key] = parseFloat(el.value);
    }
    body.limit = 50;
    return body;
  }

  function renderTable(results) {
    const tbody = document.querySelector("#deal-table tbody");
    tbody.innerHTML = "";
    results.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.id}</td>
        <td>${fmtMoney(r.price)}</td>
        <td>${fmtMoney(r.predicted_price)}</td>
        <td style="color:${r.diff_pct < 0 ? '#2de08e' : '#ff3b5c'}">${r.diff_pct > 0 ? '+' : ''}${r.diff_pct.toFixed(1)}%</td>
        <td>${r.bedrooms}bd/${r.bathrooms}ba</td>
        <td>${r.sqft_living}</td>
        <td>${r.zipcode}</td>
        <td>${dealBadgeHtml(r.deal_badge)}</td>
      `;
      tr.addEventListener("click", () => loadSimilar(r.id));
      tbody.appendChild(tr);
    });
  }

  async function search() {
    const btn = document.getElementById("btn-search-deals");
    btn.textContent = "Searching...";
    btn.disabled = true;
    try {
      const res = await apiPost("/api/deal-finder", collectFilters());
      document.getElementById("deal-summary").innerHTML =
        `<b>${res.count.toLocaleString()}</b> matching properties · Expected market price for this archetype: <b>${fmtMoney(res.expected_market_price)}</b>`;
      renderTable(res.results);
      renderScatter(res.results);
      renderBadgeBreakdown(res.results);
    } catch (e) {
      alert("Search failed: " + e.message);
    } finally {
      btn.textContent = "Search";
      btn.disabled = false;
    }
  }

  async function loadSimilar(id) {
    const el = document.getElementById("similar-result");
    el.innerHTML = `<div class="empty-state">Loading similar properties for #${id}...</div>`;
    try {
      const res = await apiGet(`/api/similar/${id}`);
      let html = `<div class="summary-line">Base: <b>#${res.base.id}</b> — ${fmtMoney(res.base.price)}, ${res.base.sqft_living}sqft, ${res.base.bedrooms}bd/${res.base.bathrooms}ba, grade ${res.base.grade}, zip ${res.base.zipcode}</div>`;
      html += `<div class="table-wrap"><table class="data-table"><thead><tr><th>ID</th><th>Price</th><th>Sqft</th><th>Beds/Baths</th><th>Grade</th><th>Zip</th><th>Similarity</th></tr></thead><tbody>`;
      res.similar.forEach((s) => {
        html += `<tr><td>${s.id}</td><td>${fmtMoney(s.price)}</td><td>${s.sqft_living}</td><td>${s.bedrooms}bd/${s.bathrooms}ba</td><td>${s.grade}</td><td>${s.zipcode}</td><td>${(s.similarity * 100).toFixed(1)}%</td></tr>`;
      });
      html += `</tbody></table></div>`;
      el.innerHTML = html;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Could not load similar properties</div>`;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("btn-search-deals").addEventListener("click", search);
    search();
  });
})();
