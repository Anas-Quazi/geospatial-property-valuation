(function () {
  let vMap, vMarker, attnMap, breakdownChart, zipCompareChart, attnWeightChart;
  let lastNeighbors = [];

  function initMaps() {
    if (vMap) return;
    vMap = L.map("v-map", { attributionControl: false }).setView([47.6205, -122.3493], 10);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { maxZoom: 18 }).addTo(vMap);
    vMarker = L.circleMarker([47.6205, -122.3493], { radius: 7, color: "#00f2c3", fillColor: "#00f2c3", fillOpacity: 0.9 }).addTo(vMap);
    vMap.on("click", (e) => {
      document.getElementById("v-lat").value = e.latlng.lat.toFixed(4);
      document.getElementById("v-long").value = e.latlng.lng.toFixed(4);
      vMarker.setLatLng(e.latlng);
    });

    attnMap = L.map("attention-map", { attributionControl: false }).setView([47.6205, -122.3493], 11);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { maxZoom: 18 }).addTo(attnMap);
  }

  function collectInput() {
    return {
      lat: parseFloat(document.getElementById("v-lat").value),
      long: parseFloat(document.getElementById("v-long").value),
      sqft_living: parseInt(document.getElementById("v-sqft-living").value),
      sqft_lot: parseInt(document.getElementById("v-sqft-lot").value),
      bedrooms: parseInt(document.getElementById("v-beds").value),
      bathrooms: parseFloat(document.getElementById("v-baths").value),
      floors: parseFloat(document.getElementById("v-floors").value),
      yr_built: parseInt(document.getElementById("v-yrbuilt").value),
      condition: parseInt(document.getElementById("v-condition").value),
      grade: parseInt(document.getElementById("v-grade").value),
    };
  }

  function renderBreakdownChart(structural, spatial) {
    const ctx = document.getElementById("chart-breakdown").getContext("2d");
    if (breakdownChart) breakdownChart.destroy();
    breakdownChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Structural Value", "Spatial / Neighborhood Premium"],
        datasets: [{
          data: [structural, spatial],
          backgroundColor: [THEME.teal, THEME.purple],
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { callback: (v) => "$" + (v / 1000) + "k" }, grid: { color: THEME.grid } },
          y: { grid: { display: false } },
        },
      },
    });
  }

  function renderZipComparison(zc) {
    const wrap = document.getElementById("chart-zip-compare").closest(".card");
    if (!zc) { wrap.querySelector(".chart-wrap").innerHTML = '<div class="empty-state">No zip data available for this location</div>'; return; }
    const ctx = document.getElementById("chart-zip-compare").getContext("2d");
    if (zipCompareChart) zipCompareChart.destroy();
    zipCompareChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Living Sqft", "Grade", "Price / Sqft"],
        datasets: [
          {
            label: `This Property`,
            data: [zc.subject.sqft_living, zc.subject.grade, zc.subject.price_per_sqft],
            backgroundColor: THEME.teal, borderRadius: 4,
          },
          {
            label: `Zip ${zc.zipcode} Avg (n=${zc.count})`,
            data: [zc.neighborhood_avg.sqft_living, zc.neighborhood_avg.grade, zc.neighborhood_avg.price_per_sqft],
            backgroundColor: THEME.purple, borderRadius: 4,
          },
        ],
      },
      options: {
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
        scales: {
          y: { grid: { color: THEME.grid } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function renderAttentionWeights(neighbors) {
    const ctx = document.getElementById("chart-attention-weights").getContext("2d");
    if (attnWeightChart) attnWeightChart.destroy();
    attnWeightChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: neighbors.map((n) => "#" + n.id),
        datasets: [{
          label: "Attention weight",
          data: neighbors.map((n) => n.attention_weight),
          backgroundColor: neighbors.map((_, i) => i === 0 ? THEME.teal : "rgba(0,242,195," + (0.85 - i * 0.08) + ")"),
          borderRadius: 4,
        }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { title: { display: true, text: "Weight" }, grid: { color: THEME.grid } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function renderAttentionMap(lat, long, neighbors) {
    attnMap.eachLayer((l) => { if (!(l instanceof L.TileLayer)) attnMap.removeLayer(l); });
    attnMap.setView([lat, long], 12);
    L.circleMarker([lat, long], { radius: 9, color: THEME.magenta, fillColor: THEME.magenta, fillOpacity: 1 })
      .bindTooltip("Target Property", { permanent: false }).addTo(attnMap);

    const maxW = Math.max(...neighbors.map((n) => n.attention_weight));
    neighbors.forEach((n) => {
      const w = n.attention_weight / maxW;
      L.polyline([[lat, long], [n.lat, n.long]], {
        color: THEME.teal, weight: 1 + w * 5, opacity: 0.25 + w * 0.55,
      }).addTo(attnMap);
      L.circleMarker([n.lat, n.long], {
        radius: 5 + w * 6, color: THEME.teal, fillColor: THEME.teal, fillOpacity: 0.4 + w * 0.5,
      }).bindTooltip(`${fmtMoney(n.price)} · weight ${n.attention_weight.toFixed(3)}`).addTo(attnMap);
    });
  }

  function renderNeighborList(neighbors) {
    lastNeighbors = neighbors;
    const list = document.getElementById("neighbor-list");
    list.innerHTML = "";
    neighbors.forEach((n, i) => {
      const row = document.createElement("div");
      row.className = "neighbor-row";
      row.innerHTML = `<span>#${i + 1} · ${fmtMoney(n.price)} · ${n.bedrooms}bd/${n.bathrooms}ba · ${n.sqft_living}sqft</span>
                        <span class="n-weight">w=${n.attention_weight.toFixed(3)}</span>`;
      row.addEventListener("click", () => showNeighborDetail(n));
      list.appendChild(row);
    });
  }

  function showNeighborDetail(n) {
    const el = document.getElementById("neighbor-detail");
    el.classList.remove("hidden");
    el.innerHTML = `
      <div><b>Property #${n.id}</b></div>
      <div>Price: ${fmtMoney(n.price)}</div>
      <div>Specs: ${n.bedrooms} bed / ${n.bathrooms} bath / ${n.sqft_living} sqft</div>
      <div>Grade ${n.grade} · Condition ${n.condition} · Built ${n.yr_built}</div>
      <div>Distance: ${n.distance_mi.toFixed(2)} mi</div>
      <div>Attention weight: ${n.attention_weight.toFixed(4)}</div>
    `;
  }

  async function runValuation() {
    const input = collectInput();
    const btn = document.getElementById("btn-valuate");
    btn.textContent = "Running...";
    btn.disabled = true;
    try {
      const res = await apiPost("/api/valuate", input);
      document.getElementById("v-result-empty").classList.add("hidden");
      document.getElementById("v-result").classList.remove("hidden");
      document.getElementById("v-price").textContent = fmtMoney(res.predicted_price);
      document.getElementById("v-range").textContent = `${fmtMoney(res.confidence_low)} — ${fmtMoney(res.confidence_high)}`;
      renderBreakdownChart(res.structural_value, res.spatial_value);
      renderAttentionMap(input.lat, input.long, res.neighbors);
      renderNeighborList(res.neighbors);
      renderZipComparison(res.zip_comparison);
      renderAttentionWeights(res.neighbors);
      vMap.setView([input.lat, input.long], 12);
      vMarker.setLatLng([input.lat, input.long]);
    } catch (e) {
      alert("Valuation failed: " + e.message);
    } finally {
      btn.textContent = "Run Valuation";
      btn.disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    setTimeout(initMaps, 50);
    document.getElementById("btn-valuate").addEventListener("click", runValuation);
  });
  window.addEventListener("tabshown", (e) => {
    if (e.detail === "tab1" && vMap) { vMap.invalidateSize(); attnMap.invalidateSize(); }
  });
})();
