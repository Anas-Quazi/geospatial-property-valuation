(function () {
  let whatifChart, waterfallChart;
  const sliderPairs = [
    ["s-sqft", "s-sqft-val", (v) => v],
    ["s-bed", "s-bed-val", (v) => v],
    ["s-bath", "s-bath-val", (v) => v],
    ["s-grade", "s-grade-val", (v) => "+" + v],
    ["s-cond", "s-cond-val", (v) => "+" + v],
  ];

  function collectBase() {
    return {
      lat: parseFloat(document.getElementById("w-lat").value),
      long: parseFloat(document.getElementById("w-long").value),
      sqft_living: parseInt(document.getElementById("w-sqft-living").value),
      sqft_lot: parseInt(document.getElementById("w-sqft-lot").value),
      bedrooms: parseInt(document.getElementById("w-beds").value),
      bathrooms: parseFloat(document.getElementById("w-baths").value),
      yr_built: parseInt(document.getElementById("w-yrbuilt").value),
      condition: parseInt(document.getElementById("w-condition").value),
      grade: parseInt(document.getElementById("w-grade").value),
    };
  }

  function collectDelta() {
    return {
      add_sqft: parseFloat(document.getElementById("s-sqft").value),
      add_beds: parseFloat(document.getElementById("s-bed").value),
      add_baths: parseFloat(document.getElementById("s-bath").value),
      grade_upgrade: parseFloat(document.getElementById("s-grade").value),
      condition_upgrade: parseFloat(document.getElementById("s-cond").value),
    };
  }

  function renderChart(before, after) {
    const ctx = document.getElementById("chart-whatif").getContext("2d");
    if (whatifChart) whatifChart.destroy();
    whatifChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Before", "After"],
        datasets: [{ data: [before, after], backgroundColor: [THEME.blue, THEME.teal], borderRadius: 4 }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { ticks: { callback: (v) => "$" + (v / 1000) + "k" }, grid: { color: THEME.grid } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function renderWaterfall(steps) {
    const card = document.getElementById("w-waterfall-card");
    if (!steps || steps.length < 2) { card.classList.add("hidden"); return; }
    card.classList.remove("hidden");
    const ctx = document.getElementById("chart-waterfall").getContext("2d");
    if (waterfallChart) waterfallChart.destroy();
    const colors = [THEME.blue, THEME.teal, THEME.purple, THEME.orange, THEME.magenta, THEME.green];
    waterfallChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: steps.map((s) => s.label),
        datasets: [{
          label: "Cumulative value",
          data: steps.map((s) => s.price),
          backgroundColor: steps.map((_, i) => colors[i % colors.length]),
          borderRadius: 4,
        }],
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                if (ctx.dataIndex === 0) return "";
                const gain = steps[ctx.dataIndex].price - steps[ctx.dataIndex - 1].price;
                return (gain >= 0 ? "+" : "") + fmtMoney(gain) + " vs previous step";
              },
            },
          },
        },
        scales: {
          y: { ticks: { callback: (v) => "$" + (v / 1000) + "k" }, grid: { color: THEME.grid } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  async function recalc() {
    const btn = document.getElementById("btn-recalculate");
    btn.textContent = "Calculating...";
    btn.disabled = true;
    try {
      const body = {
        base: collectBase(),
        delta: collectDelta(),
        renovation_cost: parseFloat(document.getElementById("w-cost").value) || null,
      };
      const res = await apiPost("/api/whatif", body);
      document.getElementById("w-result-empty").classList.add("hidden");
      document.getElementById("w-result").classList.remove("hidden");
      document.getElementById("w-before").textContent = fmtMoney(res.before_price);
      document.getElementById("w-after").textContent = fmtMoney(res.after_price);
      const roiEl = document.getElementById("w-roi");
      let roiHtml = `Value gain: <b>${fmtMoney(res.value_gain)}</b> (${fmtPct(res.value_gain_pct)})`;
      if (res.roi_pct !== null && res.roi_pct !== undefined) {
        const cls = res.roi_pct >= 0 ? "roi-pos" : "roi-neg";
        roiHtml += ` &nbsp;·&nbsp; ROI: <span class="${cls}">${res.roi_pct.toFixed(1)}%</span>`;
      }
      roiEl.innerHTML = roiHtml;
      renderChart(res.before_price, res.after_price);
      renderWaterfall(res.waterfall);
    } catch (e) {
      alert("Simulation failed: " + e.message);
    } finally {
      btn.textContent = "Recalculate";
      btn.disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    sliderPairs.forEach(([sliderId, valId, fmt]) => {
      const slider = document.getElementById(sliderId);
      const label = document.getElementById(valId);
      slider.addEventListener("input", () => { label.textContent = fmt(slider.value); });
    });
    document.getElementById("btn-recalculate").addEventListener("click", recalc);
  });
})();
