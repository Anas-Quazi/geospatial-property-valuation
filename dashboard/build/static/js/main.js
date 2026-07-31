document.addEventListener("DOMContentLoaded", () => {
  // tab switching
  const btns = document.querySelectorAll(".tab-btn");
  const sections = document.querySelectorAll(".tab-content");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      sections.forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
      window.dispatchEvent(new CustomEvent("tabshown", { detail: btn.dataset.tab }));
    });
  });

  // top summary pills
  apiGet("/api/summary").then((d) => {
    document.getElementById("stat-count").textContent = d.total_properties.toLocaleString();
    document.getElementById("stat-median").textContent = fmtMoney(d.median_price);
    document.getElementById("stat-gnn-mape").textContent = fmtPct(d.gnn_mape, 2);
    document.getElementById("stat-xgb-mape").textContent = fmtPct(d.xgb_mape, 2);
    document.getElementById("stat-improve").textContent = fmtPct(d.improvement_pct, 1);
  }).catch(console.error);
});
