const THEME = {
  teal: "#00f2c3",
  magenta: "#ff2ea6",
  purple: "#a855f7",
  orange: "#ff9f1c",
  red: "#ff3b5c",
  green: "#2de08e",
  blue: "#4d8dff",
  grid: "#161616",
  text: "#7c8f8c",
};

if (window.Chart) {
  Chart.defaults.color = THEME.text;
  Chart.defaults.font.family = "'SFMono-Regular', 'JetBrains Mono', monospace";
  Chart.defaults.font.size = 10;
  Chart.defaults.borderColor = THEME.grid;
  Chart.defaults.plugins.legend.labels.color = THEME.text;
}

function fmtMoney(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return "$" + Math.round(v).toLocaleString();
}
function fmtPct(v, digits = 1) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return v.toFixed(digits) + "%";
}
function dealBadgeHtml(label) {
  const cls = label === "Undervalued" ? "badge-undervalued" : (label === "Overpriced" ? "badge-overpriced" : "badge-fair");
  return `<span class="badge ${cls}">${label}</span>`;
}
async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
