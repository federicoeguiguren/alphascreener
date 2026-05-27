/* ═══════════════════════════════════════════
   AlphaScreener – Frontend Logic
   ═══════════════════════════════════════════ */

// ── Grade helpers ─────────────────────────────────────────────────

function gradeLabel(g) {
  if (g >= 9) return "Conviction Buy";
  if (g >= 7) return "Buy";
  if (g >= 5) return "Hold / Neutral";
  if (g >= 3) return "Sell";
  return "Strong Sell";
}

function gradeClass(g) {
  const n = Math.round(g);
  return `grade-${Math.max(1, Math.min(10, n))}`;
}

// ── Table: search + filter + sort ────────────────────────────────

(function initTable() {
  const table = document.getElementById("stocks-table");
  if (!table) return;

  const tbody      = document.getElementById("stocks-tbody");
  const searchBox  = document.getElementById("search-box");
  const sectorSel  = document.getElementById("sector-filter");
  const gradeSel   = document.getElementById("grade-filter");
  const clearBtn   = document.getElementById("clear-filters");
  const countEl    = document.getElementById("visible-count");

  let sortCol = "overall";
  let sortAsc = false;

  // Number all rows initially
  function reNumber() {
    const rows = [...tbody.querySelectorAll("tr:not(.hidden)")];
    rows.forEach((r, i) => {
      const numCell = r.querySelector(".row-num");
      if (numCell) numCell.textContent = i + 1;
    });
    if (countEl) countEl.textContent = `Showing ${rows.length} stocks`;
  }

  function applyFilters() {
    const q      = (searchBox?.value || "").toLowerCase();
    const sec    = (sectorSel?.value || "").toLowerCase();
    const gRange = gradeSel?.value || "";

    let lo = 0, hi = 11;
    if (gRange === "9") { lo = 9; hi = 11; }
    else if (gRange === "7") { lo = 7; hi = 9; }
    else if (gRange === "5") { lo = 5; hi = 7; }
    else if (gRange === "3") { lo = 3; hi = 5; }
    else if (gRange === "1") { lo = 0; hi = 3; }

    [...tbody.querySelectorAll("tr")].forEach(row => {
      const ticker  = (row.dataset.ticker  || "").toLowerCase();
      const sector  = (row.dataset.sector  || "").toLowerCase();
      const overall = parseFloat(row.dataset.overall || 0);

      const matchQ = !q || ticker.includes(q) || sector.includes(q);
      const matchS = !sec || sector === sec;
      const matchG = !gRange || (overall >= lo && overall < hi);

      row.classList.toggle("hidden", !(matchQ && matchS && matchG));
    });

    reNumber();
  }

  // Sorting
  function sortTable(col) {
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = col === "ticker" || col === "sector"; }

    const rows = [...tbody.querySelectorAll("tr")];
    rows.sort((a, b) => {
      let av, bv;
      if (col === "ticker" || col === "sector") {
        av = a.dataset[col] || "";
        bv = b.dataset[col] || "";
        return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      av = parseFloat(a.dataset[col] || 0);
      bv = parseFloat(b.dataset[col] || 0);
      return sortAsc ? av - bv : bv - av;
    });
    rows.forEach(r => tbody.appendChild(r));
    reNumber();
  }

  // Column header click
  table.querySelectorAll("th[data-sort]").forEach(th => {
    th.addEventListener("click", () => sortTable(th.dataset.sort));
  });

  searchBox?.addEventListener("input",  applyFilters);
  sectorSel?.addEventListener("change", applyFilters);
  gradeSel?.addEventListener("change",  applyFilters);
  clearBtn?.addEventListener("click", () => {
    if (searchBox)  searchBox.value  = "";
    if (sectorSel)  sectorSel.value  = "";
    if (gradeSel)   gradeSel.value   = "";
    applyFilters();
  });

  // Initial render
  reNumber();
})();


// ── Pipeline runner ───────────────────────────────────────────────

(function initPipelineBtn() {
  const btn      = document.getElementById("run-pipeline-btn");
  const statusEl = document.getElementById("pipeline-status");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled    = true;
    btn.textContent = "Running… (this may take a few minutes)";
    if (statusEl) {
      statusEl.className = "pipeline-status";
      statusEl.textContent = "Pipeline started — scoring top 50 stocks…";
      statusEl.classList.remove("hidden");
    }

    try {
      const res  = await fetch("/api/run-pipeline");
      const data = await res.json();
      if (data.status === "success") {
        if (statusEl) {
          statusEl.className   = "pipeline-status ok";
          statusEl.textContent =
            `✅ Done! Scored ${data.tickers_ok} stocks on ${data.date}. Reload to see results.`;
        }
        btn.textContent = "✅ Done — reload page";
        btn.onclick = () => location.reload();
        btn.disabled = false;
      } else {
        throw new Error(data.message || "Unknown error");
      }
    } catch (err) {
      if (statusEl) {
        statusEl.className   = "pipeline-status err";
        statusEl.textContent = `❌ Error: ${err.message}`;
      }
      btn.disabled    = false;
      btn.textContent = "Retry Pipeline";
    }
  });
})();


// ── Tooltip: grade badge hover ────────────────────────────────────

(function initTooltips() {
  const signalDescriptions = {
    "book_to_market":   "Book value ÷ price. Higher means cheaper relative to assets.",
    "earnings_yield":   "EPS ÷ price (inverse P/E). Higher means more earnings per dollar invested.",
    "gp_ratio":         "Gross profit ÷ total assets. Measures operational profitability.",
    "roa":              "Net income ÷ total assets. Return on assets — overall efficiency.",
    "current_ratio":    "Current assets ÷ current liabilities. Liquidity measure.",
    "cash_flow_quality":"Operating cash flow ÷ net income. Higher = earnings backed by cash.",
    "piotroski":        "9-point financial health score. Higher is stronger balance sheet.",
    "earnings_quality": "Inverse of accruals ratio. Low accruals = higher earnings quality.",
    "mom_12":           "12-month price return. Trailing momentum signal.",
    "mom_1":            "1-month price return. Short-term trend signal.",
    "volatility":       "Annualised daily return std dev. Lower is safer.",
    "beta":             "Market sensitivity. Lower beta = less market exposure.",
    "leverage":         "Total debt ÷ equity. Lower leverage means a stronger balance sheet.",
    "skewness":         "Return distribution skewness. Negative skew is riskier.",
  };

  // Simple inline tooltip via title attribute enhancement
  document.querySelectorAll(".grade-badge[title]").forEach(el => {
    const grade = parseFloat(el.textContent.trim());
    if (!isNaN(grade)) {
      el.title = `${el.title} (${gradeLabel(grade)})`;
    }
  });
})();


// ── Auto-refresh check (every 60s, silent) ────────────────────────

(function autoRefresh() {
  // Only refresh if there's a scores table already loaded
  if (!window.SCORES_LOADED || window.SCORES_LOADED === 0) return;

  const CHECK_INTERVAL = 60_000;
  let lastDate = null;

  async function checkForUpdates() {
    try {
      const res  = await fetch("/api/scores");
      const data = await res.json();
      if (!data.length) return;
      const newest = data[0].date;
      if (!lastDate) { lastDate = newest; return; }
      if (newest !== lastDate) location.reload();
    } catch { /* silent */ }
  }

  setInterval(checkForUpdates, CHECK_INTERVAL);
  checkForUpdates();
})();
