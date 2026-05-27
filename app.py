"""
AlphaScreener - Weekly S&P 500 Stock Grading Tool
Grades ~500 stocks on Value, Quality, Momentum, Safety (1-10 scale).
"""

import sqlite3
import json
import os
import time
import logging
from datetime import datetime, date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, render_template, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)


# ── Jinja2 template helpers ───────────────────────────────────────

def _grade_class(g) -> str:
    """Return CSS class name for a 1-10 grade."""
    if g is None:
        return "grade-1"
    n = max(1, min(10, round(float(g))))
    return f"grade-{n}"


def _grade_label(g) -> str:
    if g is None:
        return "N/A"
    g = float(g)
    if g >= 9: return "Conviction Buy"
    if g >= 7: return "Buy"
    if g >= 5: return "Hold / Neutral"
    if g >= 3: return "Sell"
    return "Strong Sell"


app.jinja_env.globals["grade_class"] = _grade_class
app.jinja_env.globals["grade_label"] = _grade_label
app.jinja_env.globals["now"]         = datetime.utcnow
DB_PATH = os.path.join(os.path.dirname(__file__), "alphascreener.db")

# --- Configurable limits ---
PIPELINE_LIMIT = int(os.environ.get("PIPELINE_LIMIT", 50))  # set to 500 for full run


# ─────────────────────────── DATABASE ────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS scores (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT NOT NULL,
            date           TEXT NOT NULL,
            overall_grade  REAL,
            value_grade    REAL,
            quality_grade  REAL,
            momentum_grade REAL,
            safety_grade   REAL,
            price          REAL,
            sector         TEXT,
            week_return    REAL,
            UNIQUE(ticker, date)
        );

        CREATE TABLE IF NOT EXISTS signals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT NOT NULL,
            date         TEXT NOT NULL,
            signal_name  TEXT NOT NULL,
            signal_value REAL,
            signal_grade REAL,
            UNIQUE(ticker, date, signal_name)
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            ended_at   TEXT,
            status     TEXT,
            tickers_ok INTEGER,
            tickers_err INTEGER
        );
    """)
    conn.commit()
    conn.close()


# ─────────────────────────── S&P 500 LIST ────────────────────────

_SP500_CACHE: list | None = None
_SP500_CACHE_DATE: date | None = None


def get_sp500_tickers() -> list[dict]:
    """Fetch S&P 500 list from Wikipedia (cached per calendar day)."""
    global _SP500_CACHE, _SP500_CACHE_DATE
    today = date.today()
    if _SP500_CACHE and _SP500_CACHE_DATE == today:
        return _SP500_CACHE

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, attrs={"id": "constituents"})
        df = tables[0]
        df.columns = [c.strip() for c in df.columns]
        # Column names vary slightly across Wikipedia edits
        ticker_col  = next(c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower())
        sector_col  = next(c for c in df.columns if "sector" in c.lower() or "gics sector" in c.lower())
        result = [
            {"ticker": row[ticker_col].replace(".", "-"), "sector": row[sector_col]}
            for _, row in df.iterrows()
        ]
        _SP500_CACHE = result
        _SP500_CACHE_DATE = today
        log.info("Fetched %d tickers from Wikipedia", len(result))
        return result
    except Exception as exc:
        log.error("Wikipedia fetch failed: %s", exc)
        # Fallback minimal list so the app still works
        return [
            {"ticker": t, "sector": "Unknown"}
            for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B",
                      "LLY", "AVGO", "TSLA"]
        ]


# ─────────────────────────── SIGNAL CALCULATORS ──────────────────

def _safe(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def compute_signals(ticker: str, sector: str) -> dict | None:
    """
    Download yfinance data and return a dict of {signal_name: raw_value}.
    Returns None if data is insufficient.
    """
    try:
        stk = yf.Ticker(ticker)
        info = stk.info or {}

        # --- Price history ---
        hist = stk.history(period="2y", auto_adjust=True)
        hist.index = hist.index.tz_localize(None)  # Remove timezone info
        
        if hist.empty or len(hist) < 60:
            return None
        closes = hist["Close"].dropna()

        price = _safe(closes.iloc[-1])
        week_return = _safe((closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else np.nan)

        # 12-month and 1-month momentum
        mom_12 = _safe(closes.iloc[-1] / closes.iloc[-253] - 1 if len(closes) >= 253 else np.nan)
        mom_1  = _safe(closes.iloc[-1] / closes.iloc[-22]  - 1 if len(closes) >= 22  else np.nan)

        # Volatility (annualised daily std)
        daily_rets = closes.pct_change().dropna()
        vol = _safe(daily_rets.std() * np.sqrt(252))

        # Beta - simplified, just use info.get("beta")
        beta = _safe(info.get("beta", np.nan))

        # Skewness
        skew = _safe(daily_rets[-252:].skew())

        # Financials
        bs  = stk.balance_sheet
        inc = stk.income_stmt
        cf  = stk.cashflow

        def sheet_val(df, *labels):
            for lbl in labels:
                for col in (df.columns if df is not None and not df.empty else []):
                    if lbl in df.index:
                        v = _safe(df.loc[lbl, col])
                        if not np.isnan(v):
                            return v
            return np.nan

        total_assets     = sheet_val(bs,  "Total Assets")
        total_equity     = sheet_val(bs,  "Stockholders Equity", "Total Stockholder Equity")
        total_debt       = sheet_val(bs,  "Total Debt", "Long Term Debt")
        current_assets   = sheet_val(bs,  "Current Assets")
        current_liab     = sheet_val(bs,  "Current Liabilities")
        cash_ops         = sheet_val(cf,  "Operating Cash Flow", "Cash From Operations")
        gross_profit     = sheet_val(inc, "Gross Profit")
        revenue          = sheet_val(inc, "Total Revenue")
        net_income       = sheet_val(inc, "Net Income")

        market_cap = _safe(info.get("marketCap", np.nan))
        book_val   = _safe(info.get("bookValue",  np.nan))
        eps        = _safe(info.get("trailingEps", np.nan))

        # VALUE
        book_to_market = _safe(book_val / price if price > 0 else np.nan)
        earnings_yield = _safe(eps / price       if price > 0 else np.nan)

        # QUALITY
        gp_ratio       = _safe(gross_profit / total_assets if total_assets > 0 else np.nan)
        roa            = _safe(net_income   / total_assets if total_assets > 0 else np.nan)
        current_ratio  = _safe(current_assets / current_liab if current_liab > 0 else np.nan)
        cfq            = _safe(cash_ops      / net_income    if net_income  != 0 else np.nan)  # cash-flow quality

        # Piotroski F-score (simplified, 0-9 → normalised to 0-1)
        piotroski = _piotroski(roa, cash_ops, total_assets, net_income,
                               total_debt, current_assets, current_liab, gross_profit, revenue)

        # Earnings quality = |accruals| inverted
        accruals = _safe((net_income - cash_ops) / total_assets if total_assets > 0 else np.nan)
        earnings_quality = _safe(-abs(accruals) if not np.isnan(accruals) else np.nan)

        # SAFETY
        leverage = _safe(total_debt / total_equity if total_equity > 0 else np.nan)

        return {
            # value
            "book_to_market":   book_to_market,
            "earnings_yield":   earnings_yield,
            # quality
            "gp_ratio":         gp_ratio,
            "roa":              roa,
            "current_ratio":    current_ratio,
            "cash_flow_quality": cfq,
            "piotroski":        piotroski,
            "earnings_quality": earnings_quality,
            # momentum
            "mom_12":           mom_12,
            "mom_1":            mom_1,
            # safety
            "volatility":       vol,
            "beta":             beta,
            "leverage":         leverage,
            "skewness":         skew,
            # meta
            "_price":           price,
            "_week_return":     week_return,
            "_sector":          sector,
        }
    except Exception as exc:
        log.warning("compute_signals(%s) failed: %s", ticker, exc)
        return None


def _piotroski(roa, cash_ops, total_assets, net_income,
               total_debt, current_assets, current_liab,
               gross_profit, revenue) -> float:
    """Simplified Piotroski F-score normalised to [0, 1]."""
    score = 0
    if not np.isnan(roa)           and roa > 0:           score += 1
    if not np.isnan(cash_ops)      and cash_ops > 0:       score += 1
    if not np.isnan(net_income)    and net_income > 0:     score += 1
    if not np.isnan(total_debt)    and total_debt < total_assets * 0.5: score += 1
    if not np.isnan(current_assets) and not np.isnan(current_liab) and current_liab > 0:
        if current_assets / current_liab > 1.5:            score += 1
    if not np.isnan(gross_profit)  and not np.isnan(revenue) and revenue > 0:
        if gross_profit / revenue > 0.3:                   score += 1
    score += 3  # partial credit so no stock is always 0
    return min(score / 9, 1.0)


# ─────────────────────────── GRADING ─────────────────────────────

VALUE_SIGNALS    = ["book_to_market", "earnings_yield"]
QUALITY_SIGNALS  = ["gp_ratio", "roa", "current_ratio", "cash_flow_quality",
                    "piotroski", "earnings_quality"]
MOMENTUM_SIGNALS = ["mom_12", "mom_1"]
SAFETY_SIGNALS   = ["volatility", "beta", "leverage", "skewness"]

# Higher-is-better vs lower-is-better
LOWER_IS_BETTER  = {"volatility", "beta", "leverage", "skewness"}


def rank_to_grade(series: pd.Series) -> pd.Series:
    """Convert a raw-signal series to 1-10 grades via percentile rank."""
    ranks  = series.rank(pct=True, na_option="keep")
    grades = (ranks * 9 + 1).clip(1, 10)
    return grades


def grade_signals(all_signals: dict[str, dict]) -> dict[str, dict]:
    """
    all_signals: {ticker: {signal_name: raw_value, ...}}
    Returns: {ticker: {signal_name: grade_1_to_10, ...}}
    """
    signal_names = (VALUE_SIGNALS + QUALITY_SIGNALS +
                    MOMENTUM_SIGNALS + SAFETY_SIGNALS)
    df = pd.DataFrame(
        {ticker: {s: v for s, v in sigs.items() if not s.startswith("_")}
         for ticker, sigs in all_signals.items()}
    ).T  # shape: (n_stocks, n_signals)

    grades_df = pd.DataFrame(index=df.index)
    for sig in signal_names:
        if sig not in df.columns:
            grades_df[sig] = np.nan
            continue
        col = df[sig]
        if sig in LOWER_IS_BETTER:
            col = -col  # invert so rank works correctly
        grades_df[sig] = rank_to_grade(col)

    return grades_df.to_dict(orient="index")


def compute_category_grades(signal_grades: dict) -> dict:
    """Weighted average of signal grades into category grades."""
    def avg(*names):
        vals = [signal_grades.get(n) for n in names
                if signal_grades.get(n) is not None and not np.isnan(signal_grades.get(n, np.nan))]
        return float(np.mean(vals)) if vals else np.nan

    value    = avg(*VALUE_SIGNALS)
    quality  = avg(*QUALITY_SIGNALS)
    momentum = avg(*MOMENTUM_SIGNALS)
    safety   = avg(*SAFETY_SIGNALS)

    parts = []
    weights = [(value, 0.30), (quality, 0.35), (momentum, 0.20), (safety, 0.15)]
    total_w = 0.0
    for v, w in weights:
        if not np.isnan(v):
            parts.append(v * w)
            total_w += w

    overall = sum(parts) / total_w if total_w > 0 else np.nan
    return {
        "overall":  round(overall,  2) if not np.isnan(overall)  else None,
        "value":    round(value,    2) if not np.isnan(value)    else None,
        "quality":  round(quality,  2) if not np.isnan(quality)  else None,
        "momentum": round(momentum, 2) if not np.isnan(momentum) else None,
        "safety":   round(safety,   2) if not np.isnan(safety)   else None,
    }


# ─────────────────────────── PIPELINE ────────────────────────────

def run_pipeline(limit: int = PIPELINE_LIMIT) -> dict:
    """
    Full scoring pipeline:
      1. Fetch S&P 500 tickers
      2. Download signals for each ticker
      3. Grade signals relative to universe
      4. Save to DB
    """
    run_date = date.today().isoformat()
    started  = datetime.utcnow().isoformat()
    log.info("Pipeline started – limit=%d", limit)

    tickers_info = get_sp500_tickers()[:limit]

    # Step 1: raw signals
    raw: dict[str, dict] = {}
    errors = []
    for i, item in enumerate(tickers_info):
        ticker = item["ticker"]
        sector = item["sector"]
        log.info("[%d/%d] %s", i + 1, len(tickers_info), ticker)
        sigs = compute_signals(ticker, sector)
        if sigs:
            raw[ticker] = sigs
        else:
            errors.append(ticker)
        time.sleep(0.3)  # be polite to Yahoo Finance

    if not raw:
        return {"status": "error", "message": "No data retrieved"}

    # Step 2: grade
    signal_grades_map = grade_signals(raw)

    # Step 3: persist
    conn = get_db()
    c    = conn.cursor()

    ok_count = 0
    for ticker, sigs in raw.items():
        sg = signal_grades_map.get(ticker, {})
        cats = compute_category_grades(sg)

        price       = sigs.get("_price")
        week_return = sigs.get("_week_return")
        sector      = sigs.get("_sector", "Unknown")

        # Upsert scores
        c.execute("""
            INSERT INTO scores
                (ticker, date, overall_grade, value_grade, quality_grade,
                 momentum_grade, safety_grade, price, sector, week_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                overall_grade  = excluded.overall_grade,
                value_grade    = excluded.value_grade,
                quality_grade  = excluded.quality_grade,
                momentum_grade = excluded.momentum_grade,
                safety_grade   = excluded.safety_grade,
                price          = excluded.price,
                sector         = excluded.sector,
                week_return    = excluded.week_return
        """, (ticker, run_date,
              cats["overall"], cats["value"], cats["quality"],
              cats["momentum"], cats["safety"],
              price, sector, week_return))

        # Upsert signals
        all_signal_names = (VALUE_SIGNALS + QUALITY_SIGNALS +
                            MOMENTUM_SIGNALS + SAFETY_SIGNALS)
        for sig_name in all_signal_names:
            raw_val   = sigs.get(sig_name)
            grade_val = sg.get(sig_name)
            c.execute("""
                INSERT INTO signals (ticker, date, signal_name, signal_value, signal_grade)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date, signal_name) DO UPDATE SET
                    signal_value = excluded.signal_value,
                    signal_grade = excluded.signal_grade
            """, (ticker, run_date, sig_name, raw_val, grade_val))

        ok_count += 1

    # Record run
    ended = datetime.utcnow().isoformat()
    c.execute("""
        INSERT INTO pipeline_runs (started_at, ended_at, status, tickers_ok, tickers_err)
        VALUES (?, ?, ?, ?, ?)
    """, (started, ended, "success", ok_count, len(errors)))

    conn.commit()
    conn.close()

    log.info("Pipeline done – ok=%d err=%d", ok_count, len(errors))
    return {
        "status":      "success",
        "date":        run_date,
        "tickers_ok":  ok_count,
        "tickers_err": len(errors),
        "errors":      errors[:20],
    }


# ─────────────────────────── HELPERS ─────────────────────────────

def latest_date(conn) -> str | None:
    row = conn.execute("SELECT MAX(date) as d FROM scores").fetchone()
    return row["d"] if row else None


def get_latest_scores(conn) -> list[dict]:
    d = latest_date(conn)
    if not d:
        return []
    rows = conn.execute(
        "SELECT * FROM scores WHERE date = ? ORDER BY overall_grade DESC NULLS LAST",
        (d,)
    ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────── ROUTES ──────────────────────────────

@app.route("/")
def index():
    conn = get_db()
    d    = latest_date(conn)
    conn.close()
    return render_template("index.html", last_updated=d or "Never")


@app.route("/summary")
def summary():
    conn    = get_db()
    scores  = get_latest_scores(conn)
    d       = latest_date(conn)
    conn.close()
    return render_template("summary.html", scores=scores, last_updated=d or "Never")


@app.route("/top-picks")
def top_picks():
    conn   = get_db()
    scores = get_latest_scores(conn)
    d      = latest_date(conn)
    conn.close()
    valid  = [s for s in scores if s.get("overall_grade") is not None]
    buys   = sorted(valid, key=lambda x: x["overall_grade"], reverse=True)[:10]
    sells  = sorted(valid, key=lambda x: x["overall_grade"])[:10]
    return render_template("top_picks.html", buys=buys, sells=sells, last_updated=d or "Never")


@app.route("/sectors")
def sectors_view():
    conn   = get_db()
    scores = get_latest_scores(conn)
    d      = latest_date(conn)
    conn.close()
    sector_data = _build_sector_data(scores)
    return render_template("sectors.html", sectors=sector_data, last_updated=d or "Never")


# ── API endpoints ──

@app.route("/api/scores")
def api_scores():
    conn   = get_db()
    scores = get_latest_scores(conn)
    conn.close()
    return jsonify(scores)


@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    ticker = ticker.upper()
    conn   = get_db()
    d      = latest_date(conn)
    if not d:
        conn.close()
        return jsonify({"error": "No data"}), 404

    score = conn.execute(
        "SELECT * FROM scores WHERE ticker=? AND date=?", (ticker, d)
    ).fetchone()
    if not score:
        conn.close()
        return jsonify({"error": "Ticker not found"}), 404

    sigs = conn.execute(
        "SELECT signal_name, signal_value, signal_grade FROM signals WHERE ticker=? AND date=?",
        (ticker, d)
    ).fetchall()
    conn.close()

    return jsonify({
        "score":   dict(score),
        "signals": [dict(s) for s in sigs],
    })


@app.route("/api/run-pipeline")
def api_run_pipeline():
    limit = int(request.args.get("limit", PIPELINE_LIMIT))
    result = run_pipeline(limit=limit)
    return jsonify(result)


@app.route("/api/sectors")
def api_sectors():
    conn   = get_db()
    scores = get_latest_scores(conn)
    conn.close()
    return jsonify(_build_sector_data(scores))


def _build_sector_data(scores: list[dict]) -> list[dict]:
    """Aggregate scores by sector."""
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for s in scores:
        sec = s.get("sector") or "Unknown"
        if s.get("overall_grade") is not None:
            buckets[sec].append(s)

    result = []
    for sector, stocks in sorted(buckets.items()):
        grades = [s["overall_grade"] for s in stocks]
        avg    = round(float(np.mean(grades)), 2)
        best   = max(stocks, key=lambda x: x["overall_grade"])
        worst  = min(stocks, key=lambda x: x["overall_grade"])

        # Grade distribution buckets
        dist = {
            "9-10": sum(1 for g in grades if g >= 9),
            "7-8":  sum(1 for g in grades if 7 <= g < 9),
            "5-6":  sum(1 for g in grades if 5 <= g < 7),
            "3-4":  sum(1 for g in grades if 3 <= g < 5),
            "1-2":  sum(1 for g in grades if g < 3),
        }

        result.append({
            "sector":       sector,
            "avg_grade":    avg,
            "stock_count":  len(stocks),
            "high_count":   sum(1 for g in grades if g >= 7),
            "low_count":    sum(1 for g in grades if g <= 3),
            "best":         {"ticker": best["ticker"], "grade": best["overall_grade"]},
            "worst":        {"ticker": worst["ticker"], "grade": worst["overall_grade"]},
            "distribution": dist,
        })

    return result


# ─────────────────────────── MAIN ────────────────────────────────

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)