import asyncio
import json
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================================================
# CONFIG
# =========================================================
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

POLL_SECONDS = 30  # fixed
DEFAULT_SLUGS = [
    "khamenei-out-as-supreme-leader-of-iran-by-june-30-747",
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Streamlit)",
    "Accept": "application/json,text/plain,*/*",
}

# =========================================================
# HELPERS
# =========================================================
def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"

def as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []

def extract_yes_no_tokens(market: dict) -> Tuple[Optional[str], Optional[str]]:
    toks = market.get("clobTokenIds") or market.get("clob_token_ids")
    toks = [str(t) for t in as_list(toks) if t not in (None, "", "null")]
    if len(toks) >= 2:
        return toks[0], toks[1]
    if len(toks) == 1:
        return toks[0], None
    return None, None

def leaning_label(p_yes: Optional[float]) -> str:
    if p_yes is None:
        return "—"
    if p_yes > 0.5:
        return "Leans YES"
    if p_yes < 0.5:
        return "Leans NO"
    return "Even"

def trend_label(delta: Optional[float], deadband: float = 0.0025) -> str:
    if delta is None or not np.isfinite(delta):
        return "—"
    if delta > deadband:
        return "Trending YES"
    if delta < -deadband:
        return "Trending NO"
    return "Flat"

def compute_trend(hist: pd.DataFrame, lookback_minutes: int = 60) -> Tuple[Optional[float], Optional[float]]:
    if hist is None or hist.empty:
        return None, None
    now = pd.Timestamp.now(tz="UTC")
    current = float(hist.iloc[-1]["price"])
    target = now - pd.Timedelta(minutes=lookback_minutes)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])
    return current, current - past

# =========================================================
# API (sync)
# =========================================================
@st.cache_data(ttl=900)
def gamma_market_by_slug(slug: str) -> Optional[dict]:
    r = requests.get(
        f"{GAMMA_BASE}/markets/slug/{slug}",
        timeout=20,
        headers=DEFAULT_HEADERS,
    )
    if r.status_code != 200:
        return None
    j = r.json()
    return j[0] if isinstance(j, list) and j else j

def _sync_clob_midpoint(token_id: str) -> Optional[float]:
    r = requests.get(
        f"{CLOB_BASE}/midpoint",
        params={"token_id": token_id},
        timeout=10,
        headers=DEFAULT_HEADERS,
    )
    if r.status_code != 200:
        return None
    return safe_float(r.json().get("mid"))

def _sync_clob_price_history(token_id: str, interval="1w", fidelity_min=15) -> pd.DataFrame:
    r = requests.get(
        f"{CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity_min},
        timeout=20,
        headers=DEFAULT_HEADERS,
    )
    r.raise_for_status()
    hist = r.json().get("history", [])
    df = pd.DataFrame(hist)
    if df.empty:
        return pd.DataFrame(columns=["ts", "price"])
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    return df[["ts", "price"]].dropna().sort_values("ts")

# =========================================================
# ASYNC WRAPPERS
# =========================================================
async def clob_midpoint_async(token_id: str) -> Optional[float]:
    return await asyncio.to_thread(_sync_clob_midpoint, token_id)

async def clob_history_async(token_id: str) -> pd.DataFrame:
    return await asyncio.to_thread(_sync_clob_price_history, token_id)

async def fetch_market_snapshot(slug: str) -> Dict:
    market = gamma_market_by_slug(slug)
    if not market:
        return {"slug": slug, "error": "market not found"}

    title = market.get("question") or "(no title)"
    end_date = market.get("endDateIso") or market.get("endDate") or "—"
    volume = market.get("volumeNum") or market.get("volume") or "—"

    yes_token, no_token = extract_yes_no_tokens(market)
    if not yes_token:
        return {"slug": slug, "title": title, "error": "missing YES token id"}

    mid_task = asyncio.create_task(clob_midpoint_async(yes_token))
    hist_task = asyncio.create_task(clob_history_async(yes_token))

    yes_mid = await mid_task
    hist = await hist_task
    _, delta_1h = compute_trend(hist, 60)

    return {
        "slug": slug,
        "title": title,
        "end_date": end_date,
        "volume": volume,
        "yes_mid": yes_mid,
        "delta_1h": delta_1h,
        "leaning": leaning_label(yes_mid),
        "trend": trend_label(delta_1h),
        "hist": hist if isinstance(hist, pd.DataFrame) and not hist.empty else None,
    }

async def fetch_all_snapshots(slugs: List[str]) -> List[Dict]:
    return await asyncio.gather(*[fetch_market_snapshot(s) for s in slugs])

def run_async(coro):
    return asyncio.run(coro)

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Polytracker", layout="wide")
st.title("Polymarket Tracker")

# Auto-refresh (same as your working app)
st_autorefresh(interval=POLL_SECONDS * 1000, key="auto_refresh_polytracker")

with st.sidebar:
    st.caption(f"Streamlit {st.__version__}")
    st.caption(f"Auto-refresh: {POLL_SECONDS}s")

    slugs_text = st.text_area(
        "Market slugs (one per line)",
        value="\n".join(DEFAULT_SLUGS),
        height=120,
    )
    slugs = [s.strip() for s in slugs_text.splitlines() if s.strip()]
    debug = st.checkbox("Show debug", value=False)

if not slugs:
    st.info("Add at least one market slug.")
    st.stop()

with st.spinner(f"Fetching {len(slugs)} market(s)..."):
    snapshots = run_async(fetch_all_snapshots(slugs))

# =========================================================
# DASHBOARD
# =========================================================
st.subheader("Dashboard")

rows = []
for s in snapshots:
    if s.get("error"):
        rows.append({
            "Market": s.get("title") or s["slug"],
            "Leaning": "—",
            "Trend (1h)": "—",
            "YES now": "—",
            "Δ 1h": "—",
            "Status": f"Error: {s['error']}",
        })
    else:
        rows.append({
            "Market": s["title"],
            "Leaning": s["leaning"],
            "Trend (1h)": s["trend"],
            "YES now": pct(s["yes_mid"]),
            "Δ 1h": pct(s["delta_1h"]),
            "Status": "OK",
        })

df = pd.DataFrame(rows)
st.dataframe(df, width="stretch")

st.caption("Leaning = YES vs 50%. Trend = change in YES probability over the last hour.")

# =========================================================
# DETAILS
# =========================================================
st.subheader("Details")

valid = [s for s in snapshots if not s.get("error")]
if not valid:
    st.info("No valid markets.")
    st.stop()

pick = st.selectbox("Choose a market", valid, format_func=lambda x: x["title"])

c1, c2, c3 = st.columns(3)
c1.metric("YES (now)", pct(pick.get("yes_mid")))
c2.metric("Leaning", pick.get("leaning"))
c3.metric("Trend (1h)", pick.get("trend"), delta=pct(pick.get("delta_1h")))

st.write(f"End date: {pick.get('end_date')}")
st.write(f"Volume: {pick.get('volume')}")

st.markdown("### YES price history")
hist = pick.get("hist")
if isinstance(hist, pd.DataFrame) and not hist.empty:
    st.line_chart(hist.set_index("ts")["price"])
else:
    st.info("No history available.")

if debug:
    st.json(pick)
