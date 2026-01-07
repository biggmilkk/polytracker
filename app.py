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

POLL_SECONDS = 30
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

def pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.2f}%"

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

def leaning_label(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return "Leans YES" if p > 0.5 else "Leans NO" if p < 0.5 else "Even"

def trend_label(delta: Optional[float], deadband: float = 0.0025) -> str:
    if delta is None or not np.isfinite(delta):
        return "—"
    if delta > deadband:
        return "Trending YES"
    if delta < -deadband:
        return "Trending NO"
    return "Flat"

def compute_trend(hist: pd.DataFrame, lookback_minutes: int = 60):
    if hist is None or hist.empty:
        return None
    now = pd.Timestamp.now(tz="UTC")
    current = float(hist.iloc[-1]["price"])
    target = now - pd.Timedelta(minutes=lookback_minutes)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])
    return current - past

# =========================================================
# API
# =========================================================
@st.cache_data(ttl=900)
def gamma_market_by_slug(slug: str) -> Optional[dict]:
    r = requests.get(f"{GAMMA_BASE}/markets/slug/{slug}", timeout=20)
    if r.status_code != 200:
        return None
    j = r.json()
    return j[0] if isinstance(j, list) and j else j

def _sync_mid(token_id: str) -> Optional[float]:
    r = requests.get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id}, timeout=10)
    if r.status_code != 200:
        return None
    return safe_float(r.json().get("mid"))

def _sync_hist(token_id: str) -> pd.DataFrame:
    r = requests.get(
        f"{CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": "1w", "fidelity": 15},
        timeout=20,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("history", []))
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    return df[["ts", "price"]].dropna()

async def fetch_snapshot(slug: str) -> Dict:
    market = gamma_market_by_slug(slug)
    if not market:
        return {"slug": slug, "error": "market not found"}

    yes_token, _ = extract_yes_no_tokens(market)
    if not yes_token:
        return {"slug": slug, "error": "missing YES token"}

    mid_task = asyncio.to_thread(_sync_mid, yes_token)
    hist_task = asyncio.to_thread(_sync_hist, yes_token)

    yes_mid, hist = await asyncio.gather(mid_task, hist_task)

    return {
        "slug": slug,
        "title": market.get("question", slug),
        "end_date": market.get("endDateIso") or market.get("endDate"),
        "volume": market.get("volumeNum"),
        "yes_mid": yes_mid,
        "delta_1h": compute_trend(hist),
        "hist": hist,
    }

def run_async(coro):
    return asyncio.run(coro)

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Polytracker", layout="wide")
st.title("Polymarket Tracker")

st_autorefresh(interval=POLL_SECONDS * 1000, key="refresh")

with st.sidebar:
    slugs_text = st.text_area("Market slugs", "\n".join(DEFAULT_SLUGS))
    slugs = [s.strip() for s in slugs_text.splitlines() if s.strip()]

snapshots = run_async(asyncio.gather(*[fetch_snapshot(s) for s in slugs]))

# Build lookup dict (NO DataFrames in widgets)
by_slug = {s["slug"]: s for s in snapshots if "error" not in s}

# ---------------- DASHBOARD ----------------
st.subheader("Dashboard")

rows = []
for s in by_slug.values():
    rows.append({
        "Market": s["title"],
        "Leaning": leaning_label(s["yes_mid"]),
        "Trend": trend_label(s["delta_1h"]),
        "YES": pct(s["yes_mid"]),
    })

st.dataframe(pd.DataFrame(rows), width="stretch")

# ---------------- DETAILS ----------------
st.subheader("Details")

pick_slug = st.selectbox("Choose market", list(by_slug.keys()),
                          format_func=lambda k: by_slug[k]["title"])

pick = by_slug[pick_slug]

st.metric("YES", pct(pick["yes_mid"]))
st.metric("Trend (1h)", trend_label(pick["delta_1h"]), delta=pct(pick["delta_1h"]))

st.write(f"End date: {pick['end_date']}")
st.write(f"Volume: {pick['volume']}")

hist = pick["hist"]
if isinstance(hist, pd.DataFrame) and not hist.empty:
    st.line_chart(hist.set_index("ts")["price"])
else:
    st.info("No history available.")
