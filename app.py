import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

MARKET_SLUG_DEFAULT = "khamenei-out-as-supreme-leader-of-iran-by-june-30-747"
POLL_SECONDS = 30

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Streamlit)",
    "Accept": "application/json,text/plain,*/*",
}

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%"

def _as_list(x) -> list:
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

def extract_yes_no_token_ids(market: dict) -> Tuple[Optional[str], Optional[str]]:
    toks = market.get("clobTokenIds") or market.get("clob_token_ids")
    toks = [str(t) for t in _as_list(toks) if t not in (None, "", "null")]
    return (toks[0], toks[1]) if len(toks) >= 2 else (toks[0], None) if toks else (None, None)

def lean_label(p: Optional[float]) -> str:
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

# -----------------------------
# API (cached)
# -----------------------------
@st.cache_data(ttl=900)
def gamma_market_by_slug(slug: str) -> Optional[dict]:
    r = requests.get(f"{GAMMA_BASE}/markets/slug/{slug}", headers=DEFAULT_HEADERS, timeout=20)
    if r.status_code != 200:
        return None
    j = r.json()
    return j[0] if isinstance(j, list) and j else j

@st.cache_data(ttl=15)
def clob_midpoint(token_id: str) -> Optional[float]:
    r = requests.get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id}, timeout=10)
    if r.status_code != 200:
        return None
    return _safe_float(r.json().get("mid"))

@st.cache_data(ttl=600)
def clob_prices_history(token_id: str, interval="1w", fidelity_min=15) -> pd.DataFrame:
    r = requests.get(
        f"{CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity_min},
        timeout=20,
    )
    r.raise_for_status()
    hist = r.json().get("history", [])
    df = pd.DataFrame(hist)
    if df.empty:
        return pd.DataFrame(columns=["ts", "price"])
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    return df[["ts", "price"]].dropna().sort_values("ts")

def compute_trend(hist: pd.DataFrame, lookback_minutes: int = 60):
    if hist.empty:
        return None, None
    now = pd.Timestamp.now(tz="UTC")
    current = float(hist.iloc[-1]["price"])
    target = now - pd.Timedelta(minutes=lookback_minutes)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])
    return current, current - past

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Polytracker", layout="wide")
st.title("Polymarket Tracker")

# Fixed polling
st.autorefresh(interval=POLL_SECONDS * 1000, key="poll")

with st.sidebar:
    slug = st.text_input("Market slug", value=MARKET_SLUG_DEFAULT)
    debug = st.checkbox("Show debug")

market = gamma_market_by_slug(slug)
if not market:
    st.error("Market not found")
    st.stop()

question = market.get("question", "(no title)")
end_date = market.get("endDateIso") or market.get("endDate")
volume = market.get("volumeNum") or market.get("volume")

yes_token, no_token = extract_yes_no_token_ids(market)
if not yes_token:
    st.error("YES token not found")
    st.stop()

yes_mid = clob_midpoint(yes_token)
hist = clob_prices_history(yes_token)
cur, delta = compute_trend(hist, 60)

# -----------------------------
# Dashboard
# -----------------------------
st.subheader("Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Market", question)
c2.metric("Leaning", lean_label(yes_mid))
c3.metric("Trend (1h)", trend_label(delta), delta=_pct(delta))
c4.metric("YES (now)", _pct(yes_mid))

st.divider()

# -----------------------------
# Details
# -----------------------------
st.subheader("Details")

d1, d2, d3 = st.columns(3)
d1.metric("YES", _pct(yes_mid))
d2.metric("NO", _pct(1 - yes_mid if yes_mid is not None else None))
d3.metric("End date", str(end_date))

st.write(f"Volume: {volume}")

st.markdown("### YES price history")
st.line_chart(hist.set_index("ts")["price"])

if debug:
    st.json(market)
