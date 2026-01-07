import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Minimal config
# -----------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Your market (from the URL you shared)
MARKET_SLUG_DEFAULT = "khamenei-out-as-supreme-leader-of-iran-by-june-30-747"

# Fixed polling interval (per your request)
POLL_SECONDS = 30

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Streamlit) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari/537.36",
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
    """Normalize list-ish fields (actual list or JSON-encoded list string)."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []

def extract_yes_no_token_ids(market: dict) -> Tuple[Optional[str], Optional[str]]:
    toks = market.get("clobTokenIds") or market.get("clob_token_ids") or market.get("clobTokenIDs")
    toks_list = [str(t) for t in _as_list(toks) if t not in (None, "", "null")]
    yes_id = toks_list[0] if len(toks_list) >= 1 else None
    no_id = toks_list[1] if len(toks_list) >= 2 else None
    return yes_id, no_id

def _lean_label(p_yes: Optional[float]) -> str:
    if p_yes is None:
        return "—"
    if p_yes > 0.5:
        return "Leans YES"
    if p_yes < 0.5:
        return "Leans NO"
    return "Even"

def _trend_label(delta: Optional[float], deadband: float = 0.0025) -> str:
    """
    delta is (current - past). deadband avoids flip-flopping on tiny moves.
    """
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
    """GET https://gamma-api.polymarket.com/markets/slug/{slug}"""
    r = requests.get(
        f"{GAMMA_BASE}/markets/slug/{slug}",
        timeout=25,
        headers=DEFAULT_HEADERS,
    )
    if r.status_code != 200:
        return None
    j = r.json()
    if isinstance(j, list):
        return j[0] if j else None
    return j if isinstance(j, dict) else None

@st.cache_data(ttl=15)
def clob_midpoint(token_id: str) -> Optional[float]:
    """GET https://clob.polymarket.com/midpoint?token_id=... -> {"mid":"0.1234"}"""
    r = requests.get(
        f"{CLOB_BASE}/midpoint",
        params={"token_id": token_id},
        timeout=15,
        headers=DEFAULT_HEADERS,
    )
    if r.status_code != 200:
        return None
    return _safe_float(r.json().get("mid"))

@st.cache_data(ttl=600)
def clob_prices_history(token_id: str, interval: str = "1w", fidelity_min: int = 15) -> pd.DataFrame:
    """
    GET https://clob.polymarket.com/prices-history?market=<token>&interval=1w&fidelity=15
    -> {"history":[{"t":..., "p":...}, ...]}
    """
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

def compute_trend_from_history(hist: pd.DataFrame, lookback_minutes: int = 60) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (current_price, delta_vs_lookback).
    delta = current - price_at_or_before(now - lookback)
    """
    if hist is None or hist.empty:
        return None, None

    hist = hist.dropna().sort_values("ts")
    current = float(hist.iloc[-1]["price"])

    target = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(minutes=lookback_minutes)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])
    return current, (current - past)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Polytracker — Single Market", layout="wide")
st.title("Polymarket Tracker")

# Fixed polling interval, no user setting
try:
    st.autorefresh(interval=POLL_SECONDS * 1000, key="poll")
except Exception:
    pass

# Sidebar: keep minimal controls, no polling slider
with st.sidebar:
    st.header("Market")
    slug = st.text_input("Market slug", value=MARKET_SLUG_DEFAULT)
    show_debug = st.checkbox("Show debug", value=False)
    st.caption(f"Auto-refresh: every {POLL_SECONDS}s")
    if st.button("Force refresh now"):
        st.cache_data.clear()
        st.rerun()

market = gamma_market_by_slug(slug)
if not market:
    st.error("Could not load market from Gamma. Check slug or network.")
    st.stop()

question = market.get("question") or "(no question)"
end_date = market.get("endDateIso") or market.get("endDate") or "—"
volume_num = market.get("volumeNum") or market.get("volume") or "—"

yes_token, no_token = extract_yes_no_token_ids(market)
if not yes_token:
    st.error("This market did not return a usable YES token id (clobTokenIds).")
    if show_debug:
        st.json(market)
    st.stop()

# --- Top-level DASHBOARD ---
st.subheader("Dashboard")

# Pull current midpoint and recent history to compute trend
yes_mid = clob_midpoint(yes_token)
# Use 1w window with 15m fidelity for a stable, cheap trend signal
hist_for_trend = clob_prices_history(yes_token, interval="1w", fidelity_min=15)
cur_hist, delta_60m = compute_trend_from_history(hist_for_trend, lookback_minutes=60)

lean = _lean_label(yes_mid)
trend = _trend_label(delta_60m, deadband=0.0025)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Market", question)  # title only, no slug shown on page
with c2:
    st.metric("Leaning", lean)
with c3:
    st.metric("Trend (1h)", trend, delta=_pct(delta_60m, 2) if delta_60m is not None else None)
with c4:
    # show the actual current probability too (use midpoint; fallback to history last)
    p = yes_mid if yes_mid is not None else cur_hist
    st.metric("YES (now)", _pct(p))

st.caption("Lean = whether YES is above or below 50%. Trend = change in YES over the last hour (from price history).")

st.divider()

# --- Detailed page content (below) ---
st.subheader("Details")

colA, colB, colC = st.columns(3)
with colA:
    st.metric("YES midpoint", _pct(yes_mid))
with colB:
    no_mid = (1.0 - yes_mid) if yes_mid is not None else None
    st.metric("NO midpoint", _pct(no_mid))
with colC:
    st.metric("End date", str(end_date))

st.write(f"Volume: {volume_num}")

st.markdown("### YES price history")
hist_interval = st.selectbox("History window", ["1d", "1w", "max"], index=1)
fidelity = st.selectbox("Fidelity (minutes)", [1, 2, 5, 10, 15, 30, 60], index=4)

hist = clob_prices_history(yes_token, interval=hist_interval, fidelity_min=int(fidelity))
if hist.empty:
    st.info("No price history returned for this token.")
else:
    st.line_chart(hist.set_index("ts")["price"])

if show_debug:
    st.markdown("### Debug")
    st.write("YES token id:", yes_token)
    st.write("NO token id:", no_token)
    st.json(market)
