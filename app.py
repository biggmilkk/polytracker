import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Market from:
# https://polymarket.com/event/khamenei-out-as-supreme-leader-of-iran-by-june-30-747
MARKET_SLUG = "khamenei-out-as-supreme-leader-of-iran-by-june-30-747"

POLL_SECONDS = 30  # fixed, no UI control

# =========================================================
# HELPERS
# =========================================================
def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%"

def as_list(x) -> list:
    """
    Gamma sometimes returns list fields as:
    - real lists
    - JSON-encoded strings
    Normalize both.
    """
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
    """
    deadband avoids noisy flips on tiny moves (~0.25%)
    """
    if delta is None or not np.isfinite(delta):
        return "—"
    if delta > deadband:
        return "Trending YES"
    if delta < -deadband:
        return "Trending NO"
    return "Flat"

# =========================================================
# API CALLS (CACHED)
# =========================================================
@st.cache_data(ttl=900)
def gamma_market_by_slug(slug: str) -> Optional[dict]:
    """
    Full market object (includes conditionId, clobTokenIds, etc.)
    """
    r = requests.get(f"{GAMMA_BASE}/markets/slug/{slug}", timeout=20)
    if r.status_code != 200:
        return None
    j = r.json()
    return j[0] if isinstance(j, list) and j else j

@st.cache_data(ttl=15)
def clob_midpoint(token_id: str) -> Optional[float]:
    """
    YES/NO midpoint from Polymarket CLOB
    """
    r = requests.get(
        f"{CLOB_BASE}/midpoint",
        params={"token_id": token_id},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    return safe_float(r.json().get("mid"))

@st.cache_data(ttl=600)
def clob_price_history(
    token_id: str,
    interval: str = "1w",
    fidelity_min: int = 15,
) -> pd.DataFrame:
    """
    Price history for trend detection
    """
    r = requests.get(
        f"{CLOB_BASE}/prices-history",
        params={
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity_min,
        },
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

def compute_trend(
    hist: pd.DataFrame,
    lookback_minutes: int = 60,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns:
      current_price, delta_vs_lookback
    """
    if hist.empty:
        return None, None

    now = pd.Timestamp.now(tz="UTC")
    current = float(hist.iloc[-1]["price"])

    target = now - pd.Timedelta(minutes=lookback_minutes)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])

    return current, current - past

# =========================================================
# STREAMLIT APP
# =========================================================
st.set_page_config(page_title="Polymarket Tracker", layout="wide")
st.title("Polymarket Tracker")

# Native auto-refresh (Streamlit ≥ 1.27)
st.autorefresh(interval=POLL_SECONDS * 1000, key="poll")

with st.sidebar:
    st.caption(f"Auto-refresh every {POLL_SECONDS} seconds")
    debug = st.checkbox("Show debug")

# ---------------------------------------------------------
# LOAD MARKET
# ---------------------------------------------------------
market = gamma_market_by_slug(MARKET_SLUG)
if not market:
    st.error("Failed to load market from Polymarket.")
    st.stop()

question = market.get("question", "(no title)")
end_date = market.get("endDateIso") or market.get("endDate")
volume = market.get("volumeNum") or market.get("volume")

yes_token, no_token = extract_yes_no_tokens(market)
if not yes_token:
    st.error("YES token ID not found.")
    st.stop()

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
yes_mid = clob_midpoint(yes_token)
hist = clob_price_history(yes_token)
current_hist_price, delta_1h = compute_trend(hist, lookback_minutes=60)

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
st.subheader("Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Market", question)

with c2:
    st.metric("Leaning", leaning_label(yes_mid))

with c3:
    st.metric(
        "Trend (1h)",
        trend_label(delta_1h),
        delta=pct(delta_1h),
    )

with c4:
    st.metric("YES (now)", pct(yes_mid))

st.caption(
    "Leaning = YES vs 50%. Trend = change in YES probability over the last hour."
)

st.divider()

# ---------------------------------------------------------
# DETAILS
# ---------------------------------------------------------
st.subheader("Details")

d1, d2, d3 = st.columns(3)

with d1:
    st.metric("YES", pct(yes_mid))

with d2:
    st.metric("NO", pct(1 - yes_mid if yes_mid is not None else None))

with d3:
    st.metric("End date", str(end_date))

st.write(f"Volume: {volume}")

st.markdown("### YES price history")
st.line_chart(hist.set_index("ts")["price"])

if debug:
    st.markdown("### Debug")
    st.json(market)
