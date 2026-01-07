import json
import time
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Minimal config
# -----------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# This is the exact Polymarket event slug from your URL:
MARKET_SLUG = "khamenei-out-as-supreme-leader-of-iran-by-june-30-747"

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

def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"

def _as_list(x) -> list:
    """
    Gamma sometimes returns list fields as actual lists, or JSON-encoded strings.
    Normalize to a Python list.
    """
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
    toks = (
        market.get("clobTokenIds")
        or market.get("clob_token_ids")
        or market.get("clobTokenIDs")
    )
    toks_list = [str(t) for t in _as_list(toks) if t not in (None, "", "null")]
    yes_id = toks_list[0] if len(toks_list) >= 1 else None
    no_id = toks_list[1] if len(toks_list) >= 2 else None
    return yes_id, no_id

# -----------------------------
# API (cached)
# -----------------------------
@st.cache_data(ttl=900)
def gamma_market_by_slug(slug: str) -> Optional[dict]:
    """
    Full market object (includes conditionId, clobTokenIds, etc.)
    GET https://gamma-api.polymarket.com/markets/slug/{slug}
    """
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
    """
    GET https://clob.polymarket.com/midpoint?token_id=...
    -> {"mid":"0.1234"} (string)
    """
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

# -----------------------------
# Streamlit UI (minimal)
# -----------------------------
st.set_page_config(page_title="Polytracker (Single Market)", layout="wide")
st.title("Polymarket Tracker — Single Market (Minimal)")

with st.sidebar:
    st.header("Settings")
    slug = st.text_input("Market slug", value=MARKET_SLUG)
    poll_seconds = st.slider("Polling interval (seconds)", 10, 300, 30, step=5)
    show_debug = st.checkbox("Show debug (raw market JSON)", value=False)

    st.divider()
    if st.button("Force refresh now"):
        st.cache_data.clear()
        st.success("Cache cleared. Reloading…")
        st.rerun()

# Auto-refresh (polling)
try:
    st.autorefresh(interval=poll_seconds * 1000, key="poll")
except Exception:
    pass

market = gamma_market_by_slug(slug)

if not market:
    st.error("Could not load market from Gamma. Check slug or network.")
    st.stop()

question = market.get("question") or "(no question)"
end_date = market.get("endDateIso") or market.get("endDate") or "—"
volume_num = market.get("volumeNum") or market.get("volume") or "—"
last_trade = _safe_float(market.get("lastTradePrice"))

yes_token, no_token = extract_yes_no_token_ids(market)

st.subheader(question)
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("End date", str(end_date))
with colB:
    st.metric("Volume", str(volume_num))
with colC:
    st.metric("Last trade (Yes)", _pct(last_trade))
with colD:
    st.metric("Slug", slug)

if not yes_token:
    st.error("This market did not return a usable YES token id (clobTokenIds).")
    if show_debug:
        st.json(market)
    st.stop()

# Current prices
yes_mid = clob_midpoint(yes_token)
no_mid = (1.0 - yes_mid) if yes_mid is not None else None

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("YES midpoint", _pct(yes_mid))
with c2:
    st.metric("NO midpoint", _pct(no_mid))
with c3:
    st.caption("Midpoints are polled from the free CLOB endpoint.")

# History chart (YES)
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
