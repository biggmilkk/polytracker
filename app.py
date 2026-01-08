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
TILES_PER_ROW = 4

DEFAULT_SLUGS = [
    "khamenei-out-as-supreme-leader-of-iran-by-june-30-747",
]

# Keep dashboard light (trend uses last hour delta computed from this)
DASH_INTERVAL = "1d"
DASH_FIDELITY_MIN = 5  # minutes

DEADBAND = 0.0025  # 0.25%

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
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
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

def clamp01(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))

def delta_1h_from_hist(hist: pd.DataFrame) -> Optional[float]:
    if hist is None or hist.empty:
        return None
    now = pd.Timestamp.now(tz="UTC")
    current = float(hist.iloc[-1]["price"])
    target = now - pd.Timedelta(minutes=60)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])
    return current - past

def trend_badge(delta: Optional[float]) -> str:
    """
    Minimal graphical trend:
      - Up green ▲ for delta > deadband
      - Down red ▼ for delta < -deadband
      - Flat gray ▬ otherwise
    """
    if delta is None or not np.isfinite(delta):
        return "<span style='color:#6c757d;'>•</span> <span style='color:#6c757d;font-size:12px;'>Δ1h —</span>"

    if delta > DEADBAND:
        icon, color = "▲", "#198754"
    elif delta < -DEADBAND:
        icon, color = "▼", "#dc3545"
    else:
        icon, color = "▬", "#6c757d"

    return f"""
    <span style="font-size:16px;color:{color};font-weight:700;">{icon}</span>
    <span style="font-size:12px;color:#6c757d;margin-left:6px;">Δ1h {pct(delta)}</span>
    """

def polymarket_event_url(slug: str) -> str:
    # Works for standard event pages; if Polymarket changes routing later, we can update centrally here.
    return f"https://polymarket.com/event/{slug}"

# =========================================================
# API (cached)
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

def _sync_midpoint(token_id: str) -> Optional[float]:
    r = requests.get(
        f"{CLOB_BASE}/midpoint",
        params={"token_id": token_id},
        timeout=10,
        headers=DEFAULT_HEADERS,
    )
    if r.status_code != 200:
        return None
    return safe_float(r.json().get("mid"))

def _sync_history(token_id: str, interval: str, fidelity_min: int) -> pd.DataFrame:
    r = requests.get(
        f"{CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity_min},
        timeout=20,
        headers=DEFAULT_HEADERS,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("history", []))
    if df.empty:
        return pd.DataFrame(columns=["ts", "price"])
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    return df[["ts", "price"]].dropna().sort_values("ts")

# =========================================================
# ASYNC SNAPSHOTS (dashboard only)
# =========================================================
async def fetch_snapshot(slug: str) -> Dict:
    market = gamma_market_by_slug(slug)
    if not market:
        return {"slug": slug, "error": "market not found"}

    title = market.get("question") or slug
    yes_token, _ = extract_yes_no_tokens(market)
    if not yes_token:
        return {"slug": slug, "title": title, "error": "missing YES token"}

    # concurrent I/O
    mid_task = asyncio.to_thread(_sync_midpoint, yes_token)
    hist_task = asyncio.to_thread(_sync_history, yes_token, DASH_INTERVAL, DASH_FIDELITY_MIN)
    yes_mid, hist = await asyncio.gather(mid_task, hist_task)

    return {
        "slug": slug,
        "title": title,
        "yes_mid": yes_mid,
        "delta_1h": delta_1h_from_hist(hist),
        "url": polymarket_event_url(slug),
    }

async def fetch_all(slugs: List[str]) -> List[Dict]:
    return await asyncio.gather(*(fetch_snapshot(s) for s in slugs))

def run_coro(coro):
    return asyncio.run(coro)

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Polytracker", layout="wide")
st.title("Polymarket Tracker")

st_autorefresh(interval=POLL_SECONDS * 1000, key="refresh")

with st.sidebar:
    st.header("Search / Add market (coming soon)")
    st.text_input("Search", value="", placeholder="Paste Polymarket URL or type keywords…", disabled=True)
    st.caption("This sidebar will become your market search & add flow.")

# Hardcoded list for now (later you’ll replace with search/add)
slugs = DEFAULT_SLUGS

with st.spinner("Fetching markets…"):
    snapshots = run_coro(fetch_all(slugs))

ok = [s for s in snapshots if not s.get("error")]
errs = [s for s in snapshots if s.get("error")]

if errs:
    with st.expander("Some markets failed to load"):
        st.dataframe(pd.DataFrame(errs), width="stretch")

if not ok:
    st.info("No valid markets to display.")
    st.stop()

# =========================================================
# DASHBOARD (TILES ONLY)
# =========================================================
def render_tile(m: Dict):
    yes_mid = m.get("yes_mid")
    delta = m.get("delta_1h")

    with st.container(border=True):
        # Title as a hyperlink
        st.markdown(f"**[{m['title']}]({m['url']})**")

        # Minimal trend indicator
        st.markdown(trend_badge(delta), unsafe_allow_html=True)

        # YES/NO + bar
        a, b = st.columns(2)
        with a:
            st.metric("YES", pct(yes_mid))
        with b:
            st.metric("NO", pct(1 - yes_mid if yes_mid is not None else None))

        st.progress(clamp01(yes_mid))

        # Small call-to-action link
        st.markdown(f"[Open on Polymarket]({m['url']})")

for i in range(0, len(ok), TILES_PER_ROW):
    row = st.columns(TILES_PER_ROW, gap="small")
    for j in range(TILES_PER_ROW):
        idx = i + j
        if idx >= len(ok):
            break
        with row[j]:
            render_tile(ok[idx])
