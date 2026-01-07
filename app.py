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

DETAIL_INTERVAL = "max"
DETAIL_FIDELITY_MIN = 30
DASH_INTERVAL = "1d"
DASH_FIDELITY_MIN = 5

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
    if delta is None or not np.isfinite(delta):
        return "<span style='color:#6c757d;'>•</span>"

    if delta > DEADBAND:
        icon, color = "▲", "#198754"
    elif delta < -DEADBAND:
        icon, color = "▼", "#dc3545"
    else:
        icon, color = "▬", "#6c757d"

    return f"""
    <span style="font-size:16px;color:{color};font-weight:700;">
        {icon}
    </span>
    <span style="font-size:12px;color:#6c757d;margin-left:6px;">
        Δ1h {pct(delta)}
    </span>
    """

# =========================================================
# API (cached)
# =========================================================
@st.cache_data(ttl=900)
def gamma_market_by_slug(slug: str) -> Optional[dict]:
    r = requests.get(f"{GAMMA_BASE}/markets/slug/{slug}", timeout=20, headers=DEFAULT_HEADERS)
    if r.status_code != 200:
        return None
    j = r.json()
    return j[0] if isinstance(j, list) and j else j

def _sync_midpoint(token_id: str) -> Optional[float]:
    r = requests.get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id}, timeout=10)
    if r.status_code != 200:
        return None
    return safe_float(r.json().get("mid"))

def _sync_history(token_id: str, interval: str, fidelity_min: int) -> pd.DataFrame:
    r = requests.get(
        f"{CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity_min},
        timeout=20,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("history", []))
    if df.empty:
        return pd.DataFrame(columns=["ts", "price"])
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    return df[["ts", "price"]].dropna().sort_values("ts")

@st.cache_data(ttl=600)
def detail_history_cached(token_id: str) -> pd.DataFrame:
    return _sync_history(token_id, DETAIL_INTERVAL, DETAIL_FIDELITY_MIN)

# =========================================================
# ASYNC SNAPSHOTS
# =========================================================
async def fetch_snapshot(slug: str) -> Dict:
    market = gamma_market_by_slug(slug)
    if not market:
        return {"slug": slug, "error": "market not found"}

    title = market.get("question") or slug
    end_date = market.get("endDateIso") or market.get("endDate")
    volume = market.get("volumeNum") or market.get("volume")

    yes_token, _ = extract_yes_no_tokens(market)
    if not yes_token:
        return {"slug": slug, "title": title, "error": "missing YES token"}

    mid_task = asyncio.to_thread(_sync_midpoint, yes_token)
    hist_task = asyncio.to_thread(_sync_history, yes_token, DASH_INTERVAL, DASH_FIDELITY_MIN)
    yes_mid, hist = await asyncio.gather(mid_task, hist_task)

    return {
        "slug": slug,
        "title": title,
        "yes_token": yes_token,
        "yes_mid": yes_mid,
        "delta_1h": delta_1h_from_hist(hist),
        "end_date": end_date,
        "volume": volume,
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

if "selected_slug" not in st.session_state:
    st.session_state.selected_slug = None

slugs = DEFAULT_SLUGS

with st.spinner("Fetching markets…"):
    snapshots = run_coro(fetch_all(slugs))

by_slug = {s["slug"]: s for s in snapshots if "error" not in s}

# =========================================================
# DASHBOARD (TILES)
# =========================================================
items = list(by_slug.values())

def render_tile(m: Dict):
    is_open = st.session_state.selected_slug == m["slug"]

    with st.container(border=True):
        st.markdown(f"**{m['title']}**")
        st.markdown(trend_badge(m["delta_1h"]), unsafe_allow_html=True)

        a, b = st.columns(2)
        a.metric("YES", pct(m["yes_mid"]))
        b.metric("NO", pct(1 - m["yes_mid"] if m["yes_mid"] is not None else None))

        st.progress(clamp01(m["yes_mid"]))

        label = "Hide details" if is_open else "View details"
        if st.button(label, key=f"toggle_{m['slug']}"):
            st.session_state.selected_slug = None if is_open else m["slug"]

for i in range(0, len(items), TILES_PER_ROW):
    row = st.columns(TILES_PER_ROW, gap="small")
    for j in range(TILES_PER_ROW):
        idx = i + j
        if idx < len(items):
            with row[j]:
                render_tile(items[idx])

# =========================================================
# INLINE DETAILS (ONLY WHEN OPEN)
# =========================================================
if st.session_state.selected_slug:
    pick = by_slug.get(st.session_state.selected_slug)
    if pick:
        st.divider()

        st.markdown(f"### {pick['title']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("YES", pct(pick["yes_mid"]))
        c2.metric("NO", pct(1 - pick["yes_mid"] if pick["yes_mid"] is not None else None))
        c3.markdown(trend_badge(pick["delta_1h"]), unsafe_allow_html=True)

        st.write(f"End date: {pick['end_date']}")
        st.write(f"Volume: {pick['volume']}")

        with st.spinner("Loading detailed history…"):
            hist_full = detail_history_cached(pick["yes_token"])

        if not hist_full.empty:
            st.line_chart(hist_full.set_index("ts")["price"], height=320)
        else:
            st.info("No detailed history available.")
