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
TILES_PER_ROW = 7  # fixed as requested

DEFAULT_SLUGS = [
    "khamenei-out-as-supreme-leader-of-iran-by-june-30-747",
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Streamlit)",
    "Accept": "application/json,text/plain,*/*",
}

# Trend/sparkline for dashboard: keep it light
DASH_INTERVAL = "1d"
DASH_FIDELITY_MIN = 5  # minutes

# Details view: heavier history
DETAIL_INTERVAL = "max"
DETAIL_FIDELITY_MIN = 30  # minutes

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

def trend_icon(delta: Optional[float], deadband: float = 0.0025) -> str:
    if delta is None or not np.isfinite(delta):
        return "•"
    if delta > deadband:
        return "↑"
    if delta < -deadband:
        return "↓"
    return "→"

def clamp01(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))

def compute_delta(hist: pd.DataFrame, lookback_minutes: int = 60) -> Optional[float]:
    if hist is None or hist.empty:
        return None
    now = pd.Timestamp.now(tz="UTC")
    current = float(hist.iloc[-1]["price"])
    target = now - pd.Timedelta(minutes=lookback_minutes)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])
    return current - past

def pill(text: str, kind: str) -> str:
    colors = {
        "yes": ("#0f5132", "#d1e7dd"),
        "no": ("#842029", "#f8d7da"),
        "flat": ("#41464b", "#e2e3e5"),
        "muted": ("#41464b", "#f1f3f5"),
    }
    fg, bg = colors.get(kind, colors["muted"])
    return f"""
    <span style="
      display:inline-block;
      padding:2px 10px;
      border-radius:999px;
      font-size:12px;
      color:{fg};
      background:{bg};
      border:1px solid rgba(0,0,0,0.08);
      white-space:nowrap;
    ">{text}</span>
    """

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

@st.cache_data(ttl=600)
def detail_history_cached(token_id: str) -> pd.DataFrame:
    return _sync_history(token_id, DETAIL_INTERVAL, DETAIL_FIDELITY_MIN)

# =========================================================
# ASYNC SNAPSHOTS (dashboard-light)
# =========================================================
async def fetch_snapshot(slug: str) -> Dict:
    market = gamma_market_by_slug(slug)
    if not market:
        return {"slug": slug, "error": "market not found"}

    title = market.get("question") or slug
    end_date = market.get("endDateIso") or market.get("endDate") or "—"
    volume = market.get("volumeNum") or market.get("volume") or "—"

    yes_token, no_token = extract_yes_no_tokens(market)
    if not yes_token:
        return {"slug": slug, "title": title, "error": "missing YES token id"}

    # Concurrent I/O (threaded) for speed as you scale markets
    mid_task = asyncio.to_thread(_sync_midpoint, yes_token)
    hist_task = asyncio.to_thread(_sync_history, yes_token, DASH_INTERVAL, DASH_FIDELITY_MIN)
    yes_mid, hist = await asyncio.gather(mid_task, hist_task)

    delta_1h = compute_delta(hist, 60)

    return {
        "slug": slug,
        "title": title,
        "end_date": end_date,
        "volume": volume,
        "yes_token": yes_token,
        "no_token": no_token,
        "yes_mid": yes_mid,
        "delta_1h": delta_1h,
        "leaning": leaning_label(yes_mid),
        "trend": trend_label(delta_1h),
        "hist_light": hist,  # used for sparkline only
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

# Auto-refresh tick (non-blocking)
st_autorefresh(interval=POLL_SECONDS * 1000, key="auto_refresh_polytracker")

# Session defaults
if "selected_slug" not in st.session_state:
    st.session_state.selected_slug = None
if "details_open" not in st.session_state:
    st.session_state.details_open = False

with st.sidebar:
    st.caption(f"Auto-refresh: every {POLL_SECONDS}s")
    slugs_text = st.text_area(
        "Market slugs (one per line)",
        value="\n".join(DEFAULT_SLUGS),
        height=160,
    )
    slugs = [s.strip() for s in slugs_text.splitlines() if s.strip()]
    spark_points = st.slider("Sparkline points", 10, 120, 40, step=10)
    show_meta = st.checkbox("Show end/vol on tiles", value=False)
    debug = st.checkbox("Show debug", value=False)

if not slugs:
    st.info("Add at least one market slug.")
    st.stop()

with st.spinner(f"Fetching {len(slugs)} market(s)..."):
    snapshots = run_coro(fetch_all(slugs))

# Build widget-safe lookup
by_slug: Dict[str, Dict] = {}
errors: List[Dict] = []
for s in snapshots:
    if s.get("error"):
        errors.append(s)
    else:
        by_slug[s["slug"]] = s

# =========================================================
# DASHBOARD (TILES, 7 PER ROW)
# =========================================================
st.subheader("Dashboard")

if errors:
    with st.expander("Some markets failed to load"):
        st.dataframe(pd.DataFrame(errors), width="stretch")

if not by_slug:
    st.info("No valid markets to display.")
    st.stop()

items = list(by_slug.values())

def render_tile(m: Dict):
    yes_mid = m.get("yes_mid")
    delta = m.get("delta_1h")

    lean = m.get("leaning", "—")
    if lean == "Leans YES":
        lean_pill = pill(lean, "yes")
    elif lean == "Leans NO":
        lean_pill = pill(lean, "no")
    elif lean == "Even":
        lean_pill = pill(lean, "flat")
    else:
        lean_pill = pill(lean, "muted")

    tr = m.get("trend", "—")
    tr_icon = trend_icon(delta)
    if tr == "Trending YES":
        tr_pill = pill(f"{tr_icon} {tr}", "yes")
    elif tr == "Trending NO":
        tr_pill = pill(f"{tr_icon} {tr}", "no")
    elif tr == "Flat":
        tr_pill = pill(f"{tr_icon} {tr}", "flat")
    else:
        tr_pill = pill(f"{tr_icon} {tr}", "muted")

    with st.container(border=True):
        st.markdown(f"**{m.get('title','(no title)')}**")

        a, b = st.columns([1, 1])
        with a:
            st.markdown(lean_pill, unsafe_allow_html=True)
        with b:
            st.markdown(tr_pill, unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("YES", pct(yes_mid))
        with c2:
            st.metric("NO", pct(1 - yes_mid if yes_mid is not None else None))

        st.progress(clamp01(yes_mid))
        st.caption(f"Δ 1h: {pct(delta)}")

        if show_meta:
            st.caption(f"End: {m.get('end_date','—')} • Vol: {m.get('volume','—')}")

        # light sparkline
        hist = m.get("hist_light")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            tail = hist.tail(int(spark_points)).set_index("ts")["price"]
            st.line_chart(tail, height=90)
        else:
            st.caption("No history")

        # Clicking this should open details expander and load details lazily
        if st.button("View details", key=f"view_{m['slug']}"):
            st.session_state.selected_slug = m["slug"]
            st.session_state.details_open = True

# Create unlimited rows of 7 tiles
for i in range(0, len(items), TILES_PER_ROW):
    row = st.columns(TILES_PER_ROW, gap="small")
    for j in range(TILES_PER_ROW):
        idx = i + j
        if idx >= len(items):
            break
        with row[j]:
            render_tile(items[idx])

st.caption("Leaning = YES vs 50%. Trend = change in YES probability over the last hour (from price history).")

# =========================================================
# DETAILS (COLLAPSED + LAZY)
# =========================================================
with st.expander("Details", expanded=bool(st.session_state.details_open)):
    if not st.session_state.selected_slug or st.session_state.selected_slug not in by_slug:
        st.info("Click **View details** on a tile to load the detailed view.")
    else:
        pick_slug = st.session_state.selected_slug
        pick = by_slug[pick_slug]

        yes_mid = pick.get("yes_mid")
        delta = pick.get("delta_1h")

        st.markdown(f"### {pick.get('title')}")

        c1, c2, c3 = st.columns(3)
        c1.metric("YES (now)", pct(yes_mid))
        c2.metric("Leaning", pick.get("leaning", "—"))
        c3.metric("Trend (1h)", pick.get("trend", "—"), delta=pct(delta))

        st.write(f"End date: {pick.get('end_date')}")
        st.write(f"Volume: {pick.get('volume')}")

        # Lazy-load heavy history ONLY when expander is opened via View details
        st.markdown("#### YES price history (max, 30m fidelity)")
        with st.spinner("Loading detailed history..."):
            hist_full = detail_history_cached(pick["yes_token"])

        if isinstance(hist_full, pd.DataFrame) and not hist_full.empty:
            st.line_chart(hist_full.set_index("ts")["price"], height=260)
        else:
            st.info("No detailed history returned.")

        # Optional: allow closing the expander state
        colA, colB = st.columns([1, 3])
        with colA:
            if st.button("Close details"):
                st.session_state.details_open = False

        if debug:
            st.markdown("#### Debug snapshot")
            st.json({k: v for k, v in pick.items() if k != "hist_light"})
