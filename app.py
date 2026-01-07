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

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Streamlit)",
    "Accept": "application/json,text/plain,*/*",
}

# Dashboard is light; details are heavier
DASH_INTERVAL = "1d"
DASH_FIDELITY_MIN = 5
DETAIL_INTERVAL = "max"
DETAIL_FIDELITY_MIN = 30

# Trend sensitivity
DEADBAND = 0.0025  # 0.25%

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
    return float(max(0.0, min(1.0, x)))

def delta_1h_from_hist(hist: pd.DataFrame, lookback_minutes: int = 60) -> Optional[float]:
    if hist is None or hist.empty:
        return None
    now = pd.Timestamp.now(tz="UTC")
    current = float(hist.iloc[-1]["price"])
    target = now - pd.Timedelta(minutes=lookback_minutes)
    older = hist[hist["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist.iloc[0]["price"])
    return current - past

def trend_badge(delta: Optional[float]) -> str:
    """
    Minimal graphical trend:
      - Up green ▲ for delta > deadband
      - Down red ▼ for delta < -deadband
      - Flat gray ▬ otherwise
    Returns HTML snippet.
    """
    if delta is None or not np.isfinite(delta):
        return """
        <span style="font-size:16px;color:#6c757d;">•</span>
        <span style="font-size:12px;color:#6c757d;margin-left:6px;">—</span>
        """

    if delta > DEADBAND:
        icon, color = "▲", "#198754"  # green
    elif delta < -DEADBAND:
        icon, color = "▼", "#dc3545"  # red
    else:
        icon, color = "▬", "#6c757d"  # gray

    return f"""
    <span style="font-size:16px;color:{color};font-weight:700;">{icon}</span>
    <span style="font-size:12px;color:#6c757d;margin-left:6px;">Δ1h {pct(delta)}</span>
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

    # concurrent I/O
    mid_task = asyncio.to_thread(_sync_midpoint, yes_token)
    hist_task = asyncio.to_thread(_sync_history, yes_token, DASH_INTERVAL, DASH_FIDELITY_MIN)
    yes_mid, hist = await asyncio.gather(mid_task, hist_task)

    delta_1h = delta_1h_from_hist(hist, 60)

    return {
        "slug": slug,
        "title": title,
        "end_date": end_date,
        "volume": volume,
        "yes_token": yes_token,
        "no_token": no_token,
        "yes_mid": yes_mid,
        "delta_1h": delta_1h,
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

st_autorefresh(interval=POLL_SECONDS * 1000, key="auto_refresh_polytracker")

# Session state for toggle details
if "selected_slug" not in st.session_state:
    st.session_state.selected_slug = None
if "details_open" not in st.session_state:
    st.session_state.details_open = False

with st.sidebar:
    st.header("Search / Add market (coming soon)")
    _ = st.text_input("Search", value="", placeholder="Paste Polymarket URL or type keywords…", disabled=True)
    st.caption("This panel will become your market search & add flow.")

# Hardcoded slugs for now (you can later replace this with search/add)
slugs = DEFAULT_SLUGS

with st.spinner(f"Fetching {len(slugs)} market(s)..."):
    snapshots = run_coro(fetch_all(slugs))

# Split errors vs ok
by_slug: Dict[str, Dict] = {}
errors: List[Dict] = []
for s in snapshots:
    if s.get("error"):
        errors.append(s)
    else:
        by_slug[s["slug"]] = s

# =========================================================
# DASHBOARD (TILES, 4 PER ROW)
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

    with st.container(border=True):
        # Title
        st.markdown(f"**{m.get('title','(no title)')}**")

        # Minimal trend indicator (graphical)
        st.markdown(trend_badge(delta), unsafe_allow_html=True)

        # YES/NO + bar
        a, b = st.columns(2)
        with a:
            st.metric("YES", pct(yes_mid))
        with b:
            st.metric("NO", pct(1 - yes_mid if yes_mid is not None else None))

        st.progress(clamp01(yes_mid))

        # Toggle button
        is_open = st.session_state.details_open and st.session_state.selected_slug == m["slug"]
        label = "Hide details" if is_open else "View details"

        if st.button(label, key=f"toggle_{m['slug']}"):
            # if clicking the currently-open tile -> close
            if is_open:
                st.session_state.details_open = False
                st.session_state.selected_slug = None
            else:
                st.session_state.details_open = True
                st.session_state.selected_slug = m["slug"]

# Grid with unlimited rows
for i in range(0, len(items), TILES_PER_ROW):
    row = st.columns(TILES_PER_ROW, gap="small")
    for j in range(TILES_PER_ROW):
        idx = i + j
        if idx >= len(items):
            break
        with row[j]:
            render_tile(items[idx])

# =========================================================
# DETAILS (COLLAPSED + LAZY LOAD)
# =========================================================
with st.expander("Details", expanded=bool(st.session_state.details_open)):
    if not st.session_state.details_open or not st.session_state.selected_slug:
        st.info("Click **View details** on any tile to load details.")
    else:
        slug = st.session_state.selected_slug
        pick = by_slug.get(slug)

        if not pick:
            st.warning("Selected market is no longer available.")
        else:
            yes_mid = pick.get("yes_mid")
            delta = pick.get("delta_1h")

            st.markdown(f"### {pick.get('title')}")

            c1, c2, c3 = st.columns(3)
            c1.metric("YES (now)", pct(yes_mid))
            c2.metric("NO (now)", pct(1 - yes_mid if yes_mid is not None else None))
            c3.markdown(trend_badge(delta), unsafe_allow_html=True)

            st.write(f"End date: {pick.get('end_date')}")
            st.write(f"Volume: {pick.get('volume')}")

            st.markdown("#### YES price history (max, 30m fidelity)")
            with st.spinner("Loading detailed history..."):
                hist_full = detail_history_cached(pick["yes_token"])

            if isinstance(hist_full, pd.DataFrame) and not hist_full.empty:
                st.line_chart(hist_full.set_index("ts")["price"], height=300)
            else:
                st.info("No detailed history returned.")

            # convenience close
            if st.button("Close details panel"):
                st.session_state.details_open = False
                st.session_state.selected_slug = None
