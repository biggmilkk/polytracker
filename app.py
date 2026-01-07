import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Config (public, free endpoints)
# -----------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

DEFAULT_KEYWORDS = [
    "venezuela", "maduro", "iran", "taiwan", "china", "russia", "ukraine",
    "sanctions", "airstrike", "coup", "invasion", "hostage", "missile",
    "nuclear", "ceasefire", "detain", "extradition",
]

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _now() -> float:
    return time.time()

def _pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.1f}%"

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (robust volatility proxy)."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

# -----------------------------
# API calls (cached)
# -----------------------------
@st.cache_data(ttl=30)
def clob_midpoint(token_id: str) -> Optional[float]:
    """GET /midpoint?token_id=... -> {"mid": "0.65"}"""
    r = requests.get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id}, timeout=15)
    if r.status_code != 200:
        return None
    j = r.json()
    return _safe_float(j.get("mid"))

@st.cache_data(ttl=600)
def clob_prices_history(token_id: str, interval: str = "1w", fidelity_min: int = 5) -> pd.DataFrame:
    """
    GET /prices-history?market=<token>&interval=1w&fidelity=5
    Response: {"history":[{"t":..., "p":...}, ...]}
    interval options include 1m, 1h, 6h, 1d, 1w, max; fidelity is minutes.
    """
    params = {"market": token_id, "interval": interval, "fidelity": fidelity_min}
    r = requests.get(f"{CLOB_BASE}/prices-history", params=params, timeout=20)
    r.raise_for_status()
    hist = r.json().get("history", [])
    df = pd.DataFrame(hist)
    if df.empty:
        return pd.DataFrame(columns=["t", "p", "ts", "price"])
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    df = df[["ts", "price"]].dropna().sort_values("ts")
    return df

@st.cache_data(ttl=900)
def gamma_markets_page(limit: int = 200, offset: int = 0) -> List[dict]:
    """
    GET https://gamma-api.polymarket.com/markets?limit=...&offset=...
    """
    r = requests.get(f"{GAMMA_BASE}/markets", params={"limit": limit, "offset": offset}, timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=900)
def data_holders_for_condition_ids(condition_ids: List[str], limit: int = 20, min_balance: int = 1) -> List[dict]:
    """
    GET /holders?market=<comma-separated condition IDs>&limit=20&minBalance=...
    Returns list like: [{"token": "...", "holders":[...]}]
    Note: the API expects condition IDs via `market` parameter. :contentReference[oaicite:8]{index=8}
    """
    params = {
        "market": ",".join(condition_ids),
        "limit": limit,
        "minBalance": min_balance,
    }
    r = requests.get(f"{DATA_BASE}/holders", params=params, timeout=20)
    if r.status_code != 200:
        return []
    return r.json()

# -----------------------------
# Domain model
# -----------------------------
@dataclass
class TrackedMarket:
    market_id: str
    slug: str
    title: str
    condition_id: str
    yes_token_id: str
    no_token_id: Optional[str]
    liquidity_num: Optional[float]
    volume_num: Optional[float]
    is_open: bool
    enable_orderbook: bool

# -----------------------------
# Discovery
# -----------------------------
def _extract_title(m: dict) -> str:
    return m.get("question") or m.get("title") or m.get("name") or "(untitled)"

def _extract_condition_id(m: dict) -> Optional[str]:
    # Gamma docs describe condition ids; field names may vary in responses.
    return (
        m.get("conditionId")
        or m.get("condition_id")
        or (m.get("conditionIds")[0] if isinstance(m.get("conditionIds"), list) and m.get("conditionIds") else None)
        or (m.get("condition_ids")[0] if isinstance(m.get("condition_ids"), list) and m.get("condition_ids") else None)
    )

def _extract_tokens(m: dict) -> Tuple[Optional[str], Optional[str]]:
    # Gamma responses include clobTokenIds (typically YES/NO) :contentReference[oaicite:9]{index=9}
    toks = m.get("clobTokenIds") or m.get("clob_token_ids") or []
    if not isinstance(toks, list) or len(toks) == 0:
        return None, None
    yes_id = str(toks[0])
    no_id = str(toks[1]) if len(toks) > 1 else None
    return yes_id, no_id

def _market_is_open(m: dict) -> bool:
    # Gamma has multiple status flags; we keep it permissive.
    if "active" in m:
        return bool(m.get("active"))
    if "closed" in m:
        return not bool(m.get("closed"))
    return True

def discover_markets(keywords: List[str], max_pages: int = 5, page_size: int = 200) -> List[TrackedMarket]:
    kws = [k.strip().lower() for k in keywords if k.strip()]
    out: List[TrackedMarket] = []
    seen = set()

    for page in range(max_pages):
        offset = page * page_size
        items = gamma_markets_page(limit=page_size, offset=offset)
        if not items:
            break

        for m in items:
            title = _extract_title(m)
            slug = (m.get("slug") or "").strip()
            hay = f"{title} {slug}".lower()

            if kws and not any(k in hay for k in kws):
                continue

            condition_id = _extract_condition_id(m)
            yes_id, no_id = _extract_tokens(m)
            if not condition_id or not yes_id:
                continue

            enable_orderbook = bool(m.get("enableOrderBook", m.get("enable_order_book", True)))
            is_open = _market_is_open(m)

            market_id = str(m.get("id") or m.get("marketId") or m.get("market_id") or slug or condition_id)
            if market_id in seen:
                continue
            seen.add(market_id)

            out.append(
                TrackedMarket(
                    market_id=market_id,
                    slug=slug,
                    title=title,
                    condition_id=condition_id,
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                    liquidity_num=_safe_float(m.get("liquidityNum") or m.get("liquidity_num")),
                    volume_num=_safe_float(m.get("volumeNum") or m.get("volume_num")),
                    is_open=is_open,
                    enable_orderbook=enable_orderbook,
                )
            )

    return out

# -----------------------------
# Baselines + scoring
# -----------------------------
def baseline_from_history(df: pd.DataFrame) -> dict:
    """
    Returns baseline stats for anomaly detection:
      - typical_abs_delta_30m
      - typical_abs_delta_5m
      - robust_vol (MAD of returns)
    """
    if df is None or df.empty or len(df) < 5:
        return {"typ_abs_30m": None, "typ_abs_5m": None, "mad_ret": None}

    # assume fidelity already roughly uniform; compute deltas
    prices = df["price"].to_numpy(dtype=float)
    # returns (differences in prob space)
    d = np.diff(prices)
    if len(d) == 0:
        return {"typ_abs_30m": None, "typ_abs_5m": None, "mad_ret": None}

    # map “steps” into 5m and 30m based on timestamp spacing
    ts = df["ts"].astype("int64").to_numpy() / 1e9
    step = np.median(np.diff(ts)) if len(ts) > 2 else 300.0
    step = max(60.0, float(step))  # avoid nonsense
    k5 = max(1, int(round(300.0 / step)))
    k30 = max(1, int(round(1800.0 / step)))

    abs_d = np.abs(d)

    def _abs_move_k(k: int) -> Optional[float]:
        if len(prices) <= k:
            return None
        moves = np.abs(prices[k:] - prices[:-k])
        return float(np.median(moves)) if len(moves) else None

    return {
        "typ_abs_5m": _abs_move_k(k5),
        "typ_abs_30m": _abs_move_k(k30),
        "mad_ret": float(_mad(d)),
    }

def anomaly_score(current_mid: Optional[float], hist_df: Optional[pd.DataFrame]) -> Tuple[float, dict]:
    """
    Score in [0, 100] (roughly) using:
      - current move over ~30m vs typical
      - robust volatility proxy
    """
    if current_mid is None or hist_df is None or hist_df.empty:
        return 0.0, {"reason": "no data"}

    base = baseline_from_history(hist_df)
    typ30 = base.get("typ_abs_30m")
    mad_ret = base.get("mad_ret")

    # approximate 30m-ago price from history tail
    # (history endpoint ends at “now-ish”; we use last timestamp <= now-30m if possible)
    target = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(minutes=30)
    past = None
    older = hist_df[hist_df["ts"] <= target]
    if not older.empty:
        past = float(older.iloc[-1]["price"])
    else:
        # fallback: earliest point
        past = float(hist_df.iloc[0]["price"])

    move30 = abs(float(current_mid) - past)

    # normalize
    denom = typ30 if (typ30 is not None and typ30 > 1e-6) else (mad_ret if (mad_ret is not None and mad_ret > 1e-6) else 0.02)
    z = move30 / denom

    # squash into 0..100
    score = 100.0 * (1.0 - math.exp(-z / 2.0))  # gentle curve
    score = float(max(0.0, min(100.0, score)))

    details = {
        "move30": move30,
        "typ_abs_30m": typ30,
        "mad_ret": mad_ret,
        "z_like": z,
        "past30": past,
    }
    return score, details

# -----------------------------
# Holders / concentration
# -----------------------------
def concentration_metrics(holders_payload: List[dict]) -> dict:
    """
    Convert /holders response into simple concentration metrics.
    We use:
      - top1_share of sum(topN)
      - top3_share of sum(topN)
      - hh_index over topN amounts (normalized)
    """
    if not holders_payload:
        return {}

    # payload is list of tokens; we aggregate across returned tokens
    all_amounts = []
    for item in holders_payload:
        holders = item.get("holders") or []
        for h in holders:
            amt = h.get("amount")
            if amt is None:
                continue
            try:
                all_amounts.append(float(amt))
            except Exception:
                pass

    if not all_amounts:
        return {}

    amounts = np.array(sorted(all_amounts, reverse=True), dtype=float)
    s = float(amounts.sum())
    if s <= 0:
        return {}

    top1 = float(amounts[0] / s)
    top3 = float(amounts[:3].sum() / s) if len(amounts) >= 3 else float(amounts.sum() / s)
    p = amounts / s
    hhi = float(np.sum(p * p))  # 1.0 is max concentration

    return {"top1_share": top1, "top3_share": top3, "hhi": hhi, "n": int(len(amounts))}

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Polytracker (Geopolitics)", layout="wide")

if "watchlist" not in st.session_state:
    st.session_state.watchlist: Dict[str, TrackedMarket] = {}
if "history_cache" not in st.session_state:
    st.session_state.history_cache: Dict[str, pd.DataFrame] = {}
if "baseline_cache" not in st.session_state:
    st.session_state.baseline_cache: Dict[str, dict] = {}
if "holders_cache" not in st.session_state:
    st.session_state.holders_cache: Dict[str, dict] = {}  # condition_id -> metrics+raw timestamp
if "last_poll" not in st.session_state:
    st.session_state.last_poll = 0.0

st.title("Polymarket Geopolitics Tracker (Polling + Baselines + Concentration)")

with st.sidebar:
    st.header("Settings")

    keywords = st.text_area(
        "Keywords (one per line)",
        value="\n".join(DEFAULT_KEYWORDS),
        height=200,
    ).splitlines()
    keywords = [k.strip() for k in keywords if k.strip()]

    max_pages = st.slider("Discovery pages (200 per page)", 1, 20, 5)
    poll_seconds = st.slider("Polling interval (seconds)", 30, 600, 120, step=30)
    max_markets = st.slider("Max markets to monitor", 5, 200, 40)

    hist_interval = st.selectbox("Baseline history window", ["1d", "1w", "max"], index=1)
    fidelity = st.selectbox("History fidelity (minutes)", [1, 2, 5, 10, 15, 30], index=2)

    score_threshold = st.slider("Anomaly alert threshold", 0, 100, 60)
    holders_min_balance = st.number_input("Holders minBalance", min_value=0, max_value=999999, value=1, step=1)

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        do_discover = st.button("Discover markets")
    with colB:
        clear = st.button("Clear watchlist")

    if clear:
        st.session_state.watchlist = {}
        st.session_state.history_cache = {}
        st.session_state.baseline_cache = {}
        st.session_state.holders_cache = {}
        st.success("Cleared.")

tabs = st.tabs(["Discover", "Monitor", "Market detail"])

# -----------------------------
# Discover
# -----------------------------
with tabs[0]:
    st.subheader("Discover markets from keywords")

    if do_discover:
        with st.spinner("Searching Gamma markets..."):
            markets = discover_markets(keywords=keywords, max_pages=max_pages, page_size=200)

        # light ranking: prioritize open + orderbook + liquidity/volume
        def rank(m: TrackedMarket) -> float:
            liq = m.liquidity_num or 0.0
            vol = m.volume_num or 0.0
            open_bonus = 1.0 if m.is_open else 0.0
            ob_bonus = 1.0 if m.enable_orderbook else 0.0
            return open_bonus * 10 + ob_bonus * 5 + math.log1p(liq) + 0.5 * math.log1p(vol)

        markets = sorted(markets, key=rank, reverse=True)

        st.write(f"Found **{len(markets)}** matching markets. Showing top 200.")
        df = pd.DataFrame(
            [{
                "title": m.title,
                "slug": m.slug,
                "open": m.is_open,
                "orderbook": m.enable_orderbook,
                "liquidity": m.liquidity_num,
                "volume": m.volume_num,
                "conditionId": m.condition_id,
                "YES token": m.yes_token_id,
                "NO token": m.no_token_id,
            } for m in markets[:200]]
        )
        st.dataframe(df, use_container_width=True)

        pick = st.multiselect(
            "Add to watchlist",
            options=list(range(min(len(markets), 200))),
            format_func=lambda i: markets[i].title,
        )
        if st.button("Add selected"):
            for i in pick:
                m = markets[i]
                st.session_state.watchlist[m.market_id] = m
            st.success(f"Watchlist now has {len(st.session_state.watchlist)} markets.")

    if st.session_state.watchlist:
        st.info(f"Current watchlist size: {len(st.session_state.watchlist)}")

# -----------------------------
# Monitor (polling + baselines + conditional holders fetch)
# -----------------------------
with tabs[1]:
    st.subheader("Monitor")

    if not st.session_state.watchlist:
        st.warning("Your watchlist is empty. Go to Discover and add markets.")
    else:
        # autorefresh at poll_seconds
        st.caption("This page auto-refreshes for polling. (No websockets, slower + cheaper.)")
        try:
            st.autorefresh(interval=poll_seconds * 1000, key="poll")
        except Exception:
            # older Streamlit: fallback to manual refresh
            st.button("Refresh now")

        # limit to max_markets (priority: open+orderbook+liq)
        watch = list(st.session_state.watchlist.values())

        def watch_rank(m: TrackedMarket) -> float:
            return (1 if m.is_open else 0) * 10 + (1 if m.enable_orderbook else 0) * 5 + math.log1p(m.liquidity_num or 0)

        watch = sorted(watch, key=watch_rank, reverse=True)[:max_markets]

        rows = []
        anomalous_condition_ids = []

        for m in watch:
            token_id = m.yes_token_id  # track YES by default (you can change this logic)
            mid = clob_midpoint(token_id)

            # history/baseline cached per token
            if token_id not in st.session_state.history_cache:
                # fetch lazily to keep initial polling light
                hist = clob_prices_history(token_id, interval=hist_interval, fidelity_min=int(fidelity))
                st.session_state.history_cache[token_id] = hist
                st.session_state.baseline_cache[token_id] = baseline_from_history(hist)

            hist_df = st.session_state.history_cache.get(token_id)
            score, details = anomaly_score(mid, hist_df)

            # trigger holders fetch on anomaly
            if score >= score_threshold:
                anomalous_condition_ids.append(m.condition_id)

            base = st.session_state.baseline_cache.get(token_id, {})
            rows.append({
                "score": score,
                "mid": mid,
                "title": m.title,
                "slug": m.slug,
                "open": m.is_open,
                "liquidity": m.liquidity_num,
                "volume": m.volume_num,
                "move30": details.get("move30"),
                "typ_move30": base.get("typ_abs_30m"),
                "conditionId": m.condition_id,
                "YES token": token_id,
            })

        df = pd.DataFrame(rows).sort_values("score", ascending=False)

        # Fetch holders for anomalous markets (batched by condition IDs)
        # Data API expects comma-separated condition IDs via `market` param. :contentReference[oaicite:10]{index=10}
        if anomalous_condition_ids:
            # only fetch for up to 10 conditions per refresh to keep it cheap
            unique = list(dict.fromkeys(anomalous_condition_ids))[:10]
            payload = data_holders_for_condition_ids(unique, limit=20, min_balance=int(holders_min_balance))

            metrics = concentration_metrics(payload)
            ts = _now()
            for cid in unique:
                st.session_state.holders_cache[cid] = {"ts": ts, "payload": payload, "metrics": metrics}

        # Display
        st.dataframe(
            df.assign(
                mid=df["mid"].apply(lambda x: _pct(x) if x is not None else "—"),
                move30=df["move30"].apply(lambda x: _pct(x) if x is not None else "—"),
                typ_move30=df["typ_move30"].apply(lambda x: _pct(x) if x is not None else "—"),
            )[
                ["score", "mid", "move30", "typ_move30", "title", "open", "liquidity", "volume", "slug", "conditionId", "YES token"]
            ],
            use_container_width=True,
        )

        st.markdown("#### Concentration (only fetched for markets above the anomaly threshold)")
        if st.session_state.holders_cache:
            # show latest cache entries
            conc_rows = []
            for cid, obj in st.session_state.holders_cache.items():
                met = obj.get("metrics", {})
                conc_rows.append({
                    "conditionId": cid,
                    "checked": pd.to_datetime(obj.get("ts", 0), unit="s"),
                    "top1_share": met.get("top1_share"),
                    "top3_share": met.get("top3_share"),
                    "hhi": met.get("hhi"),
                    "n": met.get("n"),
                })
            conc_df = pd.DataFrame(conc_rows).sort_values("checked", ascending=False)
            if not conc_df.empty:
                conc_df["top1_share"] = conc_df["top1_share"].apply(lambda x: _pct(x) if x is not None else "—")
                conc_df["top3_share"] = conc_df["top3_share"].apply(lambda x: _pct(x) if x is not None else "—")
                st.dataframe(conc_df, use_container_width=True)
        else:
            st.caption("No concentration checks triggered yet.")

# -----------------------------
# Market detail
# -----------------------------
with tabs[2]:
    st.subheader("Market detail")

    if not st.session_state.watchlist:
        st.warning("Add markets in Discover first.")
    else:
        options = list(st.session_state.watchlist.values())
        pick = st.selectbox("Choose a market", options=options, format_func=lambda m: m.title)
        token_id = pick.yes_token_id

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current midpoint (YES)", _pct(clob_midpoint(token_id)))
        with col2:
            st.write("Condition ID")
            st.code(pick.condition_id)
        with col3:
            st.write("Token ID (YES)")
            st.code(token_id)

        st.markdown("### Price history (for baseline + visualization)")
        hist = clob_prices_history(token_id, interval=hist_interval, fidelity_min=int(fidelity))
        if hist.empty:
            st.info("No history returned for this token.")
        else:
            st.line_chart(hist.set_index("ts")["price"])

            base = baseline_from_history(hist)
            st.markdown("### Baseline stats")
            st.write({
                "typical_abs_move_5m": _pct(base.get("typ_abs_5m")) if base.get("typ_abs_5m") is not None else None,
                "typical_abs_move_30m": _pct(base.get("typ_abs_30m")) if base.get("typ_abs_30m") is not None else None,
                "MAD_of_step_returns": base.get("mad_ret"),
            })

        st.markdown("### Holders / concentration (on demand)")
        if st.button("Fetch holders now"):
            payload = data_holders_for_condition_ids([pick.condition_id], limit=20, min_balance=int(holders_min_balance))
            met = concentration_metrics(payload)
            st.session_state.holders_cache[pick.condition_id] = {"ts": _now(), "payload": payload, "metrics": met}

        hc = st.session_state.holders_cache.get(pick.condition_id)
        if hc:
            met = hc.get("metrics", {})
            st.write({
                "checked": pd.to_datetime(hc.get("ts", 0), unit="s"),
                "top1_share": _pct(met.get("top1_share")) if met.get("top1_share") is not None else None,
                "top3_share": _pct(met.get("top3_share")) if met.get("top3_share") is not None else None,
                "HHI (top holders)": met.get("hhi"),
                "n (holders counted)": met.get("n"),
            })
            with st.expander("Raw holders payload"):
                st.json(hc.get("payload"))
        else:
            st.caption("No holders snapshot for this market yet.")
