import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

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

# Browser-ish headers help on hosted environments
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Streamlit; +https://streamlit.io) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
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

def _now() -> float:
    return time.time()

def _pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.1f}%"

def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]

# -----------------------------
# API calls (cached)
# -----------------------------
@st.cache_data(ttl=30)
def clob_midpoint(token_id: str) -> Optional[float]:
    r = requests.get(
        f"{CLOB_BASE}/midpoint",
        params={"token_id": token_id},
        timeout=15,
        headers=DEFAULT_HEADERS,
    )
    if r.status_code != 200:
        return None
    j = r.json()
    return _safe_float(j.get("mid"))

@st.cache_data(ttl=600)
def clob_prices_history(token_id: str, interval: str = "1w", fidelity_min: int = 5) -> pd.DataFrame:
    params = {"market": token_id, "interval": interval, "fidelity": fidelity_min}
    r = requests.get(
        f"{CLOB_BASE}/prices-history",
        params=params,
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
    df = df[["ts", "price"]].dropna().sort_values("ts")
    return df

@st.cache_data(ttl=900)
def gamma_public_search(
    q: str,
    page: int = 0,
    limit_per_type: int = 50,
    keep_closed_markets: int = 0,
    events_status: str = "active",
    cache: bool = True,
    optimized: bool = True,
) -> dict:
    params = {
        "q": q,
        "page": page,
        "limit_per_type": limit_per_type,
        "keep_closed_markets": keep_closed_markets,
        "events_status": events_status,
        "cache": str(cache).lower(),
        "optimized": str(optimized).lower(),
        "search_tags": "false",
        "search_profiles": "false",
    }
    r = requests.get(
        f"{GAMMA_BASE}/public-search",
        params=params,
        timeout=25,
        headers=DEFAULT_HEADERS,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=900)
def gamma_market_by_slug(slug: str) -> Optional[dict]:
    """
    Official endpoint:
      GET https://gamma-api.polymarket.com/markets/slug/{slug}
    This returns full market data (incl. clobTokenIds / condition ids). :contentReference[oaicite:1]{index=1}
    """
    if not slug:
        return None
    r = requests.get(
        f"{GAMMA_BASE}/markets/slug/{slug}",
        timeout=25,
        headers=DEFAULT_HEADERS,
    )
    if r.status_code != 200:
        return None
    j = r.json()
    # Some SDKs return a list; normalize to dict
    if isinstance(j, list):
        return j[0] if j else None
    return j if isinstance(j, dict) else None

@st.cache_data(ttl=900)
def data_holders_for_condition_ids(condition_ids: List[str], limit: int = 20, min_balance: int = 1) -> List[dict]:
    if not condition_ids:
        return []
    params = {
        "market": ",".join(condition_ids),
        "limit": limit,
        "minBalance": min_balance,
    }
    r = requests.get(
        f"{DATA_BASE}/holders",
        params=params,
        timeout=20,
        headers=DEFAULT_HEADERS,
    )
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
# Discovery helpers
# -----------------------------
def _extract_title(m: dict) -> str:
    return m.get("question") or m.get("title") or m.get("name") or "(untitled)"

def _extract_condition_id(m: dict) -> Optional[str]:
    return (
        m.get("conditionId")
        or m.get("condition_id")
        or (m.get("conditionIds")[0] if isinstance(m.get("conditionIds"), list) and m.get("conditionIds") else None)
        or (m.get("condition_ids")[0] if isinstance(m.get("condition_ids"), list) and m.get("condition_ids") else None)
        or (m.get("condition_ids") if isinstance(m.get("condition_ids"), str) else None)
    )

def _extract_tokens(m: dict) -> Tuple[Optional[str], Optional[str]]:
    toks = m.get("clobTokenIds") or m.get("clob_token_ids") or []
    if not isinstance(toks, list) or len(toks) == 0:
        return None, None
    yes_id = str(toks[0])
    no_id = str(toks[1]) if len(toks) > 1 else None
    return yes_id, no_id

def _market_is_open(m: dict) -> bool:
    if "active" in m:
        return bool(m.get("active"))
    if "closed" in m:
        return not bool(m.get("closed"))
    return True

def _collect_slugs_from_search_payload(payload: dict) -> List[str]:
    slugs: List[str] = []
    events = payload.get("events") or []
    if isinstance(events, dict):
        events = [events]
    for ev in events:
        markets = ev.get("markets") or []
        if isinstance(markets, dict):
            markets = [markets]
        for m in markets:
            s = m.get("slug")
            if s:
                slugs.append(str(s))
    return slugs

def discover_markets(
    keywords: List[str],
    max_pages: int = 3,
    limit_per_type: int = 50,
    keep_closed_markets: bool = False,
    debug: bool = False,
) -> Tuple[List[TrackedMarket], dict]:
    """
    Pipeline:
      1) /public-search -> slugs (slim markets)
      2) /markets/slug/{slug} -> full market objects (clobTokenIds, condition ids)
    """
    kws = [k.strip() for k in keywords if k.strip()]
    keep_closed = 1 if keep_closed_markets else 0

    debug_info = {
        "slugs_collected": 0,
        "slugs_sample": [],
        "hydrated_ok": 0,
        "hydrated_failed": 0,
        "hydrated_missing_ids": 0,
        "example_hydrated_market_keys": None,
    }

    if not kws:
        return [], debug_info

    all_slugs: List[str] = []
    first_search_payload = None

    for kw_i, kw in enumerate(kws):
        for page in range(max_pages):
            res = gamma_public_search(
                q=kw,
                page=page,
                limit_per_type=limit_per_type,
                keep_closed_markets=keep_closed,
                events_status="active",
                cache=True,
                optimized=True,
            )
            if debug and kw_i == 0 and page == 0:
                first_search_payload = res

            slugs = _collect_slugs_from_search_payload(res)
            if not slugs:
                break
            all_slugs.extend(slugs)

            pag = res.get("pagination") or {}
            if pag and not pag.get("hasMore", False):
                break

    # de-dupe slugs
    all_slugs = list(dict.fromkeys(all_slugs))
    debug_info["slugs_collected"] = len(all_slugs)
    debug_info["slugs_sample"] = all_slugs[:10]

    if debug and first_search_payload is not None:
        debug_info["first_search_payload"] = first_search_payload

    if not all_slugs:
        return [], debug_info

    out: List[TrackedMarket] = []
    seen = set()

    # hydrate only up to a reasonable cap to avoid slow discovery
    # you can increase this if you want
    hydrate_cap = min(len(all_slugs), 300)

    for slug in all_slugs[:hydrate_cap]:
        full = gamma_market_by_slug(slug)
        if not full:
            debug_info["hydrated_failed"] += 1
            continue

        debug_info["hydrated_ok"] += 1
        if debug_info["example_hydrated_market_keys"] is None:
            debug_info["example_hydrated_market_keys"] = sorted(list(full.keys()))

        title = _extract_title(full)
        condition_id = _extract_condition_id(full)
        yes_id, no_id = _extract_tokens(full)

        if not condition_id or not yes_id:
            debug_info["hydrated_missing_ids"] += 1
            continue

        enable_orderbook = bool(full.get("enableOrderBook", full.get("enable_order_book", True)))
        is_open = _market_is_open(full)

        market_id = str(full.get("id") or full.get("marketId") or full.get("market_id") or slug or condition_id)
        if market_id in seen:
            continue
        seen.add(market_id)

        out.append(
            TrackedMarket(
                market_id=market_id,
                slug=slug,
                title=title,
                condition_id=str(condition_id),
                yes_token_id=str(yes_id),
                no_token_id=str(no_id) if no_id is not None else None,
                liquidity_num=_safe_float(full.get("liquidityNum") or full.get("liquidity_num") or full.get("liquidity")),
                volume_num=_safe_float(full.get("volumeNum") or full.get("volume_num") or full.get("volume")),
                is_open=is_open,
                enable_orderbook=enable_orderbook,
            )
        )

    return out, debug_info

# -----------------------------
# Baselines + scoring
# -----------------------------
def baseline_from_history(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 5:
        return {"typ_abs_30m": None, "typ_abs_5m": None, "mad_ret": None}

    prices = df["price"].to_numpy(dtype=float)
    d = np.diff(prices)
    if len(d) == 0:
        return {"typ_abs_30m": None, "typ_abs_5m": None, "mad_ret": None}

    ts = df["ts"].astype("int64").to_numpy() / 1e9
    step = np.median(np.diff(ts)) if len(ts) > 2 else 300.0
    step = max(60.0, float(step))
    k5 = max(1, int(round(300.0 / step)))
    k30 = max(1, int(round(1800.0 / step)))

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
    if current_mid is None or hist_df is None or hist_df.empty:
        return 0.0, {"reason": "no data"}

    base = baseline_from_history(hist_df)
    typ30 = base.get("typ_abs_30m")
    mad_ret = base.get("mad_ret")

    target = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(minutes=30)
    older = hist_df[hist_df["ts"] <= target]
    past = float(older.iloc[-1]["price"]) if not older.empty else float(hist_df.iloc[0]["price"])

    move30 = abs(float(current_mid) - past)

    denom = (
        typ30 if (typ30 is not None and typ30 > 1e-6)
        else (mad_ret if (mad_ret is not None and mad_ret > 1e-6) else 0.02)
    )
    z = move30 / denom

    score = 100.0 * (1.0 - math.exp(-z / 2.0))
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
    if not holders_payload:
        return {}

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
    hhi = float(np.sum(p * p))

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
    st.session_state.holders_cache: Dict[str, dict] = {}

st.title("Polymarket Geopolitics Tracker (Polling + Baselines + Concentration)")

with st.sidebar:
    st.header("Settings")

    keywords = st.text_area(
        "Keywords (one per line)",
        value="\n".join(DEFAULT_KEYWORDS),
        height=200,
    ).splitlines()
    keywords = [k.strip() for k in keywords if k.strip()]

    max_pages = st.slider("Discovery pages per keyword", 1, 10, 3)
    limit_per_type = st.slider("Search results per page", 10, 100, 50, step=10)
    keep_closed = st.checkbox("Include closed markets in discovery", value=False)
    debug_discovery = st.checkbox("Debug discovery", value=True)

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

with tabs[0]:
    st.subheader("Discover markets from keywords")

    if do_discover:
        with st.spinner("Discovering (search → hydrate slugs)…"):
            markets, dbg = discover_markets(
                keywords=keywords,
                max_pages=max_pages,
                limit_per_type=limit_per_type,
                keep_closed_markets=keep_closed,
                debug=debug_discovery,
            )

        st.write(f"Found **{len(markets)}** matching markets. Showing top 200.")

        if debug_discovery:
            with st.expander("Debug info"):
                st.json(dbg)

        # ranking
        def rank(m: TrackedMarket) -> float:
            liq = m.liquidity_num or 0.0
            vol = m.volume_num or 0.0
            open_bonus = 1.0 if m.is_open else 0.0
            ob_bonus = 1.0 if m.enable_orderbook else 0.0
            return open_bonus * 10 + ob_bonus * 5 + math.log1p(liq) + 0.5 * math.log1p(vol)

        markets = sorted(markets, key=rank, reverse=True)

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

with tabs[1]:
    st.subheader("Monitor")

    if not st.session_state.watchlist:
        st.warning("Your watchlist is empty. Go to Discover and add markets.")
    else:
        st.caption("This page auto-refreshes for polling.")
        try:
            st.autorefresh(interval=poll_seconds * 1000, key="poll")
        except Exception:
            st.button("Refresh now")

        watch = list(st.session_state.watchlist.values())

        def watch_rank(m: TrackedMarket) -> float:
            return (1 if m.is_open else 0) * 10 + (1 if m.enable_orderbook else 0) * 5 + math.log1p(m.liquidity_num or 0)

        watch = sorted(watch, key=watch_rank, reverse=True)[:max_markets]

        rows = []
        anomalous_condition_ids = []

        for m in watch:
            token_id = m.yes_token_id
            mid = clob_midpoint(token_id)

            if token_id not in st.session_state.history_cache:
                hist = clob_prices_history(token_id, interval=hist_interval, fidelity_min=int(fidelity))
                st.session_state.history_cache[token_id] = hist
                st.session_state.baseline_cache[token_id] = baseline_from_history(hist)

            hist_df = st.session_state.history_cache.get(token_id)
            score, details = anomaly_score(mid, hist_df)

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

        if anomalous_condition_ids:
            unique = list(dict.fromkeys(anomalous_condition_ids))[:10]
            payload = data_holders_for_condition_ids(unique, limit=20, min_balance=int(holders_min_balance))
            metrics = concentration_metrics(payload)
            ts = _now()
            for cid in unique:
                st.session_state.holders_cache[cid] = {"ts": ts, "payload": payload, "metrics": metrics}

        st.dataframe(
            df.assign(
                mid=df["mid"].apply(lambda x: _pct(x) if x is not None else "—"),
                move30=df["move30"].apply(lambda x: _pct(x) if x is not None else "—"),
                typ_move30=df["typ_move30"].apply(lambda x: _pct(x) if x is not None else "—"),
            )[[
                "score", "mid", "move30", "typ_move30", "title", "open",
                "liquidity", "volume", "slug", "conditionId", "YES token"
            ]],
            use_container_width=True,
        )

        st.markdown("#### Concentration (only fetched for markets above the anomaly threshold)")
        if st.session_state.holders_cache:
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

        st.markdown("### Price history")
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
