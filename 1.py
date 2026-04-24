"""
sentiment_engine.py — Multi-Source Sentiment Field h(t) Computation
E.W. Research / Zerve Hackathon
=====================================================================

Computes external field h ∈ [-1, 1] from four independent sources:
  1. Alpha Vantage News Sentiment
  2. Options Market (IV percentile + put/call ratio) via Alpha Vantage
  3. Reddit/Social Sentiment (Reddit API)
  4. LLM Scoring (Claude reads headlines, returns structured score)

Each event type uses a custom weighting scheme for the four signals.
Sources degrade gracefully — missing signals are excluded from fusion.
"""

from __future__ import annotations
import os
import time
import warnings
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS  —  fill these in or set as environment variables
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "LMBZD9PA8K5P5ICI")
FINNHUB_KEY       = os.getenv("FINNHUB_KEY", "d6jki3hr01qkvh5qac8gd6jki3hr01qkvh5qac90")
# Reddit removed — social signal dropped in favour of 3-source fusion

# ─────────────────────────────────────────────────────────────────────────────
# FINBERT — lazy-loaded on first use, cached globally
# Model: ProsusAI/finbert  (free, ~440MB, runs on CPU)
# pip install transformers torch
# ─────────────────────────────────────────────────────────────────────────────

_finbert_pipeline = None

def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            print("    [llm] Loading FinBERT (first call only, ~440MB)...")
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,          # return all three class scores
                device=-1,           # CPU; change to 0 for GPU
                truncation=True,
                max_length=512,
            )
            print("    [llm] FinBERT ready.")
        except Exception as e:
            print(f"    [llm] FinBERT load failed: {e}")
            _finbert_pipeline = None
    return _finbert_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# PER-EVENT-TYPE FUSION WEIGHTS
# Each vector is [news, options, social, llm] — they are L1-normalised after
# dropping missing sources, so partial availability still works.
# ─────────────────────────────────────────────────────────────────────────────

EVENT_WEIGHTS: dict[str, list[float]] = {
    #                news   options  social  llm
    "earnings":   [0.0,    0.35,    0.0,   0.65],
    "fed":        [0.0,    0.10,    0.0,   0.90],
    "cpi":        [0.0,    0.10,    0.0,   0.90],
    "drug_trial": [0.0,    0.45,    0.0,   0.55],
    "merger":     [0.0,    0.40,    0.0,   0.60],
}
DEFAULT_WEIGHTS = [0.0, 0.35, 0.0, 0.65]


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — FINNHUB NEWS
# Free tier: 60 calls/min, full historical archive, no recency wall.
# Returns headlines for FinBERT scoring — no pre-scored sentiment from Finnhub.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_finnhub_news(ticker: str, event_date: str,
                       lookback_days: int = 7,
                       event_type: str = "earnings") -> tuple[Optional[float], list[str]]:
    """
    Fetches company news from Finnhub for [event_date - lookback_days, event_date].
    Returns (None, headlines[]) — h is computed downstream by FinBERT.
    Finnhub free tier: 60 req/min, full historical data.
    Uses event-type specific keyword filters and falls back to general news search
    for low-coverage tickers.
    """
    event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    from_dt  = event_dt - timedelta(days=lookback_days)

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}"
        f"&from={from_dt.strftime('%Y-%m-%d')}"
        f"&to={event_dt.strftime('%Y-%m-%d')}"
        f"&token={FINNHUB_KEY}"
    )

    # Event-type specific keyword filters
    EVENT_KEYWORDS = {
        "earnings": {
            'earnings', 'estimates', 'eps', 'revenue', 'guidance',
            'forecast', 'outlook', 'quarter', 'beat', 'miss', 'profit',
            'results', 'expectation', 'analyst', 'consensus', 'whisper',
            'sales', 'income', 'margin', 'growth', 'raised', 'lowered',
            'above', 'below', 'topped',
        },
        "fed": {
            'fed', 'federal reserve', 'fomc', 'rate', 'inflation', 'interest',
            'powell', 'cut', 'hike', 'hold', 'pause', 'pivot', 'hawkish',
            'dovish', 'monetary', 'cpi', 'pce', 'jobs', 'employment', 'gdp',
        },
        "cpi": {
            'inflation', 'cpi', 'pce', 'prices', 'consumer price', 'core',
            'fed', 'rate', 'hot', 'cool', 'above', 'below', 'estimates',
        },
        "drug_trial": {
            'fda', 'approval', 'approved', 'pdufa', 'trial', 'drug', 'therapy',
            'phase', 'clinical', 'efficacy', 'safety', 'reject', 'complete response',
            'adcom', 'advisory', 'nda', 'bla', 'breakthrough',
        },
        "merger": {
            'merger', 'acquisition', 'deal', 'acquire', 'buyout', 'takeover',
            'shareholder', 'vote', 'close', 'regulatory', 'antitrust', 'doj',
            'ftc', 'approval', 'synergy', 'premium', 'offer',
        },
    }
    keywords = EVENT_KEYWORDS.get(event_type, EVENT_KEYWORDS["earnings"])

    try:
        r    = requests.get(url, timeout=15)
        data = r.json()

        if not isinstance(data, list):
            print(f"    [news] Finnhub error: {str(data)[:80]}")
            return None, []

        # For low-coverage tickers with no results, try Finnhub general news search
        if not data:
            print(f"    [news] Finnhub: no articles for {ticker}, trying news search...")
            search_url = (
                f"https://finnhub.io/api/v1/news"
                f"?category=general"
                f"&token={FINNHUB_KEY}"
            )
            r2   = requests.get(search_url, timeout=15)
            data = r2.json() if isinstance(r2.json(), list) else []
            # Filter to articles mentioning the ticker or company
            data = [a for a in data if ticker.upper() in (a.get("headline","") + a.get("summary","")).upper()]

        if not data:
            print(f"    [news] Finnhub: no articles found")
            return None, []

        # Combine headline + summary
        headlines = []
        for a in data[:50]:
            title   = a.get("headline", "")
            summary = a.get("summary", "")
            text    = title + (". " + summary[:120] if summary else "")
            if title:
                headlines.append(text)

        # Apply event-type keyword filter
        filtered = [h for h in headlines if any(k in h.lower() for k in keywords)]

        # Fall back to all headlines if filter removes everything
        if len(filtered) < 3:
            print(f"    [news] Finnhub: {len(data)} articles → {len(headlines)} headlines (no filter applied)")
            return None, headlines[:20]

        print(f"    [news] Finnhub: {len(data)} articles → {len(filtered)}/{len(headlines)} relevant headlines")
        return None, filtered[:20]

    except Exception as e:
        print(f"    [news] Finnhub exception: {e}")
        return None, []


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — OPTIONS MARKET (Alpha Vantage)
# Computes h from IV percentile and put/call skew.
# For drug trials: IV spike pre-event is the dominant signal.
# For mergers: call skew / deal spread proxy.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_av_options(ticker: str, event_date: str, event_type: str) -> Optional[float]:
    """
    Fetches option chain via AV HISTORICAL_OPTIONS endpoint.
    Returns h ∈ [-1, 1] derived from:
      - put/call IV ratio  (skew)
      - put/call volume ratio
      - IV percentile vs 30-day trailing (for drug_trial)
    """
    event_dt = datetime.strptime(event_date, "%Y-%m-%d")

    # Use the closest available options snapshot before the event
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=HISTORICAL_OPTIONS"
        f"&symbol={ticker}"
        f"&date={event_dt.strftime('%Y-%m-%d')}"
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )

    try:
        r    = requests.get(url, timeout=20)
        data = r.json()

        if "data" not in data:
            msg = data.get("Note", data.get("Information", "no data key"))
            print(f"    [options] AV error: {str(msg)[:80]}")
            return None

        chain = data["data"]
        if not chain:
            print("    [options] empty chain")
            return None

        # Separate calls and puts near the money
        calls, puts = [], []
        for opt in chain:
            try:
                iv  = float(opt.get("implied_volatility", 0) or 0)
                vol = float(opt.get("volume", 0) or 0)
                typ = opt.get("type", "").lower()
                if iv <= 0:
                    continue
                if typ == "call":
                    calls.append({"iv": iv, "volume": vol})
                elif typ == "put":
                    puts.append({"iv": iv, "volume": vol})
            except (ValueError, TypeError):
                continue

        if not calls or not puts:
            print("    [options] insufficient call/put data")
            return None

        call_iv  = np.mean([c["iv"]  for c in calls])
        put_iv   = np.mean([p["iv"]  for p in puts])
        call_vol = np.sum([c["volume"] for c in calls]) + 1e-6
        put_vol  = np.sum([p["volume"] for p in puts])  + 1e-6

        # Put/call IV skew: >1 means puts priced higher (bearish), <1 bullish
        iv_skew   = put_iv / call_iv        # >1 → bearish
        pc_ratio  = put_vol / call_vol      # >1 → bearish

        # Normalise to [-1, 1]
        # iv_skew = 1.0 → neutral; 1.5+ → very bearish; 0.5- → very bullish
        iv_signal  = float(np.clip((1.0 - iv_skew) * 2.0, -1, 1))
        # pc_ratio = 1.0 → neutral; 2.0 → bearish; 0.5 → bullish
        pc_signal  = float(np.clip((1.0 - pc_ratio), -1, 1))

        # For drug trials: high absolute IV signals uncertainty; use sign of skew
        if event_type == "drug_trial":
            avg_iv    = (call_iv + put_iv) / 2.0
            # IV >0.8 in biotech is normal; >1.5 is extreme uncertainty
            iv_norm   = float(np.clip((avg_iv - 0.8) / 0.7, -1, 1))
            h = 0.4 * iv_signal + 0.4 * pc_signal - 0.2 * iv_norm
        elif event_type == "merger":
            # For mergers call skew is the key signal
            h = 0.6 * iv_signal + 0.4 * pc_signal
        else:
            h = 0.5 * iv_signal + 0.5 * pc_signal

        h = float(np.clip(h, -1, 1))
        print(f"    [options] iv_skew={iv_skew:.3f}  pc_ratio={pc_ratio:.3f}  "
              f"iv_sig={iv_signal:+.3f}  pc_sig={pc_signal:+.3f} → h={h:+.3f}")
        return h

    except Exception as e:
        print(f"    [options] exception: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — REDDIT / SOCIAL SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

# Social source removed — no Reddit required.


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4 — LOCAL NLP SCORING (FinBERT)
# Free, runs entirely offline after the one-time model download (~440 MB).
# ProsusAI/finbert is fine-tuned on financial news for pos/neg/neutral.
#
# Event-type adjustments:
#   fed / cpi  — invert polarity: "positive" macro news (strong economy) can
#                mean hawkish Fed = bearish for rate cuts
#   drug_trial — "positive" = approval signal; standard polarity
#   merger     — "positive" = deal completion signal; standard polarity
#   earnings   — standard polarity
# ─────────────────────────────────────────────────────────────────────────────

# Maps event type → polarity multiplier applied to the raw FinBERT score.
# +1 = standard (positive sentiment → positive h)
# -1 = inverted (positive macro sentiment → hawkish = negative h for rate-cut events)
FINBERT_POLARITY: dict[str, float] = {
    "earnings":   +1.0,
    "fed":        -0.5,   # partial invert: strong econ news → less likely to cut
    "cpi":        -1.0,   # positive sentiment about economy → CPI likely hot → above est
    "drug_trial": +1.0,
    "merger":     +1.0,
}

# Sector-aware polarity adjustments for earnings events.
# Some sectors have systematic macro noise that inverts or dampens FinBERT signal:
#   semis / tech hardware — tariff/supply chain fear dominates coverage even on beats
#   pharma               — drug pipeline news is positive but unrelated to quarterly EPS
# Multiplier is applied ON TOP of FINBERT_POLARITY for earnings events only.
SECTOR_POLARITY: dict[str, float] = {
    # Semiconductors — tariff headlines score negative; dampen bearish signal
    "MU":    0.4,   # high macro noise
    "ASML":  0.4,
    "TXN":   0.6,
    "NVDA":  0.7,
    "AMD":   0.6,
    "INTC":  0.5,
    "QCOM":  0.6,
    "AMAT":  0.5,
    "KLAC":  0.5,
    "LRCX":  0.5,
    # Pharma — pipeline/regulatory news dominates; dampen signal
    "MRK":   0.4,
    "PFE":   0.4,
    "ABBV":  0.5,
    "BMY":   0.4,
    "AZN":   0.5,
    "LLY":   0.7,   # less noisy, direct drug news
    "BIIB":  0.6,
    # Mega-cap tech — heavy non-earnings coverage; slight dampening
    "AAPL":  0.8,
    "MSFT":  0.8,
    "GOOGL": 0.8,
    "META":  0.9,
    "AMZN":  0.8,
    "TSLA":  0.6,   # heavy macro/Musk noise
}

# Minimum |h| to make a prediction. Below this, signal is too weak to call.
H_CONFIDENCE_THRESHOLD = 0.15

def fetch_finbert_score(ticker: str, event_type: str,
                        headlines: list[str]) -> Optional[float]:
    """
    Runs FinBERT over up to 12 headlines and returns h ∈ [-1, 1].
    Applies sector-aware polarity dampening for noisy sectors.
    Returns None if |h| < H_CONFIDENCE_THRESHOLD (too uncertain to call).
    """
    if not headlines:
        print("    [llm] no headlines to score")
        return None

    pipe = _get_finbert()
    if pipe is None:
        print("    [llm] FinBERT unavailable, skipping")
        return None

    scores = []
    for headline in headlines[:20]:
        try:
            result = pipe(headline[:512])
            # result is a list of lists when top_k=None: [[{label, score}, ...]]
            classes = result[0] if isinstance(result[0], list) else result
            label_map = {d["label"].lower(): d["score"] for d in classes}

            pos  = label_map.get("positive", 0.0)
            neg  = label_map.get("negative", 0.0)
            neut = label_map.get("neutral",  0.0)

            raw    = pos - neg                    # ∈ (-1, 1)
            weight = 1.0 - neut                   # down-weight neutral headlines
            scores.append((raw, weight))
        except Exception as e:
            print(f"    [llm] FinBERT error on headline: {e}")
            continue

    if not scores:
        return None

    total_w    = sum(w for _, w in scores) + 1e-9
    weighted_h = sum(r * w for r, w in scores) / total_w
    polarity   = FINBERT_POLARITY.get(event_type, 1.0)
    h          = float(np.clip(weighted_h * polarity, -1, 1))

    # Apply sector-aware dampening for earnings events
    if event_type == "earnings":
        sector_mult = SECTOR_POLARITY.get(ticker.upper(), 1.0)
        h = float(np.clip(h * sector_mult, -1, 1))
        if sector_mult < 1.0:
            print(f"    [llm]  sector dampening {ticker} × {sector_mult:.1f} → h={h:+.3f}")

    avg_raw = np.mean([r for r, _ in scores])
    print(f"    [llm]  FinBERT {len(scores)} headlines  "
          f"avg_raw={avg_raw:+.3f}  polarity={polarity:+.1f}  → h={h:+.3f}")

    # Confidence threshold — below this the signal is too weak to call
    if abs(h) < H_CONFIDENCE_THRESHOLD:
        print(f"    [llm]  |h|={abs(h):.3f} below threshold {H_CONFIDENCE_THRESHOLD} → NO PREDICTION")
        return None

    return h


# ─────────────────────────────────────────────────────────────────────────────
# FUSION
# ─────────────────────────────────────────────────────────────────────────────

def fuse_signals(signals: dict[str, Optional[float]],
                 event_type: str) -> tuple[float, dict]:
    """
    signals: {"news": float|None, "options": float|None,
              "social": float|None, "llm": float|None}

    Returns (h_fused, breakdown_dict).
    Available signals are re-weighted proportionally.
    """
    keys    = ["news", "options", "social", "llm"]
    weights = EVENT_WEIGHTS.get(event_type, DEFAULT_WEIGHTS)

    available   = [(k, w, signals[k]) for k, w in zip(keys, weights) if signals[k] is not None]
    if not available:
        print("    [fuse] all sources unavailable, h=0.0")
        return 0.0, {"h": 0.0, "sources_used": [], "weights_used": {}}

    total_w = sum(w for _, w, _ in available)
    h_fused = sum(w * v for _, w, v in available) / total_w

    sources_used  = {k: round(v, 4) for k, _, v in available}
    weights_used  = {k: round(w / total_w, 3) for k, w, _ in available}

    h_fused = float(np.clip(h_fused, -1, 1))
    print(f"    [fuse] sources={list(sources_used.keys())}  h_fused={h_fused:+.4f}")

    return h_fused, {
        "h": round(h_fused, 4),
        "sources_used": sources_used,
        "weights_used": weights_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def compute_h(event: dict, lookback_days: int = 7,
              av_sleep: float = 0.5) -> tuple[float, dict]:
    """
    Master function: given an event dict, returns (h, breakdown).
    Calls all four sources with event-type-aware logic.

    event keys: ticker, date, type, desc
    """
    ticker     = event["ticker"]
    date       = event["date"]
    etype      = event["type"]
    desc       = event["desc"]

    print(f"  Computing h for: {desc}")

    # ── Source 1: Finnhub news → headlines for FinBERT ──
    h_news, headlines = fetch_finnhub_news(ticker, date, lookback_days, event_type=etype)
    time.sleep(av_sleep)

    # ── Source 2: Options ──
    h_options = fetch_av_options(ticker, date, etype)
    time.sleep(av_sleep)

    # ── Source 3: Social — removed ──
    h_social = None

    # ── Source 4: FinBERT (local, free) ──
    h_llm = fetch_finbert_score(ticker, etype, headlines)

    # ── Fusion ──
    h, breakdown = fuse_signals(
        {"news": h_news, "options": h_options, "social": h_social, "llm": h_llm},
        etype,
    )

    breakdown["source_raw"] = {
        "news":    round(h_news,    4) if h_news    is not None else None,
        "options": round(h_options, 4) if h_options is not None else None,
        "social":  round(h_social,  4) if h_social  is not None else None,
        "llm":     round(h_llm,     4) if h_llm     is not None else None,
    }

    return h, breakdown