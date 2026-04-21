"""
data_fetcher.py — Social Sentiment & Stock Data Pipeline

Data sources:
  • Google Trends  (pytrends)  — search interest as proxy for public attention
  • yfinance                   — historical OHLCV stock prices
  • [Extensible]  X/Twitter API, Instagram Graph API, Reddit PRAW

Sentiment derivation:
  Google Trends interest [0–100] is converted to a normalised sentiment
  score h ∈ [-1, 1]:
      h = (recent_7d_avg − historical_avg) / (historical_avg + ε)
  Positive h → rising public interest → bullish external field
  Negative h → fading interest      → bearish external field
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# STOCK DATA
# ─────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """
    Fetch OHLCV data from Yahoo Finance.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    or None on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=True)
        if hist.empty or len(hist) < 10:
            return None
        return hist
    except Exception as e:
        print(f"[data_fetcher] yfinance error for {ticker}: {e}")
        return None


def compute_returns(price_series: pd.Series, normalise: bool = True) -> np.ndarray:
    """
    Compute log returns from a price series.
    If normalise=True, z-score them (zero mean, unit variance) so they
    are on the same scale as magnetisation.
    """
    log_ret = np.log(price_series).diff().dropna().values
    if normalise and log_ret.std() > 0:
        log_ret = (log_ret - log_ret.mean()) / log_ret.std()
    return log_ret


def stock_summary(hist: pd.DataFrame) -> dict:
    """Return key price statistics for display."""
    closes = hist["Close"]
    returns = closes.pct_change().dropna()

    current = float(closes.iloc[-1])
    prev_week = float(closes.iloc[-6]) if len(closes) >= 6 else current
    prev_month = float(closes.iloc[-22]) if len(closes) >= 22 else current

    vol_annual = float(returns.std() * np.sqrt(252) * 100)

    return {
        "current_price": round(current, 2),
        "change_7d_pct": round((current / prev_week - 1) * 100, 2),
        "change_30d_pct": round((current / prev_month - 1) * 100, 2),
        "volatility_annual_pct": round(vol_annual, 1),
        "avg_volume": int(hist["Volume"].tail(20).mean()),
    }


# ─────────────────────────────────────────────────────────────
# GOOGLE TRENDS  (social contagion proxy)
# ─────────────────────────────────────────────────────────────

def fetch_google_trends(
    keyword: str,
    timeframe: str = "today 3-m",
    retries: int = 3,
    delay: float = 2.0,
) -> pd.Series | None:
    """
    Fetch Google Trends interest-over-time for a single keyword.

    Returns a pd.Series (DatetimeIndex → interest [0–100]),
    or None if the fetch fails.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("[data_fetcher] pytrends not installed. Run: pip install pytrends")
        return None

    for attempt in range(retries):
        try:
            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            pytrends.build_payload([keyword], timeframe=timeframe, geo="US")
            df = pytrends.interest_over_time()

            if df.empty:
                return None
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            return df[keyword].astype(float)

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"[data_fetcher] Google Trends failed for '{keyword}': {e}")
                return None

    return None


def trends_to_sentiment_history(
    trends_series: pd.Series,
    window: int = 7,
) -> np.ndarray:
    """
    Convert a Google Trends series → array of normalised sentiment
    scores h ∈ [-1, 1], one per week.

    Score at time t:
        h_t = (mean[t-w:t] − mean[0:t-w]) / (mean[0:t-w] + ε)

    Positive → recent surge in interest (bullish pressure)
    Negative → declining interest (bearish pressure)
    """
    vals = trends_series.values.astype(float)
    scores = []
    for t in range(window, len(vals)):
        recent = vals[max(0, t - window):t].mean()
        baseline = vals[:max(1, t - window)].mean()
        score = (recent - baseline) / (baseline + 1e-6)
        scores.append(float(np.clip(score, -1.0, 1.0)))
    return np.array(scores) if scores else np.zeros(1)


def current_sentiment_score(trends_series: pd.Series, window: int = 7) -> float:
    """
    Latest sentiment score from the tail of a trends series.
    Returns h ∈ [-1, 1].
    """
    vals = trends_series.values.astype(float)
    if len(vals) < window + 1:
        return 0.0
    recent = vals[-window:].mean()
    baseline = vals[:-window].mean()
    score = (recent - baseline) / (baseline + 1e-6)
    return float(np.clip(score, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────
# MOCK / FALLBACK SENTIMENT
# ─────────────────────────────────────────────────────────────

def mock_sentiment_history(
    n_points: int,
    base_score: float = 0.0,
    noise: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate synthetic sentiment scores for offline testing.
    Models a mean-reverting process around base_score.
    """
    rng = np.random.default_rng(seed)
    scores = [base_score]
    for _ in range(n_points - 1):
        # Ornstein-Uhlenbeck style
        prev = scores[-1]
        new = prev * 0.8 + base_score * 0.2 + rng.normal(0, noise)
        scores.append(float(np.clip(new, -1.0, 1.0)))
    return np.array(scores)


def derive_sentiment_from_returns(returns: np.ndarray, smoothing: int = 3) -> np.ndarray:
    """
    Fallback: derive a sentiment proxy directly from price momentum
    when social data is unavailable.

    Uses a rolling sign-weighted momentum of log returns.
    """
    if len(returns) == 0:
        return np.zeros(1)
    smoothed = np.convolve(returns, np.ones(smoothing) / smoothing, mode="valid")
    # Normalise to [-1, 1]
    mx = np.abs(smoothed).max()
    if mx > 0:
        smoothed = smoothed / mx
    return smoothed.clip(-1, 1)


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE: one ticker
# ─────────────────────────────────────────────────────────────

def get_stock_and_sentiment(
    ticker: str,
    keyword: str | None = None,
    period: str = "6mo",
    use_trends: bool = True,
    verbose: bool = True,
) -> dict:
    """
    High-level function: fetch everything needed for one stock.

    Returns
    -------
    dict with keys:
        ticker, keyword,
        stock_hist     : pd.DataFrame or None
        returns        : np.ndarray  (normalised log returns)
        summary        : dict of price stats
        sentiment_hist : np.ndarray  (h values, aligned to returns length)
        current_h      : float       (today's sentiment score)
        data_source    : str         ("google_trends" | "price_momentum" | "mock")
    """
    if keyword is None:
        keyword = ticker

    result = {
        "ticker": ticker,
        "keyword": keyword,
        "stock_hist": None,
        "returns": np.array([]),
        "summary": {},
        "sentiment_hist": np.array([]),
        "current_h": 0.0,
        "data_source": "mock",
    }

    # ── Stock prices ──────────────────────────────────────────────────────
    if verbose:
        print(f"  📈 Fetching {ticker} price history…")
    hist = fetch_stock_data(ticker, period=period)
    result["stock_hist"] = hist

    if hist is not None and len(hist) >= 10:
        returns = compute_returns(hist["Close"], normalise=True)
        result["returns"] = returns
        result["summary"] = stock_summary(hist)
    else:
        if verbose:
            print(f"  ⚠️  Failed to fetch prices for {ticker}, using mock data.")
        returns = mock_sentiment_history(60, noise=0.4)
        result["returns"] = returns

    n_hist = len(result["returns"])

    # ── Social sentiment ──────────────────────────────────────────────────
    if use_trends:
        if verbose:
            print(f"  🔍 Fetching Google Trends for '{keyword}'…")
        trends = fetch_google_trends(keyword)

        if trends is not None and len(trends) >= 14:
            sent_hist = trends_to_sentiment_history(trends, window=7)
            current_h = current_sentiment_score(trends, window=7)
            result["data_source"] = "google_trends"
            if verbose:
                print(f"  ✅ Trends OK  (current h = {current_h:+.3f})")
        else:
            if verbose:
                print(f"  ⚠️  Trends unavailable — deriving from price momentum.")
            sent_hist = derive_sentiment_from_returns(result["returns"])
            current_h = float(sent_hist[-1]) if len(sent_hist) else 0.0
            result["data_source"] = "price_momentum"
    else:
        sent_hist = derive_sentiment_from_returns(result["returns"])
        current_h = float(sent_hist[-1]) if len(sent_hist) else 0.0
        result["data_source"] = "price_momentum"

    # Align lengths for calibration
    min_len = min(len(sent_hist), n_hist)
    result["sentiment_hist"] = sent_hist[:min_len]
    result["returns"] = result["returns"][:min_len]
    result["current_h"] = current_h

    return result
