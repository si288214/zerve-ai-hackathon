"""
position_sizer.py — Kelly Criterion Position Sizing

Converts a model signal (BULLISH / BEARISH / NEUTRAL) + confidence score
into a concrete dollar amount and share count to trade.

Mathematics
-----------
Kelly Criterion:  f* = (b·p − q) / b

  p  = probability of being correct  (derived from model confidence)
  q  = 1 − p  (probability of being wrong)
  b  = payoff ratio  (expected gain / expected loss)
       → estimated from recent volatility: higher vol = lower b

We use Half-Kelly (f*/2) by default — this cuts the theoretical maximum
position in half, which dramatically reduces variance while keeping ~75%
of the optimal geometric growth rate.  In practice, full Kelly almost
always over-bets because model probabilities are never perfectly calibrated.

Position caps:
  • Maximum 25% of portfolio per stock
  • Minimum position: ignore if Kelly < 2% (noise)

Risk levels:
  HIGH    Kelly ≥ 15%
  MEDIUM  Kelly  8–15%
  LOW     Kelly  2– 8%
  NONE    Kelly < 2%  (don't trade)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# CORE KELLY FORMULA
# ─────────────────────────────────────────────────────────────

def kelly_fraction(
    confidence_pct: float,
    win_loss_ratio: float = 1.5,
    kelly_multiplier: float = 0.5,
    max_position: float = 0.25,
) -> float:
    """
    Compute the Kelly-optimal fraction of portfolio to allocate.

    Parameters
    ----------
    confidence_pct   : model confidence in [50, 95] %
    win_loss_ratio   : expected gain / expected loss  (b in Kelly formula)
    kelly_multiplier : fraction of full Kelly to use  (0.5 = half-Kelly)
    max_position     : hard cap on fraction of portfolio per position

    Returns
    -------
    fraction ∈ [0, max_position]  — fraction of portfolio to deploy
    """
    # Map confidence → probability: 50% confidence = no edge, 95% = strong edge
    # Rescale from [50, 95] → [0.50, 0.95] (confidence already in this range)
    p = min(confidence_pct / 100.0, 0.95)
    q = 1.0 - p
    b = win_loss_ratio

    kelly_full = (b * p - q) / b  # Kelly formula

    # Apply multiplier (half-Kelly by default) and floor at 0
    fraction = max(0.0, kelly_full * kelly_multiplier)

    return min(fraction, max_position)


def volatility_to_win_loss_ratio(volatility_annual_pct: float) -> float:
    """
    Map annualised volatility → win/loss ratio (b).

    High volatility stocks have larger expected moves but more uncertainty,
    so we lower b.  Low volatility stocks are steadier, so b is higher.

    Mapping:
      vol  10% → b = 2.0  (low-vol, like bonds — reliable small gains)
      vol  25% → b = 1.5  (mid-vol, typical growth stock)
      vol  50% → b = 1.0  (high-vol, speculative — even odds)
      vol  80% → b = 0.7  (very high vol — odds slightly against)
    """
    vol = np.clip(volatility_annual_pct, 5.0, 100.0)
    # Linear interpolation on log scale
    b = 2.5 - 0.018 * vol
    return float(np.clip(b, 0.6, 2.5))


# ─────────────────────────────────────────────────────────────
# FULL POSITION RECOMMENDATION
# ─────────────────────────────────────────────────────────────

def compute_position(
    signal: str,
    confidence_pct: float,
    current_price: float,
    portfolio_value: float,
    volatility_annual_pct: float = 25.0,
    kelly_multiplier: float = 0.5,
) -> dict:
    """
    Compute a full position recommendation.

    Parameters
    ----------
    signal               : "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence_pct       : model confidence score  [50–95]
    current_price        : latest stock price in USD
    portfolio_value      : total portfolio value in USD
    volatility_annual_pct: annualised volatility % (from stock_summary)
    kelly_multiplier     : 0.5 = half-Kelly (recommended), 1.0 = full Kelly

    Returns
    -------
    dict with keys:
        action          : "BUY" | "SELL" | "HOLD"
        dollars         : recommended trade size in USD
        shares          : approximate number of shares
        fraction_pct    : % of portfolio to allocate
        risk_level      : "HIGH" | "MEDIUM" | "LOW" | "NONE"
        kelly_raw       : raw Kelly fraction before capping
        win_loss_ratio  : b used in Kelly formula
        rationale       : human-readable explanation string
    """
    # NEUTRAL → always hold
    if signal == "NEUTRAL":
        return {
            "action":        "HOLD",
            "dollars":       0.0,
            "shares":        0.0,
            "fraction_pct":  0.0,
            "risk_level":    "NONE",
            "kelly_raw":     0.0,
            "win_loss_ratio": 0.0,
            "rationale":     "No clear herd consensus. Stay in cash or hold existing position.",
        }

    b = volatility_to_win_loss_ratio(volatility_annual_pct)
    f = kelly_fraction(
        confidence_pct,
        win_loss_ratio=b,
        kelly_multiplier=kelly_multiplier,
    )

    # Below 2% Kelly → signal too weak to act on
    if f < 0.02:
        return {
            "action":        "HOLD",
            "dollars":       0.0,
            "shares":        0.0,
            "fraction_pct":  round(f * 100, 1),
            "risk_level":    "NONE",
            "kelly_raw":     round(f, 4),
            "win_loss_ratio": round(b, 2),
            "rationale":     "Signal exists but Kelly fraction < 2% — edge too small to trade.",
        }

    dollars = f * portfolio_value
    shares  = dollars / current_price if current_price > 0 else 0.0

    # Risk level
    if f >= 0.15:
        risk_level = "HIGH"
    elif f >= 0.08:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    action = "BUY" if signal == "BULLISH" else "SELL"

    p = min(confidence_pct / 100.0, 0.95)
    rationale = (
        f"Kelly formula: f* = ({b:.1f}×{p:.2f} − {1-p:.2f}) / {b:.1f} = {f/kelly_multiplier:.3f} "
        f"→ Half-Kelly = {f:.3f} ({f*100:.1f}% of portfolio). "
        f"Win/loss ratio b={b:.2f} derived from {volatility_annual_pct:.0f}% annualised volatility."
    )

    return {
        "action":        action,
        "dollars":       round(dollars, 2),
        "shares":        round(shares, 3),
        "fraction_pct":  round(f * 100, 1),
        "risk_level":    risk_level,
        "kelly_raw":     round(f, 4),
        "win_loss_ratio": round(b, 2),
        "rationale":     rationale,
    }


# ─────────────────────────────────────────────────────────────
# PORTFOLIO SUMMARY
# ─────────────────────────────────────────────────────────────

def portfolio_summary(positions: list[dict], portfolio_value: float) -> dict:
    """
    Aggregate position recommendations across all stocks.

    Returns total capital deployed, cash remaining, and risk breakdown.
    """
    total_dollars = sum(p["dollars"] for p in positions)
    total_pct     = sum(p["fraction_pct"] for p in positions)

    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    for p in positions:
        risk_counts[p["risk_level"]] = risk_counts.get(p["risk_level"], 0) + 1

    return {
        "total_deployed_dollars": round(total_dollars, 2),
        "total_deployed_pct":     round(total_pct, 1),
        "cash_remaining_dollars": round(max(0, portfolio_value - total_dollars), 2),
        "cash_remaining_pct":     round(max(0, 100 - total_pct), 1),
        "risk_counts":            risk_counts,
        "over_allocated":         total_pct > 100,
    }
