"""
backtest.py — Ising Market Model Signal Backtest
E.W. Research / Zerve Hackathon
=================================================

Tests whether the Blume-Capel Ising model + Google Trends h(t)
has predictive signal over historical periods.

Strategy:
  - Each week: derive h(t) from Google Trends window, run Ising MC
  - Magnetisation M > +threshold  -> LONG next week
  - Magnetisation M < -threshold  -> SHORT next week
  - Otherwise                     -> FLAT
  - Evaluate: Sharpe, hit rate, L/S spread, max drawdown, IC

Usage:
    python backtest.py

Requires:
    pip install yfinance pytrends numpy pandas scipy matplotlib networkx
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

# ── Import your existing modules (same directory) ─────────────────────────────
from ising_model import IsingInvestorNetwork, magnetization_to_signal
from data_fetcher import (
    fetch_stock_data,
    fetch_google_trends,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BACKTEST_STOCKS = {
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",
    "AAPL":  "Apple",
    "META":  "Meta",
    "AMZN":  "Amazon",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMD":   "AMD",
    "SPY":   "S&P 500",
    "QQQ":   "NASDAQ",
}

CFG = {
    "period":        "2y",   # yfinance history window
    "n_investors":   100,    # smaller for backtest speed
    "D":             0.25,   # Blume-Capel anisotropy
    "n_equil":       120,    # MC equilibration sweeps
    "n_samples":     25,     # MC production samples
    "sample_every":  3,
    "mag_threshold": 0.08,   # |M| > threshold triggers signal
    "forward_weeks": 1,      # predict N weeks ahead
    "beta_fixed":    1.2,    # fixed beta (skip calibration for speed)
    "trends_window": 7,      # weeks of trends used to compute h
    "use_trends":    True,   # False -> use price momentum as h
    "seed":          42,
}

# ─────────────────────────────────────────────────────────────────────────────
# BUILD ROLLING SIGNAL HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_history(ticker: str, keyword: str, cfg: dict):
    """
    Walk through weekly history. At each step:
      1. Compute h(t) from Google Trends up to time t
      2. Run Ising MC -> magnetization M(t)
      3. Record 1-week forward return

    Returns a DataFrame or None.
    """
    print(f"\n{'─'*60}")
    print(f"  {ticker}  ({keyword})")
    print(f"{'─'*60}")

    # ── Price data ────────────────────────────────────────────────────────
    hist = fetch_stock_data(ticker, period=cfg["period"])
    if hist is None or len(hist) < 30:
        print(f"  [!] Insufficient price data")
        return None

    closes = hist["Close"].resample("W-FRI").last().dropna()
    print(f"  Price: {len(closes)} weekly closes")

    # ── Google Trends ─────────────────────────────────────────────────────
    trends_weekly = None
    data_source = "price_momentum"

    if cfg["use_trends"]:
        print(f"  Fetching Google Trends for '{keyword}'...")
        raw_trends = fetch_google_trends(keyword, timeframe="today 5-y")
        time.sleep(30)  # respect rate limits

        if raw_trends is not None and len(raw_trends) >= 14:
            trends_weekly = raw_trends.resample("W-FRI").mean().dropna()
            data_source = "google_trends"
            print(f"  Trends: {len(trends_weekly)} weekly observations  OK")
        else:
            print(f"  Trends unavailable -> using price momentum")

    # ── Ising model (reused across steps) ────────────────────────────────
    model = IsingInvestorNetwork(
        n_investors=cfg["n_investors"],
        J=1.0,
        beta=cfg["beta_fixed"],
        D=cfg["D"],
        seed=cfg["seed"],
    )

    # ── Align dates ───────────────────────────────────────────────────────
    w = cfg["trends_window"]
    fw = cfg["forward_weeks"]

    if trends_weekly is not None:
        common = sorted(closes.index.intersection(trends_weekly.index))
    else:
        common = sorted(closes.index)

    if len(common) < w + fw + 2:
        print(f"  [!] Not enough aligned data ({len(common)})")
        return None

    print(f"  Running {len(common) - w - fw} weekly simulations...")

    records = []
    for i in range(w, len(common) - fw):
        date_t   = common[i]
        date_fwd = common[i + fw]

        # ── Compute h(t) ──────────────────────────────────────────────
        if trends_weekly is not None:
            window = trends_weekly.reindex(common[i - w:i]).dropna().values.astype(float)
            if len(window) >= 4:
                recent   = window[-3:].mean()
                baseline = window[:-3].mean()
                h_t = float(np.clip((recent - baseline) / (baseline + 1e-6), -1, 1))
            else:
                h_t = 0.0
        else:
            price_win = closes.reindex(common[max(0, i - w):i]).dropna().values
            if len(price_win) >= 3:
                lr = np.diff(np.log(price_win + 1e-9))
                h_t = float(np.clip(lr.mean() / (lr.std() + 1e-8), -1, 1))
            else:
                h_t = 0.0

        # ── Ising MC ──────────────────────────────────────────────────
        model.set_sentiment_field(h_t)
        mags = model.run(
            n_equil=cfg["n_equil"],
            n_samples=cfg["n_samples"],
            sample_every=cfg["sample_every"],
        )
        mag_mean = float(np.mean(mags))
        mag_std  = float(np.std(mags))

        signal, confidence = magnetization_to_signal(
            mag_mean, mag_std,
            buy_threshold=cfg["mag_threshold"],
            sell_threshold=-cfg["mag_threshold"],
        )

        # ── Forward return ────────────────────────────────────────────
        try:
            p_t   = float(closes.loc[date_t])
            p_fwd = float(closes.loc[date_fwd])
            fwd_ret = float(np.log(p_fwd / p_t))
        except Exception:
            continue

        hit = (
                (signal == "BULLISH" and fwd_ret > 0) or
                (signal == "BEARISH" and fwd_ret < 0)
        )

        records.append({
            "date":          date_t,
            "ticker":        ticker,
            "data_source":   data_source,
            "h":             round(h_t, 4),
            "magnetization": round(mag_mean, 4),
            "mag_std":       round(mag_std, 4),
            "signal":        signal,
            "confidence":    confidence,
            "forward_return":round(fwd_ret, 6),
            "hit":           hit,
        })

    if not records:
        print(f"  [!] No records generated")
        return None

    df = pd.DataFrame(records)
    b = (df["signal"] == "BULLISH").sum()
    s = (df["signal"] == "BEARISH").sum()
    n = (df["signal"] == "NEUTRAL").sum()
    print(f"  Signals: BULLISH={b}  BEARISH={s}  NEUTRAL={n}  total={len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    active = df[df["signal"] != "NEUTRAL"].copy()
    if len(active) < 5:
        return {"error": "insufficient_signals"}

    long_df  = active[active["signal"] == "BULLISH"]
    short_df = active[active["signal"] == "BEARISH"]

    long_rets  = long_df["forward_return"].values
    short_rets = -short_df["forward_return"].values  # flip for short P&L

    strat_rets = np.concatenate([long_rets, short_rets])
    hits = active["hit"].values

    ic, ic_pval = stats.pearsonr(active["magnetization"].values, active["forward_return"].values)

    ann_sharpe = (strat_rets.mean() / (strat_rets.std() + 1e-8)) * np.sqrt(52)

    cum = np.cumprod(1 + strat_rets)
    roll_max = np.maximum.accumulate(cum)
    max_dd = float(((cum - roll_max) / (roll_max + 1e-9)).min())

    avg_long  = long_rets.mean()  if len(long_rets)  > 0 else np.nan
    avg_short_raw = short_df["forward_return"].values
    avg_short = avg_short_raw.mean() if len(avg_short_raw) > 0 else np.nan
    ls_spread = avg_long - avg_short if (not np.isnan(avg_long) and not np.isnan(avg_short)) else np.nan

    from scipy.stats import binomtest
    bt = binomtest(int(hits.sum()), len(hits), 0.5, alternative="greater")

    return {
        "n_signals":     len(active),
        "n_long":        len(long_rets),
        "n_short":       len(short_rets),
        "hit_rate":      round(float(hits.mean()), 4),
        "hit_pval":      round(float(bt.pvalue), 4),
        "sharpe_ann":    round(float(ann_sharpe), 3),
        "max_drawdown":  round(float(max_dd), 4),
        "avg_long_ret":  round(float(avg_long),  5) if not np.isnan(avg_long)  else None,
        "avg_short_ret": round(float(avg_short), 5) if not np.isnan(avg_short) else None,
        "ls_spread":     round(float(ls_spread), 5) if not np.isnan(ls_spread) else None,
        "ic":            round(float(ic), 4),
        "ic_pval":       round(float(ic_pval), 4),
        "cum_return":    round(float(np.prod(1 + strat_rets) - 1), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_analysis(all_dfs):
    combined = pd.concat(all_dfs, ignore_index=True)
    active = combined[combined["signal"] != "NEUTRAL"].copy()
    active["strategy_ret"] = np.where(
        active["signal"] == "BULLISH",
        active["forward_return"],
        -active["forward_return"],
        )
    weekly = (
        active.groupby("date")["strategy_ret"]
        .mean()
        .reset_index()
        .rename(columns={"strategy_ret": "portfolio_ret"})
        .sort_values("date")
    )
    weekly["cum_ret"]  = (1 + weekly["portfolio_ret"]).cumprod()
    weekly["drawdown"] = (
            (weekly["cum_ret"] - weekly["cum_ret"].cummax())
            / (weekly["cum_ret"].cummax() + 1e-9)
    )
    return weekly


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_dfs, metrics_dict, portfolio_df, save_path="backtest_results.png"):
    valid_tickers = [t for t, m in metrics_dict.items() if "error" not in m]

    fig = plt.figure(figsize=(20, 16), facecolor="#0d0d0d")
    fig.suptitle(
        "Ising Market Model  —  Signal Backtest  |  E.W. Research",
        fontsize=16, color="#e8d5b0", fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    gold = "#e8d5b0"; red = "#e05252"; grn = "#4eca7e"; grey = "#666666"

    def sax(ax, title):
        ax.set_facecolor("#111111")
        ax.set_title(title, color=gold, fontsize=10)
        ax.tick_params(colors=grey)
        for sp in ax.spines.values():
            sp.set_color("#333333")

    # Panel 1: cumulative portfolio return
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(portfolio_df["date"], portfolio_df["cum_ret"], color=gold, lw=2)
    ax1.axhline(1.0, color=grey, lw=0.8, ls="--")
    ax1.fill_between(portfolio_df["date"], portfolio_df["cum_ret"], 1,
                     where=(portfolio_df["cum_ret"] >= 1), alpha=0.15, color=grn)
    ax1.fill_between(portfolio_df["date"], portfolio_df["cum_ret"], 1,
                     where=(portfolio_df["cum_ret"] < 1), alpha=0.15, color=red)
    sax(ax1, "Portfolio Cumulative Return (Equal-Weight L/S)")
    ax1.set_ylabel("Growth of $1", color=grey)

    # Panel 2: drawdown
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between(portfolio_df["date"], portfolio_df["drawdown"], 0,
                     color=red, alpha=0.7)
    sax(ax2, "Portfolio Drawdown")
    ax2.set_ylabel("Drawdown", color=grey)

    # Panel 3: hit rates
    ax3 = fig.add_subplot(gs[1, 0])
    hit_rates = [metrics_dict[t]["hit_rate"] * 100 for t in valid_tickers]
    colors3 = [grn if h >= 52 else (red if h < 48 else grey) for h in hit_rates]
    ax3.barh(valid_tickers, hit_rates, color=colors3, edgecolor="#222")
    ax3.axvline(50, color=grey, lw=1, ls="--")
    ax3.set_xlabel("Hit Rate %", color=grey)
    sax(ax3, "Hit Rate by Ticker")

    # Panel 4: Sharpe
    ax4 = fig.add_subplot(gs[1, 1])
    sharpes = [metrics_dict[t]["sharpe_ann"] for t in valid_tickers]
    colors4 = [grn if s > 0.3 else (red if s < -0.1 else grey) for s in sharpes]
    ax4.barh(valid_tickers, sharpes, color=colors4, edgecolor="#222")
    ax4.axvline(0, color=grey, lw=1)
    ax4.axvline(0.5, color=grn, lw=0.8, ls="--", alpha=0.5)
    ax4.set_xlabel("Annualised Sharpe", color=grey)
    sax(ax4, "Sharpe Ratio")

    # Panel 5: IC
    ax5 = fig.add_subplot(gs[1, 2])
    ics = [metrics_dict[t]["ic"] for t in valid_tickers]
    colors5 = [grn if ic > 0.05 else (red if ic < -0.05 else grey) for ic in ics]
    ax5.barh(valid_tickers, ics, color=colors5, edgecolor="#222")
    ax5.axvline(0, color=grey, lw=1)
    ax5.set_xlabel("IC = Pearson ρ(M, fwd_ret)", color=grey)
    sax(ax5, "Information Coefficient")

    # Panel 6: M vs fwd return scatter
    ax6 = fig.add_subplot(gs[2, :2])
    combined_df = pd.concat(all_dfs, ignore_index=True)
    active_df = combined_df[combined_df["signal"] != "NEUTRAL"]
    ax6.scatter(
        active_df["magnetization"],
        active_df["forward_return"] * 100,
        c=active_df["magnetization"], cmap="RdYlGn",
        alpha=0.35, s=10, vmin=-0.5, vmax=0.5,
        )
    x = active_df["magnetization"].values
    y = active_df["forward_return"].values * 100
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax6.plot(xs, slope * xs + intercept, color=gold, lw=2,
             label=f"OLS  r={r_val:.3f}  p={p_val:.3f}")
    ax6.axhline(0, color=grey, lw=0.8, ls="--")
    ax6.axvline(0, color=grey, lw=0.8, ls="--")
    ax6.set_xlabel("Magnetization M", color=grey)
    ax6.set_ylabel("Fwd Return %", color=grey)
    ax6.legend(facecolor="#1a1a1a", labelcolor=gold, fontsize=9)
    sax(ax6, "M vs 1-Week Forward Return (all tickers pooled)")

    # Panel 7: summary table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    rows = []
    for t in valid_tickers:
        m = metrics_dict[t]
        ls = m.get("ls_spread")
        rows.append([
            t,
            f"{m['hit_rate']*100:.1f}%",
            f"{m['sharpe_ann']:.2f}",
            f"{m['ic']:+.3f}",
            f"{ls*100:.2f}%" if ls is not None else "N/A",
        ])
    tbl = ax7.table(
        cellText=rows,
        colLabels=["Ticker", "Hit%", "Sharpe", "IC", "L/S"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1a1a" if r > 0 else "#2a2a2a")
        cell.set_text_props(color=gold if r == 0 else grey)
        cell.set_edgecolor("#333333")
    sax(ax7, "Performance Summary")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"  Chart saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(metrics_dict):
    print("\n" + "=" * 80)
    print("  ISING MARKET MODEL  —  BACKTEST RESULTS  |  E.W. Research")
    print("=" * 80)
    hdr = "{:<8} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}"
    print(hdr.format("Ticker", "N", "Hit%", "Hit_p", "Sharpe", "IC", "IC_p", "L/S Sprd"))
    print("-" * 80)
    for t, m in metrics_dict.items():
        if "error" in m:
            print(f"  {t:<6}  ERROR: {m['error']}")
            continue
        ls = m.get("ls_spread")
        print(hdr.format(
            t,
            m["n_signals"],
            f"{m['hit_rate']*100:.1f}%",
            f"{m['hit_pval']:.3f}",
            f"{m['sharpe_ann']:.3f}",
            f"{m['ic']:+.3f}",
            f"{m['ic_pval']:.3f}",
            f"{ls*100:.2f}%" if ls is not None else "N/A",
        ))
    print("=" * 80)

    valid = {t: m for t, m in metrics_dict.items() if "error" not in m}
    if valid:
        avg_hit = np.mean([m["hit_rate"]   for m in valid.values()])
        avg_sh  = np.mean([m["sharpe_ann"] for m in valid.values()])
        avg_ic  = np.mean([m["ic"]         for m in valid.values()])
        ls_vals = [m["ls_spread"] for m in valid.values() if m.get("ls_spread") is not None]
        avg_ls  = np.mean(ls_vals) if ls_vals else float("nan")
        print(f"\n  AVERAGES:  Hit={avg_hit*100:.1f}%  Sharpe={avg_sh:.3f}  IC={avg_ic:+.3f}  L/S Spread={avg_ls*100:.2f}%")
        print()
        print("  SIGNAL GUIDE:")
        print("  IC > 0.05, IC_p < 0.10  ->  statistically meaningful predictive signal")
        print("  Hit > 52%, Hit_p < 0.10 ->  directional accuracy above chance")
        print("  Sharpe > 0.50           ->  viable as a trading signal")
        print("  L/S Spread > 0          ->  model successfully ranks direction")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Ising Market Model  —  Signal Backtest")
    print(f"  Period    : {CFG['period']}")
    print(f"  Forward   : {CFG['forward_weeks']} week(s)")
    print(f"  Beta      : {CFG['beta_fixed']}  (fixed)")
    print(f"  Threshold : +/-{CFG['mag_threshold']}")
    print(f"  Data      : {'Google Trends + yfinance' if CFG['use_trends'] else 'Price momentum + yfinance'}")
    print("=" * 60)

    all_dfs = []
    metrics_dict = {}

    for ticker, keyword in BACKTEST_STOCKS.items():
        df = build_signal_history(ticker, keyword, CFG)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            metrics_dict[ticker] = compute_metrics(df)
        else:
            metrics_dict[ticker] = {"error": "no_data"}

    if not all_dfs:
        print("\n[!] No data collected. Check internet/yfinance/pytrends.")
        raise SystemExit(1)

    # Portfolio
    portfolio_df = portfolio_analysis(all_dfs)

    # Print results table
    print_table(metrics_dict)

    # Portfolio stats
    pr = portfolio_df["portfolio_ret"].values
    port_sharpe = (pr.mean() / (pr.std() + 1e-8)) * np.sqrt(52)
    port_cum    = float(portfolio_df["cum_ret"].iloc[-1]) - 1
    port_maxdd  = float(portfolio_df["drawdown"].min())

    print(f"\n  PORTFOLIO (equal-weight L/S):")
    print(f"  Sharpe (ann.) : {port_sharpe:.3f}")
    print(f"  Cum. Return   : {port_cum*100:.2f}%")
    print(f"  Max Drawdown  : {port_maxdd*100:.2f}%")
    print(f"  Weeks tracked : {len(portfolio_df)}")

    # Save raw signals
    pd.concat(all_dfs, ignore_index=True).to_csv("backtest_signals.csv", index=False)
    print("\n  Raw signals saved -> backtest_signals.csv")

    # Chart
    print("  Generating charts...")
    plot_results(all_dfs, metrics_dict, portfolio_df, "backtest_results.png")

    print("\nBacktest complete.")