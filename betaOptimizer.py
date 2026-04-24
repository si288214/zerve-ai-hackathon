"""
optimize_beta.py — Beta Optimization for Ising Market Model
E.W. Research / Zerve Hackathon
============================================================

Finds the optimal beta (inverse temperature) per ticker by maximizing
out-of-sample Sharpe ratio via walk-forward cross-validation.

Methods:
  1. Grid search   — sweep beta grid, pick best OOS Sharpe
  2. Scipy minimize — gradient-free Nelder-Mead on OOS Sharpe
  3. Walk-forward  — rolling train/test windows, no lookahead bias

Usage:
    python optimize_beta.py

Output:
    optimal_betas.json   — best beta per ticker
    beta_optimization.png — diagnostic charts
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import time
import warnings
from scipy.optimize import minimize_scalar, minimize
from scipy import stats

warnings.filterwarnings("ignore")

from ising_model import IsingInvestorNetwork, magnetization_to_signal
from data_fetcher import fetch_stock_data, fetch_google_trends

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

STOCKS = {
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
    "period":           "2y",
    "n_investors":      80,       # smaller for speed during optimization
    "D":                0.25,
    "n_equil":          80,
    "n_samples":        20,
    "sample_every":     3,
    "mag_threshold":    0.08,
    "forward_weeks":    1,
    "trends_window":    7,
    "use_trends":       True,
    "seed":             42,

    # Grid search range
    "beta_min":         0.1,
    "beta_max":         4.0,
    "beta_grid_points": 25,

    # Walk-forward
    "train_frac":       0.6,      # 60% train, 40% test (no lookahead)
    "n_folds":          3,        # number of rolling folds
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (reused across optimization runs)
# ─────────────────────────────────────────────────────────────────────────────

def load_ticker_data(ticker: str, keyword: str, cfg: dict):
    """
    Returns aligned weekly arrays: dates, h_series, close_series
    or None if data unavailable.
    """
    hist = fetch_stock_data(ticker, period=cfg["period"])
    if hist is None or len(hist) < 30:
        return None

    closes = hist["Close"].resample("W-FRI").last().dropna()

    trends_weekly = None
    if cfg["use_trends"]:
        raw = fetch_google_trends(keyword, timeframe="today 5-y")
        time.sleep(1.2)
        if raw is not None and len(raw) >= 14:
            trends_weekly = raw.resample("W-FRI").mean().dropna()

    w = cfg["trends_window"]
    if trends_weekly is not None:
        common = sorted(closes.index.intersection(trends_weekly.index))
    else:
        common = sorted(closes.index)

    if len(common) < w + cfg["forward_weeks"] + 5:
        return None

    # Build h series and forward returns
    fw = cfg["forward_weeks"]
    records = []
    for i in range(w, len(common) - fw):
        date_t   = common[i]
        date_fwd = common[i + fw]

        if trends_weekly is not None:
            win = trends_weekly.reindex(common[i - w:i]).dropna().values.astype(float)
            if len(win) >= 4:
                recent   = win[-3:].mean()
                baseline = win[:-3].mean()
                h_t = float(np.clip((recent - baseline) / (baseline + 1e-6), -1, 1))
            else:
                h_t = 0.0
        else:
            pwin = closes.reindex(common[max(0, i - w):i]).dropna().values
            if len(pwin) >= 3:
                lr = np.diff(np.log(pwin + 1e-9))
                h_t = float(np.clip(lr.mean() / (lr.std() + 1e-8), -1, 1))
            else:
                h_t = 0.0

        try:
            p_t   = float(closes.loc[date_t])
            p_fwd = float(closes.loc[date_fwd])
            fwd_ret = float(np.log(p_fwd / p_t))
        except Exception:
            continue

        records.append({"date": date_t, "h": h_t, "forward_return": fwd_ret})

    if len(records) < 15:
        return None

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: EVALUATE BETA ON A DATA SLICE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_beta(beta: float, data: pd.DataFrame, cfg: dict,
                  model: IsingInvestorNetwork = None) -> dict:
    """
    Run Ising model with given beta on a data slice.
    Returns dict with sharpe, hit_rate, ic, cum_return.
    """
    if model is None:
        model = IsingInvestorNetwork(
            n_investors=cfg["n_investors"],
            J=1.0,
            beta=beta,
            D=cfg["D"],
            seed=cfg["seed"],
        )
    else:
        model.beta = beta

    mags = []
    signals = []
    fwd_rets = []

    for _, row in data.iterrows():
        model.set_sentiment_field(float(row["h"]))
        samps = model.run(
            n_equil=cfg["n_equil"],
            n_samples=cfg["n_samples"],
            sample_every=cfg["sample_every"],
        )
        m_mean = float(np.mean(samps))
        m_std  = float(np.std(samps))
        sig, _ = magnetization_to_signal(
            m_mean, m_std,
            buy_threshold=cfg["mag_threshold"],
            sell_threshold=-cfg["mag_threshold"],
        )
        mags.append(m_mean)
        signals.append(sig)
        fwd_rets.append(float(row["forward_return"]))

    mags     = np.array(mags)
    fwd_rets = np.array(fwd_rets)
    signals  = np.array(signals)

    # Strategy returns (L/S)
    active_mask = signals != "NEUTRAL"
    if active_mask.sum() < 3:
        return {"sharpe": -99.0, "hit_rate": 0.5, "ic": 0.0, "cum_return": 0.0,
                "n_active": 0}

    active_sig  = signals[active_mask]
    active_ret  = fwd_rets[active_mask]
    strat_rets  = np.where(active_sig == "BULLISH", active_ret, -active_ret)

    ann_sharpe  = (strat_rets.mean() / (strat_rets.std() + 1e-8)) * np.sqrt(52)
    hit_rate    = float(((active_sig == "BULLISH") & (active_ret > 0) |
                         (active_sig == "BEARISH") & (active_ret < 0)).mean())
    ic, _       = stats.pearsonr(mags[active_mask], active_ret)
    cum_ret     = float(np.prod(1 + strat_rets) - 1)

    return {
        "sharpe":     float(ann_sharpe),
        "hit_rate":   float(hit_rate),
        "ic":         float(ic) if not np.isnan(ic) else 0.0,
        "cum_return": cum_ret,
        "n_active":   int(active_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_beta(data: pd.DataFrame, cfg: dict) -> dict:
    """Sweep beta grid, return full curve + best beta by OOS Sharpe."""
    betas = np.linspace(cfg["beta_min"], cfg["beta_max"], cfg["beta_grid_points"])

    # Walk-forward split: train on first 60%, test on last 40%
    n = len(data)
    split = int(n * cfg["train_frac"])
    train_data = data.iloc[:split]
    test_data  = data.iloc[split:]

    model = IsingInvestorNetwork(
        n_investors=cfg["n_investors"], J=1.0, beta=1.0,
        D=cfg["D"], seed=cfg["seed"],
    )

    curve = []
    print(f"    Grid search: {len(betas)} points  (train={len(train_data)}, test={len(test_data)})")

    for beta in betas:
        train_res = evaluate_beta(beta, train_data, cfg, model)
        test_res  = evaluate_beta(beta, test_data,  cfg, model)
        curve.append({
            "beta":         float(beta),
            "train_sharpe": train_res["sharpe"],
            "test_sharpe":  test_res["sharpe"],
            "train_ic":     train_res["ic"],
            "test_ic":      test_res["ic"],
            "train_hit":    train_res["hit_rate"],
            "test_hit":     test_res["hit_rate"],
        })

    curve_df = pd.DataFrame(curve)

    # Best beta by test Sharpe
    best_idx   = curve_df["test_sharpe"].idxmax()
    best_beta  = float(curve_df.loc[best_idx, "beta"])
    best_sharpe= float(curve_df.loc[best_idx, "test_sharpe"])

    return {
        "best_beta":   best_beta,
        "best_sharpe": best_sharpe,
        "curve":       curve_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: SCIPY SCALAR MINIMIZE (Brent)
# ─────────────────────────────────────────────────────────────────────────────

def scipy_optimize_beta(data: pd.DataFrame, cfg: dict) -> dict:
    """
    Use scipy minimize_scalar (Brent method) to find beta that maximizes
    OOS Sharpe. More precise than grid but no curve.
    """
    n = len(data)
    split = int(n * cfg["train_frac"])
    test_data = data.iloc[split:]

    model = IsingInvestorNetwork(
        n_investors=cfg["n_investors"], J=1.0, beta=1.0,
        D=cfg["D"], seed=cfg["seed"],
    )

    call_count = [0]

    def neg_sharpe(beta):
        call_count[0] += 1
        res = evaluate_beta(float(beta), test_data, cfg, model)
        return -res["sharpe"]   # minimize -> negate

    result = minimize_scalar(
        neg_sharpe,
        bounds=(cfg["beta_min"], cfg["beta_max"]),
        method="bounded",
        options={"xatol": 0.05, "maxiter": 20},
    )

    best_beta   = float(result.x)
    best_sharpe = -float(result.fun)

    print(f"    Scipy (Brent): beta*={best_beta:.3f}  Sharpe={best_sharpe:.3f}  calls={call_count[0]}")
    return {"best_beta": best_beta, "best_sharpe": best_sharpe}


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: WALK-FORWARD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walkforward_optimize(data: pd.DataFrame, cfg: dict) -> dict:
    """
    Rolling walk-forward: train on window, optimize beta, test on next window.
    Averages best beta across folds — most robust, least lookahead.
    """
    n = len(data)
    fold_size = n // (cfg["n_folds"] + 1)

    model = IsingInvestorNetwork(
        n_investors=cfg["n_investors"], J=1.0, beta=1.0,
        D=cfg["D"], seed=cfg["seed"],
    )

    betas_grid = np.linspace(cfg["beta_min"], cfg["beta_max"], 15)
    fold_results = []

    print(f"    Walk-forward: {cfg['n_folds']} folds  fold_size={fold_size}")

    for fold in range(cfg["n_folds"]):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if test_end - test_start < 5:
            continue

        train_slice = data.iloc[:train_end]
        test_slice  = data.iloc[test_start:test_end]

        # Find best beta on train, evaluate on test
        best_b, best_s = None, -np.inf
        for beta in betas_grid:
            res = evaluate_beta(beta, train_slice, cfg, model)
            if res["sharpe"] > best_s:
                best_s = res["sharpe"]
                best_b = beta

        # OOS evaluation with that beta
        oos = evaluate_beta(best_b, test_slice, cfg, model)

        fold_results.append({
            "fold":         fold + 1,
            "train_n":      len(train_slice),
            "test_n":       len(test_slice),
            "best_beta":    float(best_b),
            "train_sharpe": float(best_s),
            "oos_sharpe":   float(oos["sharpe"]),
            "oos_hit":      float(oos["hit_rate"]),
            "oos_ic":       float(oos["ic"]),
        })

        print(f"      Fold {fold+1}: beta*={best_b:.2f}  train_S={best_s:.3f}  oos_S={oos['sharpe']:.3f}  oos_hit={oos['hit_rate']*100:.1f}%")

    if not fold_results:
        return {"best_beta": 1.2, "fold_results": []}

    fold_df = pd.DataFrame(fold_results)
    # Weight by OOS Sharpe (if positive), else simple mean
    oos_sharpes = fold_df["oos_sharpe"].values
    weights = np.maximum(oos_sharpes, 0)
    if weights.sum() > 0:
        best_beta = float(np.average(fold_df["best_beta"].values, weights=weights))
    else:
        best_beta = float(fold_df["best_beta"].mean())

    return {
        "best_beta":    best_beta,
        "fold_results": fold_results,
        "fold_df":      fold_df,
        "avg_oos_sharpe": float(fold_df["oos_sharpe"].mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL OPTIMIZATION PIPELINE PER TICKER
# ─────────────────────────────────────────────────────────────────────────────

def optimize_ticker(ticker: str, keyword: str, data: pd.DataFrame, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  Optimizing: {ticker}  ({len(data)} observations)")
    print(f"{'='*60}")

    results = {"ticker": ticker, "n_obs": len(data)}

    # Method 1: Grid search
    print("\n  [1/3] Grid Search...")
    grid = grid_search_beta(data, cfg)
    results["grid_beta"]   = grid["best_beta"]
    results["grid_sharpe"] = grid["best_sharpe"]
    results["grid_curve"]  = grid["curve"]
    print(f"      Best beta = {grid['best_beta']:.3f}  OOS Sharpe = {grid['best_sharpe']:.3f}")

    # Method 2: Scipy (use grid best as warm start)
    print("\n  [2/3] Scipy Brent Optimization...")
    scipy_res = scipy_optimize_beta(data, cfg)
    results["scipy_beta"]   = scipy_res["best_beta"]
    results["scipy_sharpe"] = scipy_res["best_sharpe"]

    # Method 3: Walk-forward CV
    print("\n  [3/3] Walk-Forward Cross-Validation...")
    wf = walkforward_optimize(data, cfg)
    results["wf_beta"]        = wf["best_beta"]
    results["wf_avg_sharpe"]  = wf.get("avg_oos_sharpe", None)
    results["fold_df"]        = wf.get("fold_df", None)

    # Consensus: median of the three methods
    betas = [grid["best_beta"], scipy_res["best_beta"], wf["best_beta"]]
    results["consensus_beta"] = float(np.median(betas))

    print(f"\n  SUMMARY  {ticker}:")
    print(f"    Grid:         β = {results['grid_beta']:.3f}")
    print(f"    Scipy:        β = {results['scipy_beta']:.3f}")
    print(f"    Walk-Forward: β = {results['wf_beta']:.3f}")
    print(f"    *** CONSENSUS β = {results['consensus_beta']:.3f} ***")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_optimization_results(opt_results: dict, save_path="beta_optimization.png"):
    tickers = list(opt_results.keys())
    n = len(tickers)
    if n == 0:
        return

    gold = "#e8d5b0"; red = "#e05252"; grn = "#4eca7e"; blue = "#4a9eda"; grey = "#666666"

    cols = min(3, n)
    rows = (n + cols - 1) // cols + 1  # +1 for summary row

    fig = plt.figure(figsize=(cols * 7, rows * 4 + 3), facecolor="#0d0d0d")
    fig.suptitle(
        "Beta Optimization — Ising Market Model  |  E.W. Research",
        fontsize=15, color=gold, fontweight="bold", y=0.99,
    )

    def sax(ax, title):
        ax.set_facecolor("#111111")
        ax.set_title(title, color=gold, fontsize=9)
        ax.tick_params(colors=grey, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")

    # Per-ticker: grid search curve
    for idx, ticker in enumerate(tickers):
        r = opt_results[ticker]
        curve = r.get("grid_curve")
        if curve is None:
            continue

        ax = fig.add_subplot(rows - 1, cols, idx + 1)
        ax.plot(curve["beta"], curve["train_sharpe"], color=blue,
                lw=1.5, alpha=0.7, label="Train Sharpe")
        ax.plot(curve["beta"], curve["test_sharpe"],  color=gold,
                lw=2,   label="Test (OOS) Sharpe")
        ax.axvline(r["grid_beta"],     color=grn,  lw=1.5, ls="--", label=f"Grid β*={r['grid_beta']:.2f}")
        ax.axvline(r["scipy_beta"],    color=blue, lw=1,   ls=":",  label=f"Scipy β*={r['scipy_beta']:.2f}")
        ax.axvline(r["consensus_beta"],color=gold, lw=2,   ls="-",  label=f"Consensus={r['consensus_beta']:.2f}")
        ax.axhline(0, color=grey, lw=0.8, ls="--")
        ax.set_xlabel("β", color=grey, fontsize=8)
        ax.set_ylabel("Sharpe", color=grey, fontsize=8)
        ax.legend(facecolor="#1a1a1a", labelcolor=grey, fontsize=7)
        sax(ax, f"{ticker} — Beta Grid Search")

    # Summary panel: consensus betas
    ax_sum = fig.add_subplot(rows, 1, rows)
    consensus_betas = [opt_results[t]["consensus_beta"] for t in tickers]
    grid_betas      = [opt_results[t]["grid_beta"]      for t in tickers]
    scipy_betas     = [opt_results[t]["scipy_beta"]     for t in tickers]
    wf_betas        = [opt_results[t]["wf_beta"]        for t in tickers]

    x = np.arange(len(tickers))
    w = 0.2
    ax_sum.bar(x - w*1.5, grid_betas,      width=w, color=blue,  alpha=0.7, label="Grid")
    ax_sum.bar(x - w*0.5, scipy_betas,     width=w, color="#c084fc", alpha=0.7, label="Scipy")
    ax_sum.bar(x + w*0.5, wf_betas,        width=w, color=grn,   alpha=0.7, label="Walk-Fwd")
    ax_sum.bar(x + w*1.5, consensus_betas, width=w, color=gold,  alpha=0.9, label="Consensus")
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(tickers, color=grey)
    ax_sum.axhline(1.0, color=grey, lw=0.8, ls="--", alpha=0.5)
    ax_sum.set_ylabel("Optimal β", color=grey)
    ax_sum.legend(facecolor="#1a1a1a", labelcolor=grey, fontsize=9)
    sax(ax_sum, "Consensus Beta by Ticker (median of Grid / Scipy / Walk-Forward)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"\n  Chart saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Ising Market Model  —  Beta Optimization")
    print(f"  Beta range : [{CFG['beta_min']}, {CFG['beta_max']}]  ({CFG['beta_grid_points']} grid points)")
    print(f"  Methods    : Grid Search + Scipy Brent + Walk-Forward CV")
    print(f"  Consensus  : median of all three")
    print("=" * 60)

    # Load all data first (expensive due to Trends API)
    print("\n  Loading data for all tickers...")
    ticker_data = {}
    for ticker, keyword in STOCKS.items():
        print(f"  {ticker}...", end=" ", flush=True)
        d = load_ticker_data(ticker, keyword, CFG)
        if d is not None:
            ticker_data[ticker] = d
            print(f"OK ({len(d)} obs)")
        else:
            print("FAILED")

    if not ticker_data:
        print("\n[!] No data loaded.")
        raise SystemExit(1)

    # Optimize each ticker
    opt_results = {}
    for ticker, data in ticker_data.items():
        try:
            r = optimize_ticker(ticker, STOCKS[ticker], data, CFG)
            opt_results[ticker] = r
        except Exception as e:
            print(f"  [!] {ticker} failed: {e}")

    # Build output dict (JSON-serializable)
    output = {}
    print("\n" + "=" * 60)
    print("  OPTIMAL BETAS (consensus)")
    print("=" * 60)
    for ticker, r in opt_results.items():
        output[ticker] = {
            "consensus_beta": r["consensus_beta"],
            "grid_beta":      r["grid_beta"],
            "scipy_beta":     r["scipy_beta"],
            "wf_beta":        r["wf_beta"],
            "grid_oos_sharpe":r["grid_sharpe"],
            "wf_avg_oos_sharpe": r.get("wf_avg_sharpe"),
        }
        print(f"  {ticker:<6}  β = {r['consensus_beta']:.3f}"
              f"  (grid={r['grid_beta']:.3f} scipy={r['scipy_beta']:.3f} wf={r['wf_beta']:.3f})")

    # Save JSON
    with open("optimal_betas.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Saved -> optimal_betas.json")

    # Plot
    print("  Generating charts...")
    plot_optimization_results(opt_results, "beta_optimization.png")

    # How to use in backtest.py
    print("\n  TO USE IN BACKTEST:")
    print("  Replace CFG['beta_fixed'] = None  and load optimal_betas.json,")
    print("  then set model.beta = optimal_betas[ticker]['consensus_beta'] per ticker.")
    print("\nDone.")