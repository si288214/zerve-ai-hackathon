"""
event_backtest.py — Binary Event Prediction via Ising Social Contagion
E.W. Research / Zerve Hackathon  (v2 — multi-source h)
=======================================================================

Predicts binary outcomes (earnings beat/miss, CPI above/below, Fed hike/hold,
drug trial approval, merger close) using a fused sentiment field h(t):

  h = w_news * h_news  +  w_options * h_options
    + w_social * h_social  +  w_llm * h_llm

Weights are event-type specific (see sentiment_engine.EVENT_WEIGHTS).
All sources degrade gracefully if keys / data are unavailable.

Usage:
    pip install requests transformers torch anthropic
    python event_backtest.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
import requests
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings("ignore")

from ising_model import IsingInvestorNetwork, magnetization_to_signal
from sentiment_engine import compute_h
from event_types import (
    H_PRESETS, get_all_events, get_lookback,
    DRUG_TRIAL_EVENTS, MERGER_EVENTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# LEGACY EVENTS (earnings + macro)
# ─────────────────────────────────────────────────────────────────────────────

EVENTS = [
    # ── EARNINGS ──────────────────────────────────────────────────────────────
    {"date": "2024-02-21", "type": "earnings", "ticker": "NVDA", "outcome": 1, "desc": "NVDA Q4 2024 beat"},
    {"date": "2024-05-22", "type": "earnings", "ticker": "NVDA", "outcome": 1, "desc": "NVDA Q1 2025 beat"},
    {"date": "2024-08-28", "type": "earnings", "ticker": "NVDA", "outcome": 1, "desc": "NVDA Q2 2025 beat"},
    {"date": "2024-11-20", "type": "earnings", "ticker": "NVDA", "outcome": 1, "desc": "NVDA Q3 2025 beat"},
    {"date": "2024-01-25", "type": "earnings", "ticker": "TSLA", "outcome": 0, "desc": "TSLA Q4 2023 miss"},
    {"date": "2024-04-23", "type": "earnings", "ticker": "TSLA", "outcome": 0, "desc": "TSLA Q1 2024 miss"},
    {"date": "2024-07-23", "type": "earnings", "ticker": "TSLA", "outcome": 1, "desc": "TSLA Q2 2024 beat"},
    {"date": "2024-10-23", "type": "earnings", "ticker": "TSLA", "outcome": 1, "desc": "TSLA Q3 2024 beat"},
    {"date": "2024-02-01", "type": "earnings", "ticker": "META", "outcome": 1, "desc": "META Q4 2023 beat"},
    {"date": "2024-04-24", "type": "earnings", "ticker": "META", "outcome": 0, "desc": "META Q1 2024 miss"},
    {"date": "2024-07-31", "type": "earnings", "ticker": "META", "outcome": 1, "desc": "META Q2 2024 beat"},
    {"date": "2024-10-30", "type": "earnings", "ticker": "META", "outcome": 1, "desc": "META Q3 2024 beat"},
    {"date": "2024-02-01", "type": "earnings", "ticker": "AAPL", "outcome": 1, "desc": "AAPL Q1 2024 beat"},
    {"date": "2024-05-02", "type": "earnings", "ticker": "AAPL", "outcome": 1, "desc": "AAPL Q2 2024 beat"},
    {"date": "2024-08-01", "type": "earnings", "ticker": "AAPL", "outcome": 1, "desc": "AAPL Q3 2024 beat"},
    {"date": "2024-10-31", "type": "earnings", "ticker": "AAPL", "outcome": 1, "desc": "AAPL Q4 2024 beat"},
    {"date": "2024-01-30", "type": "earnings", "ticker": "MSFT", "outcome": 1, "desc": "MSFT Q2 FY24 beat"},
    {"date": "2024-04-25", "type": "earnings", "ticker": "MSFT", "outcome": 1, "desc": "MSFT Q3 FY24 beat"},
    {"date": "2024-07-30", "type": "earnings", "ticker": "MSFT", "outcome": 1, "desc": "MSFT Q4 FY24 beat"},
    {"date": "2024-10-30", "type": "earnings", "ticker": "MSFT", "outcome": 0, "desc": "MSFT Q1 FY25 miss"},
    # ── FED ───────────────────────────────────────────────────────────────────
    {"date": "2024-01-31", "type": "fed", "ticker": "SPY", "outcome": 0, "desc": "Fed hold Jan 2024"},
    {"date": "2024-03-20", "type": "fed", "ticker": "SPY", "outcome": 0, "desc": "Fed hold Mar 2024"},
    {"date": "2024-05-01", "type": "fed", "ticker": "SPY", "outcome": 0, "desc": "Fed hold May 2024"},
    {"date": "2024-06-12", "type": "fed", "ticker": "SPY", "outcome": 0, "desc": "Fed hold Jun 2024"},
    {"date": "2024-07-31", "type": "fed", "ticker": "SPY", "outcome": 0, "desc": "Fed hold Jul 2024"},
    {"date": "2024-09-18", "type": "fed", "ticker": "SPY", "outcome": 1, "desc": "Fed cut 50bps Sep 2024"},
    {"date": "2024-11-07", "type": "fed", "ticker": "SPY", "outcome": 1, "desc": "Fed cut 25bps Nov 2024"},
    {"date": "2024-12-18", "type": "fed", "ticker": "SPY", "outcome": 1, "desc": "Fed cut 25bps Dec 2024"},
    # ── CPI ───────────────────────────────────────────────────────────────────
    {"date": "2024-01-11", "type": "cpi", "ticker": "SPY", "outcome": 0, "desc": "CPI above est Jan 2024"},
    {"date": "2024-02-13", "type": "cpi", "ticker": "SPY", "outcome": 0, "desc": "CPI above est Feb 2024"},
    {"date": "2024-03-12", "type": "cpi", "ticker": "SPY", "outcome": 0, "desc": "CPI above est Mar 2024"},
    {"date": "2024-04-10", "type": "cpi", "ticker": "SPY", "outcome": 0, "desc": "CPI above est Apr 2024"},
    {"date": "2024-05-15", "type": "cpi", "ticker": "SPY", "outcome": 1, "desc": "CPI below est May 2024"},
    {"date": "2024-06-12", "type": "cpi", "ticker": "SPY", "outcome": 1, "desc": "CPI below est Jun 2024"},
    {"date": "2024-07-11", "type": "cpi", "ticker": "SPY", "outcome": 1, "desc": "CPI below est Jul 2024"},
    {"date": "2024-08-14", "type": "cpi", "ticker": "SPY", "outcome": 1, "desc": "CPI below est Aug 2024"},
    {"date": "2024-09-11", "type": "cpi", "ticker": "SPY", "outcome": 1, "desc": "CPI below est Sep 2024"},
    {"date": "2024-10-10", "type": "cpi", "ticker": "SPY", "outcome": 0, "desc": "CPI above est Oct 2024"},
    {"date": "2024-11-13", "type": "cpi", "ticker": "SPY", "outcome": 0, "desc": "CPI above est Nov 2024"},
    {"date": "2024-12-11", "type": "cpi", "ticker": "SPY", "outcome": 0, "desc": "CPI above est Dec 2024"},
]

# Combine all event types
ALL_EVENTS = EVENTS + DRUG_TRIAL_EVENTS + MERGER_EVENTS

CFG = {
    "n_investors":   120,
    "D":             0.25,
    "beta":          0.4,
    "n_equil":       150,
    "n_samples":     40,
    "sample_every":  3,
    "seed":          42,
}


# ─────────────────────────────────────────────────────────────────────────────
# ISING PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def ising_predict(h: float, cfg: dict) -> dict:
    model = IsingInvestorNetwork(
        n_investors=cfg["n_investors"],
        J=1.0,
        beta=cfg["beta"],
        D=cfg["D"],
        seed=cfg["seed"],
    )
    model.set_sentiment_field(h)
    mags = model.run(
        n_equil=cfg["n_equil"],
        n_samples=cfg["n_samples"],
        sample_every=cfg["sample_every"],
    )
    mag_mean = float(np.mean(mags))
    mag_std  = float(np.std(mags))
    prob     = (mag_mean + 1) / 2

    return {
        "magnetization": round(mag_mean, 4),
        "mag_std":       round(mag_std, 4),
        "probability":   round(prob, 4),
        "prediction":    1 if prob >= 0.5 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    valid = [r for r in results if r.get("probability") is not None]
    if len(valid) < 3:
        return {"error": "insufficient_data"}

    outcomes = np.array([r["outcome"]     for r in valid])
    probs    = np.array([r["probability"] for r in valid])
    preds    = np.array([r["prediction"]  for r in valid])

    accuracy     = float((preds == outcomes).mean())
    brier        = float(np.mean((probs - outcomes) ** 2))
    eps          = 1e-7
    log_loss     = float(-np.mean(
        outcomes * np.log(probs + eps) + (1 - outcomes) * np.log(1 - probs + eps)
    ))
    baseline_acc = float((outcomes == round(outcomes.mean())).mean())

    bins     = np.linspace(0, 1, 6)
    cal_data = []
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            cal_data.append({
                "mean_prob":    float(probs[mask].mean()),
                "mean_outcome": float(outcomes[mask].mean()),
                "n":            int(mask.sum()),
            })

    by_type = {}
    for r in valid:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        if r["prediction"] == r["outcome"]:
            by_type[t]["correct"] += 1
    for t in by_type:
        by_type[t]["accuracy"] = round(by_type[t]["correct"] / by_type[t]["total"], 3)

    # Source breakdown
    source_counts = {"fused": 0, "preset": 0}
    for r in valid:
        src = r.get("h_source", "preset")
        source_counts[src] = source_counts.get(src, 0) + 1

    # Per-source-component accuracy
    fused_results = [r for r in valid if r.get("h_source") == "fused"]

    return {
        "n":              len(valid),
        "source_counts":  source_counts,
        "accuracy":       round(accuracy, 4),
        "baseline_acc":   round(baseline_acc, 4),
        "brier_score":    round(brier, 4),
        "log_loss":       round(log_loss, 4),
        "by_type":        by_type,
        "cal_data":       cal_data,
        "fused_accuracy": round(float((
            np.array([r["prediction"] for r in fused_results]) ==
            np.array([r["outcome"]    for r in fused_results])
        ).mean()), 4) if fused_results else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "earnings":   "#4a9eda",
    "fed":        "#e8d5b0",
    "cpi":        "#c49ae8",
    "drug_trial": "#4eca7e",
    "merger":     "#f0a030",
}

def plot_results(results, metrics, save_path="event_backtest_results.png"):
    valid = [r for r in results if r.get("probability") is not None]
    if not valid:
        return

    gold = "#e8d5b0"; red = "#e05252"; grn = "#4eca7e"; grey = "#666666"; blue = "#4a9eda"

    fig = plt.figure(figsize=(20, 16), facecolor="#0d0d0d")
    fig.suptitle(
        "Ising Event Prediction — Earnings / Macro / Drug Trials / M&A  |  E.W. Research (v2)",
        fontsize=15, color=gold, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

    def sax(ax, title):
        ax.set_facecolor("#111111")
        ax.set_title(title, color=gold, fontsize=9)
        ax.tick_params(colors=grey)
        for sp in ax.spines.values():
            sp.set_color("#333333")

    # ── Panel 1: prob vs actual ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    probs    = [r["probability"] for r in valid]
    outcomes = [r["outcome"]     for r in valid]
    ev_colors = [TYPE_COLORS.get(r["type"], grey) for r in valid]
    bar_alpha = [0.85 if r["prediction"] == r["outcome"] else 0.4 for r in valid]

    for i, (p, c, a) in enumerate(zip(probs, ev_colors, bar_alpha)):
        ax1.bar(i, p, color=c, alpha=a, edgecolor="#222")

    ax1.scatter(range(len(valid)), outcomes, color=gold, s=30, zorder=5, label="Actual")
    ax1.axhline(0.5, color=grey, lw=1, ls="--", label="0.5 threshold")

    # Legend for event types
    for etype, ec in TYPE_COLORS.items():
        ax1.bar(0, 0, color=ec, alpha=0.85, label=etype)

    ax1.set_xticks(range(len(valid)))
    ax1.set_xticklabels([r["desc"][:16] for r in valid], rotation=45,
                        ha="right", fontsize=6, color=grey)
    ax1.set_ylabel("P(positive outcome)", color=grey)
    ax1.legend(facecolor="#1a1a1a", labelcolor=gold, fontsize=7, ncol=3)
    sax(ax1, f"Predicted Probability vs Actual  (bright=correct, dim=wrong)  "
             f"Acc={metrics['accuracy']*100:.1f}%")

    # ── Panel 2: calibration ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    cal = metrics.get("cal_data", [])
    if len(cal) > 1:
        cx = [c["mean_prob"]    for c in cal]
        cy = [c["mean_outcome"] for c in cal]
        ax2.plot([0, 1], [0, 1], color=grey, lw=1, ls="--", label="Perfect")
        ax2.plot(cx, cy, color=gold, lw=2, marker="o", markersize=6, label="Model")
        ax2.fill_between(cx, cy, cx, alpha=0.15, color=gold)
    ax2.set_xlabel("Mean predicted prob", color=grey)
    ax2.set_ylabel("Mean actual outcome", color=grey)
    ax2.legend(facecolor="#1a1a1a", labelcolor=gold, fontsize=8)
    sax(ax2, "Calibration Curve")

    # ── Panel 3: accuracy by type ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    types  = list(metrics["by_type"].keys())
    accs   = [metrics["by_type"][t]["accuracy"] * 100 for t in types]
    ns     = [metrics["by_type"][t]["total"]           for t in types]
    c3     = [TYPE_COLORS.get(t, grey) for t in types]
    bars   = ax3.barh(types, accs, color=c3, edgecolor="#222", alpha=0.8)
    ax3.axvline(50, color=grey, lw=1, ls="--")
    ax3.axvline(metrics["baseline_acc"] * 100, color=blue, lw=1, ls=":",
                label=f"Baseline {metrics['baseline_acc']*100:.0f}%")
    for bar, n in zip(bars, ns):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"n={n}", va="center", color=grey, fontsize=8)
    ax3.set_xlabel("Accuracy %", color=grey)
    ax3.legend(facecolor="#1a1a1a", labelcolor=grey, fontsize=8)
    sax(ax3, "Accuracy by Event Type")

    # ── Panel 4: h source breakdown ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    # Show per-source raw h distributions for fused events
    fused = [r for r in valid if r.get("h_source") == "fused" and r.get("h_breakdown")]
    if fused:
        src_names = ["news", "options", "social", "llm"]
        src_vals  = {s: [] for s in src_names}
        for r in fused:
            raw = r["h_breakdown"].get("source_raw", {})
            for s in src_names:
                v = raw.get(s)
                if v is not None:
                    src_vals[s].append(v)

        positions = range(len(src_names))
        bps = ax4.boxplot(
            [src_vals[s] for s in src_names],
            positions=list(positions),
            widths=0.5,
            patch_artist=True,
        )
        colors4 = [blue, "#f0a030", grn, "#c49ae8"]
        for patch, c in zip(bps["boxes"], colors4):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "medians", "fliers"]:
            for line in bps[element]:
                line.set_color(grey)
        ax4.set_xticks(list(positions))
        ax4.set_xticklabels(src_names, color=grey)
        ax4.axhline(0, color=grey, lw=0.8, ls="--")
        ax4.set_ylabel("h value", color=grey)
    else:
        ax4.text(0.5, 0.5, "No fused events\n(all preset)", ha="center",
                 va="center", color=grey, transform=ax4.transAxes)
    sax(ax4, "h Distribution by Source Component")

    # ── Panel 5: h vs actual outcome (scatter by type) ───────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    for etype, ec in TYPE_COLORS.items():
        sub = [r for r in valid if r["type"] == etype]
        if sub:
            hs  = [r["h"]      for r in sub]
            ocs = [r["outcome"] for r in sub]
            ax5.scatter(hs, ocs, color=ec, alpha=0.7, s=30, label=etype)
    ax5.axvline(0, color=grey, lw=0.8, ls="--")
    ax5.set_xlabel("Sentiment h", color=grey)
    ax5.set_ylabel("Outcome", color=grey)
    ax5.set_yticks([0, 1])
    ax5.legend(facecolor="#1a1a1a", labelcolor=grey, fontsize=7)
    sax(ax5, "h vs Outcome by Event Type")

    # ── Panel 6: h vs M scatter ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    hs   = [r["h"]             for r in valid]
    mags = [r["magnetization"] for r in valid]
    cols = [TYPE_COLORS.get(r["type"], grey) for r in valid]
    ax6.scatter(hs, mags, c=cols, alpha=0.7, s=40)
    if len(set(hs)) > 1:
        slope, intercept, r_val, p_val, _ = stats.linregress(hs, mags)
        xs = np.linspace(min(hs), max(hs), 100)
        ax6.plot(xs, slope * xs + intercept, color=gold, lw=2,
                 label=f"OLS r={r_val:.3f} p={p_val:.3f}")
        ax6.legend(facecolor="#1a1a1a", labelcolor=gold, fontsize=9)
    ax6.axhline(0, color=grey, lw=0.8, ls="--")
    ax6.axvline(0, color=grey, lw=0.8, ls="--")
    ax6.set_xlabel("Fused Sentiment h", color=grey)
    ax6.set_ylabel("Magnetization M", color=grey)
    sax(ax6, "h vs Magnetization  (colour = event type)")

    # ── Panel 7: summary table ───────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    fused_acc = metrics.get("fused_accuracy")
    summary = [
        ["Metric",        "Value"],
        ["Total events",  str(metrics["n"])],
        ["Fused h",       str(metrics["source_counts"].get("fused", 0))],
        ["Preset h",      str(metrics["source_counts"].get("preset", 0))],
        ["Accuracy",      f"{metrics['accuracy']*100:.1f}%"],
        ["Baseline",      f"{metrics['baseline_acc']*100:.1f}%"],
        ["Lift",          f"+{(metrics['accuracy']-metrics['baseline_acc'])*100:.1f}%"],
        ["Brier",         f"{metrics['brier_score']:.3f}"],
        ["Log Loss",      f"{metrics['log_loss']:.3f}"],
        ["Fused-only Acc",f"{fused_acc*100:.1f}%" if fused_acc else "N/A"],
    ]
    tbl = ax7.table(cellText=summary[1:], colLabels=summary[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1a1a" if r > 0 else "#2a2a2a")
        cell.set_text_props(color=gold if r == 0 else grey)
        cell.set_edgecolor("#333333")
    sax(ax7, "Performance Summary")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"  Chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Ising Event Prediction v2 — E.W. Research")
    print(f"  Events  : {len(ALL_EVENTS)}")
    print(f"  h source: fused (news + options + social + llm)  |  fallback: preset")
    print(f"  Beta    : {CFG['beta']}")
    print("=" * 65)

    results = []

    for i, event in enumerate(ALL_EVENTS):
        print(f"\n[{i+1}/{len(ALL_EVENTS)}] {event['desc']}  ({event['date']})")

        lookback = get_lookback(event["type"])

        # Use preset h for historical events (live APIs don't support >1 month lookback)
        # Only attempt live fusion if no preset exists (i.e. a new/future event)
        preset_h = H_PRESETS.get(event["desc"])

        if preset_h is not None:
            h        = preset_h
            h_source = "preset"
            breakdown = {}
            print(f"  Using preset h={h:+.3f}")
        else:
            # Try multi-source fusion for new events not in presets
            try:
                h, breakdown = compute_h(event, lookback_days=lookback, av_sleep=0.5)
                all_failed = not breakdown.get("sources_used")
                h_source = "error" if all_failed else "fused"
            except Exception as e:
                print(f"  [!] compute_h failed: {e}")
                h_source = "error"
                breakdown = {}

            if h_source == "error":
                h        = 0.0
                breakdown = {}
                print(f"  [!] all sources failed, h=0.0")

        # Ising prediction
        pred    = ising_predict(h, CFG)
        correct = pred["prediction"] == event["outcome"]
        print(f"  h={h:+.3f}  M={pred['magnetization']:+.4f}  "
              f"P={pred['probability']:.3f}  "
              f"pred={pred['prediction']}  actual={event['outcome']}  "
              f"{'✓ CORRECT' if correct else '✗ WRONG'}")

        results.append({
            **event,
            "h":            h,
            "h_source":     h_source,
            "h_breakdown":  breakdown,
            **pred,
        })

    # Metrics
    metrics = compute_metrics(results)

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Total events  : {metrics['n']}")
    sc = metrics.get("source_counts", {})
    print(f"  Fused h       : {sc.get('fused', 0)}  |  Preset: {sc.get('preset', 0)}")
    print(f"  Accuracy      : {metrics['accuracy']*100:.1f}%  "
          f"(baseline: {metrics['baseline_acc']*100:.1f}%)")
    print(f"  Lift          : +{(metrics['accuracy']-metrics['baseline_acc'])*100:.1f}%")
    print(f"  Brier Score   : {metrics['brier_score']:.3f}")
    print(f"  Log Loss      : {metrics['log_loss']:.3f}")
    if metrics.get("fused_accuracy"):
        print(f"  Fused-only Acc: {metrics['fused_accuracy']*100:.1f}%")
    print(f"\n  By event type:")
    for t, v in metrics["by_type"].items():
        print(f"    {t:<14}  {v['accuracy']*100:.1f}%  (n={v['total']})")

    # Save results
    df = pd.DataFrame(results)
    # Flatten breakdown for CSV
    df["h_news"]    = df["h_breakdown"].apply(
        lambda x: x.get("source_raw", {}).get("news")    if isinstance(x, dict) else None)
    df["h_options"] = df["h_breakdown"].apply(
        lambda x: x.get("source_raw", {}).get("options") if isinstance(x, dict) else None)
    df["h_social"]  = df["h_breakdown"].apply(
        lambda x: x.get("source_raw", {}).get("social")  if isinstance(x, dict) else None)
    df["h_llm"]     = df["h_breakdown"].apply(
        lambda x: x.get("source_raw", {}).get("llm")     if isinstance(x, dict) else None)
    df.drop(columns=["h_breakdown"], inplace=True)

    df.to_csv("event_predictions.csv", index=False)
    print("\n  Saved → event_predictions.csv")
    plot_results(results, metrics)
    print("\nDone.")
