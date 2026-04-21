"""
app.py — Ising Social Contagion Stock Predictor
================================================
Streamlit application that combines:

  1. Social sentiment data  (Google Trends → external field h)
  2. Blume-Capel Ising model on a scale-free investor network
  3. Monte Carlo β-calibration against historical returns
  4. Prediction signal + confidence score for each stock

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings, time
warnings.filterwarnings("ignore")

from ising_model import (
    IsingInvestorNetwork,
    calibrate_beta,
    magnetization_to_signal,
)
from data_fetcher import get_stock_and_sentiment

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ising Stock Predictor",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark theme overrides */
  [data-testid="stAppViewContainer"] { background: #0f172a; }
  [data-testid="stSidebar"] { background: #1e293b; }
  .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

  /* Prediction cards */
  .pred-card {
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .card-bullish { background: linear-gradient(135deg,#052e16,#14532d); border-color:#22c55e44; }
  .card-bearish { background: linear-gradient(135deg,#2d0a0a,#450a0a); border-color:#ef444444; }
  .card-neutral { background: linear-gradient(135deg,#0f172a,#1e293b); border-color:#64748b44; }

  .card-ticker  { font-size: 1.4rem; font-weight: 800; letter-spacing: 1px; }
  .card-signal  { font-size: 1.0rem; font-weight: 700; padding: 3px 12px;
                  border-radius: 20px; display: inline-block; margin-top: 4px; }
  .sig-bullish  { background:#166534; color:#bbf7d0; }
  .sig-bearish  { background:#7f1d1d; color:#fecaca; }
  .sig-neutral  { background:#334155; color:#cbd5e1; }
  .card-meta    { font-size: 0.78rem; color: #94a3b8; margin-top: 6px; }

  /* Section headers */
  .section-header {
    font-size: 1.1rem; font-weight: 700;
    color: #e2e8f0; margin: 20px 0 8px;
    border-left: 3px solid #6366f1; padding-left: 10px;
  }

  /* Metric override */
  [data-testid="metric-container"] { background: #1e293b; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

DEFAULT_STOCKS = {
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",
    "AAPL":  "Apple",
    "META":  "Meta",
    "AMZN":  "Amazon",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMD":   "AMD",
}

SIGNAL_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
SIGNAL_COLOR = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "NEUTRAL": "#64748b"}


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧲 Ising Stock Predictor")
    st.markdown("*Social contagion model for market prediction*")
    st.divider()

    st.markdown("### 📋 Stock Selection")
    selected_tickers = st.multiselect(
        "Choose stocks to analyse",
        options=list(DEFAULT_STOCKS.keys()),
        default=["NVDA", "TSLA", "AAPL", "META"],
        format_func=lambda t: f"{t} — {DEFAULT_STOCKS.get(t, t)}",
    )
    custom_ticker = st.text_input(
        "Or add a custom ticker (e.g. PLTR)",
        placeholder="TICKER",
    ).upper().strip()
    if custom_ticker and custom_ticker not in selected_tickers:
        selected_tickers.append(custom_ticker)

    st.divider()
    st.markdown("### ⚙️ Model Parameters")

    n_investors = st.slider(
        "Investor network size  (N)",
        min_value=50, max_value=300, value=120, step=10,
        help="Number of simulated investors. Larger = more accurate, slower.",
    )
    D_anisotropy = st.slider(
        "Anisotropy  D  (action bias)",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Energy cost of being non-neutral. Higher → investors prefer clear positions.",
    )
    n_beta_points = st.select_slider(
        "β calibration grid points",
        options=[6, 8, 10, 12, 15],
        value=8,
        help="More points → better calibration, longer runtime.",
    )
    use_trends = st.checkbox("Use Google Trends sentiment", value=True,
        help="Uncheck to derive sentiment from price momentum only (faster, no API).")

    st.divider()
    run_btn = st.button("🚀  Run Analysis", type="primary", use_container_width=True)
    st.caption(f"Estimated time: ~{len(selected_tickers) * 30}–{len(selected_tickers) * 60}s")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown("# 🧲 Ising Social Contagion Stock Predictor")
    st.markdown(
        "Uses a **Blume-Capel Ising model** on a scale-free investor network "
        "to simulate how sentiment spreads through social media (Google Trends, "
        "X / Instagram) and predict stock direction via Monte Carlo simulation."
    )
with col_t2:
    st.markdown(f"<div style='text-align:right;color:#64748b;padding-top:30px;'>"
                f"Last run: {datetime.now().strftime('%H:%M  %d %b %Y')}</div>",
                unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────
# PHYSICS EXPLAINER (collapsible)
# ─────────────────────────────────────────────────────────────

with st.expander("📐 Model physics — click to expand", expanded=False):
    st.markdown(r"""
**Hamiltonian:**

$$H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j \;-\; h \sum_i \sigma_i \;+\; D \sum_i \sigma_i^2$$

| Symbol | Meaning | Source |
|--------|---------|--------|
| $\sigma_i \in \{-1, 0, +1\}$ | Investor spin (sell / hold / buy) | Model state |
| $J$ | Coupling strength (peer influence) | Fixed = 1 |
| $\beta$ | Inverse temperature (noise level) | **Calibrated via Monte Carlo** |
| $h$ | External field (social sentiment) | Google Trends / X / Instagram |
| $D$ | Anisotropy (bias toward action) | User-configurable |

**Monte Carlo calibration:** We sweep $\beta \in [0.1, 3.0]$, run the Metropolis algorithm for each value, and pick the $\beta$ that maximises the **Pearson correlation** between model magnetisation $M = \langle\sigma\rangle$ and actual log-returns.

**Signal derivation:**
- $M > +0.12$ → **BULLISH** (herd buying)
- $M < -0.12$ → **BEARISH** (herd selling)
- Otherwise  → **NEUTRAL**

Confidence scales with the signal-to-noise ratio $|M| / \sigma_M$.
    """)


# ─────────────────────────────────────────────────────────────
# MAIN PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=1800)   # cache 30 min
def run_prediction_pipeline(
    ticker: str,
    keyword: str,
    n_investors: int,
    D: float,
    n_beta_points: int,
    use_trends: bool,
) -> dict:
    """Full pipeline: data → calibration → simulation → signal."""

    result = {
        "ticker": ticker,
        "keyword": keyword,
        "prediction": "N/A",
        "confidence": 0.0,
        "magnetization": 0.0,
        "mag_std": 0.0,
        "mag_samples": [],
        "calibrated_beta": 1.0,
        "model_correlation": 0.0,
        "beta_curve": [],
        "spin_counts": {},
        "summary": {},
        "current_h": 0.0,
        "data_source": "unknown",
        "error": None,
    }

    try:
        # ── 1. Fetch data ──────────────────────────────────────────────
        data = get_stock_and_sentiment(
            ticker, keyword=keyword, period="4mo",
            use_trends=use_trends, verbose=False,
        )
        result["summary"] = data["summary"]
        result["current_h"] = data["current_h"]
        result["data_source"] = data["data_source"]

        returns_arr = data["returns"]
        sent_arr = data["sentiment_hist"]

        # ── 2. Beta calibration ────────────────────────────────────────
        beta_range = np.linspace(0.1, 3.0, n_beta_points)
        if len(sent_arr) >= 3 and len(returns_arr) >= 3:
            best_beta, best_corr, curve = calibrate_beta(
                list(sent_arr),
                list(returns_arr),
                beta_range=beta_range,
                n_investors=n_investors,
                n_equil=120,
                n_samples=20,
            )
        else:
            best_beta, best_corr, curve = 1.0, 0.0, []

        result["calibrated_beta"] = round(best_beta, 3)
        result["model_correlation"] = round(best_corr, 3)
        result["beta_curve"] = curve

        # ── 3. Final simulation with calibrated β ──────────────────────
        model = IsingInvestorNetwork(
            n_investors=n_investors, J=1.0, beta=best_beta, D=D,
        )
        model.set_sentiment_field(data["current_h"])
        mag_samples = model.run(n_equil=250, n_samples=60, sample_every=5)

        mag_mean = float(np.mean(mag_samples))
        mag_std = float(np.std(mag_samples))

        result["magnetization"] = round(mag_mean, 4)
        result["mag_std"] = round(mag_std, 4)
        result["mag_samples"] = mag_samples.tolist()
        result["spin_counts"] = model.get_spin_counts()

        # Attach network figure (serialise separately)
        result["_model"] = model   # Plotly fig generated on demand

        # ── 4. Signal ──────────────────────────────────────────────────
        signal, confidence = magnetization_to_signal(mag_mean, mag_std)
        result["prediction"] = signal
        result["confidence"] = confidence

    except Exception as e:
        result["error"] = str(e)
        result["prediction"] = "ERROR"

    return result


# ─────────────────────────────────────────────────────────────
# HELPERS — VISUALISATION
# ─────────────────────────────────────────────────────────────

def prediction_card_html(r: dict) -> str:
    sig = r["prediction"]
    css_class = {"BULLISH": "card-bullish", "BEARISH": "card-bearish"}.get(sig, "card-neutral")
    sig_css = {"BULLISH": "sig-bullish", "BEARISH": "sig-bearish"}.get(sig, "sig-neutral")
    emoji = SIGNAL_EMOJI.get(sig, "❓")
    color = SIGNAL_COLOR.get(sig, "#64748b")

    price_str = (f"${r['summary'].get('current_price', 'N/A')}"
                 if r["summary"] else "N/A")
    chg = r["summary"].get("change_7d_pct", 0) if r["summary"] else 0
    chg_arrow = "▲" if chg >= 0 else "▼"
    chg_color = "#22c55e" if chg >= 0 else "#ef4444"

    src_icons = {"google_trends": "📊 Google Trends", "price_momentum": "📈 Price Momentum",
                 "mock": "🔮 Simulated"}
    src_label = src_icons.get(r["data_source"], r["data_source"])

    return f"""
<div class="pred-card {css_class}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <span class="card-ticker" style="color:{color};">{r['ticker']}</span>
      <br>
      <span class="card-signal {sig_css}">{emoji} {sig}</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;">{price_str}</div>
      <div style="color:{chg_color};font-size:0.85rem;">{chg_arrow} {abs(chg):.1f}% (7d)</div>
    </div>
  </div>
  <div class="card-meta" style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:4px;">
    <span>Confidence: <b style="color:#e2e8f0;">{r['confidence']:.0f}%</b></span>
    <span>Magnetisation M: <b style="color:#e2e8f0;">{r['magnetization']:+.3f}</b></span>
    <span>Calibrated β: <b style="color:#e2e8f0;">{r['calibrated_beta']}</b></span>
    <span>Model ρ: <b style="color:#e2e8f0;">{r['model_correlation']:+.3f}</b></span>
    <span>Sentiment h: <b style="color:#e2e8f0;">{r['current_h']:+.3f}</b></span>
    <span>Data: {src_label}</span>
  </div>
</div>
"""


def make_spin_distribution_fig(results: list[dict]) -> go.Figure:
    tickers = [r["ticker"] for r in results if r["spin_counts"]]
    buys = [r["spin_counts"].get("buy", 0) for r in results if r["spin_counts"]]
    neuts = [r["spin_counts"].get("neutral", 0) for r in results if r["spin_counts"]]
    sells = [r["spin_counts"].get("sell", 0) for r in results if r["spin_counts"]]

    fig = go.Figure()
    totals = [b + n + s for b, n, s in zip(buys, neuts, sells)]
    fig.add_bar(name="Buy (+1)", x=tickers,
                y=[b / t * 100 for b, t in zip(buys, totals)],
                marker_color="#22c55e")
    fig.add_bar(name="Neutral (0)", x=tickers,
                y=[n / t * 100 for n, t in zip(neuts, totals)],
                marker_color="#64748b")
    fig.add_bar(name="Sell (−1)", x=tickers,
                y=[s / t * 100 for s, t in zip(sells, totals)],
                marker_color="#ef4444")

    fig.update_layout(
        barmode="stack",
        title="Spin Distribution per Stock (% of Investor Network)",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="#1e293b"),
        yaxis=dict(title="% Investors", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_magnetization_fig(results: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, r in enumerate(results):
        if not r["mag_samples"]:
            continue
        samples = r["mag_samples"]
        fig.add_trace(go.Scatter(
            y=samples,
            mode="lines",
            name=r["ticker"],
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.85,
        ))

    fig.add_hline(y=0.12, line_dash="dot", line_color="#22c55e",
                  annotation_text="Bullish threshold", annotation_position="top right")
    fig.add_hline(y=-0.12, line_dash="dot", line_color="#ef4444",
                  annotation_text="Bearish threshold", annotation_position="bottom right")
    fig.add_hline(y=0, line_color="#475569", line_width=1)

    fig.update_layout(
        title="Magnetisation M = ⟨σ⟩ over Monte Carlo Samples",
        xaxis_title="MC Sample",
        yaxis_title="Magnetisation M",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="#1e293b"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_beta_calibration_fig(results: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, r in enumerate(results):
        if not r["beta_curve"]:
            continue
        betas = [d["beta"] for d in r["beta_curve"]]
        corrs = [d["correlation"] for d in r["beta_curve"]]
        best_b = r["calibrated_beta"]

        fig.add_trace(go.Scatter(
            x=betas, y=corrs,
            mode="lines+markers",
            name=r["ticker"],
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
        ))
        # Mark calibrated beta
        if best_b in betas:
            idx = betas.index(best_b)
            fig.add_trace(go.Scatter(
                x=[best_b], y=[corrs[idx]],
                mode="markers",
                marker=dict(symbol="star", size=14, color=colors[i % len(colors)],
                            line=dict(width=1, color="white")),
                showlegend=False,
                name=f"{r['ticker']} β*",
            ))

    fig.add_hline(y=0, line_color="#475569", line_width=1)

    fig.update_layout(
        title="β Calibration — Pearson ρ(M, returns) vs Inverse Temperature β",
        xaxis_title="β  (inverse temperature)",
        yaxis_title="Correlation  ρ",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="#1e293b"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def make_market_gauge(results: list[dict]) -> go.Figure:
    valid = [r for r in results if r["prediction"] not in ("N/A", "ERROR")]
    if not valid:
        return go.Figure()

    # Weighted average magnetisation → overall market sentiment
    avg_m = np.mean([r["magnetization"] for r in valid])
    label = "BULLISH" if avg_m > 0.08 else ("BEARISH" if avg_m < -0.08 else "NEUTRAL")
    gauge_color = SIGNAL_COLOR[label]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(avg_m * 100, 1),
        delta={"reference": 0, "suffix": "%"},
        title={"text": f"Overall Market Sentiment<br><span style='font-size:0.8em;color:#94a3b8;'>{label}</span>"},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": gauge_color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [-100, -12], "color": "#450a0a"},
                {"range": [-12,  12], "color": "#1e293b"},
                {"range": [ 12, 100], "color": "#052e16"},
            ],
            "threshold": {
                "line": {"color": "#f8fafc", "width": 3},
                "thickness": 0.8,
                "value": avg_m * 100,
            },
        },
        number={"suffix": "%", "font": {"color": gauge_color, "size": 28}},
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=260,
    )
    return fig


def make_confidence_bar(results: list[dict]) -> go.Figure:
    valid = [r for r in results if r["prediction"] not in ("N/A", "ERROR")]
    tickers = [r["ticker"] for r in valid]
    confs = [r["confidence"] for r in valid]
    colors = [SIGNAL_COLOR[r["prediction"]] for r in valid]

    fig = go.Figure(go.Bar(
        x=tickers, y=confs,
        marker_color=colors,
        text=[f"{c:.0f}%" for c in confs],
        textposition="outside",
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="#475569",
                  annotation_text="Baseline (50%)")
    fig.update_layout(
        title="Prediction Confidence per Stock",
        yaxis=dict(range=[0, 110], gridcolor="#334155", title="Confidence %"),
        xaxis=dict(gridcolor="#334155"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# RUN — either show placeholder or results
# ─────────────────────────────────────────────────────────────

if not run_btn:
    # ── Idle state ─────────────────────────────────────────────────────
    st.info(
        "👈  Configure stocks & parameters in the sidebar, then click **Run Analysis**.\n\n"
        "The model will:\n"
        "1. Fetch stock prices (yfinance) + Google Trends sentiment\n"
        "2. Calibrate the inverse temperature β via Monte Carlo simulation\n"
        "3. Run the Blume-Capel Ising model to convergence\n"
        "4. Output BULLISH / BEARISH / NEUTRAL predictions with confidence scores"
    )

    # Show a demo network
    st.markdown('<div class="section-header">Demo — Investor Network (random spins)</div>',
                unsafe_allow_html=True)
    demo_model = IsingInvestorNetwork(n_investors=80, D=0.25)
    demo_model.reset(seed=99)
    st.plotly_chart(demo_model.get_network_plotly(), use_container_width=True)

else:
    # ── RUN ────────────────────────────────────────────────────────────
    if not selected_tickers:
        st.warning("Please select at least one stock ticker.")
        st.stop()

    all_results = []
    total_stocks = len(selected_tickers)

    # Progress UI
    progress_bar = st.progress(0.0, text="Starting simulation…")
    status_text = st.empty()

    for idx, ticker in enumerate(selected_tickers):
        keyword = DEFAULT_STOCKS.get(ticker, ticker)
        status_text.markdown(
            f"**⏳ Processing {ticker}** ({idx+1}/{total_stocks}) — "
            f"fetching data → calibrating β → running Ising MC…"
        )
        res = run_prediction_pipeline(
            ticker=ticker,
            keyword=keyword,
            n_investors=n_investors,
            D=D_anisotropy,
            n_beta_points=n_beta_points,
            use_trends=use_trends,
        )
        all_results.append(res)
        progress_bar.progress((idx + 1) / total_stocks,
                              text=f"✅ {ticker} done — {res['prediction']}")

    progress_bar.empty()
    status_text.success(f"✅ Analysis complete for {total_stocks} stock(s). "
                        f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # ── OVERVIEW METRICS ──────────────────────────────────────────────
    st.divider()
    valid_results = [r for r in all_results if r["prediction"] not in ("N/A", "ERROR")]
    bullish = sum(1 for r in valid_results if r["prediction"] == "BULLISH")
    bearish = sum(1 for r in valid_results if r["prediction"] == "BEARISH")
    neutral = sum(1 for r in valid_results if r["prediction"] == "NEUTRAL")
    avg_conf = np.mean([r["confidence"] for r in valid_results]) if valid_results else 0

    gauge_col, met_col = st.columns([1.6, 1])
    with gauge_col:
        st.plotly_chart(make_market_gauge(valid_results), use_container_width=True)
    with met_col:
        st.metric("🟢 Bullish", bullish)
        st.metric("🔴 Bearish", bearish)
        st.metric("⚪ Neutral", neutral)
        st.metric("📊 Avg Confidence", f"{avg_conf:.0f}%")

    # ── PREDICTION CARDS ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Stock Predictions</div>',
                unsafe_allow_html=True)

    n_cols = min(3, len(all_results))
    card_cols = st.columns(n_cols)
    for i, r in enumerate(all_results):
        with card_cols[i % n_cols]:
            st.markdown(prediction_card_html(r), unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Ising Network Visualisation",
        "📈 Magnetisation Dynamics",
        "⚙️  β Calibration",
        "📊 Spin & Confidence Breakdown",
    ])

    with tab1:
        st.markdown(
            "Each node is an investor. **Green** = bullish (+1), "
            "**Grey** = neutral (0), **Red** = bearish (−1). "
            "Node size ∝ social influence (degree in scale-free network)."
        )
        ticker_for_net = st.selectbox(
            "Select stock for network view",
            [r["ticker"] for r in valid_results],
            key="net_sel",
        )
        # Re-build model for viz (cached model may have been GC'd)
        sel_r = next((r for r in valid_results if r["ticker"] == ticker_for_net), None)
        if sel_r:
            viz_model = IsingInvestorNetwork(
                n_investors=n_investors, beta=sel_r["calibrated_beta"],
                D=D_anisotropy,
            )
            viz_model.set_sentiment_field(sel_r["current_h"])
            viz_model.run(n_equil=250, n_samples=1, sample_every=1)
            st.plotly_chart(viz_model.get_network_plotly(), use_container_width=True)

    with tab2:
        st.markdown(
            "Magnetisation **M = ⟨σ⟩** sampled after equilibration. "
            "Values above **+0.12** → BULLISH, below **−0.12** → BEARISH."
        )
        st.plotly_chart(make_magnetization_fig(valid_results), use_container_width=True)

    with tab3:
        st.markdown(
            "Pearson correlation between model magnetisation and real returns, "
            "swept over $\\beta$. The ★ star marks the calibrated optimum."
        )
        st.plotly_chart(make_beta_calibration_fig(valid_results), use_container_width=True)

        # Summary table
        calib_df = pd.DataFrame([{
            "Ticker": r["ticker"],
            "Calibrated β": r["calibrated_beta"],
            "Pearson ρ": r["model_correlation"],
            "Sentiment h": round(r["current_h"], 3),
            "Data source": r["data_source"],
        } for r in valid_results])
        st.dataframe(calib_df, use_container_width=True, hide_index=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_spin_distribution_fig(valid_results),
                            use_container_width=True)
        with c2:
            st.plotly_chart(make_confidence_bar(valid_results),
                            use_container_width=True)

    # ── EXPORT ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("📥 Export raw results (JSON)"):
        export_data = [{
            k: v for k, v in r.items() if k != "_model"
        } for r in all_results]
        import json
        st.download_button(
            "Download predictions.json",
            data=json.dumps({
                "generated_at": datetime.now().isoformat(),
                "model_config": {
                    "n_investors": n_investors,
                    "D_anisotropy": D_anisotropy,
                    "n_beta_points": n_beta_points,
                    "use_google_trends": use_trends,
                },
                "predictions": export_data,
            }, indent=2),
            file_name="predictions.json",
            mime="application/json",
        )
