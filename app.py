"""
app.py — Ising Social Contagion Stock Predictor  (Live Edition)
===============================================================
Features:
  • Live auto-refresh with configurable interval
  • Real-time price fetching via yfinance fast_info
  • Kelly Criterion position sizing (buy/sell amount in $ and shares)
  • Blume-Capel Ising model with Monte Carlo beta-calibration
  • Full Streamlit dashboard with network viz and calibration charts

Run:
    streamlit run app.py
"""

import time
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")

from ising_model import (
    IsingInvestorNetwork,
    calibrate_beta,
    magnetization_to_signal,
)
from data_fetcher import get_stock_and_sentiment
from position_sizer import compute_position, portfolio_summary

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ising Live Predictor",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f172a; }
  [data-testid="stSidebar"]          { background: #1e293b; }
  .main .block-container             { padding-top: 1.2rem; max-width: 1440px; }

  .pred-card {
    border-radius: 12px; padding: 16px 18px;
    margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08);
  }
  .card-bullish { background: linear-gradient(135deg,#052e16,#14532d); border-color:#22c55e44; }
  .card-bearish { background: linear-gradient(135deg,#2d0a0a,#450a0a); border-color:#ef444444; }
  .card-neutral { background: linear-gradient(135deg,#0f172a,#1e293b); border-color:#64748b44; }

  .card-ticker { font-size:1.35rem; font-weight:800; letter-spacing:1px; }
  .card-signal { font-size:0.95rem; font-weight:700; padding:3px 11px;
                 border-radius:20px; display:inline-block; margin-top:3px; }
  .sig-bullish { background:#166534; color:#bbf7d0; }
  .sig-bearish { background:#7f1d1d; color:#fecaca; }
  .sig-neutral { background:#334155; color:#cbd5e1; }

  .action-buy  { background:#0c4a6e; color:#bae6fd; font-weight:700;
                 padding:2px 10px; border-radius:12px; font-size:0.8rem; }
  .action-sell { background:#4c0519; color:#fda4af; font-weight:700;
                 padding:2px 10px; border-radius:12px; font-size:0.8rem; }
  .action-hold { background:#1c1917; color:#a8a29e; font-weight:700;
                 padding:2px 10px; border-radius:12px; font-size:0.8rem; }

  .risk-high   { color:#ef4444; font-weight:700; }
  .risk-medium { color:#f59e0b; font-weight:700; }
  .risk-low    { color:#22c55e; font-weight:700; }
  .risk-none   { color:#64748b; }

  .card-meta { font-size:0.76rem; color:#94a3b8; margin-top:8px; }
  .section-header {
    font-size:1.05rem; font-weight:700; color:#e2e8f0;
    margin:18px 0 8px; border-left:3px solid #6366f1; padding-left:10px;
  }
  .live-badge {
    background:#dc2626; color:white; font-size:0.7rem; font-weight:700;
    padding:2px 8px; border-radius:20px;
  }
  [data-testid="metric-container"] { background:#1e293b; border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

DEFAULT_STOCKS = {
    "NVDA":"NVIDIA",  "TSLA":"Tesla",   "AAPL":"Apple",
    "META":"Meta",    "AMZN":"Amazon",  "MSFT":"Microsoft",
    "GOOGL":"Alphabet","AMD":"AMD",     "PLTR":"Palantir",
    "COIN":"Coinbase",
}

SIGNAL_EMOJI = {"BULLISH":"🟢","BEARISH":"🔴","NEUTRAL":"⚪"}
SIGNAL_COLOR = {"BULLISH":"#22c55e","BEARISH":"#ef4444","NEUTRAL":"#64748b"}
ACTION_CSS   = {"BUY":"action-buy","SELL":"action-sell","HOLD":"action-hold"}
RISK_CSS     = {"HIGH":"risk-high","MEDIUM":"risk-medium","LOW":"risk-low","NONE":"risk-none"}


# ─────────────────────────────────────────────────────────────
# LIVE PRICE FETCH
# ─────────────────────────────────────────────────────────────

def fetch_live_price(ticker):
    try:
        info = yf.Ticker(ticker).fast_info
        price = info.last_price
        return round(float(price), 2) if price else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧲 Ising Live Predictor")
    st.markdown("*Social contagion model · live analysis*")
    st.divider()

    st.markdown("### 📋 Stocks")
    selected_tickers = st.multiselect(
        "Choose stocks",
        options=list(DEFAULT_STOCKS.keys()),
        default=["NVDA","TSLA","AAPL","META"],
        format_func=lambda t: f"{t} — {DEFAULT_STOCKS.get(t, t)}",
    )
    custom = st.text_input("Add custom ticker", placeholder="e.g. PLTR").upper().strip()
    if custom and custom not in selected_tickers:
        selected_tickers.append(custom)

    st.divider()
    st.markdown("### 💼 Portfolio")
    portfolio_value = st.number_input(
        "Portfolio size (USD $)",
        min_value=1_000, max_value=10_000_000,
        value=10_000, step=1_000,
        help="Your total investable capital. Used for Kelly position sizing.",
    )
    kelly_mult = st.select_slider(
        "Kelly aggressiveness",
        options=[0.25, 0.5, 0.75, 1.0],
        value=0.5,
        format_func=lambda x: {
            0.25:"1/4 Kelly (conservative)",
            0.5: "1/2 Kelly (recommended)",
            0.75:"3/4 Kelly (aggressive)",
            1.0: "Full Kelly (risky)"
        }[x],
    )

    st.divider()
    st.markdown("### ⚙️ Model")
    n_investors  = st.slider("Investor network size (N)", 50, 300, 120, 10)
    D_anisotropy = st.slider("Anisotropy D", 0.0, 1.0, 0.25, 0.05)
    n_beta_pts   = st.select_slider("Beta grid points", [6,8,10,12,15], value=8)
    use_trends   = st.checkbox("Google Trends sentiment", value=True)

    st.divider()
    st.markdown("### 🔴 Live Mode")
    live_mode = st.toggle("Auto-refresh", value=False,
        help="Automatically re-run analysis on a timer.")
    if live_mode:
        refresh_min = st.select_slider(
            "Refresh interval",
            options=[5,10,15,30,60], value=15,
            format_func=lambda x: f"Every {x} min",
        )
    else:
        refresh_min = 15

    st.divider()
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    st.caption(f"Est. runtime: ~{len(selected_tickers)*35}–{len(selected_tickers)*70}s")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

h1, h2 = st.columns([3,1])
with h1:
    live_tag = '<span class="live-badge">● LIVE</span>' if live_mode else ""
    st.markdown(f"# 🧲 Ising Social Contagion Predictor  {live_tag}",
                unsafe_allow_html=True)
    st.markdown(
        "Blume-Capel 3-state Ising model on a scale-free investor network. "
        "Social sentiment (Google Trends) drives the external field *h*. "
        "Monte Carlo calibrates β per stock. **Kelly Criterion** sizes each position."
    )
with h2:
    st.markdown(
        f"<div style='text-align:right;color:#64748b;padding-top:28px;font-size:0.82rem;'>"
        f"Portfolio: <b style='color:#e2e8f0;'>${portfolio_value:,.0f}</b><br>"
        f"{datetime.now().strftime('%H:%M · %d %b %Y')}</div>",
        unsafe_allow_html=True,
    )
st.divider()

with st.expander("📐 Model physics + position sizing", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(r"""
**Hamiltonian:**
$$H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i + D \sum_i \sigma_i^2$$
| Symbol | Meaning |
|--------|---------|
| $\sigma_i \in \{-1,0,+1\}$ | Spin: sell / hold / buy |
| $\beta$ | Inverse temp (calibrated per stock) |
| $h$ | External field (social sentiment) |
| $D$ | Anisotropy (action vs inaction) |
        """)
    with c2:
        st.markdown(r"""
**Kelly Criterion:**
$$f^* = \frac{b \cdot p - q}{b}$$
| Symbol | Meaning |
|--------|---------|
| $p$ | Model confidence → win probability |
| $q = 1-p$ | Probability of being wrong |
| $b$ | Win/loss ratio (from volatility) |

Half-Kelly ($f^*/2$) recommended — 75% of optimal growth, far less risk.
        """)


# ─────────────────────────────────────────────────────────────
# PREDICTION PIPELINE (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=1800)
def run_prediction_pipeline(ticker, keyword, n_investors, D, n_beta_points, use_trends):
    result = {
        "ticker":ticker,"keyword":keyword,
        "prediction":"N/A","confidence":0.0,
        "magnetization":0.0,"mag_std":0.0,"mag_samples":[],
        "calibrated_beta":1.0,"model_correlation":0.0,"beta_curve":[],
        "spin_counts":{},"summary":{},"current_h":0.0,
        "data_source":"unknown","error":None,
    }
    try:
        data = get_stock_and_sentiment(
            ticker, keyword=keyword, period="4mo",
            use_trends=use_trends, verbose=False,
        )
        result["summary"]     = data["summary"]
        result["current_h"]   = data["current_h"]
        result["data_source"] = data["data_source"]

        returns_arr = data["returns"]
        sent_arr    = data["sentiment_hist"]

        beta_range = np.linspace(0.1, 3.0, n_beta_points)
        if len(sent_arr) >= 3 and len(returns_arr) >= 3:
            best_beta, best_corr, curve = calibrate_beta(
                list(sent_arr), list(returns_arr),
                beta_range=beta_range, n_investors=n_investors,
                n_equil=120, n_samples=20,
            )
        else:
            best_beta, best_corr, curve = 1.0, 0.0, []

        result["calibrated_beta"]   = round(best_beta, 3)
        result["model_correlation"] = round(best_corr, 3)
        result["beta_curve"]        = curve

        model = IsingInvestorNetwork(n_investors=n_investors, J=1.0, beta=best_beta, D=D)
        model.set_sentiment_field(data["current_h"])
        mag_samples = model.run(n_equil=250, n_samples=60, sample_every=5)

        mag_mean = float(np.mean(mag_samples))
        mag_std  = float(np.std(mag_samples))

        result["magnetization"] = round(mag_mean, 4)
        result["mag_std"]       = round(mag_std, 4)
        result["mag_samples"]   = mag_samples.tolist()
        result["spin_counts"]   = model.get_spin_counts()

        signal, confidence = magnetization_to_signal(mag_mean, mag_std,
                                                      mag_samples=mag_samples)
        result["prediction"]  = signal
        result["confidence"]  = confidence

    except Exception as e:
        result["error"]      = str(e)
        result["prediction"] = "ERROR"
    return result


# ─────────────────────────────────────────────────────────────
# CARD HTML
# ─────────────────────────────────────────────────────────────

def prediction_card_html(r, pos, live_price):
    sig      = r["prediction"]
    card_css = {"BULLISH":"card-bullish","BEARISH":"card-bearish"}.get(sig,"card-neutral")
    sig_css  = {"BULLISH":"sig-bullish","BEARISH":"sig-bearish"}.get(sig,"sig-neutral")
    color    = SIGNAL_COLOR.get(sig,"#64748b")
    emoji    = SIGNAL_EMOJI.get(sig,"❓")

    display_price = live_price or r["summary"].get("current_price")
    price_str  = f"${display_price:,.2f}" if display_price else "N/A"
    live_dot   = " <span style='color:#22c55e;font-size:0.65rem;'>● live</span>" if live_price else ""

    chg     = r["summary"].get("change_7d_pct", 0) if r["summary"] else 0
    chg_col = "#22c55e" if chg >= 0 else "#ef4444"
    chg_arr = "▲" if chg >= 0 else "▼"

    action   = pos["action"]
    act_css  = ACTION_CSS.get(action,"action-hold")
    risk_css = RISK_CSS.get(pos["risk_level"],"risk-none")

    if action in ("BUY","SELL"):
        icon = "📈" if action == "BUY" else "📉"
        pos_html = f"""
<div style="margin-top:10px;padding:10px 0 6px;border-top:1px solid #334155;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span class="{act_css}">{icon} {action}</span>
    <span style="font-size:1.1rem;font-weight:800;color:#f8fafc;">
      ${pos['dollars']:,.0f}
      <span style="font-size:0.75rem;color:#94a3b8;">({pos['fraction_pct']:.1f}%)</span>
    </span>
  </div>
  <div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">
    &approx; <b style="color:#e2e8f0;">{pos['shares']:,.2f} shares</b> @ {price_str}
    &nbsp;&middot;&nbsp; Risk: <span class="{risk_css}">{pos['risk_level']}</span>
    &nbsp;&middot;&nbsp; Kelly {pos['fraction_pct']:.1f}%
  </div>
</div>"""
    else:
        pos_html = f"""
<div style="margin-top:10px;padding:8px 0 4px;border-top:1px solid #334155;">
  <span class="{act_css}">&#9646; HOLD &mdash; no new position recommended</span>
</div>"""

    src_map = {"google_trends":"📊 Trends","price_momentum":"📈 Momentum","mock":"🔮 Simulated"}
    src = src_map.get(r["data_source"], r["data_source"])

    return f"""
<div class="pred-card {card_css}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <span class="card-ticker" style="color:{color};">{r['ticker']}</span><br>
      <span class="card-signal {sig_css}">{emoji} {sig}</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.15rem;font-weight:700;color:#e2e8f0;">{price_str}{live_dot}</div>
      <div style="color:{chg_col};font-size:0.82rem;">{chg_arr} {abs(chg):.1f}% (7d)</div>
    </div>
  </div>
  {pos_html}
  <div class="card-meta" style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:3px;">
    <span>Confidence: <b style="color:#e2e8f0;">{r['confidence']:.0f}%</b></span>
    <span>M = <b style="color:#e2e8f0;">{r['magnetization']:+.3f} &plusmn; {r['mag_std']:.3f}</b></span>
    <span>&beta; = <b style="color:#e2e8f0;">{r['calibrated_beta']}</b></span>
    <span>&rho; = <b style="color:#e2e8f0;">{r['model_correlation']:+.3f}</b></span>
    <span>h = <b style="color:#e2e8f0;">{r['current_h']:+.3f}</b></span>
    <span>{src}</span>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────

def make_market_gauge(results):
    valid = [r for r in results if r["prediction"] not in ("N/A","ERROR")]
    if not valid: return go.Figure()
    avg_m = np.mean([r["magnetization"] for r in valid])
    label = "BULLISH" if avg_m > 0.08 else ("BEARISH" if avg_m < -0.08 else "NEUTRAL")
    color = SIGNAL_COLOR[label]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(avg_m*100, 1),
        delta={"reference":0,"suffix":"%"},
        title={"text":f"Market Sentiment<br><span style='font-size:0.8em;color:#94a3b8;'>{label}</span>"},
        gauge={
            "axis":{"range":[-100,100],"tickcolor":"#475569"},
            "bar":{"color":color},
            "bgcolor":"#1e293b","bordercolor":"#334155",
            "steps":[{"range":[-100,-12],"color":"#450a0a"},
                     {"range":[-12,12],  "color":"#1e293b"},
                     {"range":[12,100],  "color":"#052e16"}],
        },
        number={"suffix":"%","font":{"color":color,"size":28}},
    ))
    fig.update_layout(paper_bgcolor="#0f172a",font=dict(color="#e2e8f0"),
                      margin=dict(l=20,r=20,t=60,b=10),height=240)
    return fig


def make_allocation_fig(positions, results, portfolio_value):
    pairs = [(r,p) for r,p in zip(results,positions)
             if r["prediction"] not in ("N/A","ERROR") and p["dollars"]>0]
    if not pairs: return go.Figure()
    tickers = [r["ticker"] for r,p in pairs]
    dollars = [p["dollars"] for r,p in pairs]
    colors  = [SIGNAL_COLOR[r["prediction"]] for r,p in pairs]
    cash    = max(0, portfolio_value - sum(dollars))
    tickers.append("Cash"); dollars.append(cash); colors.append("#334155")
    fig = go.Figure(go.Pie(
        labels=tickers, values=dollars,
        marker=dict(colors=colors,line=dict(color="#0f172a",width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        hole=0.5,
    ))
    fig.update_layout(
        title="Portfolio Allocation (Kelly Criterion)",
        paper_bgcolor="#0f172a",font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="#1e293b"),
        margin=dict(l=10,r=10,t=40,b=10),
    )
    return fig


def make_mag_fig(results):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, r in enumerate(results):
        if not r["mag_samples"]: continue
        fig.add_trace(go.Scatter(
            y=r["mag_samples"], mode="lines", name=r["ticker"],
            line=dict(color=colors[i%len(colors)],width=1.5), opacity=0.85,
        ))
    fig.add_hline(y=0.12,  line_dash="dot", line_color="#22c55e",
                  annotation_text="Bullish", annotation_position="top right")
    fig.add_hline(y=-0.12, line_dash="dot", line_color="#ef4444",
                  annotation_text="Bearish", annotation_position="bottom right")
    fig.add_hline(y=0, line_color="#475569", line_width=1)
    fig.update_layout(
        title="Magnetisation M over MC Samples",
        xaxis_title="Sample", yaxis_title="M",
        yaxis=dict(range=[-1.05,1.05],gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        paper_bgcolor="#0f172a",plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),legend=dict(bgcolor="#1e293b"),
        margin=dict(l=10,r=10,t=40,b=10),
    )
    return fig


def make_beta_fig(results):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, r in enumerate(results):
        if not r["beta_curve"]: continue
        betas = [d["beta"] for d in r["beta_curve"]]
        corrs = [d["correlation"] for d in r["beta_curve"]]
        fig.add_trace(go.Scatter(x=betas,y=corrs,mode="lines+markers",name=r["ticker"],
                                 line=dict(color=colors[i%len(colors)],width=2),marker=dict(size=5)))
        if r["calibrated_beta"] in betas:
            idx = betas.index(r["calibrated_beta"])
            fig.add_trace(go.Scatter(x=[r["calibrated_beta"]],y=[corrs[idx]],mode="markers",
                                     marker=dict(symbol="star",size=14,color=colors[i%len(colors)],
                                                 line=dict(width=1,color="white")),showlegend=False))
    fig.add_hline(y=0,line_color="#475569",line_width=1)
    fig.update_layout(
        title="Beta Calibration — rho(M, returns) vs beta",
        xaxis_title="beta",yaxis_title="Pearson rho",
        yaxis=dict(range=[-1.05,1.05],gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        paper_bgcolor="#0f172a",plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),legend=dict(bgcolor="#1e293b"),
        margin=dict(l=10,r=10,t=40,b=10),
    )
    return fig


def make_spin_fig(results):
    valid = [r for r in results if r["spin_counts"]]
    if not valid: return go.Figure()
    tickers = [r["ticker"] for r in valid]
    totals  = [sum(r["spin_counts"].values()) for r in valid]
    fig = go.Figure()
    for key, color, label in [("buy","#22c55e","Buy (+1)"),
                               ("neutral","#64748b","Neutral (0)"),
                               ("sell","#ef4444","Sell (-1)")]:
        fig.add_bar(name=label, x=tickers,
                    y=[r["spin_counts"].get(key,0)/t*100 for r,t in zip(valid,totals)],
                    marker_color=color)
    fig.update_layout(
        barmode="stack",title="Spin Distribution (% Investors)",
        yaxis=dict(title="% Investors",gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        paper_bgcolor="#0f172a",plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),legend=dict(bgcolor="#1e293b"),
        margin=dict(l=10,r=10,t=40,b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

for key, default in [
    ("last_run_ts", 0.0),
    ("all_results", []),
    ("all_positions", []),
    ("live_prices", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────
# TRIGGER
# ─────────────────────────────────────────────────────────────

now_ts     = time.time()
elapsed    = now_ts - st.session_state.last_run_ts
should_run = run_btn or (
    live_mode
    and elapsed >= refresh_min * 60
    and len(selected_tickers) > 0
)

if not should_run and not st.session_state.all_results:
    st.info(
        "Select stocks and set your portfolio size, then click **Run Analysis**.\n\n"
        "Enable **Live Mode** in the sidebar to auto-refresh on a timer."
    )
    demo = IsingInvestorNetwork(n_investors=80, D=0.25)
    demo.reset(seed=7)
    st.markdown('<div class="section-header">Demo — Investor Network</div>', unsafe_allow_html=True)
    st.plotly_chart(demo.get_network_plotly(), use_container_width=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if should_run:
    if not selected_tickers:
        st.warning("Select at least one stock.")
        st.stop()

    all_results   = []
    all_positions = []
    live_prices   = {}
    total = len(selected_tickers)
    prog  = st.progress(0.0, text="Starting...")
    status = st.empty()

    for idx, ticker in enumerate(selected_tickers):
        keyword = DEFAULT_STOCKS.get(ticker, ticker)
        status.markdown(f"**Processing {ticker}** ({idx+1}/{total}) — fetching · calibrating beta · running Ising MC...")

        res = run_prediction_pipeline(
            ticker=ticker, keyword=keyword, n_investors=n_investors,
            D=D_anisotropy, n_beta_points=n_beta_pts, use_trends=use_trends,
        )
        all_results.append(res)

        lp = fetch_live_price(ticker)
        live_prices[ticker] = lp
        effective_price = lp or res["summary"].get("current_price", 0) or 1.0

        vol = res["summary"].get("volatility_annual_pct", 25.0) if res["summary"] else 25.0
        pos = compute_position(
            signal=res["prediction"],
            confidence_pct=res["confidence"],
            current_price=effective_price,
            portfolio_value=portfolio_value,
            volatility_annual_pct=vol,
            kelly_multiplier=kelly_mult,
        )
        all_positions.append(pos)
        prog.progress((idx+1)/total, text=f"{ticker} done — {res['prediction']}")

    prog.empty()
    status.success(f"Analysis complete · {total} stock(s) · {datetime.now().strftime('%H:%M:%S')}")

    st.session_state.all_results   = all_results
    st.session_state.all_positions = all_positions
    st.session_state.live_prices   = live_prices
    st.session_state.last_run_ts   = time.time()

else:
    all_results   = st.session_state.all_results
    all_positions = st.session_state.all_positions
    live_prices   = st.session_state.live_prices


# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────

valid    = [r for r in all_results if r["prediction"] not in ("N/A","ERROR")]
bullish  = sum(1 for r in valid if r["prediction"] == "BULLISH")
bearish  = sum(1 for r in valid if r["prediction"] == "BEARISH")
neutral  = sum(1 for r in valid if r["prediction"] == "NEUTRAL")
avg_conf = np.mean([r["confidence"] for r in valid]) if valid else 0

port_sum = portfolio_summary(all_positions, portfolio_value)

st.divider()
ov1,ov2,ov3,ov4,ov5 = st.columns(5)
ov1.metric("🟢 Bullish", bullish)
ov2.metric("🔴 Bearish", bearish)
ov3.metric("⚪ Neutral", neutral)
ov4.metric("📊 Avg Confidence", f"{avg_conf:.0f}%")
ov5.metric("💵 Cash Remaining",
           f"${port_sum['cash_remaining_dollars']:,.0f}",
           f"{port_sum['cash_remaining_pct']:.0f}% free")

gauge_col, pie_col = st.columns([1, 1.4])
with gauge_col:
    st.plotly_chart(make_market_gauge(valid), use_container_width=True)
with pie_col:
    st.plotly_chart(make_allocation_fig(all_positions, all_results, portfolio_value),
                    use_container_width=True)

st.markdown('<div class="section-header">📊 Predictions &amp; Position Sizing</div>',
            unsafe_allow_html=True)

n_cols    = min(3, len(all_results))
card_cols = st.columns(n_cols)
for i, (r, pos) in enumerate(zip(all_results, all_positions)):
    with card_cols[i % n_cols]:
        lp = live_prices.get(r["ticker"])
        st.markdown(prediction_card_html(r, pos, lp), unsafe_allow_html=True)

with st.expander("🧮 Position sizing rationale"):
    rows = []
    for r, pos in zip(all_results, all_positions):
        rows.append({
            "Ticker":   r["ticker"],
            "Signal":   r["prediction"],
            "Confidence": f"{r['confidence']:.0f}%",
            "Action":   pos["action"],
            "Kelly %":  f"{pos['fraction_pct']:.1f}%",
            "Amount $": f"${pos['dollars']:,.0f}",
            "Shares":   f"{pos['shares']:.2f}",
            "Risk":     pos["risk_level"],
            "b":        pos["win_loss_ratio"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Ising Network",
    "📈 Magnetisation Dynamics",
    "⚙️ Beta Calibration",
    "📊 Spin Breakdown",
])

with tab1:
    st.markdown("Green = bullish · Grey = neutral · Red = bearish. Node size proportional to social influence.")
    if valid:
        sel = st.selectbox("Stock", [r["ticker"] for r in valid], key="net_sel")
        sel_r = next((r for r in valid if r["ticker"] == sel), None)
        if sel_r:
            viz = IsingInvestorNetwork(n_investors=n_investors,
                                       beta=sel_r["calibrated_beta"], D=D_anisotropy)
            viz.set_sentiment_field(sel_r["current_h"])
            viz.run(n_equil=250, n_samples=1, sample_every=1)
            st.plotly_chart(viz.get_network_plotly(), use_container_width=True)

with tab2:
    st.plotly_chart(make_mag_fig(valid), use_container_width=True)

with tab3:
    st.plotly_chart(make_beta_fig(valid), use_container_width=True)
    st.dataframe(pd.DataFrame([{
        "Ticker":r["ticker"],"Beta*":r["calibrated_beta"],
        "rho":r["model_correlation"],"h":round(r["current_h"],3),
        "Source":r["data_source"],
    } for r in valid]), use_container_width=True, hide_index=True)

with tab4:
    st.plotly_chart(make_spin_fig(valid), use_container_width=True)

st.divider()
with st.expander("📥 Export JSON"):
    export = {
        "generated_at": datetime.now().isoformat(),
        "portfolio_value": portfolio_value,
        "kelly_multiplier": kelly_mult,
        "model_config": {"n_investors":n_investors,"D":D_anisotropy,
                         "n_beta_pts":n_beta_pts,"use_trends":use_trends},
        "predictions": [{k:v for k,v in r.items() if k \!= "_model"} for r in all_results],
        "positions": all_positions,
        "portfolio_summary": port_sum,
    }
    st.download_button("Download predictions.json",
                       data=json.dumps(export, indent=2),
                       file_name="predictions.json", mime="application/json")


# ─────────────────────────────────────────────────────────────
# LIVE MODE COUNTDOWN + AUTO-RERUN
# ─────────────────────────────────────────────────────────────

if live_mode:
    st.divider()
    refresh_secs    = refresh_min * 60
    elapsed_now     = time.time() - st.session_state.last_run_ts
    remaining_secs  = max(0, int(refresh_secs - elapsed_now))
    progress_frac   = min(1.0, elapsed_now / refresh_secs)

    lv1, lv2 = st.columns([4, 1])
    with lv1:
        st.progress(progress_frac,
                    text=f"🔴 Live — next refresh in {remaining_secs}s (every {refresh_min} min)")
    with lv2:
        if st.button("🔄 Refresh now", use_container_width=True):
            run_prediction_pipeline.clear()
            st.session_state.last_run_ts = 0.0
            st.rerun()

    time.sleep(min(remaining_secs + 1, 10))
    st.rerun()
