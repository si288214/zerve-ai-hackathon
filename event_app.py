"""
event_app.py — Ising Event Prediction  |  E.W. Research
========================================================
Streamlit app with two pages:
  1. Predict — enter a ticker + event type, get a live prediction
  2. Dashboard — track all active predictions and their outcomes

Run:
    streamlit run event_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import json, os, time, requests
import warnings
warnings.filterwarnings("ignore")

from sentiment_engine import compute_h, fetch_finnhub_news
from event_backtest_v2 import ising_predict, CFG

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "os.getenv("GROQ_API_KEY", "")")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ising Event Predictor",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  —  dark terminal aesthetic, monospace accents
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  /* Base */
  [data-testid="stAppViewContainer"] { background: #080c14; }
  [data-testid="stSidebar"]          { background: #0d1117; border-right: 1px solid #1e2a3a; }
  .main .block-container             { padding-top: 1.5rem; max-width: 1300px; }
  html, body, [class*="css"]         { font-family: 'DM Sans', sans-serif; color: #c9d4e0; }

  /* Typography */
  h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }
  h1 { font-size: 1.8rem; color: #e8f0fe; }
  h2 { font-size: 1.2rem; color: #a8bdd4; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0d1520;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 14px 16px;
  }

  /* Prediction result cards */
  .result-card {
    border-radius: 10px;
    padding: 24px 28px;
    margin: 16px 0;
    border: 1px solid #1e2a3a;
    position: relative;
    overflow: hidden;
  }
  .result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
  }
  .result-positive { background: #061910; }
  .result-positive::before { background: #00e676; }
  .result-negative { background: #190608; }
  .result-negative::before { background: #ff1744; }
  .result-neutral  { background: #0d1520; }
  .result-neutral::before  { background: #546e7a; }
  .result-skip     { background: #0f1218; }
  .result-skip::before     { background: #ffc107; }

  .result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #546e7a;
    margin-bottom: 4px;
  }
  .result-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .result-positive .result-value { color: #00e676; }
  .result-negative .result-value { color: #ff4444; }
  .result-neutral  .result-value { color: #78909c; }
  .result-skip     .result-value { color: #ffc107; }

  .h-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #546e7a;
    margin-top: 8px;
  }

  /* Tabs */
  [data-testid="stTab"] { font-family: 'Space Mono', monospace; font-size: 0.8rem; }

  /* Buttons */
  [data-testid="baseButton-primary"] {
    background: #1565c0 !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 1px !important;
  }

  /* Input fields */
  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] select {
    background: #0d1520 !important;
    border: 1px solid #1e2a3a !important;
    color: #c9d4e0 !important;
    font-family: 'Space Mono', monospace !important;
  }

  /* Divider */
  hr { border-color: #1e2a3a; }

  /* Tag pill */
  .tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .tag-earnings   { background: #0d2137; color: #4a9eda; border: 1px solid #1e3a5c; }
  .tag-fed        { background: #1a1a0d; color: #e8d5b0; border: 1px solid #3a3a1e; }
  .tag-cpi        { background: #1a0d1a; color: #c49ae8; border: 1px solid #3a1e3a; }
  .tag-drug_trial { background: #0d1a0d; color: #4eca7e; border: 1px solid #1e3a1e; }
  .tag-merger     { background: #1a120d; color: #f0a030; border: 1px solid #3a2a1e; }

  .skip-msg {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #ffc107;
    background: #1a1500;
    border: 1px solid #3a3000;
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 10px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EVENT_TYPES = ["earnings", "fed", "cpi", "drug_trial", "merger"]

OUTCOME_LABELS = {
    "earnings":   {1: "BEAT",     0: "MISS"},
    "fed":        {1: "CUT",      0: "HOLD/HIKE"},
    "cpi":        {1: "BELOW EST",0: "ABOVE EST"},
    "drug_trial": {1: "APPROVED", 0: "REJECTED"},
    "merger":     {1: "DEAL CLOSES", 0: "DEAL BREAKS"},
}

LOOKBACK_DEFAULTS = {
    "earnings": 7, "fed": 14, "cpi": 7, "drug_trial": 14, "merger": 14,
}

DASHBOARD_FILE = "dashboard_predictions.json"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def auto_describe(ticker: str, event_type: str, event_date: str) -> str:
    try:
        _, headlines = fetch_finnhub_news(ticker, event_date, lookback_days=14, event_type=event_type)
        headline_block = "\n".join(f"- {h[:120]}" for h in headlines[:8]) if headlines else "No headlines available."
        prompt = (
            f"Based on what you know about {ticker} and these recent headlines, "
            f"write a single concise description (max 12 words) for an upcoming "
            f"{event_type.replace('_',' ')} event on {event_date}. "
            f"Be specific — include the company name, event type, and any key detail (e.g. drug name, deal target, quarter). "
            f"Return ONLY the description, no punctuation at the end.\n\nHeadlines:\n{headline_block}"
        )
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 40, "temperature": 0.3},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        dt_obj = __import__('datetime').datetime.strptime(event_date, "%Y-%m-%d")
        return f"{ticker} {event_type.replace('_',' ').title()} {dt_obj.strftime('%b %Y')}"


def load_dashboard() -> list:
    if os.path.exists(DASHBOARD_FILE):
        with open(DASHBOARD_FILE) as f:
            return json.load(f)
    return []

def save_dashboard(data: list):
    with open(DASHBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def result_card_html(label: str, value: str, h: float, prob: float,
                     card_class: str, extra: str = "") -> str:
    return f"""
    <div class="result-card {card_class}">
        <div class="result-label">{label}</div>
        <div class="result-value">{value}</div>
        <div class="h-value">h = {h:+.3f} &nbsp;|&nbsp; P = {prob:.1%}</div>
        {extra}
    </div>
    """

def tag_html(event_type: str) -> str:
    return f'<span class="tag tag-{event_type}">{event_type.replace("_"," ")}</span>'

@st.cache_data(show_spinner=False, ttl=300)
def run_prediction(ticker: str, event_date: str, event_type: str,
                   desc: str, lookback: int) -> dict:
    event = {"ticker": ticker, "date": event_date,
             "type": event_type, "desc": desc}
    h, breakdown = compute_h(event, lookback_days=lookback)
    has_signal = bool(breakdown.get("sources_used"))

    if not has_signal:
        return {
            "h": h, "breakdown": breakdown,
            "has_signal": False,
            "prediction": None, "probability": None,
            "label": None,
        }

    pred = ising_predict(h, CFG)
    label = OUTCOME_LABELS[event_type][pred["prediction"]]
    return {
        "h": h, "breakdown": breakdown,
        "has_signal": True,
        "prediction": pred["prediction"],
        "probability": pred["probability"],
        "magnetization": pred["magnetization"],
        "label": label,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚛ Ising Event Predictor")
    st.markdown("*E.W. Research*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔮 Predict", "📊 Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:0.7rem;"
        "color:#546e7a;line-height:1.8'>"
        "Model: Blume-Capel Ising<br>"
        f"β = {CFG['beta']}  |  N = {CFG['n_investors']}<br>"
        "Signal: Finnhub + FinBERT<br>"
        "Confidence threshold: |h| ≥ 0.15"
        "</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────

if page == "🔮 Predict":

    st.markdown("# ⚛ Event Prediction")
    st.markdown("Enter an upcoming event and get a probability estimate from the Ising social contagion model.")
    st.divider()

    # ── Input form ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        ticker = st.text_input(
            "Ticker", placeholder="e.g. AAPL", value="",
            help="Primary equity ticker. Use SPY for Fed/CPI events."
        ).upper().strip()

    with col2:
        event_type = st.selectbox(
            "Event Type",
            EVENT_TYPES,
            format_func=lambda x: x.replace("_", " ").title(),
        )

    with col3:
        event_date = st.date_input(
            "Event Date",
            value=date.today(),
            min_value=date.today(),
        )

    col4, col5 = st.columns([2, 1])
    with col4:
        desc = st.text_input(
            "Description",
            placeholder=f"e.g. {ticker or 'AAPL'} Q2 FY26 earnings",
            value=st.session_state.get("auto_desc", f"{ticker} {event_type.replace('_',' ')} {event_date.strftime('%b %Y')}" if ticker else ""),
        )
    with col5:
        lookback = st.number_input(
            "Lookback days",
            min_value=3, max_value=30,
            value=LOOKBACK_DEFAULTS.get(event_type, 7),
        )

    run_col, auto_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button("⚛ Run Prediction", type="primary", use_container_width=True)
    with auto_col:
        auto_btn = st.button("✨ Auto-fill desc", use_container_width=True, disabled=not bool(ticker))
    if auto_btn and ticker:
        with st.spinner("Generating description..."):
            generated = auto_describe(ticker, event_type, str(event_date))
            st.session_state["auto_desc"] = generated
            st.rerun()

    # ── Run ─────────────────────────────────────────────────────────────────
    if run_btn:
        if not ticker:
            st.warning("Please enter a ticker.")
            st.stop()

        with st.spinner(f"Fetching headlines for {ticker} and running Ising MC..."):
            result = run_prediction(
                ticker=ticker,
                event_date=str(event_date),
                event_type=event_type,
                desc=desc or f"{ticker} {event_type}",
                lookback=lookback,
            )

        st.divider()

        if not result["has_signal"]:
            st.markdown(result_card_html(
                label="PREDICTION",
                value="NO SIGNAL",
                h=result["h"],
                prob=0.5,
                card_class="result-skip",
                extra='<div class="skip-msg">⚠ Signal below confidence threshold (|h| &lt; 0.15). '
                      'Too few relevant headlines or weak sentiment consensus.</div>'
            ), unsafe_allow_html=True)
        else:
            # Determine card class
            pred = result["prediction"]
            if pred == 1:
                card_class = "result-positive"
            else:
                card_class = "result-negative"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(result_card_html(
                    label="PREDICTION",
                    value=result["label"],
                    h=result["h"],
                    prob=result["probability"],
                    card_class=card_class,
                ), unsafe_allow_html=True)
            with c2:
                st.metric("Sentiment h", f"{result['h']:+.3f}")
                st.metric("P(positive)", f"{result['probability']:.1%}")
            with c3:
                st.metric("Magnetization M", f"{result.get('magnetization', 0):+.3f}")
                st.metric("Event type", event_type.replace("_"," ").title())

            # Source breakdown
            sources = result["breakdown"].get("sources_used", {})
            if sources:
                st.divider()
                st.markdown("**Signal sources used:**")
                src_cols = st.columns(len(sources))
                for i, (src, val) in enumerate(sources.items()):
                    with src_cols[i]:
                        st.metric(src.upper(), f"{val:+.3f}")

            # Save to dashboard button
            st.divider()
            if st.button("📌 Save to Dashboard", use_container_width=False):
                dashboard = load_dashboard()
                dashboard.append({
                    "id": f"{ticker}_{event_type}_{event_date}",
                    "ticker": ticker,
                    "event_type": event_type,
                    "event_date": str(event_date),
                    "desc": desc,
                    "h": result["h"],
                    "probability": result["probability"],
                    "prediction": result["prediction"],
                    "label": result["label"],
                    "has_signal": result["has_signal"],
                    "added_at": datetime.now().isoformat(),
                    "outcome": None,
                    "correct": None,
                })
                save_dashboard(dashboard)
                st.success(f"✅ Saved {ticker} {event_type} to dashboard.")

    else:
        # Idle state
        st.markdown("""
        <div style='background:#0d1520;border:1px solid #1e2a3a;border-radius:10px;
        padding:28px 32px;margin-top:8px;font-family:DM Sans,sans-serif;'>
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;
        color:#546e7a;letter-spacing:2px;margin-bottom:16px'>HOW IT WORKS</div>
        <div style='color:#a8bdd4;line-height:2;font-size:0.9rem'>
        1. Enter a ticker, event type, and upcoming date<br>
        2. Finnhub fetches recent news headlines<br>
        3. FinBERT scores headlines for sentiment → h ∈ [-1, 1]<br>
        4. Blume-Capel Ising MC simulation runs with h as external field<br>
        5. Magnetization M maps to P(positive outcome)<br>
        6. If |h| &lt; 0.15, model abstains (insufficient signal)
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Supported event types:**")
        type_cols = st.columns(5)
        examples = {
            "earnings":   "Quarterly EPS vs consensus",
            "fed":        "Rate cut / hold / hike",
            "cpi":        "Inflation above/below est",
            "drug_trial": "FDA approval / rejection",
            "merger":     "Deal close / break",
        }
        for i, (et, ex) in enumerate(examples.items()):
            with type_cols[i]:
                st.markdown(
                    f"{tag_html(et)}<br>"
                    f"<span style='font-size:0.75rem;color:#546e7a'>{ex}</span>",
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Dashboard":

    st.markdown("# 📊 Prediction Dashboard")
    st.markdown("Track all saved predictions and log outcomes to measure live accuracy.")
    st.divider()

    dashboard = load_dashboard()

    if not dashboard:
        st.info("No predictions saved yet. Go to **🔮 Predict** and save a prediction.")
        st.stop()

    # ── Summary metrics ──────────────────────────────────────────────────────
    total    = len(dashboard)
    called   = [d for d in dashboard if d.get("has_signal")]
    resolved = [d for d in called if d.get("outcome") is not None]
    correct  = [d for d in resolved if d.get("correct")]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total saved", total)
    m2.metric("Called (signal)", len(called))
    m3.metric("Resolved", len(resolved))
    acc = len(correct) / len(resolved) * 100 if resolved else 0
    m4.metric("Live accuracy", f"{acc:.1f}%" if resolved else "—")

    st.divider()

    # ── Prediction table ────────────────────────────────────────────────────
    st.markdown("### Active Predictions")

    for i, pred in enumerate(reversed(dashboard)):
        idx = len(dashboard) - 1 - i

        # Card class based on prediction
        if not pred.get("has_signal"):
            card_class = "result-skip"
        elif pred.get("outcome") is not None:
            card_class = "result-positive" if pred.get("correct") else "result-negative"
        elif pred.get("prediction") == 1:
            card_class = "result-positive"
        else:
            card_class = "result-negative"

        with st.container():
            row1, row2 = st.columns([3, 1])
            with row1:
                outcome_str = ""
                if pred.get("outcome") is not None:
                    correct_icon = "✓" if pred.get("correct") else "✗"
                    actual = OUTCOME_LABELS.get(pred["event_type"], {}).get(pred["outcome"], str(pred["outcome"]))
                    outcome_str = f" &nbsp;|&nbsp; Actual: <b>{actual}</b> {correct_icon}"

                border_color = '#00e676' if card_class=='result-positive' else '#ff4444' if card_class=='result-negative' else '#ffc107'
                signal_html = ('<b style="color:#00e676">' + pred['label'] + '</b>') if pred.get('has_signal') else '<span style="color:#ffc107">NO SIGNAL</span>'
                prob_html = ("&nbsp;&nbsp;P=" + f"{pred['probability']:.1%}") if pred.get('probability') else ""
                card_html = (
                    f"<div style='padding:14px 18px;background:#0d1520;"
                    f"border:1px solid #1e2a3a;border-radius:8px;"
                    f"border-left:3px solid {border_color}'>"
                    f"<span style='font-family:Space Mono,monospace;font-size:1rem;"
                    f"font-weight:700;color:#e8f0fe'>{pred['ticker']}</span>"
                    f"&nbsp;&nbsp;{tag_html(pred['event_type'])}"
                    f"<span style='font-family:Space Mono,monospace;font-size:0.75rem;"
                    f"color:#546e7a'>&nbsp;{pred['event_date']}</span><br>"
                    f"<span style='font-size:0.85rem;color:#a8bdd4'>"
                    f"{signal_html}"
                    f"&nbsp;&nbsp;h={pred['h']:+.3f}"
                    f"{prob_html}"
                    f"{outcome_str}</span>"
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

            with row2:
                if pred.get("outcome") is None and pred.get("has_signal"):
                    outcome_key = f"outcome_{idx}"
                    outcome_choice = st.selectbox(
                        "Log outcome",
                        ["— pending —",
                         OUTCOME_LABELS.get(pred["event_type"], {}).get(1, "Positive"),
                         OUTCOME_LABELS.get(pred["event_type"], {}).get(0, "Negative")],
                        key=outcome_key,
                        label_visibility="collapsed",
                    )
                    if outcome_choice != "— pending —":
                        # Determine if positive or negative outcome
                        pos_label = OUTCOME_LABELS.get(pred["event_type"], {}).get(1, "")
                        actual_outcome = 1 if outcome_choice == pos_label else 0
                        dashboard[idx]["outcome"] = actual_outcome
                        dashboard[idx]["correct"] = (actual_outcome == pred["prediction"])
                        dashboard[idx]["resolved_at"] = datetime.now().isoformat()
                        save_dashboard(dashboard)
                        st.rerun()
                elif pred.get("outcome") is not None:
                    icon = "✅" if pred.get("correct") else "❌"
                    st.markdown(
                        f"<div style='text-align:center;padding:14px;"
                        f"font-family:Space Mono,monospace;font-size:1.2rem'>{icon}</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Accuracy chart ───────────────────────────────────────────────────────
    if len(resolved) >= 3:
        st.divider()
        st.markdown("### Live Accuracy Trend")

        cumulative_correct = 0
        cumulative_total   = 0
        trend_data = []
        for d in dashboard:
            if d.get("outcome") is not None and d.get("has_signal"):
                cumulative_total   += 1
                cumulative_correct += int(d.get("correct", False))
                trend_data.append({
                    "n": cumulative_total,
                    "accuracy": cumulative_correct / cumulative_total * 100,
                    "ticker": d["ticker"],
                })

        df = pd.DataFrame(trend_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["n"], y=df["accuracy"],
            mode="lines+markers",
            line=dict(color="#4a9eda", width=2),
            marker=dict(size=8, color="#4a9eda"),
            name="Accuracy",
            text=df["ticker"],
            hovertemplate="%{text}<br>n=%{x}<br>Acc=%{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=75.8, line_dash="dot", line_color="#ffc107",
                      annotation_text="Baseline (always predict positive)")
        fig.update_layout(
            paper_bgcolor="#080c14",
            plot_bgcolor="#0d1520",
            font=dict(color="#a8bdd4", family="Space Mono"),
            xaxis=dict(title="Predictions called", gridcolor="#1e2a3a"),
            yaxis=dict(title="Cumulative accuracy %", gridcolor="#1e2a3a",
                       range=[0, 105]),
            margin=dict(l=10, r=10, t=20, b=10),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Delete / export ──────────────────────────────────────────────────────
    st.divider()
    col_del, col_exp = st.columns(2)
    with col_exp:
        st.download_button(
            "📥 Export predictions.json",
            data=json.dumps(dashboard, indent=2, default=str),
            file_name="predictions.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_del:
        if st.button("🗑 Clear all predictions", use_container_width=True):
            save_dashboard([])
            st.rerun()
