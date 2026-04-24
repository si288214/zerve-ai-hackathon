"""
api.py — Ising Event Predictor REST API
========================================
FastAPI server that exposes /predict and /describe endpoints
for the landing page HTML to call.

Run alongside Streamlit:
    uvicorn api:app --port 8000 --reload

Endpoints:
    GET /predict?ticker=BSX&type=merger&date=2026-05-07&desc=...&lookback=14
    GET /describe?ticker=BSX&type=merger&date=2026-05-07
    GET /health
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os, requests as req
import warnings
warnings.filterwarnings("ignore")

from sentiment_engine import compute_h, fetch_finnhub_news
from event_backtest_v2 import ising_predict, CFG

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "os.getenv("GROQ_API_KEY", "")")

app = FastAPI(title="Ising Event Predictor API", version="1.0")

# Allow the landing page (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

OUTCOME_LABELS = {
    "earnings":   {1: "BEAT",     0: "MISS"},
    "fed":        {1: "CUT",      0: "HOLD/HIKE"},
    "cpi":        {1: "BELOW EST",0: "ABOVE EST"},
    "drug_trial": {1: "APPROVED", 0: "REJECTED"},
    "merger":     {1: "DEAL CLOSES", 0: "DEAL BREAKS"},
}


@app.get("/health")
def health():
    return {"status": "ok", "model": "blume-capel-ising", "beta": CFG["beta"]}


@app.get("/predict")
def predict(
    ticker:   str = Query(..., description="Equity ticker e.g. BSX"),
    type:     str = Query("earnings", description="Event type"),
    date:     str = Query(..., description="Event date YYYY-MM-DD"),
    desc:     str = Query("", description="Event description"),
    lookback: int = Query(14, ge=3, le=30),
):
    ticker = ticker.upper()
    event = {
        "ticker": ticker,
        "date":   date,
        "type":   type,
        "desc":   desc or f"{ticker} {type} {date}",
    }

    h, breakdown = compute_h(event, lookback_days=lookback)
    has_signal = bool(breakdown.get("sources_used"))

    if not has_signal:
        return {
            "ticker":     ticker,
            "event_type": type,
            "event_date": date,
            "has_signal": False,
            "h":          round(h, 4),
            "prediction": None,
            "probability": None,
            "label":      None,
            "magnetization": None,
            "sources":    {},
        }

    pred   = ising_predict(h, CFG)
    label  = OUTCOME_LABELS.get(type, {}).get(pred["prediction"], str(pred["prediction"]))

    return {
        "ticker":        ticker,
        "event_type":    type,
        "event_date":    date,
        "has_signal":    True,
        "h":             round(h, 4),
        "prediction":    pred["prediction"],
        "probability":   round(pred["probability"], 4),
        "label":         label,
        "magnetization": round(pred["magnetization"], 4),
        "sources":       breakdown.get("sources_used", {}),
        "weights":       breakdown.get("weights_used", {}),
        "source_raw":    breakdown.get("source_raw", {}),
    }


@app.get("/describe")
def describe(
    ticker: str = Query(...),
    type:   str = Query("earnings"),
    date:   str = Query(...),
):
    ticker = ticker.upper()
    try:
        _, headlines = fetch_finnhub_news(ticker, date, lookback_days=14, event_type=type)
        headline_block = "\n".join(f"- {h[:120]}" for h in headlines[:8]) if headlines else "No headlines available."

        prompt = (
            f"Based on what you know about {ticker} and these recent headlines, "
            f"write a single concise description (max 12 words) for an upcoming "
            f"{type.replace('_', ' ')} event on {date}. "
            f"Be specific — include the company name, event type, and any key detail "
            f"(drug name, deal counterparty, quarter, etc). "
            f"Return ONLY the description, no punctuation at the end.\n\n"
            f"Headlines:\n{headline_block}"
        )

        resp = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 40,
                "temperature": 0.3,
            },
            timeout=10,
        )
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        from datetime import datetime
        dt = datetime.strptime(date, "%Y-%m-%d")
        description = f"{ticker} {type.replace('_', ' ').title()} {dt.strftime('%b %Y')}"

    return {"ticker": ticker, "description": description}