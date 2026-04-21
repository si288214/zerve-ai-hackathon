#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Ising Social Contagion Stock Predictor — Quick Start
# ─────────────────────────────────────────────────────────────

set -e

echo "🧲 Ising Stock Predictor — Setup & Launch"
echo "─────────────────────────────────────────"

# 1. Install dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt -q

# 2. Launch Streamlit
echo ""
echo "🚀 Launching app at http://localhost:8501"
echo "   (Press Ctrl-C to stop)"
echo ""
streamlit run app.py \
  --server.port 8501 \
  --server.headless false \
  --theme.base dark \
  --theme.primaryColor "#6366f1"
