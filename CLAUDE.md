# Ising Social Contagion Stock Predictor
**Zerve AI Hackathon — Built by Luke (si288214)**

> A physics-based stock prediction engine that models investor sentiment as a
> social contagion process using the **Blume-Capel 3-state Ising model** on a
> scale-free investor network, with Monte Carlo β-calibration and live
> social sentiment data.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit dashboard
streamlit run app.py
# → opens at http://localhost:8501

# Or use the helper script
bash start.sh
```

---

## Repository Structure

```
Hackathon/
├── ising_model.py      # Physics engine — Blume-Capel model + MC calibration
├── data_fetcher.py     # Data pipeline — yfinance + Google Trends + fallbacks
├── app.py              # Streamlit dashboard — UI, visualisations, export
├── requirements.txt    # Python dependencies
├── start.sh            # One-command launcher
└── CLAUDE.md           # This file — full project context
```

---

## Model Physics

### Hamiltonian

```
H = -J Σ_{<i,j>} σ_i σ_j  -  h Σ_i σ_i  +  D Σ_i σ_i²
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| `σ_i ∈ {-1, 0, +1}` | Investor spin: sell / hold / buy | Model state |
| `J` | Coupling constant — peer influence / social contagion | Fixed = 1.0 |
| `β` | Inverse temperature — market noise level | **Calibrated via Monte Carlo** |
| `h` | External field — social media sentiment score | Google Trends / Reddit |
| `D` | Single-ion anisotropy — cost of taking a position vs staying neutral | User-configurable (default 0.25) |

This is the **Blume-Capel model** (3-state extension of standard Ising).
Standard Ising uses σ ∈ {-1, +1}; the D term introduces a neutral state σ = 0,
representing undecided / passive investors.

### Network Topology

Investor network is a **Barabási–Albert scale-free graph** (`m=3`, N nodes).
- High-degree nodes = market influencers / large social media accounts
- Hub nodes receive a stronger external field (`influencer_boost=0.4`)
- Edge = mutual social influence between two investors

Scale-free topology is realistic: real social networks follow a power-law
degree distribution where a small number of influencers connect to many followers.

### Monte Carlo Dynamics — Metropolis Algorithm

1. Precompute all neighbour sums via sparse matrix multiply: `neighbor_sums = self._adj @ self.spins`
2. For each investor i (random order), propose a new spin `σ_new ∈ {-1, 0, +1}`
3. Compute energy change ΔE
4. Accept with probability `min(1, exp(-β · ΔE))` — Metropolis criterion
5. Propagate accepted flips incrementally to cached neighbour sums

### β Calibration

Sweep β ∈ [0.1, 3.0] over N grid points (default 8). For each β:
- Run the Ising model against the historical sentiment sequence
- Measure Pearson correlation ρ between model magnetisation M and actual log-returns
- Select the β that maximises |ρ|

**Physical interpretation:**
- Low β (high temperature) → disordered, investors act randomly, mean-reverting
- High β (low temperature) → ordered, herd behaviour, strong momentum
- β_c (critical point) → phase transition, maximum susceptibility, peak contagion risk

### Prediction Signal

After calibration, run the model with today's sentiment as external field h.

```
M = ⟨σ⟩ = (1/N) Σ σ_i    ∈ [-1, 1]
```

| M value | Signal | Interpretation |
|---------|--------|----------------|
| M > P75(samples) | BULLISH | Herd tipping bullish |
| M < P25(samples) | BEARISH | Herd tipping bearish |
| Otherwise | NEUTRAL | No clear consensus |

Confidence = 50 + 45 × min(SNR/3, 1), where SNR = |M| / σ_M

---

## File-by-File Reference

### `ising_model.py`

**Classes:**
- `IsingInvestorNetwork` — core Blume-Capel model

**Key methods on `IsingInvestorNetwork`:**

| Method | Description |
|--------|-------------|
| `__init__(n_investors, J, beta, D, seed)` | Builds BA graph, sparse adj matrix, initial state |
| `set_sentiment_field(h_global, influencer_boost)` | Sets external field h per node |
| `_metropolis_sweep()` | One full sweep via vectorised `self._adj @ self.spins` |
| `_autocorrelation_time(samples)` | Estimates τ via `np.correlate`; warns if τ > sample_every |
| `run(n_equil, n_samples, sample_every)` | Burn-in + production sampling; returns M array |
| `detect_critical_beta(beta_range)` | Scans β, returns β_c where χ = N·Var(M) peaks |
| `susceptibility(samples)` | χ = N·Var(M) — peaks at phase transition |
| `get_spin_counts()` | Returns {buy, neutral, sell} counts |
| `get_network_plotly()` | Plotly network figure coloured by spin state |

**Module-level functions:**

| Function | Description |
|----------|-------------|
| `_evaluate_beta(args)` | Picklable worker for `ProcessPoolExecutor` |
| `calibrate_beta(...)` | Parallel β sweep, returns (best_beta, best_corr, curve_data) |
| `magnetization_to_signal(mag_mean, mag_std, mag_samples)` | M → BULLISH/BEARISH/NEUTRAL + confidence |

**Key implementation details:**
- `self._adj` is a `scipy.sparse.csr_array` built via `nx.to_scipy_sparse_array()`
- Sweep precomputes `self._adj @ self.spins` once (O(E) sparse matmul) instead of per-node for-loops
- `calibrate_beta` uses `ProcessPoolExecutor` with `as_completed`; falls back to sequential if process pool fails (safe inside Streamlit)
- `magnetization_to_signal` uses P25/P75 adaptive thresholds when `mag_samples` is provided; falls back to hardcoded ±0.12

---

### `data_fetcher.py`

**Functions:**

| Function | Description |
|----------|-------------|
| `fetch_stock_data(ticker, period)` | yfinance OHLCV, returns DataFrame or None |
| `compute_returns(price_series, normalise)` | Log returns, z-scored by default |
| `stock_summary(hist)` | Price stats: current, 7d/30d change, volatility, volume |
| `fetch_google_trends(keyword, timeframe, retries)` | pytrends with retry/backoff; returns pd.Series or None |
| `trends_to_sentiment_history(series, window)` | Rolling h scores ∈ [-1,1] from trends series |
| `current_sentiment_score(series, window)` | Latest h score |
| `mock_sentiment_history(n, base, noise, seed)` | Ornstein-Uhlenbeck synthetic sentiment (fallback) |
| `derive_sentiment_from_returns(returns, smoothing)` | Momentum-based sentiment proxy when trends unavailable |
| `get_stock_and_sentiment(ticker, keyword, period, use_trends)` | High-level pipeline: returns stock + sentiment data dict |

**Sentiment score formula:**
```
h_t = clip( (mean[t-7:t] - mean[0:t-7]) / (mean[0:t-7] + ε), -1, 1 )
```
Positive h → search interest surging → bullish external pressure
Negative h → search interest fading → bearish external pressure

**Data source priority:**
1. Google Trends (pytrends) — primary, weekly granularity
2. Price momentum (`derive_sentiment_from_returns`) — fallback if trends rate-limited
3. Ornstein-Uhlenbeck mock (`mock_sentiment_history`) — last resort for testing

**Extensible to:**
- Reddit PRAW (`r/wallstreetbets`, `r/investing`, `r/stocks`)
- X (Twitter) API v2 filtered stream
- FinBERT NLP on news headlines
- Instagram Graph API hashtag trends

---

### `app.py`

**Streamlit dashboard with 4 analysis tabs:**

| Tab | Content |
|-----|---------|
| 🔬 Ising Network Visualisation | Live Plotly force-graph coloured by spin state |
| 📈 Magnetisation Dynamics | M time series with buy/sell threshold lines |
| ⚙️ β Calibration | ρ(M, returns) vs β curve with calibrated β starred |
| 📊 Spin & Confidence Breakdown | Stacked spin distribution + confidence bar chart |

**Sidebar controls:**
- Stock multiselect (NVDA, TSLA, AAPL, META, AMZN, MSFT, GOOGL, AMD) + custom ticker input
- `N` investors (50–300, default 120)
- `D` anisotropy (0.0–1.0, default 0.25)
- β calibration grid points (6/8/10/12/15, default 8)
- Google Trends toggle

**Caching:** `@st.cache_data(ttl=1800)` — results cached 30 min per (ticker, params) combo.
Do NOT store non-serialisable objects (e.g. model instances) inside the cached dict — the `_model` key was removed from cache output; models are rebuilt in-place for visualisation.

**Export:** JSON download button in expandable section at page bottom.

---

## Dependencies

```
streamlit>=1.32.0      # Dashboard framework
yfinance>=0.2.40       # Stock price data (Yahoo Finance)
pytrends>=4.9.2        # Google Trends API wrapper
networkx>=3.2.0        # Graph construction (BA network, spring layout)
numpy>=1.26.0          # Numerical core
pandas>=2.2.0          # Data manipulation
plotly>=5.20.0         # Interactive charts
scipy>=1.12.0          # Sparse matrices, Pearson correlation
```

---

## Physics Engine Enhancement History

### v1 — Initial Build
- Basic Blume-Capel model with Python for-loop neighbour sums
- Sequential β calibration
- Hardcoded ±0.12 signal thresholds
- No autocorrelation diagnostics

### v2 — 5 Physics Improvements (current)

| # | Improvement | Technical Detail |
|---|-------------|-----------------|
| 1 | **Vectorised sweeps** | `self._adj = nx.to_scipy_sparse_array()` + `self._adj @ self.spins` precomputation; ~10× faster |
| 2 | **Parallel β calibration** | `ProcessPoolExecutor` + module-level `_evaluate_beta` (picklable); sequential fallback |
| 3 | **Autocorrelation diagnostics** | `_autocorrelation_time()` via `np.correlate`; `warnings.warn` when τ > `sample_every` |
| 4 | **Phase transition detector** | `detect_critical_beta()` scans χ = N·Var(M); returns β_c and full scan curve |
| 5 | **Adaptive signal thresholds** | `magnetization_to_signal(mag_samples=...)` uses P25/P75 percentiles instead of hardcoded ±0.12 |

---

## Roadmap — Next Steps

### High Priority
- [ ] **Backtesting engine** — walk-forward replay of historical signals; report hit rate, Sharpe ratio, max drawdown
- [ ] **Beta cache persistence** — save calibrated β to `beta_cache.json` (TTL 24h); skip recalibration when fresh
- [ ] **Reddit PRAW integration** — daily sentiment from `r/wallstreetbets` + `r/investing` via keyword scoring

### Medium Priority
- [ ] **Market regime indicator** — use average susceptibility across stocks to flag HIGH/LOW contagion risk
- [ ] **Multi-asset sector coupling** — block off-diagonal Hamiltonian term for inter-stock contagion (e.g. NVDA ↔ AMD)
- [ ] **EWM sentiment smoothing** — replace rolling mean with `pd.Series.ewm(span=7/30)` in `trends_to_sentiment_history`
- [ ] **FinBERT news sentiment** — NLP on financial headlines as a third sentiment source

### Production / Scale
- [ ] **FastAPI backend** — REST endpoints for prediction triggers, signal retrieval, β history, WebSocket magnetisation stream
- [ ] **React + D3.js frontend** — animated network, sector heatmap, equity curve
- [ ] **PostgreSQL β history** — store calibrated β per ticker with timestamps; flag regime changes
- [ ] **Docker deployment** — containerise full stack

---

## Key Design Decisions

**Why Blume-Capel (3-state) over standard Ising (2-state)?**
Real investors aren't binary. The neutral state σ=0 represents the large population of passive/undecided market participants. The D parameter controls how much energy it costs to leave the neutral state — high D means investors need strong peer pressure or sentiment signals to take a position.

**Why Barabási–Albert scale-free network?**
Social media follower graphs follow a power-law degree distribution. A few high-follower accounts influence many others. BA graphs reproduce this topology, making hub nodes natural proxies for market influencers like institutional accounts or high-follower traders.

**Why calibrate β per stock?**
β captures how strongly investors in a given stock's community copy each other. NVDA (strong retail community, high contagion) will calibrate to a different β than a low-profile mid-cap. Hardcoding β would lose all stock-specificity.

**Why Google Trends as the external field h?**
Google search volume is a real-time, unfiltered proxy for public attention. Surging searches for "NVIDIA" ahead of earnings → h > 0 → bullish external pressure on the network. It's free, no API key needed, and correlates strongly with retail investor attention.

**Why ProcessPoolExecutor instead of GPU?**
The Metropolis algorithm is a sequential Markov chain — step N depends on step N-1, so the inner sweep can't be trivially parallelised onto a GPU. The outer β sweep (each β value is independent) is embarrassingly parallel and maps cleanly to a process pool. For N ≤ 300 investors, CPU is fast enough; GPU would only help at N ≥ 10,000+.

---

## GitHub

Repository: https://github.com/si288214/zerve-ai-hackathon
Owner: Luke (si288214)
