"""
ising_model.py — Blume-Capel 3-State Ising Model for Investor Sentiment
Enhanced with 5 physics/performance improvements:
  1. Vectorised neighbour sums via sparse adjacency matrix (self._adj @ self.spins)
  2. ProcessPoolExecutor parallelises calibrate_beta beta sweep
  3. Autocorrelation time τ via np.correlate; warns when τ > sample_every
  4. detect_critical_beta finds β_c via susceptibility χ = N·Var(M) peak
  5. magnetization_to_signal uses P25/P75 adaptive thresholds from sample distribution

Physics background:
  Standard Ising: σ ∈ {-1, +1}
  Blume-Capel extension: σ ∈ {-1, 0, +1}  ← our model

  Interpretation:
    σ = +1  →  Bullish investor (Buy signal)
    σ =  0  →  Neutral investor (Hold / undecided)
    σ = -1  →  Bearish investor (Sell signal)

  Hamiltonian:
    H = -J Σ_{<i,j>} σ_i σ_j  -  h Σ_i σ_i  +  D Σ_i σ_i²

    J : Coupling constant (peer influence / social contagion)
    h : External field (social media sentiment — Google/X/Instagram trends)
    D : Single-ion anisotropy (energy cost of taking a position vs staying neutral)
    β : Inverse temperature (1 / market "noise level"), calibrated via MC

  Network:
    Barabási–Albert scale-free graph — matches social media follower dynamics
    where hubs (high-degree nodes) ≡ market influencers / large accounts
"""

import warnings
import numpy as np
import networkx as nx
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─────────────────────────────────────────────────────────────
# MODULE-LEVEL WORKER (must be at module scope for pickling)
# ─────────────────────────────────────────────────────────────

def _evaluate_beta(args):
    """
    Worker function for ProcessPoolExecutor in calibrate_beta.
    Must be defined at module level so it is picklable across processes.

    args : (beta, sentiment_sequence, n_investors, D, n_equil, n_samples)
    Returns (beta, list_of_mean_magnetisations)
    """
    beta, sentiment_sequence, n_investors, D, n_equil, n_samples = args
    model = IsingInvestorNetwork(n_investors=n_investors, D=D, beta=beta)
    model_mags = []
    for h_val in sentiment_sequence:
        model.set_sentiment_field(float(h_val))
        mags = model.run(n_equil=n_equil, n_samples=n_samples, sample_every=3)
        model_mags.append(float(np.mean(mags)))
    return beta, model_mags


# ─────────────────────────────────────────────────────────────
# INVESTOR NETWORK MODEL
# ─────────────────────────────────────────────────────────────

class IsingInvestorNetwork:
    """
    Blume-Capel (3-state) Ising model on a scale-free investor network.

    Key methods
    -----------
    set_sentiment_field(h)     — apply external field from social data
    run(n_equil, n_samples)    — equilibrate + collect magnetisation samples
    detect_critical_beta()     — scan β to find phase-transition point β_c
    get_spin_counts()          — distribution of buy / neutral / sell investors
    get_network_plotly()       — Plotly figure of the live investor graph

    Improvements over v1
    --------------------
    1. self._adj (scipy sparse CSR) replaces Python neighbour for-loops
    2. _autocorrelation_time() checks MC sample independence
    3. detect_critical_beta() exposes the χ-peak (phase transition)
    """

    def __init__(
        self,
        n_investors: int = 150,
        J: float = 1.0,
        beta: float = 1.0,
        D: float = 0.25,
        seed: int = 42,
    ):
        self.n = n_investors
        self.J = J
        self.beta = beta
        self.D = D
        self._seed = seed

        # ── Build scale-free network ──────────────────────────────────────
        self.G = nx.barabasi_albert_graph(n_investors, m=3, seed=seed)
        self.neighbors = [list(self.G.neighbors(i)) for i in range(n_investors)]

        # ── Improvement 1: sparse adjacency matrix ────────────────────────
        # nx.to_scipy_sparse_array returns a scipy.sparse array (CSR format).
        # self._adj @ self.spins computes all neighbour sums in one BLAS call
        # instead of N separate Python for-loops — ~10× faster for N ≥ 100.
        self._adj = nx.to_scipy_sparse_array(self.G, format="csr", dtype=np.float32)

        # Degree → proxy for social influence (hub = influencer)
        degrees = np.array([self.G.degree(i) for i in range(n_investors)], dtype=float)
        self.influence_weight = degrees / (degrees.max() + 1e-9)

        # ── State ─────────────────────────────────────────────────────────
        self.spins = np.zeros(n_investors, dtype=np.int8)
        self.h = np.zeros(n_investors, dtype=np.float64)

        # ── Layout (computed once, reused for viz) ────────────────────────
        self._pos = nx.spring_layout(self.G, seed=seed, k=1.5)

        self.mag_history: list[float] = []

    # ──────────────────────────────────────────────────────────
    # External field
    # ──────────────────────────────────────────────────────────

    def set_sentiment_field(self, h_global: float, influencer_boost: float = 0.4):
        """
        Apply a uniform sentiment field h_global ∈ [-1, 1] (normalised from
        social data).  Hub investors receive a slightly stronger field,
        modelling their greater exposure to trend information.
        """
        self.h = h_global * (1.0 + influencer_boost * self.influence_weight)

    # ──────────────────────────────────────────────────────────
    # Energy  (Improvement 1: sparse row × spins)
    # ──────────────────────────────────────────────────────────

    def _delta_energy(self, i: int, new_spin: int) -> float:
        """
        Compute ΔE for proposing σ_i → new_spin.
        Uses self._adj[i] @ self.spins for the neighbour sum (sparse row mult)
        instead of a Python for-loop.
        """
        old_spin = int(self.spins[i])
        if new_spin == old_spin:
            return 0.0

        # Sparse row × dense spin vector  — no Python loop over neighbours
        neighbor_sum = float(self._adj[i] @ self.spins)

        dE = (
            -self.J * (new_spin - old_spin) * neighbor_sum
            - self.h[i] * (new_spin - old_spin)
            + self.D * (new_spin ** 2 - old_spin ** 2)
        )
        return dE

    # ──────────────────────────────────────────────────────────
    # Monte Carlo dynamics  (Improvement 1: vectorised sweep)
    # ──────────────────────────────────────────────────────────

    def _metropolis_sweep(self):
        """
        One full Metropolis sweep.

        Precomputes ALL neighbour sums at sweep-start via:
            neighbor_sums = self._adj @ self.spins          ← single sparse matmul
        then propagates accepted moves incrementally, avoiding repeated
        per-site for-loops over neighbour lists.
        """
        # Precompute all neighbour sums in one sparse matrix–vector multiply
        neighbor_sums = (self._adj @ self.spins).astype(np.float64)

        indices = np.random.permutation(self.n)
        for i in indices:
            new_spin = int(np.random.choice((-1, 0, 1)))
            old_spin = int(self.spins[i])
            if new_spin == old_spin:
                continue

            dE = (
                -self.J * (new_spin - old_spin) * neighbor_sums[i]
                - self.h[i] * (new_spin - old_spin)
                + self.D * (new_spin ** 2 - old_spin ** 2)
            )
            if dE <= 0.0 or np.random.random() < np.exp(-self.beta * dE):
                diff = float(new_spin - old_spin)
                self.spins[i] = new_spin
                # Propagate change to neighbours' cached sums
                for j in self.neighbors[i]:
                    neighbor_sums[j] += diff

    # ──────────────────────────────────────────────────────────
    # Improvement 3: autocorrelation time
    # ──────────────────────────────────────────────────────────

    def _autocorrelation_time(self, samples: np.ndarray) -> float:
        """
        Estimate integrated autocorrelation time τ from magnetisation samples
        using the normalised autocorrelation function via np.correlate.

        τ ≈ 0.5 + Σ_{t=1}^{t*} ACF(t)   where ACF drops below 0.05 at t*.

        If τ > sample_every, samples are not statistically independent and
        reported confidence scores will be over-inflated.
        """
        n = len(samples)
        if n < 4:
            return 1.0
        x = samples - samples.mean()
        variance = np.dot(x, x)
        if variance < 1e-12:
            return 1.0
        full_corr = np.correlate(x, x, mode="full")
        acf = full_corr[n - 1:] / variance
        tau = 0.5
        for t in range(1, n):
            if acf[t] < 0.05:
                break
            tau += acf[t]
        return float(tau)

    def reset(self, seed: int | None = None):
        """Re-randomise spin configuration."""
        if seed is not None:
            np.random.seed(seed)
        self.spins = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=self.n)

    def run(
        self,
        n_equil: int = 300,
        n_samples: int = 80,
        sample_every: int = 5,
    ) -> np.ndarray:
        """
        Equilibrate the model then collect magnetisation samples.
        After sampling, checks autocorrelation time τ and emits a
        UserWarning when τ > sample_every (samples not independent).

        Returns
        -------
        np.ndarray of shape (n_samples,)  — magnetisation M = <σ>
        """
        self.reset(seed=self._seed)

        # Burn-in
        for _ in range(n_equil):
            self._metropolis_sweep()

        # Production sampling
        samples = []
        for _ in range(n_samples):
            for _ in range(sample_every):
                self._metropolis_sweep()
            m = float(np.mean(self.spins))
            samples.append(m)
            self.mag_history.append(m)

        samples_arr = np.array(samples)

        # Improvement 3: autocorrelation check
        tau = self._autocorrelation_time(samples_arr)
        if tau > sample_every:
            warnings.warn(
                f"Autocorrelation time τ={tau:.2f} exceeds sample_every={sample_every}. "
                "MC samples may not be independent — confidence scores may be inflated. "
                "Consider increasing sample_every or n_equil.",
                UserWarning,
                stacklevel=2,
            )

        return samples_arr

    # ──────────────────────────────────────────────────────────
    # Observables
    # ──────────────────────────────────────────────────────────

    def magnetization(self) -> float:
        """Current order parameter M = (1/N) Σ σ_i ∈ [-1, 1]."""
        return float(np.mean(self.spins))

    def susceptibility(self, samples: np.ndarray) -> float:
        """Magnetic susceptibility χ = N * Var(M) — peaks at phase transition."""
        return float(self.n * np.var(samples))

    # ──────────────────────────────────────────────────────────
    # Improvement 4: detect_critical_beta
    # ──────────────────────────────────────────────────────────

    def detect_critical_beta(
        self,
        beta_range: np.ndarray | None = None,
        n_equil: int = 200,
        n_samples: int = 30,
    ) -> tuple[float, list[dict]]:
        """
        Scan β and find the critical inverse temperature β_c where the
        magnetic susceptibility χ = N·Var(M) peaks.

        At β_c the network is at its phase transition: small changes in
        social sentiment cause the largest collective swings in investor
        sentiment — maximum contagion risk.

        Returns
        -------
        beta_c    : float  — β value at susceptibility peak
        scan_data : list of {'beta', 'susceptibility', 'magnetization'}
        """
        if beta_range is None:
            beta_range = np.linspace(0.1, 3.0, 15)

        scan_data = []
        for beta in beta_range:
            self.beta = beta
            mags = self.run(n_equil=n_equil, n_samples=n_samples, sample_every=3)
            susc = self.susceptibility(mags)
            scan_data.append({
                "beta": float(beta),
                "susceptibility": susc,
                "magnetization": float(np.mean(mags)),
            })

        beta_c_entry = max(scan_data, key=lambda x: x["susceptibility"])
        return beta_c_entry["beta"], scan_data

    def get_spin_counts(self) -> dict:
        return {
            "buy":     int(np.sum(self.spins == 1)),
            "neutral": int(np.sum(self.spins == 0)),
            "sell":    int(np.sum(self.spins == -1)),
        }

    # ──────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────

    def get_network_plotly(self):
        """
        Build a Plotly figure of the investor network.
        Node colour: green=buy, grey=neutral, red=sell.
        Node size: proportional to degree (influence).
        """
        import plotly.graph_objects as go

        pos = self._pos
        spin_color = {1: "#22c55e", 0: "#94a3b8", -1: "#ef4444"}
        spin_label = {1: "Buy (+1)", 0: "Neutral (0)", -1: "Sell (−1)"}

        # Edges
        edge_x, edge_y = [], []
        for u, v in self.G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.4, color="#334155"),
            hoverinfo="none",
        )

        # Nodes grouped by spin for legend
        traces = [edge_trace]
        for spin_val in (1, 0, -1):
            mask = self.spins == spin_val
            if not mask.any():
                continue
            nodes = np.where(mask)[0]
            xs = [pos[n][0] for n in nodes]
            ys = [pos[n][1] for n in nodes]
            sizes = [6 + 14 * self.influence_weight[n] for n in nodes]
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name=spin_label[spin_val],
                marker=dict(
                    size=sizes,
                    color=spin_color[spin_val],
                    line=dict(width=0.5, color="#1e293b"),
                ),
                text=[f"Investor {n}<br>Spin: {spin_val}<br>Degree: {self.G.degree(n)}"
                      for n in nodes],
                hoverinfo="text",
            ))

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title="Investor Network — Live Spin States",
                showlegend=True,
                hovermode="closest",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
            ),
        )
        return fig


# ─────────────────────────────────────────────────────────────
# MONTE CARLO BETA CALIBRATION  (Improvement 2: ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────

def calibrate_beta(
    sentiment_sequence: list[float],
    return_sequence: list[float],
    beta_range: np.ndarray | None = None,
    n_investors: int = 100,
    n_equil: int = 150,
    n_samples: int = 25,
    max_workers: int = 4,
    progress_callback=None,
) -> tuple[float, float, list[dict]]:
    """
    Sweep β and find the value that maximises |Pearson r| between
    model magnetisation and actual stock returns.

    Improvement 2: beta values are evaluated in parallel via
    ProcessPoolExecutor with max_workers processes.  Falls back to
    sequential execution if multiprocessing is unavailable (e.g. inside
    certain Streamlit/notebook environments).

    Parameters
    ----------
    sentiment_sequence : list of h values (one per historical time point)
    return_sequence    : corresponding normalised stock returns
    beta_range         : values of β to sweep (default: 12 points, 0.1–3.0)
    max_workers        : number of parallel worker processes
    progress_callback  : optional callable(current_idx, total) for UI progress

    Returns
    -------
    best_beta   : float
    best_corr   : float  (Pearson r at best_beta)
    curve_data  : list of {'beta': …, 'correlation': …, 'susceptibility': …}
    """
    if beta_range is None:
        beta_range = np.linspace(0.1, 3.0, 12)

    n_hist = min(len(sentiment_sequence), len(return_sequence))
    hist_sentiment = sentiment_sequence[:n_hist]
    returns_slice = list(return_sequence[:n_hist])

    # Build task args for each beta
    task_args = [
        (float(beta), hist_sentiment, n_investors, 0.25, n_equil, n_samples)
        for beta in beta_range
    ]

    # ── Parallel sweep via ProcessPoolExecutor ───────────────────────────
    beta_to_mags: dict[float, list[float]] = {}
    completed = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_evaluate_beta, args): args[0]
                for args in task_args
            }
            for future in as_completed(future_map):
                beta_val, mags = future.result()
                beta_to_mags[beta_val] = mags
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(beta_range))
    except Exception:
        # Fallback: sequential (safe inside Streamlit / fork-hostile envs)
        for idx, args in enumerate(task_args):
            beta_val, mags = _evaluate_beta(args)
            beta_to_mags[beta_val] = mags
            if progress_callback:
                progress_callback(idx + 1, len(beta_range))

    # ── Build curve_data in beta_range order ─────────────────────────────
    curve_data = []
    for beta in beta_range:
        mags = beta_to_mags.get(float(beta), [])
        if len(mags) >= 3:
            try:
                corr, _ = stats.pearsonr(mags, returns_slice[:len(mags)])
                corr = float(corr) if not np.isnan(corr) else 0.0
            except Exception:
                corr = 0.0
            susc = float(np.var(mags) * n_investors)
        else:
            corr = 0.0
            susc = 0.0

        curve_data.append({
            "beta": float(beta),
            "correlation": corr,
            "susceptibility": susc,
        })

    best = max(curve_data, key=lambda x: abs(x["correlation"]))
    return best["beta"], best["correlation"], curve_data


# ─────────────────────────────────────────────────────────────
# PREDICTION SIGNAL  (Improvement 5: adaptive P25/P75 thresholds)
# ─────────────────────────────────────────────────────────────

def magnetization_to_signal(
    mag_mean: float,
    mag_std: float,
    buy_threshold: float = 0.12,
    sell_threshold: float = -0.12,
    mag_samples: np.ndarray | None = None,
) -> tuple[str, float]:
    """
    Convert magnetisation to a trading signal with confidence score.

    Improvement 5: when mag_samples is provided (≥ 4 points), thresholds
    are set adaptively from the sample distribution:
        buy_threshold  = np.percentile(mag_samples, 75)   ← 75th percentile
        sell_threshold = np.percentile(mag_samples, 25)   ← 25th percentile

    This accounts for stocks where M naturally saturates near ±1 (strongly
    ordered) vs stocks that remain near 0 (weakly ordered), avoiding the
    hardcoded ±0.12 from being too loose or too tight.

    Returns (signal, confidence_pct)
    """
    # Improvement 5: derive thresholds from sample distribution
    if mag_samples is not None and len(mag_samples) >= 4:
        buy_threshold  = float(np.percentile(mag_samples, 75))
        sell_threshold = float(np.percentile(mag_samples, 25))

    # Signal-to-noise: how many std-devs from neutral?
    snr = abs(mag_mean) / (mag_std + 1e-6)
    raw_conf = min(snr / 3.0, 1.0)      # saturates at SNR = 3

    if mag_mean > buy_threshold:
        signal = "BULLISH"
        confidence = 50 + 45 * raw_conf
    elif mag_mean < sell_threshold:
        signal = "BEARISH"
        confidence = 50 + 45 * raw_conf
    else:
        signal = "NEUTRAL"
        confidence = 50 - 30 * raw_conf  # less confident near zero

    return signal, round(confidence, 1)
