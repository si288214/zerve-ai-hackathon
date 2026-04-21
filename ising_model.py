"""
ising_model.py — Blume-Capel 3-State Ising Model for Investor Sentiment

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

import numpy as np
import networkx as nx
from scipy import stats


# ─────────────────────────────────────────────────────────────
# INVESTOR NETWORK MODEL
# ─────────────────────────────────────────────────────────────

class IsingInvestorNetwork:
    """
    Blume-Capel (3-state) Ising model on a scale-free investor network.

    Key methods
    -----------
    set_sentiment_field(h)   — apply external field from social data
    run(n_equil, n_samples)  — equilibrate + collect magnetization samples
    get_spin_counts()        — distribution of buy / neutral / sell investors
    get_network_plotly()     — Plotly figure of the live investor graph
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
    # Energy
    # ──────────────────────────────────────────────────────────

    def _delta_energy(self, i: int, new_spin: int) -> float:
        """Compute ΔE for proposing σ_i → new_spin (Metropolis criterion)."""
        old_spin = int(self.spins[i])
        if new_spin == old_spin:
            return 0.0

        neighbor_sum = sum(int(self.spins[j]) for j in self.neighbors[i])

        dE = (
            -self.J * (new_spin - old_spin) * neighbor_sum
            - self.h[i] * (new_spin - old_spin)
            + self.D * (new_spin ** 2 - old_spin ** 2)
        )
        return dE

    # ──────────────────────────────────────────────────────────
    # Monte Carlo dynamics
    # ──────────────────────────────────────────────────────────

    def _metropolis_sweep(self):
        """One full Metropolis sweep (N random spin-update attempts)."""
        indices = np.random.permutation(self.n)
        for i in indices:
            new_spin = np.random.choice((-1, 0, 1))
            dE = self._delta_energy(int(i), int(new_spin))
            if dE <= 0.0 or np.random.random() < np.exp(-self.beta * dE):
                self.spins[i] = new_spin

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
        Equilibrate the model, then collect magnetization samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)  — magnetization M = <σ>
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

        return np.array(samples)

    # ──────────────────────────────────────────────────────────
    # Observables
    # ──────────────────────────────────────────────────────────

    def magnetization(self) -> float:
        """Current order parameter M = (1/N) Σ σ_i ∈ [-1, 1]."""
        return float(np.mean(self.spins))

    def susceptibility(self, samples: np.ndarray) -> float:
        """Magnetic susceptibility χ = N * Var(M)  — peaks at phase transition."""
        return float(self.n * np.var(samples))

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

        # Nodes (grouped by spin for legend)
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
# MONTE CARLO BETA CALIBRATION
# ─────────────────────────────────────────────────────────────

def calibrate_beta(
    sentiment_sequence: list[float],
    return_sequence: list[float],
    beta_range: np.ndarray | None = None,
    n_investors: int = 100,
    n_equil: int = 150,
    n_samples: int = 25,
    progress_callback=None,
) -> tuple[float, float, list[dict]]:
    """
    Sweep β and find the value that maximises |Pearson r| between
    model magnetisation and actual stock returns.

    Parameters
    ----------
    sentiment_sequence : list of h values (one per historical time point)
    return_sequence    : corresponding normalised stock returns
    beta_range         : values of β to sweep (default: 12 points, 0.1–3.0)
    progress_callback  : optional callable(current_idx, total) for UI progress

    Returns
    -------
    best_beta   : float
    best_corr   : float  (Pearson r at best_beta)
    curve_data  : list of {'beta': …, 'correlation': …, 'susceptibility': …}
    """
    if beta_range is None:
        beta_range = np.linspace(0.1, 3.0, 12)

    model = IsingInvestorNetwork(n_investors=n_investors, D=0.25)
    curve_data = []
    n_hist = min(len(sentiment_sequence), len(return_sequence))

    for idx, beta in enumerate(beta_range):
        model.beta = beta
        model_mags = []

        for h_val in sentiment_sequence[:n_hist]:
            model.set_sentiment_field(float(h_val))
            mags = model.run(n_equil=n_equil, n_samples=n_samples, sample_every=3)
            model_mags.append(float(np.mean(mags)))

        returns_slice = return_sequence[:n_hist]

        if len(model_mags) >= 3:
            try:
                corr, _ = stats.pearsonr(model_mags, returns_slice)
                corr = float(corr) if not np.isnan(corr) else 0.0
            except Exception:
                corr = 0.0
            susc = float(np.var(model_mags) * n_investors)
        else:
            corr = 0.0
            susc = 0.0

        curve_data.append({
            "beta": float(beta),
            "correlation": corr,
            "susceptibility": susc,
        })

        if progress_callback:
            progress_callback(idx + 1, len(beta_range))

    best = max(curve_data, key=lambda x: abs(x["correlation"]))
    return best["beta"], best["correlation"], curve_data


# ─────────────────────────────────────────────────────────────
# PREDICTION SIGNAL
# ─────────────────────────────────────────────────────────────

def magnetization_to_signal(
    mag_mean: float,
    mag_std: float,
    buy_threshold: float = 0.12,
    sell_threshold: float = -0.12,
) -> tuple[str, float]:
    """
    Convert magnetisation to a trading signal with confidence score.

    Returns (signal, confidence_pct)
    """
    # Signal-to-noise: how many std-devs from neutral?
    snr = abs(mag_mean) / (mag_std + 1e-6)
    raw_conf = min(snr / 3.0, 1.0)      # saturates at SNR=3

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
