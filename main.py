"""
Ising Market Model — E.W. Research
====================================
Graph-based Ising model applied to stock market dynamics.

Key mappings:
  spin +1  →  bullish investor
  spin -1  →  bearish investor
  J (coupling) →  social influence / herding strength
  h (field)    →  external signal (macro news, earnings surprise)
  T (temperature) →  market irrationality / noise

Phase transition occurs near T_c = 2J / ln(1 + sqrt(2)) ≈ 2.269J
Below T_c: coordinated herding → bubble or crash
Above T_c: disordered, noisy market
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from collections import deque

# ─────────────────────────────────────────────
# 1. GRAPH TOPOLOGY
# ─────────────────────────────────────────────

def build_market_graph(n_investors=200, graph_type="small_world", seed=42):
    """
    Build investor network.
    - small_world: Watts-Strogatz (realistic social network)
    - scale_free: Barabási-Albert (hub investors / influencers)
    - random: Erdos-Renyi baseline
    """
    rng = np.random.default_rng(seed)
    if graph_type == "small_world":
        G = nx.watts_strogatz_graph(n_investors, k=6, p=0.1, seed=seed)
    elif graph_type == "scale_free":
        G = nx.barabasi_albert_graph(n_investors, m=3, seed=seed)
    else:
        G = nx.erdos_renyi_graph(n_investors, p=0.05, seed=seed)
    return G


# ─────────────────────────────────────────────
# 2. ISING MARKET ENGINE
# ─────────────────────────────────────────────

class IsingMarket:
    def __init__(self, G, J=1.0, h=0.0, T=2.5, seed=42):
        """
        G : networkx graph of investors
        J : coupling strength (herding intensity)
        h : external field (positive = bullish macro, negative = bearish)
        T : temperature (noise / irrationality)
        """
        self.G = G
        self.J = J
        self.h = h
        self.T = T
        self.N = G.number_of_nodes()
        rng = np.random.default_rng(seed)
        self.spins = rng.choice([-1, 1], size=self.N)
        self.neighbors = {n: list(G.neighbors(n)) for n in G.nodes()}

    def local_field(self, i):
        """Net field experienced by investor i."""
        neighbor_sum = sum(self.spins[j] for j in self.neighbors[i])
        return self.J * neighbor_sum + self.h

    def metropolis_step(self):
        """Single Metropolis-Hastings update sweep."""
        for i in np.random.permutation(self.N):
            dE = 2 * self.spins[i] * self.local_field(i)
            if dE < 0 or np.random.random() < np.exp(-dE / self.T):
                self.spins[i] *= -1

    def magnetization(self):
        """Market sentiment: +1 = fully bullish, -1 = fully bearish."""
        return np.mean(self.spins)

    def order_parameter(self):
        """Absolute magnetization — measures herding regardless of direction."""
        return abs(self.magnetization())

    def simulate(self, n_steps=500, burn_in=100):
        """
        Run simulation. Returns time series of:
          - magnetization (sentiment)
          - simulated price index (cumulative sentiment)
          - susceptibility (volatility proxy)
        """
        mag_series = []
        price = 100.0
        price_series = [price]

        for step in range(n_steps):
            self.metropolis_step()
            m = self.magnetization()
            mag_series.append(m)
            # Price return ∝ net bullish signal + noise
            ret = 0.02 * m + np.random.normal(0, 0.005)
            price *= np.exp(ret)
            price_series.append(price)

        # Susceptibility ≈ variance of magnetization (volatility proxy)
        post_burn = mag_series[burn_in:]
        susceptibility = self.N * np.var(post_burn)

        return np.array(mag_series), np.array(price_series), susceptibility


# ─────────────────────────────────────────────
# 3. PHASE TRANSITION SCAN
# ─────────────────────────────────────────────

def phase_transition_scan(G, T_range=None, J=1.0, h=0.0, n_steps=300, burn_in=100, seed=42):
    """Sweep temperature and measure order parameter & susceptibility."""
    if T_range is None:
        T_range = np.linspace(0.5, 5.0, 40)

    order_params = []
    susceptibilities = []

    for T in T_range:
        market = IsingMarket(G, J=J, h=h, T=T, seed=seed)
        mags, _, chi = market.simulate(n_steps=n_steps, burn_in=burn_in)
        order_params.append(abs(np.mean(mags[burn_in:])))
        susceptibilities.append(chi)

    return T_range, np.array(order_params), np.array(susceptibilities)


# ─────────────────────────────────────────────
# 4. SHOCK PROPAGATION
# ─────────────────────────────────────────────

def shock_propagation(G, J=1.0, T=1.8, shock_step=150, shock_h=-2.0, n_steps=400, seed=42):
    """
    Simulate a negative macro shock (e.g. tariff announcement, earnings miss)
    mid-simulation. h flips negative at shock_step.
    """
    market = IsingMarket(G, J=J, h=0.0, T=T, seed=seed)
    mag_series = []
    price = 100.0
    price_series = [price]

    for step in range(n_steps):
        if step == shock_step:
            market.h = shock_h          # Shock hits
        if step == shock_step + 50:
            market.h = 0.0              # Shock fades

        market.metropolis_step()
        m = market.magnetization()
        mag_series.append(m)
        ret = 0.02 * m + np.random.normal(0, 0.005)
        price *= np.exp(ret)
        price_series.append(price)

    return np.array(mag_series), np.array(price_series)


# ─────────────────────────────────────────────
# 5. REGIME CLASSIFIER
# ─────────────────────────────────────────────

def classify_regime(magnetization_series, window=30):
    """
    Rolling regime classification based on sentiment and volatility.
    Returns: list of regime labels per timestep
    """
    regimes = []
    q = deque(maxlen=window)
    for m in magnetization_series:
        q.append(m)
        if len(q) < window:
            regimes.append("Warming Up")
            continue
        avg = np.mean(q)
        vol = np.std(q)
        if avg > 0.3 and vol < 0.15:
            regimes.append("Bubble")
        elif avg < -0.3 and vol < 0.15:
            regimes.append("Crash")
        elif vol > 0.25:
            regimes.append("High Volatility")
        elif abs(avg) < 0.1 and vol < 0.1:
            regimes.append("Efficient / Disordered")
        else:
            regimes.append("Trending")
    return regimes


# ─────────────────────────────────────────────
# 6. FULL VISUALIZATION
# ─────────────────────────────────────────────

def plot_full_analysis(graph_type="small_world"):
    G = build_market_graph(n_investors=300, graph_type=graph_type)

    fig = plt.figure(figsize=(18, 14), facecolor="#0d0d0d")
    fig.suptitle(
        f"Ising Market Model  |  {graph_type.replace('_',' ').title()} Network  |  E.W. Research",
        fontsize=16, color="#e8d5b0", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    gold = "#e8d5b0"
    red  = "#e05c5c"
    grn  = "#5ce08a"
    blue = "#5ca8e0"
    grey = "#888888"

    # ── Panel 1: Baseline simulation (T above critical) ──
    ax1 = fig.add_subplot(gs[0, :2])
    market_hot = IsingMarket(G, J=1.0, h=0.0, T=3.5)
    mags_hot, prices_hot, _ = market_hot.simulate(n_steps=400)
    ax1.plot(prices_hot, color=blue, lw=1.5, label="Price (T=3.5, disordered)")
    market_cold = IsingMarket(G, J=1.0, h=0.0, T=1.5)
    mags_cold, prices_cold, _ = market_cold.simulate(n_steps=400)
    ax1.plot(prices_cold, color=gold, lw=1.5, label="Price (T=1.5, herding)")
    ax1.set_facecolor("#111111")
    ax1.set_title("Simulated Price: Disordered vs Herding Regime", color=gold, fontsize=11)
    ax1.set_xlabel("Step", color=grey); ax1.set_ylabel("Price Index", color=grey)
    ax1.tick_params(colors=grey); ax1.legend(facecolor="#1a1a1a", labelcolor=gold, fontsize=9)
    for sp in ax1.spines.values(): sp.set_color("#333333")

    # ── Panel 2: Magnetization time series ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(mags_hot, color=blue, lw=1, alpha=0.8, label="T=3.5")
    ax2.plot(mags_cold, color=gold, lw=1, alpha=0.8, label="T=1.5")
    ax2.axhline(0, color=grey, lw=0.5, ls="--")
    ax2.set_facecolor("#111111")
    ax2.set_title("Market Sentiment (Magnetization)", color=gold, fontsize=11)
    ax2.set_xlabel("Step", color=grey); ax2.set_ylabel("⟨m⟩", color=grey)
    ax2.tick_params(colors=grey); ax2.legend(facecolor="#1a1a1a", labelcolor=gold, fontsize=9)
    for sp in ax2.spines.values(): sp.set_color("#333333")

    # ── Panel 3: Phase transition ──
    ax3 = fig.add_subplot(gs[1, :2])
    T_range, order_params, susceptibilities = phase_transition_scan(G, J=1.0)
    T_c_est = T_range[np.argmax(susceptibilities)]
    ax3_twin = ax3.twinx()
    ax3.plot(T_range, order_params, color=gold, lw=2, label="|⟨m⟩| (Order)")
    ax3_twin.plot(T_range, susceptibilities, color=red, lw=2, ls="--", label="χ (Susceptibility)")
    ax3.axvline(T_c_est, color=grn, lw=1.5, ls=":", label=f"T_c ≈ {T_c_est:.2f}")
    ax3.set_facecolor("#111111")
    ax3.set_title("Phase Transition: Herding → Disordered Market", color=gold, fontsize=11)
    ax3.set_xlabel("Temperature T (Noise/Irrationality)", color=grey)
    ax3.set_ylabel("|⟨m⟩|  Order Parameter", color=gold)
    ax3_twin.set_ylabel("χ  Susceptibility (Volatility Proxy)", color=red)
    ax3.tick_params(colors=grey); ax3_twin.tick_params(colors=red)
    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labs1+labs2, facecolor="#1a1a1a", labelcolor=gold, fontsize=9)
    for sp in ax3.spines.values(): sp.set_color("#333333")

    # ── Panel 4: Network graph (small sample) ──
    ax4 = fig.add_subplot(gs[1, 2])
    G_small = build_market_graph(n_investors=80, graph_type=graph_type)
    market_snap = IsingMarket(G_small, J=1.0, h=0.0, T=1.8)
    for _ in range(50): market_snap.metropolis_step()
    pos = nx.spring_layout(G_small, seed=42)
    spin_colors = [grn if market_snap.spins[i] == 1 else red for i in range(80)]
    nx.draw_networkx(G_small, pos=pos, ax=ax4, node_color=spin_colors,
                     node_size=30, with_labels=False, edge_color="#333333", width=0.4)
    ax4.set_facecolor("#111111")
    ax4.set_title("Investor Network\n(green=bull, red=bear)", color=gold, fontsize=10)
    for sp in ax4.spines.values(): sp.set_color("#333333")

    # ── Panel 5: Shock propagation ──
    ax5 = fig.add_subplot(gs[2, :2])
    shock_mags, shock_prices = shock_propagation(G, T=1.8, shock_step=150, shock_h=-2.5)
    regimes = classify_regime(shock_mags)
    regime_colors = {"Bubble": grn, "Crash": red, "High Volatility": "#e0a85c",
                     "Efficient / Disordered": blue, "Trending": gold, "Warming Up": grey}
    ax5.plot(shock_prices, color=gold, lw=1.5)
    ax5.axvline(150, color=red, lw=1.5, ls="--", label="Shock hits (h=−2.5)")
    ax5.axvline(200, color=grn, lw=1.5, ls="--", label="Shock fades")
    # Color background by regime
    for i, reg in enumerate(regimes):
        ax5.axvspan(i, i+1, alpha=0.08, color=regime_colors.get(reg, grey))
    ax5.set_facecolor("#111111")
    ax5.set_title("Shock Propagation: Negative Macro Event (e.g. Tariff / Earnings Miss)", color=gold, fontsize=11)
    ax5.set_xlabel("Step", color=grey); ax5.set_ylabel("Price Index", color=grey)
    ax5.tick_params(colors=grey); ax5.legend(facecolor="#1a1a1a", labelcolor=gold, fontsize=9)
    for sp in ax5.spines.values(): sp.set_color("#333333")

    # ── Panel 6: Regime distribution ──
    ax6 = fig.add_subplot(gs[2, 2])
    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    regime_counts.pop("Warming Up", None)
    labels = list(regime_counts.keys())
    vals = [regime_counts[l] for l in labels]
    bar_colors = [regime_colors.get(l, grey) for l in labels]
    ax6.barh(labels, vals, color=bar_colors, edgecolor="#222222")
    ax6.set_facecolor("#111111")
    ax6.set_title("Regime Distribution\n(Post-Shock)", color=gold, fontsize=10)
    ax6.set_xlabel("Steps", color=grey)
    ax6.tick_params(colors=grey)
    for sp in ax6.spines.values(): sp.set_color("#333333")

    plt.savefig("ising_market_analysis.png",
                dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print("Saved: ising_market_analysis.png")
    plt.show()


if __name__ == "__main__":
    print("Running Ising Market Model — E.W. Research\n")
    print("Building small-world investor network (300 nodes)...")
    G = build_market_graph(300, "small_world")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("\nBaseline simulation (T=2.5, J=1.0, h=0.0)...")
    market = IsingMarket(G, J=1.0, h=0.0, T=2.5)
    mags, prices, chi = market.simulate(n_steps=500)
    print(f"  Final sentiment: {mags[-1]:.3f}")
    print(f"  Susceptibility (vol proxy): {chi:.2f}")
    print(f"  Final price index: {prices[-1]:.2f}")

    print("\nRegime classification...")
    regimes = classify_regime(mags)
    from collections import Counter
    print("  Regime counts:", dict(Counter(regimes)))

    print("\nGenerating full analysis plot...")
    plot_full_analysis("small_world")
    print("\nDone.")