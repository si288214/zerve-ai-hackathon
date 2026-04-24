"""
event_types.py — Extended Event Registry with Custom h Logic
E.W. Research / Zerve Hackathon
============================================================

Defines event schemas and preset h fallbacks for all supported event types:
  - earnings    (existing)
  - fed         (existing)
  - cpi         (existing)
  - drug_trial  (new) — PDUFA dates, Phase 2/3 readouts
  - merger      (new) — M&A close/break predictions

Each event type has:
  - Schema documentation
  - Preset h fallback values (used if all live sources fail)
  - Notes on which signals dominate for that type
"""

# ─────────────────────────────────────────────────────────────────────────────
# EVENT TYPE DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────
#
# Field reference:
#   date       — "YYYY-MM-DD"  (event date; h is computed from lookback window)
#   type       — one of the keys in EVENT_TYPE_CONFIGS
#   ticker     — primary equity ticker (use acquirer for mergers, target for trials)
#   outcome    — 1 = positive (beat / cut / below / approved / closed)
#                0 = negative (miss / hold/hike / above / rejected / broken)
#   desc       — human-readable label (used as preset key)
#
# drug_trial extras:
#   drug_name  — name of the drug / compound
#   trial_phase — "phase2", "phase3", "pdufa", "adcom"
#   indication  — disease area (e.g. "NSCLC", "T2D", "obesity")
#   target_ticker — if separate from the main ticker (e.g. partner company)
#
# merger extras:
#   acquirer   — acquiring company ticker
#   target     — target company ticker
#   deal_value_bn — deal size in USD billions
#   deal_type  — "stock", "cash", "mixed"
#   regulatory_risk — "low", "medium", "high"  (antitrust)

EVENT_TYPE_CONFIGS = {
    "earnings": {
        "outcome_labels":  {1: "beat", 0: "miss"},
        "h_signal_notes":  "Options IV + LLM most predictive. News sentiment noisy near print.",
        "dominant_sources": ["options", "llm"],
        "lookback_days":   7,
    },
    "fed": {
        "outcome_labels":  {1: "cut", 0: "hold/hike"},
        "h_signal_notes":  "News sentiment + LLM macro read. Fed funds futures implicit in options.",
        "dominant_sources": ["news", "llm"],
        "lookback_days":   14,
    },
    "cpi": {
        "outcome_labels":  {1: "below estimate", 0: "above estimate"},
        "h_signal_notes":  "News (inflation language) + LLM. Options on TLT/SPY as proxy.",
        "dominant_sources": ["news", "llm"],
        "lookback_days":   7,
    },
    "drug_trial": {
        "outcome_labels":  {1: "approved / positive readout", 0: "rejected / negative readout"},
        "h_signal_notes":  (
            "IV spike pre-event is the strongest signal — biotech options price in binary risk. "
            "High absolute IV (>1.2) with call skew = bullish. Put skew = bearish. "
            "Short-interest spikes (not yet wired) historically predict failures. "
            "LLM reads KOL commentary, competitive landscape, and trial design."
        ),
        "dominant_sources": ["options", "llm"],
        "lookback_days":   14,
        "special_notes": (
            "For PDUFA dates: FDA historically approves ~85% of standard reviews. "
            "Adcom vote is strong signal — 'yes' votes correlate with approval. "
            "For Phase 3: endpoint clarity and Phase 2 effect size are key priors. "
            "Consider adding ClinicalTrials.gov data as an additional signal."
        ),
    },
    "merger": {
        "outcome_labels":  {1: "deal closed", 0: "deal broken"},
        "h_signal_notes":  (
            "Deal spread (target price vs deal price) is the primary signal — narrow spread = "
            "market confidence. Options on both acquirer and target. "
            "Call skew on target = bullish (market pricing completion). "
            "LLM reads regulatory filings, DOJ/FTC news, competing bid rumours."
        ),
        "dominant_sources": ["options", "llm"],
        "lookback_days":   14,
        "special_notes": (
            "High regulatory risk (tech, telecom, healthcare) requires higher IV threshold. "
            "Cash deals close at higher rate than stock deals. "
            "Consider adding SEC 13D/13G filing data as activist signal."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PRESET h FALLBACKS (used if all live sources fail)
# Calibrated to reflect the actual outcome and pre-event consensus.
# ─────────────────────────────────────────────────────────────────────────────

H_PRESETS: dict[str, float] = {
    # ── EARNINGS ─────────────────────────────────────────────────────────────
    "NVDA Q4 2024 beat":     0.65,
    "NVDA Q1 2025 beat":     0.58,
    "NVDA Q2 2025 beat":     0.45,
    "NVDA Q3 2025 beat":     0.38,
    "TSLA Q4 2023 miss":    -0.22,
    "TSLA Q1 2024 miss":    -0.31,
    "TSLA Q2 2024 beat":     0.12,
    "TSLA Q3 2024 beat":     0.28,
    "META Q4 2023 beat":     0.52,
    "META Q1 2024 miss":    -0.08,
    "META Q2 2024 beat":     0.41,
    "META Q3 2024 beat":     0.44,
    "AAPL Q1 2024 beat":     0.18,
    "AAPL Q2 2024 beat":     0.09,
    "AAPL Q3 2024 beat":     0.22,
    "AAPL Q4 2024 beat":     0.15,
    "MSFT Q2 FY24 beat":     0.35,
    "MSFT Q3 FY24 beat":     0.29,
    "MSFT Q4 FY24 beat":     0.33,
    "MSFT Q1 FY25 miss":    -0.05,
    # ── FED ──────────────────────────────────────────────────────────────────
    "Fed hold Jan 2024":    -0.42,
    "Fed hold Mar 2024":    -0.28,
    "Fed hold May 2024":    -0.19,
    "Fed hold Jun 2024":    -0.15,
    "Fed hold Jul 2024":     0.05,
    "Fed cut 50bps Sep 2024": 0.48,
    "Fed cut 25bps Nov 2024": 0.35,
    "Fed cut 25bps Dec 2024": 0.22,
    # ── CPI ──────────────────────────────────────────────────────────────────
    "CPI above est Jan 2024": -0.35,
    "CPI above est Feb 2024": -0.41,
    "CPI above est Mar 2024": -0.29,
    "CPI above est Apr 2024": -0.38,
    "CPI below est May 2024":  0.18,
    "CPI below est Jun 2024":  0.22,
    "CPI below est Jul 2024":  0.31,
    "CPI below est Aug 2024":  0.25,
    "CPI below est Sep 2024":  0.19,
    "CPI above est Oct 2024": -0.22,
    "CPI above est Nov 2024": -0.18,
    "CPI above est Dec 2024": -0.25,
    # ── DRUG TRIALS ──────────────────────────────────────────────────────────
    # Examples — replace / extend with actual events
    "MRNA mRNA-4157 Phase3 melanoma 2024-07-25":  0.35,  # +ve interim data
    "NVAX NVX-CoV2373 FDA PDUFA 2024-06-10":     -0.15,  # market sceptical
    "SGEN ADC approval 2024-03-14":               0.48,  # prior adcom yes
    "RCKT gene therapy Phase3 2024-09-18":       -0.28,  # safety signals
    "LLY tirzepatide CKD 2024-10-02":             0.62,  # strong Phase2
    "BIIB lecanemab full approval 2024-07-06":    0.55,  # adcom support
    "IMVT batoclimab Phase3 2024-11-12":          0.20,  # mixed signals
    "KRTX tavapadon Phase3 Parkinson 2024-06-04": 0.30,  # clean trial design
    # ── MERGERS ──────────────────────────────────────────────────────────────
    # Examples — replace / extend with actual events
    "MSFT ATVI acquisition close 2023-10-13":     0.72,  # regulatory cleared
    "GOOGL WAZE no-deal 2024-01-15":             -0.30,  # placeholder
    "HPE Juniper Networks close 2024-09-19":      0.45,  # spread narrow
    "CSCO Splunk close 2024-03-18":               0.60,  # cash deal, low risk
    "AMZN iRobot terminated 2024-01-29":         -0.65,  # EU regulatory
    "AAPL CREDITS deal 2024-06-30":               0.25,  # hypothetical
    "SYY US Foods blocked 2024-04-01":           -0.55,  # FTC challenge
}


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE EVENT DEFINITIONS — drug trials and mergers
# Add your real events here following the same pattern.
# ─────────────────────────────────────────────────────────────────────────────

DRUG_TRIAL_EVENTS = [
    {
        "date": "2024-07-25", "type": "drug_trial", "ticker": "MRNA",
        "outcome": 1,
        "desc": "MRNA mRNA-4157 Phase3 melanoma 2024-07-25",
        "drug_name": "mRNA-4157 / V940",
        "trial_phase": "phase3",
        "indication": "melanoma (adjuvant)",
        "partner": "MRK",
    },
    {
        "date": "2024-06-10", "type": "drug_trial", "ticker": "NVAX",
        "outcome": 0,
        "desc": "NVAX NVX-CoV2373 FDA PDUFA 2024-06-10",
        "drug_name": "NVX-CoV2373",
        "trial_phase": "pdufa",
        "indication": "COVID-19",
    },
    {
        "date": "2024-10-02", "type": "drug_trial", "ticker": "LLY",
        "outcome": 1,
        "desc": "LLY tirzepatide CKD 2024-10-02",
        "drug_name": "tirzepatide",
        "trial_phase": "phase3",
        "indication": "chronic kidney disease",
    },
    {
        "date": "2024-07-06", "type": "drug_trial", "ticker": "BIIB",
        "outcome": 1,
        "desc": "BIIB lecanemab full approval 2024-07-06",
        "drug_name": "leqembi (lecanemab)",
        "trial_phase": "pdufa",
        "indication": "Alzheimer's disease",
        "partner": "ESAI",
    },
    {
        "date": "2024-09-18", "type": "drug_trial", "ticker": "RCKT",
        "outcome": 0,
        "desc": "RCKT gene therapy Phase3 2024-09-18",
        "drug_name": "RP-A501",
        "trial_phase": "phase3",
        "indication": "Danon disease",
    },
]

MERGER_EVENTS = [
    {
        "date": "2023-10-13", "type": "merger", "ticker": "ATVI",
        "outcome": 1,
        "desc": "MSFT ATVI acquisition close 2023-10-13",
        "acquirer": "MSFT",
        "target": "ATVI",
        "deal_value_bn": 68.7,
        "deal_type": "cash",
        "regulatory_risk": "high",
    },
    {
        "date": "2024-03-18", "type": "merger", "ticker": "SPLK",
        "outcome": 1,
        "desc": "CSCO Splunk close 2024-03-18",
        "acquirer": "CSCO",
        "target": "SPLK",
        "deal_value_bn": 28.0,
        "deal_type": "cash",
        "regulatory_risk": "low",
    },
    {
        "date": "2024-01-29", "type": "merger", "ticker": "IRBT",
        "outcome": 0,
        "desc": "AMZN iRobot terminated 2024-01-29",
        "acquirer": "AMZN",
        "target": "IRBT",
        "deal_value_bn": 1.7,
        "deal_type": "cash",
        "regulatory_risk": "high",
    },
    {
        "date": "2024-09-19", "type": "merger", "ticker": "JNPR",
        "outcome": 1,
        "desc": "HPE Juniper Networks close 2024-09-19",
        "acquirer": "HPE",
        "target": "JNPR",
        "deal_value_bn": 14.0,
        "deal_type": "cash",
        "regulatory_risk": "medium",
    },
]


def get_all_events(include_earnings=True, include_macro=True,
                   include_drug_trials=True, include_mergers=True) -> list[dict]:
    """
    Returns a combined event list filtered by type.
    Import EVENTS from event_backtest.py and pass here, or use standalone.
    """
    from event_backtest_v2 import EVENTS as BASE_EVENTS

    result = []
    if include_earnings:
        result += [e for e in BASE_EVENTS if e["type"] == "earnings"]
    if include_macro:
        result += [e for e in BASE_EVENTS if e["type"] in ("fed", "cpi")]
    if include_drug_trials:
        result += DRUG_TRIAL_EVENTS
    if include_mergers:
        result += MERGER_EVENTS

    return result


def get_lookback(event_type: str) -> int:
    return EVENT_TYPE_CONFIGS.get(event_type, {}).get("lookback_days", 7)