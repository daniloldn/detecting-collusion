# src/dgp/tier0.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class Tier0Config:
    """
    Configuration for the Tier 0 synthetic data generator.

    Tier 0 goal: isolate the *core conduct mechanism* only:
    - distorted pass-through (beta differs by regime)
    - slower adjustment (kappa differs by regime)
    Everything else (trend/seasonality/missingness) comes later in Tier 1/2.
    """
    T: int = 180        # number of months kept (e.g. 15 years)
    burn_in: int = 24   # simulated months to drop to remove initialization artifacts

    # Latent cost process parameters
    rho_c_low: float = 0.6
    rho_c_high: float = 0.95
    sigma_c_low: float = 0.01
    sigma_c_high: float = 0.08

    # Jump shock parameters in the cost process (rare large shocks)
    jump_prob_low: float = 0.01
    jump_prob_high: float = 0.05
    sigma_J_low: float = 0.05
    sigma_J_high: float = 0.20

    # Price noise (idiosyncratic noise around the structural relation)
    sigma_p_low: float = 0.001
    sigma_p_high: float = 0.02

    # Regime-dependent pass-through beta ranges
    beta_C: Tuple[float, float] = (0.8, 1.2)   # competitive
    beta_T: Tuple[float, float] = (0.5, 0.9)   # tacit/soft coordination
    beta_K: Tuple[float, float] = (0.2, 0.6)   # cartel

    # Regime-dependent speed-of-adjustment kappa ranges (stickiness)
    kappa_C: Tuple[float, float] = (0.4, 0.9)
    kappa_T: Tuple[float, float] = (0.2, 0.6)
    kappa_K: Tuple[float, float] = (0.05, 0.35)

    # Markov chain persistence (probability of staying in same regime)
    # Higher -> longer episodes.
    stay_C: float = 0.97
    stay_T: float = 0.97
    stay_K: float = 0.985


def _sample_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Helper: sample a float uniformly in [low, high)."""
    return float(rng.uniform(low, high))


def sample_market_params(rng: np.random.Generator, cfg: Tier0Config, mode: str = "baseline") -> Dict:
    """
    Sample one market's structural parameters.

    Key idea: markets differ (heterogeneity), so we randomize parameters per market.
    This reduces the risk that the model learns a single narrow signature.
    """
    # cost process params
    rho_c = _sample_uniform(rng, cfg.rho_c_low, cfg.rho_c_high)
    sigma_c = _sample_uniform(rng, cfg.sigma_c_low, cfg.sigma_c_high)

    # jump process params
    jump_prob = _sample_uniform(rng, cfg.jump_prob_low, cfg.jump_prob_high)
    sigma_J = _sample_uniform(rng, cfg.sigma_J_low, cfg.sigma_J_high)

    # price noise
    sigma_p = _sample_uniform(rng, cfg.sigma_p_low, cfg.sigma_p_high)

    # regime-dependent betas and kappas
    beta = {
        "C": _sample_uniform(rng, *cfg.beta_C),
        "T": _sample_uniform(rng, *cfg.beta_T),
        "K": _sample_uniform(rng, *cfg.beta_K),
    }
    kappa = {
        "C": _sample_uniform(rng, *cfg.kappa_C),
        "T": _sample_uniform(rng, *cfg.kappa_T),
        "K": _sample_uniform(rng, *cfg.kappa_K),
    }

      # --- Stress test overrides ---
    if mode == "kappa_only":
        beta_fixed = beta["C"]
        beta = {"C": beta_fixed, "T": beta_fixed, "K": beta_fixed}

    elif mode == "beta_only":
        kappa_fixed = kappa["C"]
        kappa = {"C": kappa_fixed, "T": kappa_fixed, "K": kappa_fixed}

    elif mode != "baseline":
        raise ValueError(f"Unknown stress test mode: {mode}")

    return dict(
        rho_c=rho_c,
        sigma_c=sigma_c,
        jump_prob=jump_prob,
        sigma_J=sigma_J,
        sigma_p=sigma_p,
        beta=beta,
        kappa=kappa,
    )


def simulate_regime_path(rng: np.random.Generator, cfg: Tier0Config, T: int) -> np.ndarray:
    """
    Simulate the conduct regime path S_t using a 3-state Markov chain.

    States:
        0 = Competitive (C)
        1 = Tacit coordination (T)
        2 = Cartel (K)

    We use high "stay" probabilities so regimes occur in persistent episodes.
    """
    # Transition matrix P where P[i, j] = P(S_t=j | S_{t-1}=i)
    # These off-diagonal splits are just a reasonable default; we can tune later.
    P = np.array([
        # from C: mostly stay competitive; occasionally move to T or K
        [cfg.stay_C, (1 - cfg.stay_C) * 0.70, (1 - cfg.stay_C) * 0.30],
        # from T: can drift back to C or escalate to K
        [(1 - cfg.stay_T) * 0.50, cfg.stay_T, (1 - cfg.stay_T) * 0.50],
        # from K: cartel can break down to C or soften to T; tends to persist
        [(1 - cfg.stay_K) * 0.60, (1 - cfg.stay_K) * 0.40, cfg.stay_K],
    ], dtype=float)

    S = np.zeros(T, dtype=int)

    # Starting distribution: mostly competitive, some tacit, very rare cartel start
    S[0] = rng.choice([0, 1, 2], p=[0.75, 0.20, 0.05])

    # Draw the Markov chain forward
    for t in range(1, T):
        S[t] = rng.choice([0, 1, 2], p=P[S[t - 1]])

    return S


def simulate_market_series(
    rng: np.random.Generator,
    cfg: Tier0Config,
    market_id: int,
) -> pd.DataFrame:
    """
    Simulate one market's monthly log price series under Tier 0 DGP.

    Outputs a long-format DataFrame with:
        market_id, t, S (0/1/2), c (latent cost), p (log price)

    Important: we simulate one *continuous* market history that can switch between regimes.
    This lets you later form windows and evaluate whether the model detects transitions.
    """
    # Sample market-specific parameters (heterogeneity across markets)
    params = sample_market_params(rng, cfg)

    # Simulate a bit longer then discard burn-in to avoid start-at-zero artifacts
    T_total = cfg.burn_in + cfg.T

    # 1) regime path
    S = simulate_regime_path(rng, cfg, T_total)

    # 2) latent cost process c_t
    c = np.zeros(T_total, dtype=float)
    for t in range(1, T_total):
        # small regular shock
        u = rng.normal(0.0, params["sigma_c"])
        # occasional large jump shock
        jump = rng.normal(0.0, params["sigma_J"]) if rng.random() < params["jump_prob"] else 0.0
        # AR(1) persistence
        c[t] = params["rho_c"] * c[t - 1] + u + jump

    # 3) price process p_t
    p = np.zeros(T_total, dtype=float)

    # map numeric state to string key
    state_map = {0: "C", 1: "T", 2: "K"}

    for t in range(1, T_total):
        s = state_map[S[t]]
        kappa = params["kappa"][s]  # adjustment speed in this regime
        beta = params["beta"][s]    # pass-through in this regime

        # idiosyncratic price noise
        eps = rng.normal(0.0, params["sigma_p"])

        # Partial adjustment form:
        #   implied target price p*_t = beta * c_t
        #   actual price moves a fraction kappa toward target
        p[t] = (1 - kappa) * p[t - 1] + kappa * (beta * c[t]) + eps

    # Drop burn-in
    S = S[cfg.burn_in:]
    c = c[cfg.burn_in:]
    p = p[cfg.burn_in:]

    df = pd.DataFrame({
        "market_id": market_id,
        "t": np.arange(len(p)),
        "S": S,   # 0=C, 1=T, 2=K
        "c": c,
        "p": p,
    })

    # Save params for debugging (not stored in parquet, but useful while prototyping)
    df.attrs["params"] = params
    return df


def simulate_panel(cfg: Tier0Config, n_markets: int, seed: int = 0) -> pd.DataFrame:
    """
    Simulate many markets and stack into one DataFrame.
    """
    rng = np.random.default_rng(seed)
    dfs = [simulate_market_series(rng, cfg, m) for m in range(n_markets)]
    return pd.concat(dfs, axis=0, ignore_index=True)