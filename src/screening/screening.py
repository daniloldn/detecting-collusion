import pandas as pd
import numpy as np


def longest_run(mask: pd.Series) -> int:
    """
    Longest consecutive run of True values in a boolean Series.
    """
    max_run = 0
    current = 0

    for val in mask:
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    return max_run

def compute_market_metrics(df: pd.DataFrame, tau95: float, tau99: float, time_col: str) -> pd.DataFrame:
    """
    Compute market-level screening metrics from window-level conduct scores.
    """
    rows = []

    for market_id, g in df.groupby("Name"): #name for real, market_id for synthetic
        g = g.sort_values(time_col).copy()

        scores = g["conduct_score_centered"]

        mask95 = scores > tau95
        mask99 = scores > tau99

        rows.append({
            "market_id": market_id,
            "n_windows": len(g),
            "mean_score": scores.mean(),
            "sd_score": scores.std(ddof=0),
            "pct_above_tau95": mask95.mean(),
            "pct_above_tau99": mask99.mean(),
            "longest_run_tau95": longest_run(mask95),
            "max_score": scores.max(),
        })

    return pd.DataFrame(rows)


def compute_market_truth(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Compute market-level ground-truth cartel exposure from synthetic labels.
    """
    rows = []

    for market_id, g in df.groupby("market_id"):
        g = g.sort_values(time_col).copy()

        cartel_mask = g["state_mode"] == 2
        pure_cartel_mask = (g["state_mode"] == 2) & (g["is_pure_80"] == 1)

        rows.append({
            "market_id": market_id,
            "true_pct_cartel_windows": cartel_mask.mean(),
            "true_pct_pure_cartel_windows": pure_cartel_mask.mean(),
            "true_mean_share_K": g["share_K"].mean() if "share_K" in g.columns else np.nan,
            "true_max_share_K": g["share_K"].max() if "share_K" in g.columns else np.nan,
            "true_longest_cartel_run": longest_run(cartel_mask),
        })

    return pd.DataFrame(rows)

def compute_structural_intensity(params_df: pd.DataFrame) -> pd.DataFrame:
    out = params_df.copy()

    eps = 1e-12

    out["structural_intensity"] = (
        (out["beta_C"] - out["beta_K"]) / (out["beta_C"] + eps)
        +
        (out["kappa_C"] - out["kappa_K"]) / (out["kappa_C"] + eps)
    )

    return out[["market_id", "structural_intensity"]]


def count_episodes(mask):
    mask = pd.Series(mask).astype(bool)
    return ((mask) & (~mask.shift(fill_value=False))).sum()

def compute_market_metrics_real(df, tau95, tau99, recon_tau95, recon_tau99, market_col="Name", time_col=None):
    rows = []

    grouped = df.groupby(market_col)
    for market, g in grouped:
        if time_col is not None:
            g = g.sort_values(time_col).copy()
        else:
            g = g.copy()

        score = g["conduct_score_centered"]
        recon = g["recon_error"]

        score95 = score > tau95
        score99 = score > tau99
        recon95 = recon > recon_tau95
        recon99 = recon > recon_tau99

        rows.append({
            "market_id": market,
            "n_windows": len(g),

            "mean_score": score.mean(),
            "median_score": score.median(),
            "sd_score": score.std(ddof=0),
            "max_score": score.max(),

            "pct_score95": score95.mean(),
            "pct_score99": score99.mean(),
            "longest_run_score95": longest_run(score95),
            "longest_run_score99": longest_run(score99),
            "episodes_score95": count_episodes(score95),
            "episodes_score99": count_episodes(score99),

            "mean_recon": recon.mean(),
            "median_recon": recon.median(),
            "max_recon": recon.max(),
            "pct_recon95": recon95.mean(),
            "pct_recon99": recon99.mean(),

            "pct_score95_in_manifold": (score95 & ~recon95).mean(),
            "pct_score95_out_manifold": (score95 & recon95).mean(),
            "pct_score99_in_manifold": (score99 & ~recon99).mean(),
            "pct_score99_out_manifold": (score99 & recon99).mean(),
        })

    return pd.DataFrame(rows)