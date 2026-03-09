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

    for market_id, g in df.groupby("market_id"):
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