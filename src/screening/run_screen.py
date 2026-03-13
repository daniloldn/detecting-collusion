import pandas as pd
import numpy as np


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.screening.screening import compute_market_metrics, compute_market_truth, compute_structural_intensity
from src.utils.config import load_tier0_config
from src.utils.paths import run_dir


def main():


    #experiment = "dgp0"
    #_, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    #seed = raw_cfg["simulation"]["seed"]

    #mode = "baseline"
    #L = 18

    #base_model = run_dir(experiment, seed, mode)

    path = Path("data/processed_real")

    # Load features
    #score_path = base_model / "scoring" / f"scoring_L{L}.parquet"
    df = pd.read_parquet(path / "real_scored_L18.parquet").copy()

    #tau
    tau95 = 2.268063545227051
    tau99 = 3.7468080520629883


    market_metrics = compute_market_metrics(df, tau95=tau95, tau99=tau99, time_col="Window") # time_col = window_start for syn and Window for real
    market_eval = market_metrics.copy()
    #market_truth = compute_market_truth(df, time_col="window_start")

    #market intensity
    #params_df = pd.read_parquet(base_model / "data" / "market_params.parquet")
    #intensity_df = compute_structural_intensity(params_df)

    #market_eval = market_metrics.merge(market_truth, on="market_id", how="left")
    #merge intensity
    #market_eval = market_eval.merge(intensity_df, on="market_id", how="left")
    

    #intensity * occurance of cartels
    #market_eval["effective_structural_intensity"] = (
    #market_eval["structural_intensity"] * market_eval["true_mean_share_K"]
    #)

    print(market_eval.head())

    # Save screening
    screen_dir = path / "screen"
    screen_dir.mkdir(parents=True, exist_ok=True)

    market_eval.to_parquet(screen_dir / f"screen_L18.parquet", index=False)

 

    return None

if __name__ == "__main__":
    main()