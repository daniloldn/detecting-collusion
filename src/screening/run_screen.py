import pandas as pd
import numpy as np


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.screening.screening import compute_market_metrics, compute_market_truth
from src.utils.config import load_tier0_config
from src.utils.paths import run_dir


def main():


    experiment = "dgp0"
    _, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    seed = raw_cfg["simulation"]["seed"]

    mode = "baseline"
    L = 18

    base_model = run_dir(experiment, seed, mode)

    # Load features
    score_path = base_model / "scoring" / f"scoring_L{L}.parquet"
    df = pd.read_parquet(score_path).copy()

    #tau
    tau95 = 2.268063545227051
    tau99 = 3.7468080520629883


    market_metrics = compute_market_metrics(df, tau95=tau95, tau99=tau99, time_col="window_start")
    market_truth = compute_market_truth(df, time_col="window_start")

    market_eval = market_metrics.merge(market_truth, on="market_id", how="left")

    print(market_eval.head())

    # Save screening
    screen_dir = base_model / "screen"
    screen_dir.mkdir(parents=True, exist_ok=True)

    market_eval.to_parquet(screen_dir / f"screen_L{L}.parquet", index=False)

 

    return None

if __name__ == "__main__":
    main()