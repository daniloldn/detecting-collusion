import numpy as np
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import run_dir
from src.utils.config import load_tier0_config

experiment = "dgp0"
_, raw_cfg = load_tier0_config("configs/dgp0.yaml")
seed = raw_cfg["simulation"]["seed"]
L = 18

# thresholds from baseline
tau95 = 2.268063545227051
tau99 = 3.7468080520629883

# load calm scores (scored using baseline model)
calm = run_dir(experiment, seed, "calm_fundamentals")
df_calm = pd.read_parquet(calm / "scoring" / f"scoring_L{L}.parquet")

scores_calm = df_calm["conduct_score_centered"].dropna().to_numpy()

fpr95 = float(np.mean(scores_calm > tau95))
fpr99 = float(np.mean(scores_calm > tau99))

print("Calm windows:", len(scores_calm))
print("FPR @ tau95:", fpr95)
print("FPR @ tau99:", fpr99)