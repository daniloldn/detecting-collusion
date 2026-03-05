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

# Load baseline scored data (this is your baseline-trained scoring output)
base = run_dir(experiment, seed, "baseline")
df_base = pd.read_parquet(base / "scoring" / f"scoring_L{L}.parquet")

scores_C = df_base[df_base["state_mode"] == 0]["conduct_score_centered"].dropna().to_numpy()

tau95 = float(np.quantile(scores_C, 0.95))
tau99 = float(np.quantile(scores_C, 0.99))

print("tau95:", tau95)
print("tau99:", tau99)
print("baseline competitive count:", len(scores_C))