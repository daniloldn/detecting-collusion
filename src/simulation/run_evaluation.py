import json
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.paths import run_dir
from src.utils.config import load_tier0_config
from src.simulation.validation import separation_auc_like

def main():
    experiment = "dgp0"
    _, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    seed = raw_cfg["simulation"]["seed"]

    mode = "calm_fundamentals"
    L = 18

    base = run_dir(experiment, seed, mode)
    score_path = base / "scoring" / f"scoring_L{L}.parquet"
    df = pd.read_parquet(score_path)

    # A6 metric (only meaningful when both C and K exist)
    scores_C = df[df["state_mode"] == 0]["conduct_score_centered"].dropna().to_numpy()
    scores_K = df[df["state_mode"] == 2]["conduct_score_centered"].dropna().to_numpy()

    a6 = separation_auc_like(scores_C, scores_K, n=10000, seed=seed)

    # Summary table
    summary = df.groupby("state_mode")["conduct_score_centered"].describe()
    out_dir = base / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / f"summary_L{L}.csv")

    metrics = {"A6_P_K_gt_C": a6, "seed": seed, "mode": mode, "L": L}
    with open(out_dir / f"metrics_L{L}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved eval in:", out_dir)
    print(metrics)
    print(summary)

if __name__ == "__main__":
    main()