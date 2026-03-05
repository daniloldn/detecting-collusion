import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import run_dir
from src.utils.config import load_tier0_config
from src.data.feature_eng import feature_eng_syn

def main():
    _, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    seed = raw_cfg["simulation"]["seed"]
    experiment = "dgp0"

    for mode in ["baseline", "kappa_only", "beta_only", "calm_fundamentals"]:
        base = run_dir(experiment, seed, mode)
        win_dir = base / "data" / "windows"
        feat_dir = base / "data" / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)

        for w in (18, 24, 36):
            w_path = win_dir / f"windows_L{w}.parquet"
            dfw = pd.read_parquet(w_path)

            dff = feature_eng_syn(dfw)

            out = feat_dir / f"features_L{w}.parquet"
            dff.to_parquet(out, index=False)

        print(f"Saved features for {mode} in {feat_dir}")

if __name__ == "__main__":
    main()