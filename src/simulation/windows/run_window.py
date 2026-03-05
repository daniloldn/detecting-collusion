import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from src.simulation.windows.windows import make_windows_multi, make_windows
from src.utils.paths import run_dir
from src.utils.config import load_tier0_config

def main():
    _, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    seed = raw_cfg["simulation"]["seed"]

    experiment = "dgp0"

    for mode in ["baseline", "kappa_only", "beta_only"]:
        base = run_dir(experiment, seed, mode)

        # read series from the run folder
        df = pd.read_parquet(base / "data" / "series.parquet")

        # write windows next to it
        win_dir = base / "data" / "windows"
        win_dir.mkdir(parents=True, exist_ok=True)

        for w in (18, 24, 36):
            win_df = make_windows(df, window=w)
            out = win_dir / f"windows_L{w}.parquet"
            win_df.to_parquet(out, index=False)

        print(f"Saved windows for {mode} in {win_dir}")

if __name__ == "__main__":
    main()