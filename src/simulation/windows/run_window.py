import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from src.simulation.windows.windows import make_windows_multi, make_windows

def main():
    df = pd.read_parquet("data/interim_syn/synth_dgp0_series.parquet")

    for w in (18, 24, 36):
        win_df = make_windows(df, window=w)
        win_df.to_parquet(f"data/processed/synth_tier0_windows_L{w}.parquet")


if __name__ == "__main__":
    main()