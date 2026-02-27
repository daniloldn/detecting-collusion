import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_tier0_config
from src.simulation.dgp0 import simulate_panel



def main():
    cfg, raw_cfg = load_tier0_config("configs/dgp0.yaml")

    n_markets = raw_cfg["simulation"]["n_markets"]
    seed = raw_cfg["simulation"]["seed"]
    
    for mode in ["baseline", "kappa_only", "beta_only"]:
        df = simulate_panel(cfg, n_markets=n_markets, seed=seed, mode=mode)
        out = f"data/interim_syn/synth_dgp0_series_{mode}.parquet"
        df.to_parquet(out, index=False)
        print("Saved:", out, df.shape)

if __name__ == "__main__":
    main()