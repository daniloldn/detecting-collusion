import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import run_dir
from src.utils.config import load_tier0_config
from src.scoring.conduct_axis import compute_centroids, compute_axis, score_centered

FEATURES_5 = ["volatility", "zero_change_fraction", "max_abs_ret", "AR_1", "price_range"]

def main():
    experiment = "dgp0"
    _, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    seed = raw_cfg["simulation"]["seed"]

    mode = "kappa_only"
    L = 18

    base_model = run_dir(experiment, seed, mode)
    base_feat = run_dir(experiment, seed, "kappa_only")

    # Load features
    feat_path = base_feat / "data" / "features" / f"features_L{L}.parquet"
    df = pd.read_parquet(feat_path).dropna(subset=FEATURES_5).copy()

    # Load scaler + encoder
    model_dir = base_model / "model"
    scaler = joblib.load(model_dir / f"scaler_L{L}.pkl")
    encoder = keras.models.load_model(model_dir / f"encoder_L{L}.keras")

    X = df[FEATURES_5].to_numpy().astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)

    Z = encoder.predict(Xs, batch_size=1024, verbose=0)
    df["z1"] = Z[:, 0]
    df["z2"] = Z[:, 1]

    # Centroids from pure windows only
    pure = df[df["is_pure_80"] == 1]
    mu_C, mu_K = compute_centroids(pure, z_cols=("z1", "z2"))
    v_hat = compute_axis(mu_C, mu_K)

    # Centered conduct score
    Z2 = df[["z1", "z2"]].to_numpy()
    df["conduct_score_centered"] = score_centered(Z2, mu_C, v_hat)

    # Save artifacts
    score_dir = base_feat / "scoring"
    score_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(score_dir / f"scoring_L{L}.parquet", index=False)
    np.save(score_dir / f"mu_C_L{L}.npy", mu_C)
    np.save(score_dir / f"mu_K_L{L}.npy", mu_K)
    np.save(score_dir / f"v_hat_L{L}.npy", v_hat)

    print("Saved scoring artifacts in:", score_dir)

if __name__ == "__main__":
    main()