import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.metrics import roc_auc_score

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

    mode = "baseline"
    L = 18

    base_model = run_dir(experiment, seed, mode)
    base_feat = run_dir(experiment, seed, "baseline")


    # Load features
    feat_path = base_feat / "data" / "features" / f"features_L{L}.parquet"
    #df = pd.read_parquet(feat_path).dropna(subset=FEATURES_5).copy()
    path = Path("data/cartel_data/feature/features.parquet")
    df = pd.read_parquet(path).dropna(subset=FEATURES_5).copy()

    # Load scaler + encoder
    model_dir = base_model / "model"
    scaler = joblib.load(model_dir / f"scaler_L{L}.pkl")
    encoder = keras.models.load_model(model_dir / f"encoder_L{L}.keras")
    decoder = keras.models.load_model( model_dir / f"decoder_L{L}.keras")


    X = df[FEATURES_5].to_numpy().astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)

    #latent representation
    Z = encoder.predict(Xs, batch_size=1024, verbose=0)
    df["z1"] = Z[:, 0]
    df["z2"] = Z[:, 1]

    # Reconstruction + reconstruction error
    Xhat = decoder.predict(Z, batch_size=1024, verbose=0)
    df["recon_error"] = np.mean((Xs - Xhat) ** 2, axis=1)

    # Centroids from pure windows only
    #pure = df[df["is_pure_80"] == 1]
    #mu_C, mu_K = compute_centroids(pure, z_cols=("z1", "z2"))
    #v_hat = compute_axis(mu_C, mu_K)

    # Centered conduct score
    Z2 = df[["z1", "z2"]].to_numpy()

    #baseline line centriods
    baseline_path = base_model / "scoring"
    mu_C = np.load(baseline_path / f"mu_C_L{L}.npy")
    v_hat = np.load(baseline_path / f"v_hat_L{L}.npy")


    df["conduct_score_centered"] = score_centered(Z2, mu_C, v_hat)

    df["cartel_share"] = df["share_K"]
    df["cartel_window"] = (df["share_K"] >= 0.5).astype(int)
    df["pure_cartel_window"] = (df["share_K"] >= 0.8).astype(int)

    
    

    # Save artifacts
    #score_dir = base_feat / "scoring"
    #score_dir.mkdir(parents=True, exist_ok=True)
    score_dir = Path("data/cartel_data/score")
    score_dir.mkdir(parents=True, exist_ok=True)

   
    #np.save(score_dir / f"mu_C_L{L}.npy", mu_C)
    #np.save(score_dir / f"mu_K_L{L}.npy", mu_K)
    #np.save(score_dir / f"v_hat_L{L}.npy", v_hat)

    recon_tau95 = np.load(baseline_path / f"recon_tau95_L{L}.npy")
    recon_tau99 = np.load(baseline_path / f"recon_tau99_L{L}.npy")

    tau95 = 2.268063545227051
    tau99 = 3.7468080520629883

    # Save reconstruction thresholds from synthetic baseline
    #recon_tau95 = np.quantile(df["recon_error"], 0.95)
    #recon_tau99 = np.quantile(df["recon_error"], 0.99)

    df["recon_outside95"] = df["recon_error"] > recon_tau95
    df["recon_outside99"] = df["recon_error"] > recon_tau99

    df["score_outside95"] = df["conduct_score_centered"] > tau95
    df["score_outside99"] = df["conduct_score_centered"] > tau99


    df["cartel_and_in_manifold"] = (
    (df["cartel_window"] == 1) &
    (df["recon_error"] <= recon_tau95)
    )

    df["high_score_and_cartel"] = (
    (df["conduct_score_centered"] > tau95) &
    (df["cartel_window"] == 1)
    )


    for j, col in enumerate(FEATURES_5):
        df[f"recon_sqerr_{col}"] = (Xs[:, j] - Xhat[:, j]) ** 2


    print("\nValidation summaries")

    print("\nMean conduct score by cartel_window:")
    print(df.groupby("cartel_window")["conduct_score_centered"].mean())

    print("\nShare above tau95 by cartel_window:")
    print((df["conduct_score_centered"] > tau95).groupby(df["cartel_window"]).mean())

    print("\nMean reconstruction error by cartel_window:")
    print(df.groupby("cartel_window")["recon_error"].mean())

    print("\nSpearman correlation with cartel share:")
    print(df["conduct_score_centered"].corr(df["cartel_share"], method="spearman"))

    try:
        print("\nAUC for cartel_window:")
        print(roc_auc_score(df["cartel_window"], df["conduct_score_centered"]))
    except Exception as e:
        print("AUC failed:", e)

    #np.save(score_dir / f"recon_tau95_L{L}.npy", recon_tau95)
    #np.save(score_dir / f"recon_tau99_L{L}.npy", recon_tau99)

    df.to_parquet(score_dir / f"cartel_scoring_L{L}.parquet", index=False)

    print("Saved scoring artifacts in:", score_dir)
    print(f"Recon error thresholds: tau95={recon_tau95:.6f}, tau99={recon_tau99:.6f}")


if __name__ == "__main__":
    main()