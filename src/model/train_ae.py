import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import run_dir
from src.utils.config import load_tier0_config
from src.utils.seeding import set_global_seed
from src.model.autoencoder import PriceAutoencoder

FEATURES_5 = ["volatility", "zero_change_fraction", "max_abs_ret", "AR_1", "price_range"]

def main():
    experiment = "dgp0"
    _, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    seed = raw_cfg["simulation"]["seed"]
    set_global_seed(seed)

    mode = "kappa_only"
    L = 18

    base = run_dir(experiment, seed, mode)
    feat_path = base / "data" / "features" / f"features_L{L}.parquet"
    df = pd.read_parquet(feat_path).dropna(subset=FEATURES_5).copy()

    X = df[FEATURES_5].to_numpy().astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    X_train, X_val = train_test_split(Xs, test_size=0.2, random_state=seed)

    ae = PriceAutoencoder(input_dim=len(FEATURES_5), latent_dim=2, hidden_dims=(16, 8), latent_activation=None)
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]

    history = ae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=200,
        batch_size=256,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # Save artifacts
    model_dir = base / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, model_dir / f"scaler_L{L}.pkl")
    ae.save(model_dir / f"ae_L{L}.keras")
    ae.encoder.save(model_dir / f"encoder_L{L}.keras")

    pd.DataFrame(history.history).to_csv(model_dir / f"history_L{L}.csv", index=False)

    print("Saved model artifacts in:", model_dir)

if __name__ == "__main__":
    main()