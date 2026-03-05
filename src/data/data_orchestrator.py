import pandas as pd 
import sys
from pathlib import Path
import numpy as np
import joblib
from tensorflow import keras



# Ensure the project root is on sys.path so `src` imports work when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import file_names, load_data, load_pickle, file_names_cleaned
from src.data.clean_data import missing_observation, storing_data, storing_data_merged
from src.data.feature_eng import feature_eng_syn
from src.scoring.conduct_axis import score_centered

def main():

    #loading and cleaning dataset
    dataset_names = file_names()

    for file in dataset_names:
        # load dataset
        df = load_data(file)
        # store dataset
        save_file = file.replace(".csv", "")
        storing_data(missing_observation(df), f"{save_file}.pkl")

    #creating rolling windows and feature eng
    interim_names = file_names_cleaned()

    #empty list of dfs
    dfs = []

    for names in interim_names:
        dfs.append(load_pickle(names))

    #mergining dfs
    merged_df = pd.concat(dfs, ignore_index=True)

    feature_df = feature_eng_syn(merged_df)

    storing_data_merged(feature_df, "real_processed_18.csv")

    FEATURES_5 = ["volatility", "zero_change_fraction", "max_abs_ret", "AR_1", "price_range"]

    # ---- paths to baseline artifacts ----
    L = 18
    seed = 42
    baseline_dir = PROJECT_ROOT / "runs" / "dgp0" / f"seed_{seed}" / "baseline"

    scaler = joblib.load(baseline_dir / "model" / f"scaler_L{L}.pkl")
    encoder = keras.models.load_model(baseline_dir / "model" / f"encoder_L{L}.keras")

    mu_C = np.load(baseline_dir / "scoring" / f"mu_C_L{L}.npy")
    v_hat = np.load(baseline_dir / "scoring" / f"v_hat_L{L}.npy")

    # ---- prepare real X ----
    df_real = feature_df.dropna(subset=FEATURES_5).copy()
    X = df_real[FEATURES_5].to_numpy().astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)

    # ---- encode ----
    Z = encoder.predict(Xs, batch_size=2048, verbose=0)
    df_real["z1"] = Z[:, 0]
    df_real["z2"] = Z[:, 1]

    # ---- score ----
    Z2 = df_real[["z1", "z2"]].to_numpy()
    df_real["conduct_score_centered"] = score_centered(Z2, mu_C, v_hat)

    # ---- save scored real file ----
    out_path = PROJECT_ROOT / "data" / "processed_real" / "real_scored_L18.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_real.to_parquet(out_path, index=False)

    print("Saved scored real data:", out_path, df_real.shape)
    print(df_real["conduct_score_centered"].describe())


    


if __name__ == "__main__":
    main()