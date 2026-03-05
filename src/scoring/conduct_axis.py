import numpy as np
import pandas as pd

def compute_centroids(pure_df: pd.DataFrame, z_cols=("z1", "z2")):
    mu_C = pure_df[pure_df["state_mode"] == 0][list(z_cols)].mean().to_numpy()
    mu_K = pure_df[pure_df["state_mode"] == 2][list(z_cols)].mean().to_numpy()
    return mu_C, mu_K

def compute_axis(mu_C: np.ndarray, mu_K: np.ndarray):
    v = mu_K - mu_C
    v_hat = v / (np.linalg.norm(v) + 1e-12)
    return v_hat

def score_centered(Z: np.ndarray, mu_C: np.ndarray, v_hat: np.ndarray):
    return (Z - mu_C) @ v_hat