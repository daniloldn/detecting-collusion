import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import run_dir
from src.utils.config import load_tier0_config

STATE_MAP = {0: "Competitive", 1: "Tacit", 2: "Cartel"}
COLORS = {"Competitive": "green", "Tacit": "orange", "Cartel": "red"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    experiment = "dgp0"
    _, raw_cfg = load_tier0_config("configs/dgp0.yaml")
    seed = raw_cfg["simulation"]["seed"]

    mode_1 = "baseline"
    mode_2 = "trend_fundamentals"
    
    L = 18

    base = run_dir(experiment, seed, mode_1)
    df = pd.read_parquet(base / "scoring" / f"scoring_L{L}.parquet")

    #base = run_dir(experiment, seed, mode_1)
    #df = pd.read_parquet(base / "scoring" / f"scoring_L{L}.parquet")
    base_1 = PROJECT_ROOT / "data" / "processed_real" / f"real_scored_L{L}.parquet"
    df1= pd.read_parquet(base_1)

    # thresholds from baseline
    tau95 = 2.268063545227051
    tau99 = 3.7468080520629883

    figs_dir = base / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Latent space (subsample pure) + centroids + axis ---
    pure = df[df["is_pure_80"] == 1].copy()
    pure["state_label"] = pure["state_mode"].map(STATE_MAP)

    # subsample for visibility
    n = 1500
    pure_s = (
        pure.groupby("state_label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n), random_state=seed))
    )

    pure_s["state_label"] = pure_s["state_mode"].map(STATE_MAP)

    # centroids from pure (not subsample)
    mu_C = pure[pure["state_mode"] == 0][["z1", "z2"]].mean().to_numpy()
    mu_T = pure[pure["state_mode"] == 1][["z1", "z2"]].mean().to_numpy()
    mu_K = pure[pure["state_mode"] == 2][["z1", "z2"]].mean().to_numpy()

    fig_latent = px.scatter(
        pure_s,
        x="z1",
        y="z2",
        color="state_label",
        color_discrete_map=COLORS,
        opacity=0.01,
        title="Latent Space (Pure Windows, Subsampled) with Centroids and Conduct Axis",
        template="plotly_white",
    )

    centroids = np.vstack([mu_C, mu_T, mu_K])
    centroid_labels = ["Competitive centroid", "Tacit centroid", "Cartel centroid"]

    fig_latent.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode="markers+text",
        text=centroid_labels,
        textposition="top center",
        marker=dict(size=14, symbol="x", color="black"),
        name="Centroids",
    ))

    fig_latent.add_trace(go.Scatter(
        x=[mu_C[0], mu_K[0]],
        y=[mu_C[1], mu_K[1]],
        mode="lines",
        line=dict(width=4, dash="dash", color="black"),
        name="Conduct axis (C→K)",
    ))

    latent_path = figs_dir / f"latent_centroids_L{L}.png"
    fig_latent.write_image(str(latent_path), scale=2)
    print("Saved:", latent_path)

    # --- Plot 2: Score distribution ---
    df["state_label"] = df["state_mode"].map(STATE_MAP)

    fig_hist = px.histogram(
        df,
        x="conduct_score_centered",
        color="state_label",
        color_discrete_map=COLORS,
        nbins=70,
        opacity=0.55,
        barmode="overlay",
        title="Centered Conduct Score Distribution by Regime",
        template="plotly_white",
    )

    fig_hist.add_histogram(
    x=df1["conduct_score_centered"],
    nbinsx=70,
    name="real markets",
    marker_color="blue",
    opacity=0.35
    )

    fig_hist.add_vline(
    x=tau95,
    line_dash="dash",
    line_color="black",
    annotation_text="τ95",
    annotation_position="top"
    )

    fig_hist.add_vline(
    x=tau99,
    line_dash="dash",
    line_color="black",
    annotation_text="τ99",
    annotation_position="top"
    )

    hist_path = figs_dir / f"score_distribution_L{L}.png"
    fig_hist.write_image(str(hist_path), scale=2)
    print("Saved:", hist_path)

if __name__ == "__main__":
    main()