# Detecting Collusion

Masters dissertation project exploring machine-learning pipelines that flag potential cartel behavior in commodity price series. The repository contains two complementary workflows:

1. **Real-market ingestion** – ingest raw price files, clean missing observations, build rolling-window feature sets, and persist curated tables for downstream models.
2. **Synthetic benchmarking** – simulate labeled price paths under multiple conduct regimes (competitive, tacit, cartel) so model behavior can be stress-tested before touching sensitive real data.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `configs/dgp0.yaml` | Master configuration for the Tier‑0 synthetic data-generating process (DGP). |
| `data/` | Data lake with raw inputs, intermediate pickles, processed feature tables, and synthetic exports. |
| `notebooks/` | Exploration notebooks (baseline autoencoder diagnostics, β‑only / κ‑only stress tests). |
| `src/data/` | Real-data ingestion, cleaning, window creation, and feature engineering utilities. |
| `src/simulation/` | Synthetic DGP, labeling, and validation utilities (includes window builders for simulated panels). |
| `src/model/` | Trainable models such as the `PriceAutoencoder`. |
| `src/utils/` | Shared helpers (e.g., YAML config loader). |
| `tests/` | Placeholder for automated tests. |

## Real Data Workflow

1. **Enumerate raw files** – `src/data/load_data.py` provides `file_names()` over `data/raw_data/` and type-aware loaders for CSV/Excel/JSON.
2. **Clean missing observations** – `missing_observation()` converts price columns into ragged time series per product, dropping `NaN`s before storage.
3. **Persist interim pickles** – `storing_data()` writes each cleaned series dict to `data/interim/*.pkl` so rolling windows can be generated quickly.
4. **Window + feature engineering** – `load_pickle()` assembles sliding windows (default length 18) and `feature_eng()` computes signal features (mean/volatility of returns, coefficient of variation, rigidity share, autocorrelation, kurtosis, etc.). Results are merged across products and stored via `storing_data_merged()` under `data/processed/real_processed_18.csv`.

Run the full pipeline from the project root:

```bash
python -m src.data.data_orchestrator
```

Adjust the window length or feature recipe by editing `feature_eng()` or the `window` constant inside `load_pickle()`.

## Synthetic Data Workflow

Synthetic data helps benchmark detection models under controlled regime switches.

### Tier‑0 DGP

- `src/simulation/dgp0.py` defines `Tier0Config` plus helpers to sample market-specific structural parameters, simulate latent costs, regime paths, and log prices.
- `src/utils/config.py` loads `configs/dgp0.yaml` and instantiates `Tier0Config` for reproducibility.
- `src/simulation/run_dgp0.py` iterates over three scenarios:
	- `baseline` – heterogeneity in both pass-through (β) and adjustment speed (κ)
	- `kappa_only` – β held fixed so only stickiness changes
	- `beta_only` – κ held fixed so only pass-through changes

Execute the simulation (creates `data/interim_syn/synth_dgp0_series_<mode>.parquet`):

```bash
python -m src.simulation.run_dgp0
```

### Windowing + Labels

- `src/simulation/windows/windows.py` (`make_windows` / `make_windows_multi`) converts the long panel into overlapping windows with price vectors `Price 1..Price L`.
- `src/simulation/windows/labels.py` annotates each window with regime shares (`share_C`, `share_T`, `share_K`), the modal state, and whether ≥80% of the window belongs to a single regime (`is_pure_80`).
- `src/simulation/windows/run_window.py` materializes parquet tables for multiple window lengths (18/24/36) per scenario under `data/processed_syn/`.

Run after generating simulated series:

```bash
python -m src.simulation.windows.run_window
```

### Diagnostic Tools

- `src/simulation/validation.py` draws Plotly visualizations of price vs. cost with shaded regimes and provides `separation_auc_like()` to quantify how well a scoring rule separates competitive vs. cartel samples.
- Notebook companions (`notebooks/beta_only.ipynb`, `kappa_only.ipynb`, `simulation.ipynb`) reproduce figures and sanity checks for the stress scenarios.

## Modeling

- `src/model/autoencoder.py` contains `PriceAutoencoder`, a dense autoencoder with configurable hidden widths and latent dimensionality. It acts on engineered feature vectors (default six economic features) and can be extended for reconstruction error–based anomaly detection.
- Model training/evaluation scripts can import the processed real or synthetic windows and leverage Plotly for exploratory plots.

## Configuration & Customization

- Update `configs/dgp0.yaml` to experiment with alternative horizons (`T`), regime persistence, shock variances, or the number of markets.
- When adding new scenarios, extend the `mode` loop inside `run_dgp0.py` and `run_window.py` so downstream files follow a consistent naming pattern.
- Additional engineered features for synthetic data can be defined in `feature_eng_syn()` (currently prepared to handle log-price windows of arbitrary length).

## Getting Started

1. **Python environment** – create/activate your preferred env (e.g., `conda create -n collusion python=3.11`).
2. **Install dependencies** – minimally `pandas`, `numpy`, `pyyaml`, `tensorflow` (or `tensorflow-cpu`), `plotly`.
3. **Place raw data** – drop proprietary files into `data/raw_data/` following the expected CSV/Excel format.
4. **Run pipelines** – execute the real or synthetic workflows described above.
5. **Explore notebooks** – open the `notebooks/` folder in VS Code or Jupyter Lab for interactive analysis.

## Suggested Next Steps

- Add experiment tracking (weights & biases, MLflow) for autoencoder training.
- Extend `tests/` with regression tests that ensure feature engineering and simulation outputs remain stable when configs change.
- Create a `requirements.txt` or `environment.yml` to lock versions used in the dissertation analysis.

For questions or collaboration ideas, open an issue or leave comments in the notebooks.
