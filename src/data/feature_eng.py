import pandas as pd
import numpy as np


def row_autocorr(row: pd.Series, lag: int = 1) -> float:
        """Lagged autocorrelation for a single row of price changes."""
        if row.count() <= lag:
                return np.nan
        return row.autocorr(lag=lag)

def feature_eng(data: pd.DataFrame, windows = 18) -> pd.DataFrame:
    """creates features for df"""

    #getting price colums
    price_cols = [f"Price {i}" for i in range(1, windows)]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    price_changes = data[price_cols].pct_change(axis=1)

    #creating featues
    data["mean_change"] = price_changes.mean(axis=1)
    data["volatility"] = price_changes.std(axis=1, ddof=0)
    data["CoV_change"] = data["volatility"] / data["mean_change"].replace(0, np.nan)
    data["zero_change_fraction"] = (price_changes.abs() < 1e-6).sum(axis=1) / price_changes.shape[1]
    data["autocorr_change"] = price_changes.apply(row_autocorr,axis=1)
    data["kurtosis_change"] = price_changes.kurtosis(axis=1)

    return data