import pandas as pd
import numpy as np


def row_autocorr(row: pd.Series, lag: int = 1) -> float:
        """Lagged autocorrelation for a single row of price changes."""
        if row.count() <= lag:
                return np.nan
        return row.autocorr(lag=lag)

def feature_eng(data: pd.DataFrame, windows = 18) -> pd.DataFrame:
    """creates features for df"""
    out = data.copy()

    # detect price columns dynamically and sort by index
    price_cols = sorted(
        [c for c in out.columns if c.startswith("Price ")],
        key=lambda s: int(s.split(" ")[1])
    )

    out[price_cols] = out[price_cols].apply(pd.to_numeric, errors="coerce")
    out[price_cols] = np.log(out[price_cols])

    # log returns across months within each window
    rets = out[price_cols].diff(axis=1).iloc[:, 1:]  # drop first NaN

    out["mean_change"] = rets.mean(axis=1)
    out["volatility"] = rets.std(axis=1, ddof=0) 

    eps = 1e-6

    out["CoV_change"] = out["volatility"] / (out["mean_change"].abs() + eps)

    # “near zero” change fraction (rigidity proxy) — tune threshold later
    out["zero_change_fraction"] = (rets.abs() < 1e-3).sum(axis=1) / rets.shape[1]

    # autocorr of returns within the window
    out["AR_1"] = rets.apply(row_autocorr, axis=1)
    out["AR_2"] = rets.apply(lambda row: row_autocorr(row, lag=2), axis=1)

    out["kurtosis_change"] = rets.kurtosis(axis=1)

    # biggest change in price
    out["max_abs_ret"] = rets.abs().max(axis=1)

    #different reactions to shocks , postive and negative
    out["pos_vol"] = rets.where(rets > 0).std(axis=1)
    out["neg_vol"] = rets.where(rets<0).std(axis=1)

    #level voltality
    out["level_vol"] = out[price_cols].std(axis=1)

    #price range
    out["price_range"] = out[price_cols].max(axis=1) - out[price_cols].min(axis=1)
 
    return out


def feature_eng_syn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for windows with columns Price 1..Price L.
    Assumes Price columns are LOG PRICES.
    Produces fixed-size feature vector regardless of L.
    """
    out = df.copy()

    # detect price columns dynamically and sort by index
    price_cols = sorted(
        [c for c in out.columns if c.startswith("Price ")],
        key=lambda s: int(s.split(" ")[1])
    )

    out[price_cols] = out[price_cols].apply(pd.to_numeric, errors="coerce")
    #out[price_cols] = np.log(out[price_cols])

    # log returns across months within each window
    rets = out[price_cols].diff(axis=1).iloc[:, 1:]  # drop first NaN

    out["mean_change"] = rets.mean(axis=1)
    out["volatility"] = rets.std(axis=1, ddof=0) 

    eps = 1e-6

    out["CoV_change"] = out["volatility"] / (out["mean_change"].abs() + eps)

    # “near zero” change fraction (rigidity proxy) — tune threshold later
    out["zero_change_fraction"] = (rets.abs() < 1e-3).sum(axis=1) / rets.shape[1]

    # autocorr of returns within the window
    out["AR_1"] = rets.apply(row_autocorr, axis=1)
    out["AR_2"] = rets.apply(lambda row: row_autocorr(row, lag=2), axis=1)

    out["kurtosis_change"] = rets.kurtosis(axis=1)

    # biggest change in price
    out["max_abs_ret"] = rets.abs().max(axis=1)

    #different reactions to shocks , postive and negative
    out["pos_vol"] = rets.where(rets > 0).std(axis=1)
    out["neg_vol"] = rets.where(rets<0).std(axis=1)

    #level voltality
    out["level_vol"] = out[price_cols].std(axis=1)

    #price range
    out["price_range"] = out[price_cols].max(axis=1) - out[price_cols].min(axis=1)
 
    return out