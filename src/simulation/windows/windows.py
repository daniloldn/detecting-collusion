# src/windows/make_windows.py
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from typing import Iterable, List
from src.simulation.windows.labels import summarize_window_states


def make_windows(
    df: pd.DataFrame,
    window: int,
    price_col: str = "p",
    state_col: str = "S",
    id_col: str = "market_id",
    time_col: str = "t",
) -> pd.DataFrame:
    """
    Convert a long market panel into a row-per-window dataset.

    Input df columns required:
      - market_id, t, p, S (and optionally c)
    Output:
      - market_id, window_start, window_end, window_length
      - Price 1..Price L
      - share_C/T/K, state_mode, is_pure_80 (for diagnostics)
    """
    required = {id_col, time_col, price_col, state_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values([id_col, time_col]).copy()

    rows = []

    for market_id, g in df.groupby(id_col, sort=False):
        g = g.sort_values(time_col)
        p = g[price_col].to_numpy()
        S = g[state_col].to_numpy()

        T = len(g)
        if T < window:
            continue

        # left-aligned windows: start ... start+window-1
        for start in range(0, T - window + 1):
            end = start + window - 1

            p_win = p[start:start + window]
            S_win = S[start:start + window]

            row = {
                id_col: market_id,
                "window_start": int(g[time_col].iloc[start]),
                "window_end": int(g[time_col].iloc[end]),
                "window_length": int(window),
            }

            # price columns: Price 1..Price L
            for j, val in enumerate(p_win, start=1):
                row[f"Price {j}"] = float(val)

            # add regime summary labels (diagnostics)
            row.update(summarize_window_states(S_win))

            rows.append(row)

    return pd.DataFrame(rows)


def make_windows_multi(
    df: pd.DataFrame,
    windows: Iterable[int] = (18, 24, 36),
    **kwargs
) -> pd.DataFrame:
    """Stack windows for multiple window lengths into one DataFrame."""
    out = []
    for w in windows:
        out.append(make_windows(df, window=w, **kwargs))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()