from pathlib import Path
from typing import Sequence

import pickle
import pandas as pd
from collections.abc import Iterable


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_data"
INTERIM_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"   


def file_names() -> list[str]:
    """Return every file name inside the collusion data directory."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    return [f.name for f in DATA_DIR.iterdir() if f.is_file()]

def file_names_cleaned() -> list[str]:
    """Return every file name inside the collusion data directory."""
    if not INTERIM_DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {INTERIM_DATA_DIR}")

    return [f.name for f in INTERIM_DATA_DIR.iterdir() if f.is_file()]



def load_data(filename: str):
    """
    Load data from a file in the collusion data directory.
    Supports CSV, Excel (xls/xlsx), and JSON files.
    """
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    

def load_final(file_name:str):
    """
    Load data from a file in the collusion data directory.
    Supports CSV, Excel (xls/xlsx), and JSON files.
    """
    file_path = PROCESSED_DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_pickle(file_name:str):

    #load data
    with (INTERIM_DATA_DIR / file_name).open("rb") as fp:
        data = pickle.load(fp)

    #create intervals
    first_key = next(iter(data))
    intervals = []
    window = 18
    for start in range(len(data[first_key]) - window):
        intervals.append(list(range(start, start + window)))

    all_rows = []

    for name in list(data.keys())[1:]:
        for i, interval in enumerate(intervals):
            try:
                row = {
                "Window": i,
                "Name": name
                }
                # Add dynamic price columns
                for j, t in enumerate(interval):
                    row[f"Price {j+1}"] = data[name][t]
                all_rows.append(row)
            except (KeyError, IndexError) as exc:
                print(f"Failed at {name}, window {i}, time {interval}: {exc}")

    # Combine all rows into a single DataFrame
    df = pd.DataFrame(all_rows)

    return df