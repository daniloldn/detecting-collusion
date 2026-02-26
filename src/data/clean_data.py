import pandas as pd
import pickle
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
DATA_DIR_2 = Path(__file__).resolve().parents[2] / "data" / "processed"

def missing_observation(data:pd.DataFrame):

    "function to turn data in to time series and take out missing observations"

    series_dict = {
    col: data[col].dropna()
    for col in data.columns
    }
    return series_dict



def storing_data(data_dict: dict, filename: str = "clean_data.pkl") -> Path:

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = DATA_DIR / filename
        with file_path.open("wb") as fh:
            pickle.dump(data_dict, fh)
        return file_path
