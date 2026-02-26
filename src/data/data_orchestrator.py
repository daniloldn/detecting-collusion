import pandas as pd 
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` imports work when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import file_names, load_data, load_pickle, file_names_cleaned
from src.data.clean_data import missing_observation, storing_data, storing_data_merged
from src.data.feature_eng import feature_eng

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

    feature_df = feature_eng(merged_df)

    storing_data_merged(feature_df, "real_processed_18.csv")


    


if __name__ == "__main__":
    main()