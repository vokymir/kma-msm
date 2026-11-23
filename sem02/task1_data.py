"""Task 1 – Choice of data (Fish Market: Perch, Bream, Roach)."""

from pathlib import Path
import pandas as pd
from config import DATA_URL, SPECIES, VARIABLES

LOCAL_FILE = Path("data/Fish.csv")


# [Task 1 | data choice] Use the local CSV if it exists, otherwise the public URL.
def find_data_source():
    return LOCAL_FILE if LOCAL_FILE.exists() else DATA_URL


# [Task 1 | data choice] Read the whole CSV file.
def read_raw_data(source):
    return pd.read_csv(source)


# [Task 1 | data choice] Stop if the dataset lacks a needed column.
def check_required_columns(data):
    required = ["Species", *VARIABLES]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"V datasetu chybí sloupce: {', '.join(sorted(missing))}")


# [Task 1 | data choice] Keep only the three chosen species and the needed columns.
def select_species_and_columns(data):
    required = ["Species", *VARIABLES]
    return data.loc[data["Species"].isin(SPECIES), required].copy()


# [Task 1 | data choice] Convert the continuous variables to numbers.
def convert_to_numeric(data):
    data[VARIABLES] = data[VARIABLES].apply(pd.to_numeric, errors="coerce")
    return data


# [Task 1 | data choice] Remove rows with a missing measurement.
def drop_incomplete_rows(data):
    return data.dropna(subset=VARIABLES)


# [Task 1 | data choice] Zero weight is not a valid measurement (data error).
def drop_zero_weights(data):
    return data.loc[data["Weight"] > 0].copy()


# [Task 1 | data choice] Stop if some of the three groups is missing.
def check_all_groups_present(data):
    if data.empty or set(data["Species"].unique()) != set(SPECIES):
        raise ValueError("Data neobsahují všechny tři požadované skupiny ryb.")


# [Task 1 | data choice] Stop if any measurement is negative.
def check_no_negative_values(data):
    if (data[VARIABLES] < 0).any().any():
        raise ValueError("V datech se vyskytuje záporná hodnota měření.")


# [Task 1 | data choice] Load and validate the data (calls the steps above in order).
def load_data():
    data = read_raw_data(find_data_source())
    check_required_columns(data)
    data = select_species_and_columns(data)
    data = convert_to_numeric(data)
    data = drop_incomplete_rows(data)
    data = drop_zero_weights(data)
    check_all_groups_present(data)
    check_no_negative_values(data)
    return data


# [Task 1 | enough observations per group] Number of fish in each group.
def group_counts(data):
    return data.groupby("Species").size().rename("n")


# [Task 1 | data quality] Total count and count of zero weights (suspicious values).
def data_quality_table(data):
    return pd.DataFrame(
        {
            "kontrola": [
                "Počet vybraných pozorování",
                "Minimální hmotnost (g)",
                "Minimální výška (cm)",
            ],
            "hodnota": [
                len(data),
                float(data["Weight"].min()),
                float(data["Height"].min()),
            ],
        }
    )
