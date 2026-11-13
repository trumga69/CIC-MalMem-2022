import pandas as pd

CSV_PATH = "data/raw/Obfuscated-MalMem2022.csv"
NONE_FEATURE_COLUMS = ["Category" , "Class"]

def load_data () -> pd.DataFrame: return pd.read_csv(CSV_PATH)
def feature_columns(df: pd.DataFrame) -> list:
    return df.columns.drop(NONE_FEATURE_COLUMS).tolist()
def derive_family(df: pd.DataFrame) -> pd.Series:
    return df["Category"].str.split("-").str[0]
def derive_sample_token(df: pd.DataFrame) -> pd.Series:
    return (
        df["Category"]
        .str.extract(r"^[^-]+-[^-]+-([0-9a-fA-F]+)-[0-9]+\.raw$")[0]
        .str.lower()
    )
