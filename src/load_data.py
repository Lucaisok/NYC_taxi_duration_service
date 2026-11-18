from pathlib import Path

import pandas as pd
import requests

from src.schemas import TripSchema

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
)
DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "yellow_tripdata_2025-01.parquet"
)


def get_data_path() -> Path:
    """Return the local parquet path, downloading it first if missing."""
    if DATA_PATH.exists():
        return DATA_PATH
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    DATA_PATH.write_bytes(response.content)
    return DATA_PATH


def load_data() -> pd.DataFrame:
    """Load the trip data with lowercase column names (TLC casing varies by month) and validate against the schema."""
    df = pd.read_parquet(get_data_path())
    df.columns = df.columns.str.lower()
    df = TripSchema.validate(df)
    return df
