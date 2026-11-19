import pandas as pd

from src.config import DURATION_RANGE, TARGET
from src.features import add_features
from src.preprocess import create_target, filter_target_outliers


def test_add_features_builds_time_and_route_columns():
    X = pd.DataFrame(
        {
            "pulocationid": [161],
            "dolocationid": [236],
            # 2025-01-18 is a Saturday
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-18 08:30:00"]),
        }
    )
    out = add_features(X)

    assert "tpep_pickup_datetime" not in out.columns
    assert out.loc[0, "hour"] == 8
    assert out.loc[0, "day_of_week"] == 5
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[0, "route"] == "161_236"


def test_create_target_and_filter_outliers():
    pickup = pd.to_datetime(["2025-01-15 08:00:00"] * 3)
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            # 1 min (too short), 15 min (kept), 200 min (too long)
            "tpep_dropoff_datetime": pickup
            + pd.to_timedelta([1, 15, 200], unit="min"),
        }
    )
    filtered = filter_target_outliers(create_target(df))

    assert filtered[TARGET].tolist() == [15.0]
    assert filtered[TARGET].between(*DURATION_RANGE).all()
