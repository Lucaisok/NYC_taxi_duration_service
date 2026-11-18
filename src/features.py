def add_features(X):
    """First pipeline step: runs in training and at predict time (API).

    Turns raw inputs (FEATURES) into model features: time features from the
    pickup datetime and a pickup_dropoff route. Drops the raw datetime, since
    models can't use it directly and absolute dates don't generalize.
    """
    X = X.copy()
    pickup = X["tpep_pickup_datetime"]
    X["hour"] = pickup.dt.hour
    X["day_of_week"] = pickup.dt.day_of_week
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
    X["route"] = X["pulocationid"].astype(str) + "_" + X["dolocationid"].astype(str)
    return X.drop(columns=["tpep_pickup_datetime"])
