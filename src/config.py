# Raw model inputs (X): only info known at pickup; also what the API must receive.
FEATURES = ["pulocationid", "dolocationid", "tpep_pickup_datetime"]

TARGET = "duration"

# minutes
DURATION_RANGE = (3, 120)

TARGET_ENCODED_COLS = ["pulocationid", "dolocationid", "route"]
NUM_COLS = ["hour", "day_of_week", "is_weekend"]
