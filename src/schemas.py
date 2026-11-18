import pandera.pandas as pa

# Raw-file contract: every column the code reads, checked at load time.
# Other columns pass through unchecked (strict=False by default).
TripSchema = pa.DataFrameSchema(
    {
        # Model inputs (FEATURES in config.py)
        "pulocationid": pa.Column(int),
        "dolocationid": pa.Column(int),
        "tpep_pickup_datetime": pa.Column("datetime64[us]"),
        # Needed only to compute the TARGET (duration), then drop
        "tpep_dropoff_datetime": pa.Column("datetime64[us]"),
    },
    add_missing_columns=True,  # creates optional columns that are missing
    coerce=True,
)
