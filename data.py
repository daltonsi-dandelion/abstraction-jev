"""Data loading helpers."""
import logging

import pandas as pd

log = logging.getLogger("data")


def load_parquet(s3_uri: str, aws_profile: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file from S3 into a DataFrame (requires s3fs + pyarrow)."""
    storage_options = {"profile": aws_profile} if aws_profile else None
    log.info("Reading %s", s3_uri)
    df = pd.read_parquet(s3_uri, columns=columns, storage_options=storage_options)
    log.info("Loaded %d rows x %d cols", *df.shape)
    return df
