# common_functions.py

import logging
from datetime import datetime
from typing import Any, Optional, Union
import pandas as pd
import numpy as np

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def to_datetime(timestamp: Union[str, int, float], unit: str = 'ms') -> Optional[datetime]:
    try:
        return pd.to_datetime(timestamp, unit=unit)
    except Exception as e:
        logging.warning(f"Failed to convert timestamp: {timestamp} | Error: {e}")
        return None

def normalize_symbol(symbol: str) -> str:
    return symbol.upper().replace("/", "")

def round_price(value: float, precision: int = 2) -> float:
    return round(value, precision)

def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)

def is_valid_dataframe(df: Any) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def scale_series(series: pd.Series, method: str = "minmax") -> pd.Series:
    if method == "minmax":
        return (series - series.min()) / (series.max() - series.min())
    elif method == "zscore":
        return (series - series.mean()) / series.std(ddof=0)
    else:
        return series
