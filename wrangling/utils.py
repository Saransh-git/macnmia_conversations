import sys
from datetime import datetime, timedelta

import numpy as np
from pandas import DataFrame


def convert_to_python_datetime(datetime_str, format_str='%m/%d/%Y'):
    try:
        ret_val = datetime.strptime(datetime_str, format_str)
    except TypeError:
        ret_val = np.nan
    finally:
        return ret_val


def convert_datetime_cols_to_python_datetime(
        df, cols=['request_date', 'created_date', 'ship_date', 'settle_date'],
        conversion_func=convert_to_python_datetime
):
    for col in cols:
        df[col] = df[col].astype(np.object)  # making sure the data is string type
        df[col] = df[col].parallel_apply(conversion_func)
    return df


def convert_timedelata_to_days(delta: timedelta):
    return delta.days


def filter_based_on_ranges(
        df: DataFrame, col_name: str = None, lower_bound: int = 0, upper_bound: int = sys.maxsize
) -> DataFrame:
    if col_name is None:
        raise ValueError('No column name provided')
    return df.loc[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]
