from typing import Optional, Tuple

import pandas as pd

from ppscore.core.models import TaskType


def dtype_represents_categories(series: pd.Series) -> bool:
    """Determines if the dtype of the series represents categorical values"""
    from pandas.api.types import is_categorical_dtype  # type: ignore
    from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

    return (
        is_bool_dtype(series)
        or is_object_dtype(series)
        or is_string_dtype(series)
        or is_categorical_dtype(series)
    )


def feature_is_id(df: pd.DataFrame, x: str) -> bool:
    """Returns Boolean if the feature column x is an ID"""
    if not dtype_represents_categories(df[x]):
        return False

    category_count = df[x].value_counts().count()
    return category_count == len(df[x])


def determine_case_and_prepare_df(
    df: pd.DataFrame, x: str, y: str, sample: int = 5_000, random_seed: int = 123
) -> Tuple[pd.DataFrame, TaskType]:
    """Returns str with the name of the determined case based on the columns x and y"""
    from pandas.api.types import (
        is_datetime64_any_dtype,
        is_numeric_dtype,
        is_timedelta64_dtype,
    )

    if x == y:
        return df, TaskType.PREDICT_ITSELF

    df = df[[x, y]]
    df = df.dropna()

    if len(df) == 0:
        return df, TaskType.EMPTY_DATAFRAME_AFTER_DROPPING_NA

    df = maybe_sample(df, sample, random_seed=random_seed)

    if feature_is_id(df, x):
        return df, TaskType.FEATURE_IS_ID

    category_count = df[y].value_counts().count()
    if category_count == 1:
        return df, TaskType.TARGET_IS_CONSTANT
    if dtype_represents_categories(df[y]) and (category_count == len(df[y])):
        return df, TaskType.TARGET_IS_ID

    if dtype_represents_categories(df[y]):
        return df, TaskType.CLASSIFICATION
    if is_numeric_dtype(df[y]):
        return df, TaskType.REGRESSION

    if is_datetime64_any_dtype(df[y]) or is_timedelta64_dtype(df[y]):
        return df, TaskType.TARGET_IS_DATETIME

    return df, TaskType.TARGET_DATA_TYPE_NOT_SUPPORTED


def maybe_sample(
    df: pd.DataFrame, sample: int, random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Maybe samples the rows of the given df to have at most `sample` rows
    """
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=random_seed, replace=False)
    return df
