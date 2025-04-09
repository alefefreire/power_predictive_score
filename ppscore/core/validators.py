import pandas as pd


def validate_dataframe(df) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\n"
            f"Please convert your input to a pandas.DataFrame"
        )


def validate_column_in_df(column: str, df: pd.DataFrame) -> None:
    if not is_column_in_df(column, df):
        raise ValueError(
            f"The '{column}' argument should be the name of a dataframe column but the variable "
            f"that you passed is not a column in the given dataframe.\n"
            f"Please review the column name or your dataframe"
        )


def validate_unique_column(column: str, df: pd.DataFrame) -> None:
    if len(df[[column]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[column]].columns)} columns with the same column name {column}\n"
            f"Please adjust the dataframe and make sure that only 1 column has the name {column}"
        )


def validate_output_format(output: str) -> None:
    if output not in ["df", "list"]:
        raise ValueError(
            f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\n"""
            f"""Please adjust your input to one of the valid values"""
        )


def validate_sorted_param(sorted_param: bool) -> None:
    if sorted_param not in [True, False]:
        raise ValueError(
            f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted_param}\n"""
            f"""Please adjust your input to one of the valid values"""
        )


def is_column_in_df(column: str, df: pd.DataFrame) -> bool:
    try:
        return column in df.columns
    except Exception:
        return False
