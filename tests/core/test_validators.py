import pandas as pd
import pytest

from ppscore.core.validators import (
    is_column_in_df,
    validate_column_in_df,
    validate_dataframe,
    validate_output_format,
    validate_sorted_param,
    validate_unique_column,
)


def test_validate_dataframe():
    # Valid DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3]})
    assert validate_dataframe(df) is None

    # Invalid DataFrame
    with pytest.raises(
        TypeError, match="The 'df' argument should be a pandas.DataFrame"
    ):
        validate_dataframe([1, 2, 3])


def test_validate_column_in_df():
    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Valid column
    assert validate_column_in_df("col1", df) is None

    # Invalid column
    with pytest.raises(ValueError, match="is not a column in the given dataframe"):
        validate_column_in_df("col2", df)


def test_validate_unique_column():
    df = pd.DataFrame({"col1": [1, 2, 3], "col1.1": [4, 5, 6]})

    # Valid unique column
    assert validate_unique_column("col1", df) is None

    # Duplicate column names
    df.columns = ["col1", "col1"]
    with pytest.raises(AssertionError, match="columns with the same column name"):
        validate_unique_column("col1", df)


def test_validate_output_format():
    # Valid output formats
    assert validate_output_format("df") is None
    assert validate_output_format("list") is None

    # Invalid output format
    with pytest.raises(ValueError, match="should be one of"):
        validate_output_format("invalid")


def test_validate_sorted_param():
    # Valid sorted_param values
    assert validate_sorted_param(True) is None
    assert validate_sorted_param(False) is None

    # Invalid sorted_param value
    with pytest.raises(ValueError, match="should be one of"):
        validate_sorted_param("invalid")


def test_is_column_in_df():
    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Column exists
    assert is_column_in_df("col1", df) is True

    # Column does not exist
    assert is_column_in_df("col2", df) is False
