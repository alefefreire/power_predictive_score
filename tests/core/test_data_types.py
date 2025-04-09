import pandas as pd

from ppscore.core.data_types import determine_case_and_prepare_df
from ppscore.core.models import TaskType


def test_determine_case_and_prepare_df_predict_itself():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result_df, task_type = determine_case_and_prepare_df(df, "x", "x")
    assert task_type == TaskType.PREDICT_ITSELF


def test_determine_case_and_prepare_df_empty_after_dropping_na():
    df = pd.DataFrame({"x": [None, None], "y": [None, None]})
    result_df, task_type = determine_case_and_prepare_df(df, "x", "y")
    assert task_type == TaskType.EMPTY_DATAFRAME_AFTER_DROPPING_NA


def test_determine_case_and_prepare_df_feature_is_id():
    df = pd.DataFrame({"x": ["id1", "id2", "id3"], "y": [1, 2, 3]})
    result_df, task_type = determine_case_and_prepare_df(df, "x", "y")
    assert task_type == TaskType.FEATURE_IS_ID


def test_determine_case_and_prepare_df_target_is_constant():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 1, 1]})
    result_df, task_type = determine_case_and_prepare_df(df, "x", "y")
    assert task_type == TaskType.TARGET_IS_CONSTANT


def test_determine_case_and_prepare_df_target_is_id():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["id1", "id2", "id3"]})
    result_df, task_type = determine_case_and_prepare_df(df, "x", "y")
    assert task_type == TaskType.TARGET_IS_ID


def test_determine_case_and_prepare_df_classification():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["cat1", "cat2", "cat1"]})
    result_df, task_type = determine_case_and_prepare_df(df, "x", "y")
    assert task_type == TaskType.CLASSIFICATION


def test_determine_case_and_prepare_df_regression():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10.5, 20.5, 30.5]})
    result_df, task_type = determine_case_and_prepare_df(df, "x", "y")
    assert task_type == TaskType.REGRESSION


def test_determine_case_and_prepare_df_target_is_datetime():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        }
    )
    result_df, task_type = determine_case_and_prepare_df(df, "x", "y")
    assert task_type == TaskType.TARGET_IS_DATETIME
