import pytest

from ppscore.core.models import ScoreTask, TaskType
from ppscore.core.task_registry import get_invalid_task, get_task_registry


def test_get_task_registry_keys():
    registry = get_task_registry()
    expected_keys = {
        TaskType.REGRESSION,
        TaskType.CLASSIFICATION,
        TaskType.PREDICT_ITSELF,
        TaskType.TARGET_IS_CONSTANT,
        TaskType.TARGET_IS_ID,
        TaskType.FEATURE_IS_ID,
    }
    assert (
        set(registry.keys()) == expected_keys
    ), "Registry keys do not match expected TaskTypes"


def test_get_task_registry_values_are_score_tasks():
    registry = get_task_registry()
    for task_type, score_task in registry.items():
        assert isinstance(
            score_task, ScoreTask
        ), f"Value for {task_type} is not a ScoreTask"


def test_get_task_registry_regression_task():
    registry = get_task_registry()
    regression_task = registry[TaskType.REGRESSION]

    assert regression_task.type == TaskType.REGRESSION
    assert regression_task.metric_name == "mean absolute error"
    assert regression_task.metric_key == "neg_mean_absolute_error"
    assert regression_task.model is not None, "Regression task model should not be None"
    assert (
        regression_task.score_normalizer is not None
    ), "Regression task score_normalizer should not be None"


def test_get_task_registry_classification_task():
    registry = get_task_registry()
    classification_task = registry[TaskType.CLASSIFICATION]

    assert classification_task.type == TaskType.CLASSIFICATION
    assert classification_task.metric_name == "weighted F1"
    assert classification_task.metric_key == "f1_weighted"
    assert (
        classification_task.model is not None
    ), "Classification task model should not be None"
    assert (
        classification_task.score_normalizer is not None
    ), "Classification task score_normalizer should not be None"


def test_get_task_registry_predict_itself_task():
    registry = get_task_registry()
    predict_itself_task = registry[TaskType.PREDICT_ITSELF]

    assert predict_itself_task.type == TaskType.PREDICT_ITSELF
    assert predict_itself_task.model_score == 1
    assert predict_itself_task.baseline_score == 0
    assert predict_itself_task.ppscore == 1
    assert predict_itself_task.metric_name is None
    assert predict_itself_task.metric_key is None
    assert predict_itself_task.model is None
    assert predict_itself_task.score_normalizer is None


def test_get_invalid_task_valid_case_type():
    invalid_score = -1
    case_type = TaskType.TARGET_IS_DATETIME

    invalid_task = get_invalid_task(case_type, invalid_score)

    assert invalid_task.type == case_type
    assert invalid_task.is_valid_score is False
    assert invalid_task.model_score == invalid_score
    assert invalid_task.baseline_score == invalid_score
    assert invalid_task.ppscore == invalid_score
    assert invalid_task.metric_name is None
    assert invalid_task.metric_key is None
    assert invalid_task.model is None
    assert invalid_task.score_normalizer is None


def test_get_invalid_task_unsupported_case_type():
    invalid_score = -1
    unsupported_case_type = "UNSUPPORTED_CASE"

    with pytest.raises(Exception) as excinfo:
        get_invalid_task(unsupported_case_type, invalid_score)

    assert str(excinfo.value) == f"case_type {unsupported_case_type} is not supported"
