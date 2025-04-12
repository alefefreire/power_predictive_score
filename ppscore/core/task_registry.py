from sklearn import tree  # type: ignore

from ppscore.core.metrics import f1_normalizer, mae_normalizer
from ppscore.core.models import ScoreTask, TaskType


def get_task_registry() -> dict[TaskType, ScoreTask]:
    """Returns a dictionary with all tasks and their corresponding ScoreTask"""
    return {
        TaskType.REGRESSION: ScoreTask(
            type=TaskType.REGRESSION,
            is_valid_score=True,
            model_score=-1,  # TO_BE_CALCULATED
            baseline_score=-1,  # TO_BE_CALCULATED
            ppscore=-1,  # TO_BE_CALCULATED
            metric_name="mean absolute error",
            metric_key="neg_mean_absolute_error",
            model=tree.DecisionTreeRegressor(),
            score_normalizer=mae_normalizer,
        ),
        TaskType.CLASSIFICATION: ScoreTask(
            type=TaskType.CLASSIFICATION,
            is_valid_score=True,
            model_score=-1,  # TO_BE_CALCULATED
            baseline_score=-1,  # TO_BE_CALCULATED
            ppscore=-1,  # TO_BE_CALCULATED
            metric_name="weighted F1",
            metric_key="f1_weighted",
            model=tree.DecisionTreeClassifier(),
            score_normalizer=f1_normalizer,
        ),
        TaskType.PREDICT_ITSELF: ScoreTask(
            type=TaskType.PREDICT_ITSELF,
            is_valid_score=True,
            model_score=1,
            baseline_score=0,
            ppscore=1,
            metric_name=None,
            metric_key=None,
            model=None,
            score_normalizer=None,
        ),
        TaskType.TARGET_IS_CONSTANT: ScoreTask(
            type=TaskType.TARGET_IS_CONSTANT,
            is_valid_score=True,
            model_score=1,
            baseline_score=1,
            ppscore=0,
            metric_name=None,
            metric_key=None,
            model=None,
            score_normalizer=None,
        ),
        TaskType.TARGET_IS_ID: ScoreTask(
            type=TaskType.TARGET_IS_ID,
            is_valid_score=True,
            model_score=0,
            baseline_score=0,
            ppscore=0,
            metric_name=None,
            metric_key=None,
            model=None,
            score_normalizer=None,
        ),
        TaskType.FEATURE_IS_ID: ScoreTask(
            type=TaskType.FEATURE_IS_ID,
            is_valid_score=True,
            model_score=0,
            baseline_score=0,
            ppscore=0,
            metric_name=None,
            metric_key=None,
            model=None,
            score_normalizer=None,
        ),
    }


def get_invalid_task(case_type: TaskType, invalid_score: float) -> ScoreTask:
    """Get an invalid task for cases where calculation is not possible"""
    if case_type in get_task_registry().keys():
        return get_task_registry()[case_type]  # type: ignore
    elif case_type in [
        TaskType.TARGET_IS_DATETIME,
        TaskType.TARGET_DATA_TYPE_NOT_SUPPORTED,
        TaskType.EMPTY_DATAFRAME_AFTER_DROPPING_NA,
        TaskType.UNKNOWN_ERROR,
    ]:
        return ScoreTask(
            type=case_type,
            is_valid_score=False,
            model_score=invalid_score,
            baseline_score=invalid_score,
            ppscore=invalid_score,
            metric_name=None,
            metric_key=None,
            model=None,
            score_normalizer=None,
        )
    raise Exception(f"case_type {case_type} is not supported")
