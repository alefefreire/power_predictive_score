from enum import Enum
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    PREDICT_ITSELF = "predict_itself"
    TARGET_IS_CONSTANT = "target_is_constant"
    TARGET_IS_ID = "target_is_id"
    FEATURE_IS_ID = "feature_is_id"
    TARGET_IS_DATETIME = "target_is_datetime"
    TARGET_DATA_TYPE_NOT_SUPPORTED = "target_data_type_not_supported"
    EMPTY_DATAFRAME_AFTER_DROPPING_NA = "empty_dataframe_after_dropping_na"
    UNKNOWN_ERROR = "unknown_error"


class ScoreTask(BaseModel):
    type: TaskType
    is_valid_score: bool
    model_score: Union[float, int]
    baseline_score: Union[float, int]
    ppscore: Union[float, int]
    metric_name: Optional[str] = None
    metric_key: Optional[str] = None
    model: Any = None
    score_normalizer: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True


class PPSResult(BaseModel):
    x: str
    y: str
    ppscore: float
    case: str
    is_valid_score: bool
    metric: Optional[str] = None
    baseline_score: float
    model_score: float
    model: Any = None

    class Config:
        arbitrary_types_allowed = True
