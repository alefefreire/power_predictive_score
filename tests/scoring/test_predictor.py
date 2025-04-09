from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ppscore.core.models import PPSResult, TaskType
from ppscore.scoring.predictor import PPSCalculator


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": ["A", "B", "A", "C"],
            "target": [10, 20, 30, 40],
        }
    )


@patch("ppscore.scoring.predictor.determine_case_and_prepare_df")
@patch("ppscore.scoring.predictor.calculate_model_cv_score")
@patch("ppscore.scoring.predictor.get_invalid_task")
def test_score_classification(
    mock_get_invalid_task,
    mock_calculate_model_cv_score,
    mock_determine_case_and_prepare_df,
    sample_dataframe,
):
    # Mock dependencies
    mock_determine_case_and_prepare_df.return_value = (
        sample_dataframe,
        TaskType.CLASSIFICATION,
    )
    mock_calculate_model_cv_score.return_value = 0.8
    mock_task = MagicMock()
    mock_task.score_normalizer.return_value = (0.7, 0.5)
    mock_task.metric_name = "accuracy"
    mock_get_invalid_task.return_value = mock_task

    calculator = PPSCalculator()
    result = calculator.score(sample_dataframe, "feature2", "target")

    assert isinstance(result, PPSResult)
    assert result.ppscore == 0.7
    assert result.case == TaskType.CLASSIFICATION
    mock_calculate_model_cv_score.assert_called_once()


@patch("ppscore.scoring.predictor.determine_case_and_prepare_df")
@patch("ppscore.scoring.predictor.get_invalid_task")
def test_score_invalid_task(
    mock_get_invalid_task, mock_determine_case_and_prepare_df, sample_dataframe
):
    # Mock dependencies
    mock_determine_case_and_prepare_df.return_value = (
        sample_dataframe,
        TaskType.UNKNOWN_ERROR,
    )
    mock_task = MagicMock()
    mock_task.ppscore = 0
    mock_task.is_valid_score = False
    mock_task.metric_name = "accuracy"
    mock_get_invalid_task.return_value = mock_task

    calculator = PPSCalculator()
    result = calculator.score(sample_dataframe, "feature1", "target")

    assert isinstance(result, PPSResult)
    assert result.ppscore == 0
    assert result.case == TaskType.UNKNOWN_ERROR
    mock_get_invalid_task.assert_called_once()


@patch("ppscore.scoring.predictor.PPSCalculator.score")
def test_predictors(mock_score, sample_dataframe):
    # Mock the score method
    mock_score.return_value = PPSResult(
        x="feature1",
        y="target",
        ppscore=0.5,
        case=TaskType.CLASSIFICATION,
        is_valid_score=True,
        metric="accuracy",
        baseline_score=0.4,
        model_score=0.6,
        model=None,
    )

    calculator = PPSCalculator()
    result = calculator.predictors(sample_dataframe, "target")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two features excluding the target
    assert "ppscore" in result.columns
    mock_score.assert_called()


@patch("ppscore.scoring.predictor.PPSCalculator.score")
def test_matrix(mock_score, sample_dataframe):
    # Mock the score method
    mock_score.return_value = PPSResult(
        x="feature1",
        y="feature2",
        ppscore=0.3,
        case=TaskType.CLASSIFICATION,
        is_valid_score=True,
        metric="accuracy",
        baseline_score=0.2,
        model_score=0.4,
        model=None,
    )

    calculator = PPSCalculator()
    result = calculator.matrix(sample_dataframe)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 9  # 3x3 matrix (all column pairs)
    assert "ppscore" in result.columns
    mock_score.assert_called()
