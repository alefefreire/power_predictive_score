import numpy as np
import pandas as pd
import pytest

from ppscore.core.modelling import calculate_model_cv_score
from ppscore.core.models import ScoreTask, TaskType


@pytest.fixture
def mock_task_classification():
    return ScoreTask(
        type=TaskType.CLASSIFICATION,
        metric_key="accuracy",
        model=None,  # Mocked durante o teste
        is_valid_score=True,
        model_score=-1,
        baseline_score=-1,
        ppscore=-1,
        metric_name="accuracy",
        score_normalizer=None,
    )


@pytest.fixture
def mock_task_regression():
    return ScoreTask(
        type=TaskType.REGRESSION,
        metric_key="neg_mean_absolute_error",
        model=None,  # Mocked durante o teste
        is_valid_score=True,
        model_score=-1,
        baseline_score=-1,
        ppscore=-1,
        metric_name="mean absolute error",
        score_normalizer=None,
    )


def test_calculate_model_cv_score_classification(mocker, mock_task_classification):
    mock_cross_val_score = mocker.patch("sklearn.model_selection.cross_val_score")
    mock_cross_val_score.return_value = np.array([0.8, 0.85, 0.9])

    df = pd.DataFrame(
        {"feature": ["A", "B", "A", "C"], "target": ["yes", "no", "yes", "no"]}
    )
    mock_task_classification.model = "mock_model"  # Mock do modelo

    result = calculate_model_cv_score(
        df,
        target="target",
        feature="feature",
        task=mock_task_classification,
        cross_validation=3,
        random_seed=42,
    )

    assert result == pytest.approx(
        0.85
    ), "O score médio de cross-validation deve ser calculado corretamente"
    mock_cross_val_score.assert_called_once()


def test_calculate_model_cv_score_regression(mocker, mock_task_regression):
    mock_cross_val_score = mocker.patch("sklearn.model_selection.cross_val_score")
    mock_cross_val_score.return_value = np.array([-10, -12, -11])

    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [10, 20, 30, 40]})
    mock_task_regression.model = "mock_model"  # Mock do modelo

    result = calculate_model_cv_score(
        df,
        target="target",
        feature="feature",
        task=mock_task_regression,
        cross_validation=3,
        random_seed=42,
    )

    assert result == pytest.approx(
        -11
    ), "O score médio de cross-validation deve ser calculado corretamente"
    mock_cross_val_score.assert_called_once()


def test_calculate_model_cv_score_empty_dataframe(mock_task_classification):
    df = pd.DataFrame({"feature": [], "target": []})

    with pytest.raises(ValueError):
        calculate_model_cv_score(
            df,
            target="target",
            feature="feature",
            task=mock_task_classification,
            cross_validation=3,
            random_seed=42,
        )


def test_calculate_model_cv_score_invalid_feature(mock_task_classification):
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [1, 2, 3]})

    with pytest.raises(KeyError):
        calculate_model_cv_score(
            df,
            target="target",
            feature="invalid_feature",
            task=mock_task_classification,
            cross_validation=3,
            random_seed=42,
        )
