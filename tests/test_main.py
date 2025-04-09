from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ppscore.main import score


@patch("ppscore.main.PPSCalculator")
def test_score_valid(mock_calculator):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    mock_result = MagicMock()
    mock_result.dict.return_value = {"ppscore": 0.5, "case": "classification"}
    mock_calculator.return_value.score.return_value = mock_result

    result = score(df, "x", "y")

    assert result == {"ppscore": 0.5, "case": "classification"}
    mock_calculator.return_value.score.assert_called_once_with(
        df, "x", "y", 5000, 4, 123, 0, True
    )


def test_score_raises_attribute_error():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    with pytest.raises(
        AttributeError, match="The attribute 'task' is no longer supported"
    ):
        score(df, "x", "y", task="deprecated_task")
