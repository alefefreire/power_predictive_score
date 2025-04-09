import pandas as pd

from ppscore.core.metrics import f1_normalizer


def test_f1_normalizer_most_common_value_baseline():
    df = pd.DataFrame({"y": ["cat1", "cat1", "cat2", "cat1", "cat2"]})
    model_score = 0.8
    random_seed = 42

    ppscore, baseline_score = f1_normalizer(df, "y", model_score, random_seed)

    assert 0 <= ppscore <= 1, "PPS should be normalized between 0 and 1"
    assert baseline_score > 0, "Baseline F1 score should be greater than 0"


def test_f1_normalizer_random_baseline():
    df = pd.DataFrame({"y": ["cat1", "cat2", "cat3", "cat4", "cat5"]})
    model_score = 0.5
    random_seed = 42

    ppscore, baseline_score = f1_normalizer(df, "y", model_score, random_seed)

    assert 0 <= ppscore <= 1, "PPS should be normalized between 0 and 1"
    assert baseline_score > 0, "Baseline F1 score should be greater than 0"


def test_f1_normalizer_single_unique_value():
    df = pd.DataFrame({"y": ["cat1", "cat1", "cat1", "cat1"]})
    model_score = 0.5
    random_seed = 42

    ppscore, baseline_score = f1_normalizer(df, "y", model_score, random_seed)

    assert (
        ppscore == 0
    ), "PPS should be 0 when the model score is less than or equal to the baseline"
    assert (
        baseline_score == 1
    ), "Baseline F1 score should be 1 when all values are the same"
