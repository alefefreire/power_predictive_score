from typing import Tuple

import pandas as pd


def normalized_mae_score(model_mae: float, naive_mae: float) -> float:
    """Normalizes the model MAE score, given the baseline score"""
    if model_mae > naive_mae:
        return 0
    else:
        return 1 - (model_mae / naive_mae)


def mae_normalizer(
    df: pd.DataFrame, y: str, model_score: float, **kwargs
) -> Tuple[float, float]:
    """In case of MAE, calculates the baseline score for y and derives the PPS."""
    from sklearn.metrics import mean_absolute_error  # type: ignore

    df["naive"] = df[y].median()
    baseline_score = mean_absolute_error(df[y].to_numpy(), df["naive"].to_numpy())

    ppscore = normalized_mae_score(abs(model_score), baseline_score)
    return ppscore, baseline_score


def normalized_f1_score(model_f1: float, baseline_f1: float) -> float:
    """Normalizes the model F1 score, given the baseline score"""
    if model_f1 < baseline_f1:
        return 0
    else:
        scale_range = 1.0 - baseline_f1
        f1_diff = model_f1 - baseline_f1
        return f1_diff / scale_range


def f1_normalizer(
    df: pd.DataFrame, y: str, model_score: float, random_seed: int
) -> Tuple[float, float]:
    """In case of F1, calculates the baseline score for y and derives the PPS."""
    from sklearn import preprocessing  # type: ignore
    from sklearn.metrics import f1_score

    label_encoder = preprocessing.LabelEncoder()
    df["truth"] = label_encoder.fit_transform(df[y])
    df["most_common_value"] = df["truth"].value_counts().index[0]
    random = df["truth"].sample(frac=1, random_state=random_seed)

    baseline_score = max(
        f1_score(df["truth"], df["most_common_value"], average="weighted"),
        f1_score(df["truth"], random, average="weighted"),
    )

    ppscore = normalized_f1_score(model_score, baseline_score)
    return ppscore, baseline_score
