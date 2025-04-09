import numpy as np

from ppscore.core.data_types import dtype_represents_categories
from ppscore.core.models import TaskType


def calculate_model_cv_score(
    df, target, feature, task, cross_validation, random_seed, **kwargs
):
    """Calculates the mean model score based on cross-validation"""
    from sklearn import preprocessing  # type: ignore
    from sklearn.model_selection import cross_val_score  # type: ignore

    metric = task.metric_key
    model = task.model

    # Shuffle the rows for better cross-validation
    df = df.sample(frac=1, random_state=random_seed, replace=False)

    # Preprocess target
    if task.type == TaskType.CLASSIFICATION:
        label_encoder = preprocessing.LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target])
        target_series = df[target]
    else:
        target_series = df[target]

    # Preprocess feature
    if dtype_represents_categories(df[feature]):
        one_hot_encoder = preprocessing.OneHotEncoder()
        array = df[feature].__array__()
        sparse_matrix = one_hot_encoder.fit_transform(array.reshape(-1, 1))
        feature_input = sparse_matrix
    else:
        # Reshaping needed because there is only 1 feature
        array = df[feature].values
        if not isinstance(array, np.ndarray):  # e.g Int64 IntegerArray
            array = array.to_numpy()
        feature_input = array.reshape(-1, 1)

    # Cross-validation
    scores = cross_val_score(
        model,
        feature_input,
        target_series.to_numpy(),
        cv=cross_validation,
        scoring=metric,
    )

    return scores.mean()
