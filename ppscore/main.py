from ppscore.scoring.predictor import PPSCalculator


def score(
    df,
    x,
    y,
    task=None,  # Deprecated parameter
    sample=5_000,
    cross_validation=4,
    random_seed=123,
    invalid_score=0,
    catch_errors=True,
):
    """Compatibility function for existing code"""
    if task is not None:
        raise AttributeError(
            "The attribute 'task' is no longer supported because it led to confusion and inconsistencies.\n"
            "The task of the model is now determined based on the data types of the columns."
        )

    calculator = PPSCalculator()
    result = calculator.score(
        df, x, y, sample, cross_validation, random_seed, invalid_score, catch_errors
    )

    # Convert Pydantic model to dict for backwards compatibility
    return result.dict()


def predictors(df, y, output="df", sorted=True, **kwargs):
    """Compatibility function for existing code"""
    calculator = PPSCalculator()
    return calculator.predictors(df, y, output, sorted, **kwargs)


def matrix(df, output="df", sorted=False, **kwargs):
    """Compatibility function for existing code"""
    calculator = PPSCalculator()
    return calculator.matrix(df, output, sorted, **kwargs)
