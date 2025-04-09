import pandas as pd

from ppscore.core.data_types import determine_case_and_prepare_df
from ppscore.core.modelling import calculate_model_cv_score
from ppscore.core.models import PPSResult, TaskType
from ppscore.core.task_registry import get_invalid_task, get_task_registry
from ppscore.core.validators import (
    validate_column_in_df,
    validate_dataframe,
    validate_output_format,
    validate_sorted_param,
    validate_unique_column,
)


class PPSCalculator:
    def __init__(self):
        self.task_registry = get_task_registry()

    def score(
        self,
        df,
        x,
        y,
        sample=5_000,
        cross_validation=4,
        random_seed=123,
        invalid_score=0,
        catch_errors=True,
    ):
        """
        Calculate the Predictive Power Score (PPS) for "x predicts y"
        """
        validate_dataframe(df)
        validate_column_in_df(x, df)
        validate_unique_column(x, df)
        validate_column_in_df(y, df)
        validate_unique_column(y, df)

        if random_seed is None:
            from random import random

            random_seed = int(random() * 1000)

        try:
            return self._calculate_score(
                df, x, y, sample, cross_validation, random_seed, invalid_score
            )
        except Exception as exception:
            if catch_errors:
                case_type = TaskType.UNKNOWN_ERROR
                task = get_invalid_task(case_type, invalid_score)
                return PPSResult(
                    x=x,
                    y=y,
                    ppscore=task.ppscore,
                    case=case_type,
                    is_valid_score=task.is_valid_score,
                    metric=task.metric_name,
                    baseline_score=task.baseline_score,
                    model_score=task.model_score,
                    model=task.model,
                )
            else:
                raise exception

    def _calculate_score(
        self, df, x, y, sample, cross_validation, random_seed, invalid_score
    ):
        df, case_type = determine_case_and_prepare_df(
            df, x, y, sample=sample, random_seed=random_seed
        )

        task = get_invalid_task(case_type, invalid_score)

        if case_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            model_score = calculate_model_cv_score(
                df,
                target=y,
                feature=x,
                task=task,
                cross_validation=cross_validation,
                random_seed=random_seed,
            )
            ppscore, baseline_score = task.score_normalizer(
                df, y, model_score, random_seed=random_seed
            )
        else:
            model_score = task.model_score
            baseline_score = task.baseline_score
            ppscore = task.ppscore

        return PPSResult(
            x=x,
            y=y,
            ppscore=ppscore,
            case=case_type,
            is_valid_score=task.is_valid_score,
            metric=task.metric_name,
            baseline_score=baseline_score,
            model_score=abs(model_score),  # sklearn returns negative mae
            model=task.model,
        )

    def format_results(self, scores, output, sorted_result):
        """Format list of score dicts"""
        if sorted_result:
            scores.sort(key=lambda item: item.ppscore, reverse=True)

        if output == "df":
            scores_dict = [score.dict() for score in scores]
            df_data = pd.DataFrame(scores_dict)
            return df_data

        return scores

    def predictors(self, df, y, output="df", sorted=True, **kwargs):
        """
        Calculate PPS of all features against a target column
        """
        validate_dataframe(df)
        validate_column_in_df(y, df)
        validate_unique_column(y, df)
        validate_output_format(output)
        validate_sorted_param(sorted)

        scores = [self.score(df, column, y, **kwargs) for column in df if column != y]

        return self.format_results(scores=scores, output=output, sorted_result=sorted)

    def matrix(self, df, output="df", sorted=False, **kwargs):
        """
        Calculate the PPS matrix for all columns in the dataframe
        """
        validate_dataframe(df)
        validate_output_format(output)
        validate_sorted_param(sorted)

        scores = [self.score(df, x, y, **kwargs) for x in df for y in df]

        return self.format_results(scores=scores, output=output, sorted_result=sorted)
