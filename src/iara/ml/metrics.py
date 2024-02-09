"""
Module for model evaluation metrics and result compilation.

This module provides classes for computing evaluation metrics commonly used in model analysis,
as well as for compiling and formatting evaluation results from cross-validation and grid search.
"""
import enum
import typing
import math

import numpy as np
import sklearn.metrics as sk_metrics


class Metric(enum.Enum):
    """Enumeration representing metrics for model analysis.

    This enumeration provides a set of metrics commonly used for evaluating multiclasses models and
        a eval method to perform this evaluation.
    """
    ACCURACY = 1
    BALANCED_ACCURACY = 2
    MICRO_F1 = 3
    MACRO_F1 = 4

    def __str__(self):
        """Return the string representation of the Metric enum."""
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

    def as_label(self):
        """Return the human-readable label of the metric."""
        en_labels = {
            __class__.ACCURACY: "ACCURACY",
            __class__.BALANCED_ACCURACY: "BALANCED_ACCURACY",
            __class__.MICRO_F1: "MICRO_F1",
            __class__.MACRO_F1: "MACRO_F1",
        }
        return en_labels[self]

    def compute(self, target: typing.Iterable[int], prediction: typing.Iterable[int]) -> float:
        """Compute the metric value based on the target and prediction.

        Args:
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.

        Returns:
            float: Computed metric value.
        """
        if self == Metric.ACCURACY:
            return sk_metrics.accuracy_score(target, prediction) * 100

        if self == Metric.BALANCED_ACCURACY:
            return sk_metrics.balanced_accuracy_score(target, prediction) * 100

        if self == Metric.MICRO_F1:
            return sk_metrics.f1_score(target, prediction, average='micro') * 100

        if self == Metric.MACRO_F1:
            return sk_metrics.f1_score(target, prediction, average='macro') * 100

        raise NotImplementedError(f"Evaluation for Metric {self} is not implemented.")

    @staticmethod
    def compute_all(metric_list: typing.List['Metric'],
                    target: typing.Iterable[int],
                    prediction: typing.Iterable[int]) -> typing.Dict['Metric', float]:
        """Compute all metrics in the given list.

        Args:
            metric_list (List[Metric]): List of metrics to compute.
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.

        Returns:
            Dict[Metric, float]: Dictionary containing computed metric values for each metric
                in the list.
        """
        dict_values = {}
        for metric in metric_list:
            dict_values[metric] = metric.compute(target, prediction)
        return dict_values


class CrossValidationCompiler():
    """Class for compiling cross-validation results.

    This class compiles the results of cross-validation evaluations, including metric scores
    for each fold and each metric, and provides methods for formatting the results.
    """

    def __init__(self) -> None:
        """Initialize the CrossValidationCompiler object."""

        self._score_dict = {
            'i_fold':[],
            'n_samples':[],
            'metrics': []
        }

        for metric in Metric:
            self._score_dict[str(metric)] = []

    def add(self,
            i_fold: int,
            metric_list: typing.List['Metric'],
            target: typing.Iterable[int],
            prediction: typing.Iterable[int]) -> None:
        """Add evaluation results for a fold.

        Args:
            i_fold (int): Fold index.
            metric_list (List[Metric]): List of metrics to compute.
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.
        """

        self._score_dict['i_fold'].append(i_fold)
        self._score_dict['n_samples'].append(len(target))

        for metric, score in Metric.compute_all(metric_list, target, prediction).items():
            self._score_dict[str(metric)].append(score)

        self._score_dict['metrics'].extend(metric_list)
        self._score_dict['metrics'] = list(set(self._score_dict['metrics']))

    @staticmethod
    def str_format(values, n_samples=60, tex_format=False) -> str:
        """Format the values as a string.

        Args:
            values: Values to format.
            n_samples (int, optional): Number of samples to compute the decimal places.
                Defaults to 60.
            tex_format (bool, optional): Whether to format as LaTeX. Defaults to False.

        Returns:
            str: Formatted string.
        """
        decimal_places = int(math.log10(math.sqrt(n_samples))+1)
        if tex_format:
            return f'${np.mean(values):.{decimal_places}f} \\pm \
                {np.std(values):.{decimal_places}f}$'

        return f'{np.mean(values):.{decimal_places}f} \u00B1 {np.std(values):.{decimal_places}f}'

    @staticmethod
    def table_to_str(table: typing.List[typing.List[str]]) -> str:
        """Convert the table to a formatted string.

        Args:
            table (List[List]): Table to convert.

        Returns:
            str: Formatted string representation of the table.
        """
        num_columns = len(table[0])
        column_widths = [max(len(str(row[i])) for row in table) for i in range(num_columns)]

        formatted_rows = []
        for row in table:
            formatted_row = [str(row[i]).ljust(column_widths[i]) for i in range(num_columns)]
            formatted_rows.append('  '.join(formatted_row))

        return '\n'.join(formatted_rows)

    def metric_as_str(self, metric, tex_format=False):
        """Get the metric as a formatted string.

        Args:
            metric: Metric to format.
            tex_format (bool, optional): Whether to format as LaTeX. Defaults to False.

        Returns:
            str: Formatted string representation of the metric.
        """
        return CrossValidationCompiler.str_format(self._score_dict[str(metric)],
                                                    np.mean(self._score_dict['n_samples']),
                                                    tex_format)

    def as_str(self, tex_format=False):
        """Get the compiled results as a formatted string.

        Args:
            tex_format (bool, optional): Whether to format as LaTeX. Defaults to False.

        Returns:
            str: Formatted string representation of the compiled results.
        """
        ret = ['' for _ in self._score_dict['metrics']]
        for i, metric in enumerate(self._score_dict['metrics']):
            ret[i] = self.metric_as_str(metric, tex_format)
        return ret

    def __str__(self) -> str:
        return self.as_str()


class GridCompiler():
    """Class for compiling grid search results.

    This class compiles the results of grid search evaluations, including metric scores for each
    combination of parameters and each metric, and provides methods for formatting the results.
    """
    default_metric_list = Metric
    default_fig_size = (10, 7)

    def __init__(self, metric_list: typing.List['Metric'] = default_metric_list):
        """
        Args:
            metric_list (List[Metric], optional): List of metrics to compute.
                Defaults to all Metrics available.
        """
        self.cv_dict = {}
        self.param_dict = {}
        self.params = None
        self.metric_list = metric_list

    def add(self,
            params: typing.Dict,
            i_fold: int,
            target: typing.Iterable[int],
            prediction: typing.Iterable[int]) -> None:
        """
        Add evaluation results for a specific combination of parameters and fold.

        Args:
            grid_id (str): Identifier for the grid search combination of parameters.
            i_fold (int): Index of the fold.
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.
        """
        params_hash = hash(tuple(params.items()))
        self.params = params.keys()

        if not params_hash in self.cv_dict:
            self.cv_dict[params_hash]  = {
                'params': params,
                'cv': CrossValidationCompiler(),
            }
            self.param_dict[params_hash] = params

        self.cv_dict[params_hash]['cv'].add(i_fold = i_fold,
                                metric_list = self.metric_list,
                                target = target,
                                prediction = prediction)

    def as_table(self, tex_format=False) -> typing.List[typing.List[str]]:
        """
        Get the compiled results as a formatted table.

        Args:
            tex_format (bool, optional): Whether to format the table for LaTeX. Defaults to False.

        Returns:
            List[List[str]]: Formatted table representation of the compiled results.
        """
        table = [[''] * (len(self.params) + len(self.metric_list)) for _ in range(len(self.cv_dict)+1)]

        j = 0
        for param in self.params:
            table[0][j] = str(param).replace('_', ' ')
            j = j + 1

        for metric in self.metric_list:
            table[0][j] = metric.as_label()
            j += 1

        i = 1
        for _, cv_dict in self.cv_dict.items():
            j = 0

            for _, param_value in cv_dict['params'].items():
                table[i][j] = str(param_value)
                j = j+1

            for metric in self.metric_list:
                table[i][j] = cv_dict['cv'].metric_as_str(metric, tex_format=tex_format)
                j += 1

            i += 1

        return table

    def as_str(self, tex_format=False):
        """
        Get the compiled results as a formatted string.

        Args:
            tex_format (bool, optional): Whether to format the string for LaTeX. Defaults to False.

        Returns:
            str: Formatted string representation of the compiled results.
        """
        return CrossValidationCompiler.table_to_str(
            self.as_table(tex_format=tex_format)
        )

    def __str__(self) -> str:
        return self.as_str()
