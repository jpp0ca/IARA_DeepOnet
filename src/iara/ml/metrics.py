"""
Module for model evaluation metrics and result compilation.

This module provides classes for computing evaluation metrics commonly used in model analysis,
as well as for compiling and formatting evaluation results from cross-validation and grid search.
"""
import enum
import typing
import math

import numpy as np
import pandas as pd

import sklearn.metrics as sk_metrics
import scipy.stats as scipy

class Metric(enum.Enum):
    """Enumeration representing metrics for model analysis.

    This enumeration provides a set of metrics commonly used for evaluating multiclasses models and
        a eval method to perform this evaluation.
    """
    ACCURACY = 1
    BALANCED_ACCURACY = 2
    MICRO_F1 = 3
    MACRO_F1 = 4
    DETECTION_PROBABILITY = 5
    SP_INDEX = 6

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
            __class__.DETECTION_PROBABILITY: "DETECTION_PROBABILITY",
            __class__.SP_INDEX: "SP_INDEX",
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

        if self == Metric.DETECTION_PROBABILITY:
            cm = sk_metrics.confusion_matrix(target, prediction, labels=list(set(list(target))))
            detection_probabilities = cm.diagonal() / cm.sum(axis=1)
            return np.mean(detection_probabilities) * 100

        if self == Metric.SP_INDEX:
            cm = sk_metrics.confusion_matrix(target, prediction, labels=list(set(list(target))))
            detection_probabilities = cm.diagonal() / cm.sum(axis=1)
            geometric_mean = scipy.gmean(detection_probabilities)
            return np.sqrt(np.mean(detection_probabilities * geometric_mean)) * 100



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


class Test(enum.Enum):
    F_TEST_5x2 = 0
    STD_OVERLAY = 1

    @staticmethod
    def std_overlay(sample1: np.ndarray, sample2: np.ndarray, confidence_level: float) -> bool:
        mean1 = np.mean(sample1)
        std1 = np.std(sample1)
        mean2 = np.mean(sample2)
        std2 = np.std(sample2)
        return np.abs(mean1 - mean2) > (std1 + std2)

    @staticmethod
    def f_test_5x2(sample1: np.ndarray, sample2: np.ndarray, confidence_level: float) -> bool:
        #http://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/
        #https://www.cmpe.boun.edu.tr/~ethem/files/papers/NC110804.PDF
        if len(sample1) != 10 or len(sample2) != 10:
            raise UnboundLocalError('For Ftest_5x2 must be calculated 10 values')

        p_1_a = sample1[0::2]
        p_2_a = sample1[1::2]

        p_1_b = sample2[0::2]
        p_2_b = sample2[1::2]

        p_1 = p_1_a - p_1_b
        p_2 = p_2_a - p_2_b

        p_mean = (p_1 + p_2)/2
        s_2 = (p_1 - p_mean)**2 + (p_2 - p_mean)**2

        f = (np.sum(sample1**2) + np.sum(sample2**2)) / (2*np.sum(s_2))

        return f > f.ppf(confidence_level, 10, 5)

    def reject_equal_hipoteses(self, sample1: typing.Iterable, sample2: typing.Iterable, confidence_level = 0.95) -> bool: #return true se diferente
        return getattr(self.__class__, self.name.lower())(sample1, sample2, confidence_level)


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
            'metrics': [],
            'abs_cm': {},
            'rel_cm': {}
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

        self._score_dict['abs_cm'][i_fold] = sk_metrics.confusion_matrix(target, prediction)
        self._score_dict['rel_cm'][i_fold] = sk_metrics.confusion_matrix(target, prediction, normalize='true')

    def print_abs_cm(self):
        result = ""
        result_matrix = None
        for i_fold in self._score_dict['i_fold']:
            if result_matrix is None:
                result_matrix = self._score_dict['abs_cm'][i_fold].flatten()
            else:
                result_matrix = np.column_stack((result_matrix, self._score_dict['abs_cm'][i_fold].flatten()))

        if len(self._score_dict['i_fold']) != 1:
            mean = np.mean(result_matrix, axis=1)
            std = np.std(result_matrix, axis=1)
        else:
            mean = result_matrix
            std = np.zeros(mean.shape)
        n_elements = self._score_dict['abs_cm'][self._score_dict['i_fold'][0]].shape

        for i in range(n_elements[0]):
            for j in range(n_elements[1]):
                result += f'{mean[n_elements[1]*i + j]:.1f} +- {std[n_elements[1]*i + j]:.1f} \t'
            result += '\n'

        return result
    
    def print_rel_cm(self):
        result = ""
        result_matrix = None
        for i_fold in self._score_dict['i_fold']:
            if result_matrix is None:
                result_matrix = self._score_dict['rel_cm'][i_fold].flatten()
            else:
                result_matrix = np.column_stack((result_matrix, self._score_dict['rel_cm'][i_fold].flatten()))

        if len(self._score_dict['i_fold']) != 1:
            mean = np.mean(result_matrix, axis=1)
            std = np.std(result_matrix, axis=1)
        else:
            mean = result_matrix
            std = np.zeros(mean.shape)
        n_elements = self._score_dict['rel_cm'][self._score_dict['i_fold'][0]].shape

        for i in range(n_elements[0]):
            if i != 0:
                result += '\n'

            for j in range(n_elements[1]):
                result += f'{mean[n_elements[1]*i + j]*100:.2f} +- {std[n_elements[1]*i + j]*100:.2f} \t'

        return result

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
    default_metric_list = [Metric.SP_INDEX,
                           Metric.BALANCED_ACCURACY,
                           Metric.MACRO_F1]

    def __init__(self,
                 metric_list: typing.List['Metric'] = default_metric_list,
                 comparison_test: Test = None):
        self.cv_dict = {}
        self.param_dict = {}
        self.params = None
        self.metric_list = metric_list
        self.comparison_test = comparison_test

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

    def print_cm(self):
        ret = '------- Confusion Matrix -------------\n'

        for hash, dict in self.cv_dict.items():
            ret += f'-- {dict["params"]} --\n\n'
            ret += dict['cv'].print_abs_cm()
            ret += '\n'
            ret += dict['cv'].print_rel_cm()
            ret += '\n'
        return ret

    def __str__(self) -> str:

        
        ret = '------- Metric Table -------------\n'
        ret += self.as_str()

        return ret