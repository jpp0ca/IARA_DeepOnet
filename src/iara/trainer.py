"""
Trainer description Module

This module provides classes for configure and training.
"""
import enum
import os
import typing
import datetime
import pandas as pd

import jsonpickle

import iara.description
import iara.processing.analysis as iara_proc

class TrainingType(enum.Enum):
    """Enum defining training types."""
    WINDOW = 0
    IMAGE = 1

    def to_json(self) -> str:
        """Return the string equivalent of the enum."""
        return self.name

class DatasetSelection():
    """Class representing a filter to apply or a training target on a dataset."""
    def __init__(self,
                 column: str,
                 values: typing.List[str],
                 include_others: bool = False):
        """
        Parameters:
        - column (str): Name of the column for selection.
        - values (List[str]): List of values for selection.
        - include_others (bool, optional): Indicates whether other values should be compiled as one
            and included, make sense only when using as target. Default is False.
        """
        self.column = column
        self.values = values
        self.include_others = include_others

class TrainingDataset:
    """Class representing a training dataset."""

    def __init__(self,
                 dataset_base_dir: str,
                 dataset: iara.description.Subdataset,
                 target: DatasetSelection,
                 filters: typing.List[DatasetSelection] = None):
        """
        Parameters:
        - dataset_base_dir (str): Base directory of the dataset,
            expected to contain a directory for each label dataset (from A to J).
        - dataset (iara.description.Subdataset): Subdataset to be used.
        - target (DatasetSelection): Target selection for training.
        - filters (List[DatasetSelection], optional): List of filters to be applied.
            Default is use all dataset.
        """
        self.dataset_base_dir = dataset_base_dir
        self.dataset = dataset
        self.target = target
        self.filters = filters if filters else []

    def get_dataset_info(self) -> pd.DataFrame:
        """
        Generate a DataFrame with information from the dataset based on specified filters
            and target configuration.

        Returns:
            pd.DataFrame: DataFrame containing the dataset information after applying filters
                and target mapping.
        """
        df = self.dataset.to_dataframe()
        for filt in self.filters:
            df = df.loc[df[filt.column].isin(filt.values)]

        if not self.target.include_others:
            df = df.loc[df[self.target.column].isin(self.target.values)]

        df['Target'] = df[self.target.column].map(
            {value: index for index, value in enumerate(self.target.values)})
        df['Target'] = df['Target'].fillna(len(self.target.values))
        df['Target'] = df['Target'].astype(int)

        return df

    def __str__(self) -> str:
        return str(self.get_dataset_info())

class TrainingConfig:
    """Class representing training configuration."""

    def __init__(self,
                name: str,
                dataset: TrainingDataset,
                analysis: iara_proc.Analysis,
                analysis_parameters: dict,
                output_base_dir: str,
                training_type: TrainingType = TrainingType.WINDOW):
        """
        Parameters:
        - name (str): Unique identifier for the training configuration.
        - dataset (TrainingDataset): Training dataset.
        - analysis (iara.processing.analysis.Analysis): Analysis applied in data configuration.
        - analysis_parameters (dict) : parameter for analysis
        - output_base_dir (str): Base directory for training output.
        - training_type (TrainingType, optional): Type of training. Default is TrainingType.WINDOW.
        """
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.dataset = dataset
        self.analysis = analysis
        self.analysis_parameters = analysis_parameters
        self.output_base_dir = output_base_dir
        self.training_type = training_type

    def save(self, file_dir: str) -> None:
        """Save the TrainingConfig to a JSON file."""
        os.makedirs(file_dir, exist_ok = True)
        file_path = os.path.join(file_dir, f"{self.name}.json")
        with open(file_path, 'w', encoding="utf-8") as json_file:
            json_str = jsonpickle.encode(self, indent=4)
            json_file.write(json_str)

    @staticmethod
    def load(file_dir: str, name: str) -> 'TrainingConfig':
        """Read a JSON file and return a TrainingConfig instance."""
        file_path = os.path.join(file_dir, f"{name}.json")
        with open(file_path, 'r', encoding="utf-8") as json_file:
            json_str = json_file.read()
        return jsonpickle.decode(json_str)

    def __str__(self) -> str:
        return  f"----------- {self.name} ----------- \n{str(self.dataset)}"
