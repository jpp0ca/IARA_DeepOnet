"""
Trainer description Module

This module provides classes for configure and training.
"""
import enum
import os
import typing
import datetime
import shutil

import pandas as pd
import jsonpickle
import sklearn.model_selection as sk_selection

import iara.description
import iara.processing.dataset as iara_data_proc

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
                 dataset: iara.description.Subdataset,
                 target: DatasetSelection,
                 filters: typing.List[DatasetSelection] = None,
                 only_sample: bool = False):
        """
        Parameters:
        - dataset (iara.description.Subdataset): Subdataset to be used.
        - target (DatasetSelection): Target selection for training.
        - filters (List[DatasetSelection], optional): List of filters to be applied.
            Default is use all dataset.
        - only_sample (bool, optional): Use only data available in sample dataset. Default is False.
        """
        self.dataset = dataset
        self.target = target
        self.filters = filters if filters else []
        self.only_sample = only_sample

    def get_dataset_info(self) -> pd.DataFrame:
        """
        Generate a DataFrame with information from the dataset based on specified filters
            and target configuration.

        Returns:
            pd.DataFrame: DataFrame containing the dataset information after applying filters
                and target mapping.
        """
        df = self.dataset.to_dataframe(only_sample=self.only_sample)
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
    time_str_format = "%Y%m%d-%H%M%S"

    def __init__(self,
                name: str,
                dataset: TrainingDataset,
                dataset_processor: iara_data_proc.DatasetProcessor,
                output_base_dir: str,
                n_folds: int = 10,
                test_factor: float = 0.2):
        """
        Parameters:
        - name (str): Unique identifier for the training configuration.
        - dataset (TrainingDataset): Training dataset.
        - dataset_processor (iara.processing.dataset.DatasetProcessor): DatasetProcessor,
        - output_base_dir (str): Base directory for training output.
        - training_type (TrainingType, optional): Type of training. Default is TrainingType.WINDOW.
        - n_folds (int, optional): Number of folds in training. Default is 10.
        """
        self.timestring = datetime.datetime.now().strftime(self.time_str_format)
        self.name = name
        self.dataset = dataset
        self.dataset_processor = dataset_processor
        self.output_base_dir = os.path.join(output_base_dir, self.name)
        self.n_folds = n_folds
        self.test_factor = test_factor

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

    def get_split_datasets(self) -> \
        typing.Tuple[pd.DataFrame, typing.List[typing.Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Split the dataset into training, validation, and test sets.

        Returns:
        - test_set (pd.DataFrame): Test set with a stratified split.
        - train_val_set_list (list): List of tuples containing training and validation sets
            for each fold.
        """
        df = self.dataset.get_dataset_info()

        sss = sk_selection.StratifiedShuffleSplit(n_splits=1,
                                                  test_size=self.test_factor, random_state=42)

        for train_val_index, test_index in sss.split(df, df['Target']):
            train_val_set, test_set = df.iloc[train_val_index], df.iloc[test_index]

        # Move elements with 'Ship ID' present in test set and train_val_set set to test set
        for ship_id in test_set['Ship ID'].unique():
            ship_data = train_val_set[train_val_set['Ship ID'] == ship_id]
            test_set = pd.concat([test_set, ship_data])
            train_val_set = train_val_set[train_val_set['Ship ID'] != ship_id]

        skf = sk_selection.StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        train_val_set_list = []
        for train_idx, val_idx in skf.split(train_val_set, train_val_set['Target']):
            train_val_set_list.append((train_val_set.iloc[train_idx], train_val_set.iloc[val_idx]))

        return test_set, train_val_set_list


class Trainer():
    """Class for manage and execute training from a TrainingConfig"""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def __prepare_output_dir(self):
        """ Creates the directory tree for training, keeping backups of conflicting trainings. """
        if os.path.exists(self.config.output_base_dir):
            try:
                old_config = TrainingConfig.load(self.config.output_base_dir, self.config.name)
                if old_config.timestring == self.config.timestring:
                    return

                old_dir = os.path.join(self.config.output_base_dir, old_config.timestring)
                os.makedirs(old_dir)

                contents = os.listdir(self.config.output_base_dir)
                for item in contents:
                    item_path = os.path.join(self.config.output_base_dir, item)

                    if os.path.isdir(item_path):
                        try:
                            datetime.datetime.strptime(item, TrainingConfig.time_str_format)
                            continue
                        except ValueError:
                            pass
                    shutil.move(item_path, old_dir)

            except FileNotFoundError:
                pass

        os.makedirs(self.config.output_base_dir, exist_ok=True)
        self.config.save(self.config.output_base_dir)

    def run(self, only_first_fold: bool = False):
        """Execute training from the TrainingConfig"""
        self.__prepare_output_dir()

        test_set, train_val_set_list = self.config.get_split_datasets()

        for i_fold, (train_set, val_set) in enumerate(train_val_set_list):

            print(f'--------{i_fold}--------')
            print(train_set['Target'])

            print(self.config.dataset_processor.get_training_data(
                    train_set['ID'], train_set['Target']))

            if only_first_fold:
                break
