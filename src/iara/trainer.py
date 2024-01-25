"""
Trainer description Module

This module provides classes for configure and training machine learning models.
"""
import os
import typing
import datetime
import shutil

import pandas as pd
import jsonpickle
import sklearn.model_selection as sk_selection

import iara.description
import iara.processing.dataset as iara_data_proc


class TrainingConfig:
    """Class representing training configuration."""
    time_str_format = "%Y%m%d-%H%M%S"

    def __init__(self,
                name: str,
                dataset: iara.description.CustomDataset,
                dataset_processor: iara_data_proc.DatasetProcessor,
                output_base_dir: str,
                n_folds: int = 10,
                test_factor: float = 0.2):
        """
        Parameters:
        - name (str): A unique identifier for the training configuration.
        - dataset (iara.description.CustomDataset): The dataset used for training.
        - dataset_processor (iara.processing.dataset.DatasetProcessor):
            The DatasetProcessor for accessing and processing data in the dataset.
        - output_base_dir (str): The base directory for storing training outputs.
        - n_folds (int, optional): Number of folds for Kfold cross-validation. Default is 10.
        - test_factor (float, optional): Fraction of the dataset reserved for the test subset.
            Default is 0.2 (20%).
        """
        self.timestring = datetime.datetime.now().strftime(self.time_str_format)
        self.name = name
        self.dataset = dataset
        self.dataset_processor = dataset_processor
        self.output_base_dir = os.path.join(output_base_dir, self.name)
        self.n_folds = n_folds
        self.test_factor = test_factor

    def __str__(self) -> str:
        return  f"----------- {self.name} ----------- \n{str(self.dataset)}"

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

    def get_split_datasets(self) -> \
        typing.Tuple[pd.DataFrame, typing.List[typing.Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Split the dataset into training, validation, and test sets.

        Returns:
        - test_set (pd.DataFrame): Test set with a stratified split.
        - train_val_set_list (list): List of tuples containing training and validation sets
            for each fold in StratifiedKFold strategy
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
    """Class for managing and executing training based on a TrainingConfig"""

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
        """Execute training based on the TrainingConfig"""
        self.__prepare_output_dir()

        _, train_val_set_list = self.config.get_split_datasets()

        for i_fold, (train_set, _) in enumerate(train_val_set_list):

            print(f'--------{i_fold}--------')
            print(train_set['Target'])

            print(self.config.dataset_processor.get_training_data(
                    train_set['ID'], train_set['Target']))

            if only_first_fold:
                break
