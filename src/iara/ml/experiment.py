"""
Experiment Module

This module provides classes for configure, training and compare machine learning models.
"""
import os
import typing
import datetime
import itertools

import tqdm
import pandas as pd
import jsonpickle

import sklearn.model_selection as sk_selection

import iara.utils
import iara.records
import iara.ml.metrics as iara_metrics
import iara.ml.dataset as iara_dataset
import iara.ml.models.trainer as iara_trainer
import iara.processing.manager as iara_manager


class Config:
    """Class representing training configuration."""
    TIME_STR_FORMAT = "%Y%m%d-%H%M%S"

    def __init__(self,
                name: str,
                dataset: iara.records.CustomCollection,
                dataset_processor: iara_manager.AudioFileProcessor,
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
        self.timestring = datetime.datetime.now().strftime(self.TIME_STR_FORMAT)
        self.name = name
        self.dataset = dataset
        self.dataset_processor = dataset_processor
        self.output_base_dir = os.path.join(output_base_dir, self.name)
        self.n_folds = n_folds
        self.test_factor = test_factor

    def __str__(self) -> str:
        return  f"----------- {self.name} ----------- \n{str(self.dataset)}"

    def save(self, file_dir: str) -> None:
        """Save the Config to a JSON file."""
        os.makedirs(file_dir, exist_ok = True)
        file_path = os.path.join(file_dir, f"{self.name}.json")
        with open(file_path, 'w', encoding="utf-8") as json_file:
            json_str = jsonpickle.encode(self, indent=4)
            json_file.write(json_str)

    @staticmethod
    def load(file_dir: str, name: str) -> 'Config':
        """Read a JSON file and return a Config instance."""
        file_path = os.path.join(file_dir, f"{name}.json")
        with open(file_path, 'r', encoding="utf-8") as json_file:
            json_str = json_file.read()
        return jsonpickle.decode(json_str)

    def split_datasets(self) -> \
        typing.Tuple[pd.DataFrame, typing.List[typing.Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Split the dataset into training, validation, and test sets.

        Returns:
        - test_set (pd.DataFrame): Test set with a stratified split.
        - trn_val_set_list (list): List of tuples containing training and validation sets
            for each fold in StratifiedKFold strategy
        """
        df = self.dataset.to_df()

        sss = sk_selection.StratifiedShuffleSplit(n_splits=1,
                                                  test_size=self.test_factor, random_state=42)

        for trn_val_index, test_index in sss.split(df, df['Target']):
            trn_val_set, test_set = df.iloc[trn_val_index], df.iloc[test_index]

        # Move elements with 'Ship ID' present in test set and trn_val_set set to test set
        for ship_id in test_set['Ship ID'].unique():
            ship_data = trn_val_set[trn_val_set['Ship ID'] == ship_id]
            test_set = pd.concat([test_set, ship_data])
            trn_val_set = trn_val_set[trn_val_set['Ship ID'] != ship_id]

        skf = sk_selection.StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        trn_val_set_list = []
        for trn_idx, val_idx in skf.split(trn_val_set, trn_val_set['Target']):
            trn_val_set_list.append((trn_val_set.iloc[trn_idx], trn_val_set.iloc[val_idx]))

        return test_set, trn_val_set_list

class Manager():
    """Class for managing and executing training based on a Config for multiple trainers"""

    def __init__(self,
                 config: Config,
                 *trainers: iara_trainer.BaseTrainer) -> None:
        """
        Args:
            config (Config): The training configuration.
            trainer_list (typing.List[BaseTrainer]): A list of BaseTrainer instances to be used
                for training and evaluation in this configuration.
        """
        self.config = config
        self.trainer_list = trainers

    def __str__(self) -> str:
        return f'{self.config.name} with {len(self.trainer_list)} models'

    def __prepare_output_dir(self):
        """ Creates the directory tree for training, keeping backups of conflicting trainings. """
        if os.path.exists(self.config.output_base_dir):
            try:
                old_config = Config.load(self.config.output_base_dir, self.config.name)
                if old_config.timestring == self.config.timestring:
                    return

                iara.utils.backup_folder(base_dir=self.config.output_base_dir,
                                         time_str_format=Config.TIME_STR_FORMAT)

            except FileNotFoundError:
                pass

        os.makedirs(self.config.output_base_dir, exist_ok=True)
        self.config.save(self.config.output_base_dir)

    def is_trained(self, model_base_dir: str) -> bool:
        """
        Check if all trained models are saved in the specified directory.

        Args:
            model_base_dir (str): The directory where the trained models are expected to be saved.

        Returns:
            bool: True if all training is completed and models are saved in the directory,
                False otherwise.
        """
        for trainer in self.trainer_list:
            if not trainer.is_trained(model_base_dir):
                return False
        return True

    def fit(self, i_fold: int,
            trn_dataset_ids: typing.Iterable[int], trn_targets: typing.Iterable,
            val_dataset_ids: typing.Iterable[int], val_targets: typing.Iterable) -> None:
        """
        Fit the model for a specified fold using the provided training and validation dataset IDs
            and targets.

        Args:
            i_fold (int): The index of the fold.
            trn_dataset_ids (typing.Iterable[int]): Iterable of training dataset IDs.
            trn_targets (typing.Iterable): Iterable of training targets.
            val_dataset_ids (typing.Iterable[int]): Iterable of validation dataset IDs.
            val_targets (typing.Iterable): Iterable of validation targets.
        """
        iara.utils.set_seed()

        model_base_dir = os.path.join(self.config.output_base_dir,
                                        'model',
                                        f'{i_fold}_of_{self.config.n_folds}')

        if self.is_trained(model_base_dir):
            return

        # os.makedirs(model_base_dir, exist_ok=True)
        # merged_df = pd.concat([trn_dataset_ids, trn_targets], axis=1)
        # merged_df.to_csv(os.path.join(model_base_dir, 'merged_df.csv'), index=False)

        trn_dataset = iara_dataset.AudioDataset(self.config.dataset_processor,
                                                    trn_dataset_ids,
                                                    trn_targets)

        val_dataset = iara_dataset.AudioDataset(self.config.dataset_processor,
                                                    val_dataset_ids,
                                                    val_targets)

        for trainer in self.trainer_list if (len(self.trainer_list) == 1) else \
                            tqdm.tqdm(self.trainer_list, leave=False, desc="Trainers"):
            trainer.fit(model_base_dir=model_base_dir,
                    trn_dataset=trn_dataset,
                    val_dataset=val_dataset)

    def is_evaluated(self, dataset_id: str, eval_base_dir: str) -> bool:
        """
        Check if all models are evaluated for the specified dataset_id in the specified directory.

        Args:
            dataset_id (str): Identifier for the dataset, e.g., 'val', 'trn', 'test'.
            model_base_dir (str): The directory where the trained models are expected to be saved.

        Returns:
            bool: True if all models are evaluated in the directory,
                False otherwise.
        """
        for trainer in self.trainer_list:
            if not trainer.is_evaluated(dataset_id=dataset_id, eval_base_dir=eval_base_dir):
                return False
        return True

    def eval(self, i_fold: int, dataset_id: str,
            dataset_ids: typing.Iterable[int], targets: typing.Iterable) -> None:
        """
        Eval the model for a specified fold using the provided training and validation dataset IDs
            and targets.

        Args:
            i_fold (int): The index of the fold.
            dataset_id (str): Identifier for the dataset, e.g., 'val', 'trn', 'test'.
            dataset_ids (typing.Iterable[int]): Iterable of evaluated dataset IDs.
            targets (typing.Iterable): Iterable of evaluated targets.
        """
        model_base_dir = os.path.join(self.config.output_base_dir,
                                        'model',
                                        f'{i_fold}_of_{self.config.n_folds}')
        eval_base_dir = os.path.join(self.config.output_base_dir,
                                        'eval',
                                        f'{i_fold}_of_{self.config.n_folds}')

        if not self.is_trained(model_base_dir):
            raise FileNotFoundError(f'Models not trained in {model_base_dir}')

        if self.is_evaluated(dataset_id=dataset_id, eval_base_dir=eval_base_dir):
            return

        dataset = iara_dataset.AudioDataset(self.config.dataset_processor,
                                              dataset_ids,
                                              targets)

        for trainer in self.trainer_list if (len(self.trainer_list) == 1) else \
                            tqdm.tqdm(self.trainer_list, leave=False, desc="Trainers"):

            trainer.eval(dataset_id=dataset_id,
                         model_base_dir=model_base_dir,
                         eval_base_dir=eval_base_dir,
                         dataset=dataset)

    def compile_results(self,
                        dataset_id: str,
                        trainer: iara_trainer.BaseTrainer,
                        folds: typing.List[int] = None) -> typing.List[pd.DataFrame]:
        """Compiles evaluated results for a specified trainer.

        Args:
            dataset_id (str): Identifier for the dataset, e.g., 'val', 'trn', 'test'.
            trainer (BaseTrainer): An instance used for compilation in this
                configuration.
            folds (List[int], optional): List of fold to be evaluated.
                Defaults all folds will be executed.

        Returns:
            typing.List[pd.DataFrame]: List of DataFrame with two columns, ["Target", "Prediction"].
        """
        all_results = []
        for i_fold in range(self.config.n_folds):

            if folds and i_fold not in folds:
                continue

            eval_base_dir = os.path.join(self.config.output_base_dir,
                                            'eval',
                                            f'{i_fold}_of_{self.config.n_folds}')

            all_results.append(trainer.eval(dataset_id=dataset_id, eval_base_dir=eval_base_dir))


        return all_results

    def run(self, folds: typing.List[int] = None) -> typing.Dict:
        """Execute training based on the Config"""

        self.__prepare_output_dir()

        _, trn_val_set_list = self.config.split_datasets()

        for _ in tqdm.tqdm(range(1), leave=False, desc="Fitting models", bar_format = "{desc}"):
            for i_fold, (trn_set, val_set) in enumerate(trn_val_set_list if len(folds) == 1 else \
                                    tqdm.tqdm(trn_val_set_list,
                                              leave=False,
                                              desc="Fold")):

                if folds and i_fold not in folds:
                    continue

                self.fit(i_fold=i_fold,
                        trn_dataset_ids=trn_set['ID'],
                        trn_targets=trn_set['Target'],
                        val_dataset_ids=val_set['ID'],
                        val_targets=val_set['Target'])

        for _ in tqdm.tqdm(range(1), leave=False, desc="Evaluating models", bar_format = "{desc}"):
            for i_fold, (trn_set, val_set) in enumerate(trn_val_set_list if len(folds) == 1 else \
                                    tqdm.tqdm(trn_val_set_list,
                                              leave=False,
                                              desc="Fold")):

                if folds and i_fold not in folds:
                    continue

                self.eval(i_fold=i_fold,
                          dataset_id='val',
                          dataset_ids=val_set['ID'],
                          targets=val_set['Target'])


        result_dict = {}

        for trainer in self.trainer_list:

            results = self.compile_results(dataset_id='val',
                                          trainer=trainer,
                                          folds=folds)

            result_dict[trainer.trainer_id] = results

        return result_dict


class Comparator():
    """
    Class for comparing models based on cross-comparison in test datasets.

    This class provides functionality to compare multiple models based on cross-comparison in test
    datasets. It iterates through all combinations of two trainers from the provided list of
    trainers, and evaluates each combination on the test dataset of the second trainer.
    """
    def __init__(self, output_dir: str, manager_list: typing.List[Manager]) -> None:
        """
        Args:
            output_dir (str): The directory path where evaluation results will be saved.
            trainer_list (List[Trainer]): A list of Trainer objects containing the models to be
                compared.
        """
        self.output_dir = output_dir
        self.manager_list = manager_list

    def cross_compare_in_test(self, folds: typing.List[int] = None):
        """
        Perform cross-comparison of models on test datasets.

        Iterates through all combinations of two trainers and evaluates each combination on the test
        dataset of the second trainer.

        Args:
            folds (List[int], optional): List of fold to be evaluated.
                Defaults all folds will be executed.
        """
        grid = iara_metrics.GridCompiler()

        for trainer_1, trainer_2 in itertools.permutations(self.manager_list, 2):

            test_df1, _ = trainer_2.config.split_datasets()

            dataset = iara_dataset.AudioDataset(trainer_2.config.dataset_processor,
                                                test_df1['ID'],
                                                test_df1['Target'])

            for trainer in trainer_1.trainer_list:

                for i_fold in range(trainer_1.config.n_folds):

                    if folds and i_fold not in folds:
                        continue

                    model_base_dir = os.path.join(trainer_1.config.output_base_dir,
                                            'model',
                                            f'{i_fold}_of_{trainer_1.config.n_folds}')

                    evaluation = trainer.eval(
                        dataset_id=f'{trainer_2.config.name}_{str(trainer_1.config.name)}_{i_fold}',
                        eval_base_dir = self.output_dir,
                        model_base_dir = model_base_dir,
                        dataset = dataset)

                    grid.add(params={'': f'{trainer_1.config.name} ({trainer.trainer_id}) \
                                -> {trainer_2.config.name}'},
                             i_fold=i_fold,
                             target=evaluation['Target'],
                             prediction=evaluation['Prediction'])

        print('\n______________Comparison________________________')
        print(grid.as_str())
        print('----------------------------------------')
