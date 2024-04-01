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
                test_factor: float = 0.2,
                excludent_ship_id = True,
                outer_splits = 3):
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
        self.excludent_ship_id = excludent_ship_id
        self.outer_splits = outer_splits

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
        typing.List[typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Split the dataset into training, validation, and test sets.

        Returns:
        - list (list): List of tuples containing training, validation and test sets
            for each fold in StratifiedKFold strategy
        """
        df = self.dataset.to_df()

        split_list = []

        sss = sk_selection.StratifiedShuffleSplit(n_splits=self.outer_splits,
                                                  test_size=self.test_factor, random_state=42)

        for trn_val_index, test_index in sss.split(df, df['Target']):
            trn_val_set, test_set = df.iloc[trn_val_index], df.iloc[test_index]

            # Move elements with 'Ship ID' present in test set and trn_val_set set to test set
            if self.excludent_ship_id:
                for ship_id in test_set['Ship ID'].unique():
                    ship_data = trn_val_set[trn_val_set['Ship ID'] == ship_id]
                    test_set = pd.concat([test_set, ship_data])
                    trn_val_set = trn_val_set[trn_val_set['Ship ID'] != ship_id]

            skf = sk_selection.StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            for trn_idx, val_idx in skf.split(trn_val_set, trn_val_set['Target']):
                split_list.append((trn_val_set.iloc[trn_idx], trn_val_set.iloc[val_idx], test_set))

        return split_list

    def get_data_loader(self) -> iara_dataset.ExperimentDataLoader:
        df = self.dataset.to_df()
        return iara_dataset.ExperimentDataLoader(self.dataset_processor,
                                        df['ID'].to_list(),
                                        df['Target'].to_list())

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
        self.experiment_loader = None

    def get_experiment_loader(self) -> iara_dataset.ExperimentDataLoader:
        if self.experiment_loader is None:
            self.experiment_loader = self.config.get_data_loader()
        return self.experiment_loader

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

    def get_model_base_dir(self, i_fold: int) -> str:
        return os.path.join(self.config.output_base_dir,
                                'model',
                                f'{i_fold}_of_{self.config.n_folds}')

    def is_trained(self, i_fold: int) -> bool:
        model_base_dir = self.get_model_base_dir(i_fold)
        for trainer in self.trainer_list:
            if not trainer.is_trained(model_base_dir):
                return False
        return True

    def fit(self, i_fold: int,
            trn_dataset_ids: typing.Iterable[int],
            val_dataset_ids: typing.Iterable[int]) -> None:
        """
        Fit the model for a specified fold using the provided training and validation dataset IDs
            and targets.

        Args:
            i_fold (int): The index of the fold.
            trn_dataset_ids (typing.Iterable[int]): Iterable of training dataset IDs.
            val_dataset_ids (typing.Iterable[int]): Iterable of validation dataset IDs.
        """
        if self.is_trained(i_fold):
            return

        trn_dataset = iara_dataset.AudioDataset(self.get_experiment_loader(), trn_dataset_ids)

        val_dataset = iara_dataset.AudioDataset(self.get_experiment_loader(), val_dataset_ids)

        model_base_dir = self.get_model_base_dir(i_fold)

        for _ in tqdm.tqdm(range(1), leave=False, bar_format = "{desc}",
                        desc=f'Trn({str(trn_dataset)}) Val({str(val_dataset)})'):

            for trainer in self.trainer_list if (len(self.trainer_list) == 1) else \
                                tqdm.tqdm(self.trainer_list, leave=False, desc="Trainers"):

                iara.utils.set_seed()

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

    def eval(self, i_fold: int, dataset_id: str, dataset_ids: typing.Iterable[int]) -> None:
        """
        Eval the model for a specified fold using the provided training and validation dataset IDs
            and targets.

        Args:
            i_fold (int): The index of the fold.
            dataset_id (str): Identifier for the dataset, e.g., 'val', 'trn', 'test'.
            dataset_ids (typing.Iterable[int]): Iterable of evaluated dataset IDs.
        """
        model_base_dir = self.get_model_base_dir(i_fold)
        eval_base_dir = os.path.join(self.config.output_base_dir,
                                        'eval',
                                        f'{i_fold}_of_{self.config.n_folds}')

        if not self.is_trained(i_fold):
            raise FileNotFoundError(f'Models not trained in {model_base_dir}')

        if self.is_evaluated(dataset_id=dataset_id, eval_base_dir=eval_base_dir):
            return

        dataset = iara_dataset.AudioDataset(self.get_experiment_loader(),
                                              dataset_ids)

        for trainer in self.trainer_list if (len(self.trainer_list) == 1) else \
                            tqdm.tqdm(self.trainer_list, leave=False, desc="Trainers"):

            trainer.eval(dataset_id=dataset_id,
                         model_base_dir=model_base_dir,
                         eval_base_dir=eval_base_dir,
                         dataset=dataset)

    def compile_results(self,
                        dataset_id: str,
                        trainer_list: typing.List[iara_trainer.BaseTrainer] = None,
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

        result_dict = {}

        for trainer in trainer_list if trainer_list is not None else self.trainer_list:

            results = []
            for i_fold in range(self.config.n_folds):

                if folds and i_fold not in folds:
                    continue

                eval_base_dir = os.path.join(self.config.output_base_dir,
                                                'eval',
                                                f'{i_fold}_of_{self.config.n_folds}')

                results.append(trainer.eval(dataset_id=dataset_id, eval_base_dir=eval_base_dir))

            result_dict[trainer.trainer_id] = results

        return result_dict

    def print_dataset_details(self, id_list) -> None:

        df = self.config.dataset.to_compiled_df()
        df = df.rename(columns={'Qty': 'Total'})

        for i_fold, (trn_set, val_set, test_set) in enumerate(id_list):

            df_trn = self.config.dataset.to_compiled_df(trn_set)
            df_val = self.config.dataset.to_compiled_df(val_set)
            df_test = self.config.dataset.to_compiled_df(test_set)

            df_trn = df_trn.rename(columns={'Qty': f'Trn_{i_fold}'})
            df_val = df_val.rename(columns={'Qty': f'Val_{i_fold}'})
            df_test = df_test.rename(columns={'Qty': f'Test_{i_fold}'})

            df = pd.merge(df, df_trn, on=self.config.dataset.target.column)
            df = pd.merge(df, df_val, on=self.config.dataset.target.column)
            df = pd.merge(df, df_test, on=self.config.dataset.target.column)

            break

        print('--- Dataset ---')
        print(df)

    def run(self, folds: typing.List[int] = None) -> typing.Dict:
        """Execute training based on the Config"""
        folds = folds if folds is not None else range(self.config.n_folds)

        self.__prepare_output_dir()
        id_list = self.config.split_datasets()

        self.print_dataset_details(id_list)

        for _ in tqdm.tqdm(range(1), leave=False,
                           desc="--- Fitting models ---", bar_format = "{desc}"):
            for i_fold in folds if len(folds) == 1 else \
                                    tqdm.tqdm(folds,
                                              leave=False,
                                              desc="Fold"):
                (trn_set, val_set, test_set) = id_list[i_fold]

                self.fit(i_fold=i_fold,
                        trn_dataset_ids=trn_set['ID'].to_list(),
                        val_dataset_ids=val_set['ID'].to_list())

                self.eval(i_fold=i_fold,
                          dataset_id='trn',
                          dataset_ids=trn_set['ID'].to_list())

                self.eval(i_fold=i_fold,
                          dataset_id='val',
                          dataset_ids=val_set['ID'].to_list())

                self.eval(i_fold=i_fold,
                          dataset_id='test',
                          dataset_ids=test_set['ID'].to_list())

        return self.compile_results(dataset_id='val',
                                trainer_list=self.trainer_list,
                                folds=folds)


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
