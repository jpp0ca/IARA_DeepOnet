"""
Trainer description Module

This module provides classes for configure and training machine learning models.
"""
import os
import enum
import typing
import datetime
import shutil
import abc

import tqdm
import numpy as np
import pandas as pd
import jsonpickle
import matplotlib.pyplot as plt

import sklearn.model_selection as sk_selection
import sklearn.utils.class_weight as sk_utils

import torch
import torch.utils.data as torch_data

import iara.description
import iara.processing.dataset as iara_data_proc
import iara.ml.base_model as ml_model
import iara.utils


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

class TrainingStrategy(enum.Enum):
    """Enum defining training strategies."""
    CLASS_SPECIALIST =0,
    MULTICLASS=1

    def default_loss(self, class_weights: torch.Tensor) -> torch.nn.Module:
        """ This method returns the default loss function corresponding to the specified
                TrainingStrategy.

        Args:
            class_weights (torch.Tensor): Class weights for the loss function.

        Raises:
            NotImplementedError: If some features are still in development.

        Returns:
            torch.nn.Module: The torch loss function module.
        """
        if self == TrainingStrategy.CLASS_SPECIALIST:
            return torch.nn.BCELoss(weight=class_weights[1]/class_weights[0])

        if self == TrainingStrategy.MULTICLASS:
            return torch.nn.CrossEntropyLoss(weight=class_weights)

        raise NotImplementedError('TrainingStrategy has not default_loss implemented')

    def to_str(self, class_id: int = None) -> str:
        """This method returns a string representation that can be used to identify the model.

        Args:
            class_id (int, optional): The class identification when the model is specialized
                for a particular class. Defaults to None.

        Raises:
            NotImplementedError: If some features are still in development.

        Returns:
            str: A string to use as the model identifier.
        """
        if self == TrainingStrategy.CLASS_SPECIALIST:
            return f"specialist({class_id})"

        if self == TrainingStrategy.MULTICLASS:
            return "multiclass"

        raise NotImplementedError('TrainingStrategy has not default_loss implemented')


class OneShotTrainerInterface():
    """Interface defining methods that should be implemented by classes to serve as a trainer
        in the Trainer class."""

    @abc.abstractmethod
    def fit(self,
            output_base_dir: str,
            train_dataset: torch_data.Dataset,
            validation_dataset: torch_data.Dataset) -> None:
        """Abstract method to fit (train) the model.

        This method should be implemented by subclasses to fit (train) the model using the provided
            training and validation datasets.

        Args:
            output_base_dir (str): The base directory to save any training-related outputs
                or artifacts.
            train_dataset (torch_data.Dataset): The training dataset.
            validation_dataset (torch_data.Dataset): The validation dataset.

        Returns:
            None
        """


class NNTrainer(OneShotTrainerInterface):
    """Implementation of the OneShotTrainerInterface for training neural networks."""

    @staticmethod
    def default_optimizer_allocator(model: ml_model.BaseModel) -> torch.optim.Optimizer:
        """Allocate a default torch.optim.Optimizer for the given model, specifically the Adam
            optimizer, for the parameters of the provided model.

        Args:
            model (ml_model.BaseModel): The input model.

        Returns:
            torch.optim.Optimizer: The allocated optimizer.
        """
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    @staticmethod
    def _class_weight(train_dataset: torch_data.Dataset, class_id: int = None) -> torch.Tensor:
        targets = train_dataset.targets
        if class_id is not None:
            targets = torch.where(targets == class_id, torch.tensor(1.0), torch.tensor(0.0))

        targets = targets.numpy()
        classes=np.unique(targets)

        class_weights = sk_utils.compute_class_weight('balanced',
                                                    classes=classes,
                                                    y=targets)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        return class_weights

    def __init__(self,
                 training_strategy: TrainingStrategy,
                 model_allocator: typing.Callable[[typing.List[int]],ml_model.BaseModel],
                 batch_size: int = 64,
                 n_epochs: int = 128,
                 patience: int = None,
                 optimizer_allocator: typing.Callable[[ml_model.BaseModel],
                                                      torch.optim.Optimizer]=None,
                 loss_allocator: typing.Callable[[torch.Tensor], torch.nn.Module]=None,
                 device: typing.Union[str, torch.device] = iara.utils.get_available_device()) \
                     -> None:
        """Initialize the Trainer object with specified parameters.

        Args:
            training_strategy (TrainingStrategy): The training strategy to be used.
            model_allocator (typing.Callable[[typing.List[int]], ml_model.BaseModel]):
                A callable that allocates the model with the given architecture.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            n_epochs (int, optional): The number of epochs for training. Defaults to 128.
            patience (int, optional): The patience for early stopping. Defaults to None.
            optimizer_allocator (typing.Optional[typing.Callable[[ml_model.BaseModel],
                torch.optim.Optimizer]], optional): A callable that allocates the optimizer for
                the model. If provided, this callable will be used to allocate the optimizer.
                If not provided (defaulting to None), the Adam optimizer will be used.
            loss_allocator (typing.Optional[typing.Callable[[torch.Tensor], torch.nn.Module]],
                optional): A callable that allocates the loss function. If provided, this callable
                will be used to allocate the loss function. If not provided (defaulting to None),
                the default loss function corresponding to the specified training strategy will
                be used.
            device (typing.Union[str, torch.device], optional): The device for training
                (e.g., 'cuda' or 'cpu'). Defaults to iara.utils.get_available_device().
        """

        self.training_strategy = training_strategy
        self.model_allocator = model_allocator
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.optimizer_allocator = optimizer_allocator or NNTrainer.default_optimizer_allocator
        self.loss_allocator = loss_allocator or self.training_strategy.default_loss
        self.device = device

    def _prepare_for_training(self,
                train_dataset: torch_data.Dataset) -> typing.Dict[int,typing.Tuple[
                        ml_model.BaseModel,torch.optim.Optimizer, torch.nn.Module]]:

        input_shape = list(train_dataset[0][0].shape)
        trn_dict = {}

        if self.training_strategy == TrainingStrategy.CLASS_SPECIALIST:
            for class_id in np.unique(train_dataset.targets.numpy()):
                class_weights = self._class_weight(train_dataset, class_id).to(self.device)

                model = self.model_allocator(input_shape).to(self.device)
                optimizer = self.optimizer_allocator(model)
                loss = self.loss_allocator(class_weights)
                trn_dict[class_id] = model, optimizer, loss
            return trn_dict

        if self.training_strategy == TrainingStrategy.MULTICLASS:
            class_weights = self._class_weight(train_dataset).to(self.device)

            model = self.model_allocator(input_shape).to(self.device)
            optimizer = self.optimizer_allocator(model)
            loss = self.loss_allocator(class_weights)
            trn_dict[-1] = model, optimizer, loss
            return trn_dict


        raise NotImplementedError('TrainingStrategy has not _loss implemented')

    def fit(self,
            output_base_dir: str,
            train_dataset: torch_data.Dataset,
            validation_dataset: torch_data.Dataset) -> None:
        """Implementation of OneShotTrainerInterface.fit method, to fit (train) the model.

        Args:
            output_base_dir (str): The base directory to save any training-related outputs
                or artifacts.
            train_dataset (torch_data.Dataset): The training dataset.
            validation_dataset (torch_data.Dataset): The validation dataset.

        Returns:
            None
        """

        os.makedirs(output_base_dir, exist_ok=True)

        train_loader = torch_data.DataLoader(train_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True)
        val_loader = torch_data.DataLoader(validation_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)

        container = self._prepare_for_training(train_dataset=train_dataset).items()

        for class_id, (model, optimizer, loss_module) in container if (len(container) == 1) else \
                    tqdm.tqdm(container, leave=False, desc="Classes"):

            sufix = self.training_strategy.to_str(class_id=class_id)
            model_filename = f'{str(model)}_{sufix}'

            model_absfilename = os.path.join(output_base_dir, f'{model_filename}.pkl')
            trn_error_absfilename = os.path.join(output_base_dir, f'{model_filename}_trn.png')

            if os.path.exists(model_absfilename):
                continue

            best_val_loss = float('inf')
            epochs_without_improvement = 0
            best_model_state_dict = None

            train_loss = []
            val_loss = []
            n_epochs = 0
            for _ in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs"):
                n_epochs += 1

                for samples, targets in tqdm.tqdm(train_loader,
                                                  leave=False,
                                                  desc="Training Batchs"):

                    optimizer.zero_grad()

                    if class_id != -1:
                        targets = torch.where(targets == class_id,
                                              torch.tensor(1.0),
                                              torch.tensor(0.0))

                    targets = targets.to(self.device)
                    samples = samples.to(self.device)
                    predictions = model(samples)

                    loss = loss_module(predictions, targets)
                    loss.backward()
                    train_loss.append(loss.item())

                    optimizer.step()

                running_loss = 0.0
                with torch.no_grad():
                    for samples, targets in tqdm.tqdm(val_loader,
                                                      leave=False,
                                                      desc="Evaluating Batch"):

                        if class_id != -1:
                            targets = torch.where(targets == class_id,
                                                torch.tensor(1.0),
                                                torch.tensor(0.0))

                        targets = targets.to(self.device)
                        samples = samples.to(self.device)
                        predictions = model(samples)

                        loss = loss_module(predictions, targets)
                        running_loss += loss.item()
                        val_loss.append(loss.item())

                if running_loss < best_val_loss:
                    best_val_loss = running_loss
                    epochs_without_improvement = 0
                    best_model_state_dict = model.state_dict()
                else:
                    epochs_without_improvement += 1

                if self.patience is not None and epochs_without_improvement >= self.patience:
                    break

            if best_model_state_dict:
                model.load_state_dict(best_model_state_dict)

            train_loss, val_loss = np.array(train_loss), np.array(val_loss)

            train_batches = np.linspace(start=1, stop=n_epochs, num=len(train_loss))
            val_batches = np.linspace(start=1, stop=n_epochs, num=len(val_loss))

            plt.figure(figsize=(10, 5))
            plt.plot(train_batches, train_loss, label='Training Loss')
            plt.plot(val_batches, val_loss, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(trn_error_absfilename)

            model.save(model_absfilename)


class Trainer():
    """Class for managing and executing training based on a TrainingConfig"""

    def __init__(self,
                 config: TrainingConfig,
                 trainer_list: typing.List[OneShotTrainerInterface]) -> None:
        """
        Args:
            config (TrainingConfig): The training configuration.
            trainer_list (typing.List[OneShotTrainerInterface]): A list of OneShotTrainerInterface
                instances to be used for training and evaluation in this configuration.
        """
        self.config = config
        self.trainer_list = trainer_list

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

        for i_fold, (train_set, val_set) in enumerate(train_val_set_list) if only_first_fold  else \
                                tqdm.tqdm(enumerate(train_val_set_list), leave=False, desc="Fold"):

            train_dataset = iara_data_proc.TorchDataset(self.config.dataset_processor,
                                                        train_set['ID'],
                                                        train_set['Target'])

            val_dataset = iara_data_proc.TorchDataset(self.config.dataset_processor,
                                                        val_set['ID'],
                                                        val_set['Target'])

            output_base_dir = os.path.join(self.config.output_base_dir,
                                        f'{i_fold}_of_{self.config.n_folds}')

            for trainer in self.trainer_list if (len(self.trainer_list) == 1) else \
                                tqdm.tqdm(self.trainer_list, leave=False, desc="Trainers"):
                trainer.fit(output_base_dir=output_base_dir,
                        train_dataset=train_dataset,
                        validation_dataset=val_dataset)

            if only_first_fold:
                break
