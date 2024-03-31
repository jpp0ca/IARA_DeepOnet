"""
Module providing standardized interfaces for training models in this library.
"""
import typing
import tqdm
import math

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.utils.data as torch_data

import iara.processing.manager as iara_proc_manager


class BaseDataset(torch_data.Dataset):
    """
    Class representing a dataset and providing access to data, targets, and classes
    to standardize the interface for training models.

    This class serves as the base for defining custom datasets in PyTorch.
    It inherits from torch.utils.data.Dataset and provides common functionality
    for accessing data samples, their corresponding targets, and classes."""

    def __init__(self, data: pd.DataFrame, targets: pd.Series, classes: typing.List):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the dataset samples.
            targets (pd.Series): Series containing the corresponding targets for each sample.
            classes (typing.List): List of classes present in the dataset.
        """
        self.data = data
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if self.is_loaded():
            return self.data[index], self.targets[index]

        raise UnboundLocalError('Data not ready to be acessed')

    def is_loaded(self) -> bool:
        """
        Check if the dataset is loaded in memory or needs to be accessed in parts via a
        torch.utils.data.DataLoader.

        Returns:
            bool: True if the dataset is loaded in memory, False otherwise.
        """
        return self.data is not None


class AudioDataset(BaseDataset):
    """
    Custom dataset abstraction for audio data, bridging between torch.utils.data.Dataset and
        pandas.DataFrame/pandas.Series containing audio file descriptions and targets.

    This class facilitates the integration of PyTorch's DataLoader with data information
        provided as a pandas.DataFrame and pandas.Series.

    Attributes:
        MEMORY_LIMIT (int): Maximum size in bytes of a dataframe that can be loaded into memory.
            When a dataset exceeds this limit, the data is loaded partially as needed.
    """
    MEMORY_LIMIT = 1 * 1024 * 1024 * 1024  # gigabytes
    N_WORKERS = 8

    def __init__(self,
                 processor: iara_proc_manager.AudioFileProcessor,
                 file_ids: typing.Iterable[int],
                 targets: typing.Iterable) -> None:
        """
        Args:
            processor (iara.processing.manager.AudioFileProcessor): An instance of the
                AudioFileProcessor class responsible for processing audio data.
            file_ids (typing.Iterable[int]): An iterable of integers representing the IDs of the
                audio files used in the search for file in the dataset.
            targets (typing.Iterable): An iterable representing the target labels corresponding to
                each audio file in the dataset.
        """
        super().__init__(pd.DataFrame(), pd.Series(), np.unique(targets).tolist())

        self.processor = processor
        self.file_ids = file_ids
        self.limit_ids = [0]
        self.last_id = -1
        self.partial_data = []
        self.total_memory = 0

        with ThreadPoolExecutor(max_workers=AudioDataset.N_WORKERS) as executor:
            for dataset_id, target in tqdm.tqdm(list(zip(file_ids, targets)),
                                                desc='Processing dataset', leave=False):
                executor.submit(self.processor.get_data, dataset_id)

        for dataset_id, target in tqdm.tqdm(list(zip(file_ids, targets)),
                                            desc='Loading dataset', leave=False):
            data_df = self.processor.get_data(dataset_id)
            self.limit_ids.append(self.limit_ids[-1] + len(data_df))

            replicated_targets = pd.Series([target] * len(data_df), name='Target')
            self.targets = pd.concat([self.targets, replicated_targets], ignore_index=True)

            self.total_memory += data_df.memory_usage(deep=True).sum()

            if self.total_memory > AudioDataset.MEMORY_LIMIT:
                # if self.data is not None:
                #     print('Exceeds memory limit')
                self.data = None
            else:
                self.data = pd.concat([self.data, data_df], ignore_index=True)

        # # Uncomment to print total memory needed by keeping a dataset in memory
        # print(str(self))

        self.targets = torch.tensor(self.targets.values, dtype=torch.int64)

        if self.data is not None:
            self.data = torch.tensor(self.data.values, dtype=torch.float32)

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if self.is_loaded():
            return super().__getitem__(index)

        current_id = next(i for i, valor in enumerate(self.limit_ids) if valor > index) - 1

        if current_id != self.last_id:
            self.last_id = current_id
            self.partial_data = self.processor.get_data(self.file_ids[current_id])
            self.partial_data = torch.tensor(self.partial_data.values, dtype=torch.float32)

        return self.partial_data[index - self.limit_ids[current_id]], self.targets[index]

    def __str__(self) -> str:
        total_memory = self.total_memory
        unity = ['B', 'KB', 'MB', 'GB', 'TB']
        cont = int(math.log(total_memory, 1024))
        total_memory /= (1024 ** cont)

        return f'{self.limit_ids[-1]} windows in {total_memory} {unity[cont]}'