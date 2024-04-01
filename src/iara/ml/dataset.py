"""
Module providing standardized interfaces for training models in this library.
"""
import typing
import tqdm
import math

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.utils.data as torch_data

import iara.utils
import iara.processing.manager as iara_proc_manager


class BaseDataset(torch_data.Dataset):
    # broken in refactoring - need update
    def __init__(self):
        pass


class ExperimentDataLoader():
    """
    Custom dataset loader for audio data, should be use to pre-load all data in a experiment to RAM
        avoiding multiple load along side fold trainings:

    Attributes:
        MEMORY_LIMIT (int): Maximum size in bytes that can be loaded into memory.
            When a dataset exceeds this limit, the data is loaded partially as needed (Very low).
        N_WORKERS (int): Number of simultaneos threads to process run files and load data.
    """
    MEMORY_LIMIT = 5 * 1024 * 1024 * 1024  # gigabytes
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
        self.processor = processor
        self.file_ids = file_ids
        self.targets = targets
        self.data_map = {}
        self.size_map = {}
        self.target_map = {}
        self.memory_map = {}
        self.total_memory = 0
        self.total_samples = 0

        with ThreadPoolExecutor(max_workers=ExperimentDataLoader.N_WORKERS) as executor:
            futures = [executor.submit(self.__load, file_id, target) 
                       for file_id, target in zip(file_ids, targets)]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures),
                                    desc='Processing dataset', leave=False):
                future.result()


    def __load(self, file_id: int, target) -> None:
        """ Process or/and load a single file to data map

        Args:
            file_id (int): IDs of the audio files
            target (_type_): Target labels correspondent
        """        

        data_df = self.processor.get_data(file_id)

        memory = data_df.memory_usage(deep=True).sum()

        if self.total_memory < ExperimentDataLoader.MEMORY_LIMIT and \
                (self.total_memory + memory) > ExperimentDataLoader.MEMORY_LIMIT:
            self.data_map.clear()

        self.total_memory += memory

        if self.total_memory < ExperimentDataLoader.MEMORY_LIMIT:   
            self.data_map[file_id] = torch.tensor(data_df.values, dtype=torch.float32)

        self.total_samples += len(data_df)
        self.size_map[file_id] = len(data_df)
        self.target_map[file_id] = target
        self.memory_map[file_id] = memory

    def get(self, file_id: int, offset: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """ Return data and target from offset sample of file_id

        Args:
            file_id (int): IDs of the audio files
            offset (int): Number lower than the number os windows generated in processing of
                file_id data

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor]: data, target
        """

        if file_id in self.data_map:
            return self.data_map[file_id][offset], \
                    torch.tensor(self.target_map[file_id], dtype=torch.int64)

        data_df = self.processor.get_data(file_id)
        data_df = torch.tensor(data_df.values, dtype=torch.float32)
        return data_df[offset], torch.tensor(self.target_map[file_id], dtype=torch.int64)

    def __str__(self) -> str:
        total_memory = self.total_memory
        unity = ['B', 'KB', 'MB', 'GB', 'TB']
        cont = int(math.log(total_memory, 1024))
        total_memory /= (1024 ** cont)
        return f'{self.total_samples} windows in {total_memory} {unity[cont]}'


class AudioDataset(BaseDataset):

    def __init__(self,
                 loader: ExperimentDataLoader,
                 file_ids: typing.Iterable[int]) -> None:
        self.loader = loader
        self.file_ids = file_ids
        self.limit_ids = [0]

        for file_id in self.file_ids:
            self.limit_ids.append(self.limit_ids[-1] + loader.size_map[file_id])

    def __len__(self):
        return self.limit_ids[-1]

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        current_id = next(i for i, valor in enumerate(self.limit_ids) if valor > index) - 1
        offset_index = index - self.limit_ids[current_id]

        return self.loader.get(self.file_ids[current_id], offset_index)

    def __str__(self) -> str:
        total_memory = 0
        for file_id in self.file_ids:
            total_memory += self.loader.memory_map[file_id]
        return f'{self.limit_ids[-1]} windows in {iara.utils.str_format_bytes(total_memory)}'
