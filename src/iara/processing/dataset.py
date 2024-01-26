"""Module for handling access to processed data from a dataset.

This module defines a class, `DatasetProcessor`, which facilitates access
to processed data by providing methods for loading and retrieving dataframes
based on specified parameters. It supports the processing and normalization
of audio data, allowing users to work with either window-based or image-based
input types.

Classes:
    DatasetProcessor: Class for handling access to processed data from a dataset.
"""
import os
import enum
import typing
import tqdm

import PIL
import pandas as pd
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt
import matplotlib.colors as color
import tikzplotlib as tikz

import torch
import torch.utils.data as torch_data

import iara.processing.analysis as iara_proc
import iara.processing.prefered_number as iara_pn

def get_iara_id(file:str) -> int:
    """
    Default function to extracts the ID from the given file name.

    Parameters:
        file (str): The file name without extension.

    Returns:
        int: The extracted ID.
    """
    return int(file.rsplit('-',maxsplit=1)[-1])

class PlotType(enum.Enum):
    """Enum defining plot types."""
    SHOW_FIGURE = 0
    EXPORT_RAW = 1
    EXPORT_PLOT = 2
    EXPORT_TEX = 3

    def __str__(self):
        return str(self.name).rsplit('_', maxsplit=1)[-1].lower()

class InputType(enum.Enum):
    """Enum defining training types."""
    WINDOW = 0
    IMAGE = 1

    def __str__(self):
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

class DatasetProcessor():
    """ Class for handling acess to process data from a dataset. """

    def __init__(self,
                data_base_dir: str,
                data_processed_base_dir: str,
                normalization: iara_proc.Normalization,
                analysis: iara_proc.SpectralAnalysis,
                n_pts: int = 1024,
                n_overlap: int = 0,
                n_mels: int = 256,
                decimation_rate: int = 1,
                extract_id: typing.Callable[[str], int] = get_iara_id,
                input_type: InputType = InputType.WINDOW,
                frequency_limit: float = None
                ) -> None:
        """
        Parameters:
            data_base_dir (str): Base directory for raw data.
            data_processed_base_dir (str): Base directory for process data.
            normalization (iara_proc.Normalization): Normalization object.
            analysis (iara_proc.Analysis): Analysis object.
            n_pts (int): Number of points for use in analysis.
            n_overlap (int): Number of points to overlap for use in analysis.
            n_mels (int): Number of Mel frequency bins for use in analysis.
            decimation_rate (int): Decimation rate for use in analysis when mel based.
            extract_id (Callable[[str], str]): Function to extract ID from a file name without
                extension. Default is split based on '-' em get last part of the name
            input_type (InputType): Type of input data. Default WINDOW.
            frequency_limit (float): The frequency limit to be considered in the data
                processing result. Default is fs/2
        """
        self.data_base_dir = data_base_dir
        self.data_processed_base_dir = data_processed_base_dir
        self.normalization = normalization
        self.analysis = analysis
        self.n_pts = n_pts
        self.n_overlap = n_overlap
        self.n_mels = n_mels
        self.decimation_rate = decimation_rate
        self.extract_id = extract_id
        self.input_type = input_type
        self.frequency_limit = frequency_limit

        self.data_processed_dir = os.path.join(self.data_processed_base_dir,
                                  str(self.analysis) + "_" + str(self.input_type))

    def _find_raw_file(self, dataset_id: int) -> str:
        """
        Finds the raw file associated with the given ID.

        Parameters:
            dataset_id (int): The ID to search for.

        Returns:
            str: The path to the raw file.

        Raises:
            UnboundLocalError: If the file is not found.
        """
        for root, _, files in os.walk(self.data_base_dir):
            for file in files:
                filename, extension = os.path.splitext(file)
                if extension == ".wav" and self.extract_id(filename) == dataset_id:
                    return os.path.join(root, file)
        raise UnboundLocalError(f'file {dataset_id} not found in {self.data_base_dir}')

    def _process(self, dataset_id: int) -> typing.Tuple[np.array, np.array, np.array]:

        file = self._find_raw_file(dataset_id = dataset_id)

        fs, data = scipy_wav.read(file)

        if data.ndim != 1:
            data = data[:,0]

        power, freqs, times = self.analysis.apply(data = data,
                                                    fs = fs,
                                                    n_pts = self.n_pts,
                                                    n_overlap = self.n_overlap,
                                                    n_mels = self.n_mels,
                                                    decimation_rate = self.decimation_rate)

        power = self.normalization(power)

        if self.frequency_limit:
            index_limit = next((i for i, freq in enumerate(freqs)
                                if freq > self.frequency_limit), len(freqs))
            freqs = freqs[:index_limit]
            power = power[:index_limit,:]

        return power, freqs, times

    def get_data(self, dataset_id: int) -> pd.DataFrame:
        """
        Get data for the given ID.

        Parameters:
            dataset_id (int): The ID to get data for.

        Returns:
            pd.DataFrame: The DataFrame containing the processed data.
        """
        if self.input_type == InputType.WINDOW:

            os.makedirs(self.data_processed_dir, exist_ok=True)
            dataset_file = os.path.join(self.data_processed_dir, f'{dataset_id}.pkl')

            if os.path.exists(dataset_file):
                return pd.read_pickle(dataset_file)

            power, freqs, _ = self._process(dataset_id)

            columns = [f'f {i}' for i in range(len(freqs))]
            df = pd.DataFrame(power.T, columns=columns)
            df.to_pickle(dataset_file)

            return df

        else:
            raise NotImplementedError(f'input type {str(self.input_type)} not implemented')

    def get_training_df(self,
                          dataset_ids: typing.Iterable[int],
                          targets: typing.Iterable) -> typing.Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieve data for the given dataset IDs.

        Parameters:
            - dataset_ids (Iterable[int]): The list of IDs to fetch data for;
                a pd.Series of ints can be passed as well.
            - targets (Iterable): List of target values corresponding to the dataset IDs.
                Should have the same number of elements as dataset_ids.

        Returns:
            Tuple[pd.DataFrame, pd.Series]:
                - pd.DataFrame: The DataFrame containing the processed data.
                - pd.Series: The Series containing the target values,
                    with the same type as the target input.
        """
        result_df = pd.DataFrame()
        result_target = pd.Series()

        for local_id, target in tqdm.tqdm(
                                list(zip(dataset_ids, targets)), desc='Get data', leave=False):
            data_df = self.get_data(local_id)
            result_df = pd.concat([result_df, data_df], ignore_index=True)

            replicated_targets = pd.Series([target] * len(data_df), name='Target')
            result_target = pd.concat([result_target, replicated_targets], ignore_index=True)

        return result_df, result_target

    def get_training_pytorch_datasets(self,
                                      dataset_ids: typing.Iterable[int],
                                      targets: typing.Iterable) -> 'TorchDataset':
        """
        Retrieve data for the given dataset IDs.

        Parameters:
            - dataset_ids (Iterable[int]): The list of IDs to fetch data for;
                a pd.Series of ints can be passed as well.
            - targets (Iterable): List of target values corresponding to the dataset IDs.
                Should have the same number of elements as dataset_ids.

        Returns:
            TorchDataset: torch.utils.data.Dataset to use in a DataLoader.
        """
        return TorchDataset(self, dataset_ids, targets)

    def plot(self,
             dataset_id: typing.Union[int, typing.Iterable[int]],
             plot_type: PlotType = PlotType.EXPORT_PLOT,
             frequency_in_x_axis: bool=False,
             colormap: color.Colormap = plt.get_cmap('jet'),
             override: bool = False) -> None:
        """
        Display or save images with processed data.

        Parameters:
            dataset_id (Union[int, Iterable[int]]): ID or list of IDs of the dataset to plot.
            plot_type (PlotType): Type of plot to generate (default: PlotType.EXPORT_PLOT).
            frequency_in_x_axis (bool): If True, plot frequency values on the x-axis.
                Default: False.
            colormap (Colormap): Colormap to use for the plot. Default: 'jet'.
            override (bool): If True, override any existing saved plots. Default: False.

        Returns:
            None
        """
        if plot_type != PlotType.SHOW_FIGURE:
            output_dir = os.path.join(self.data_processed_base_dir,
                                      str(self.analysis) + "_" + str(plot_type))
            os.makedirs(output_dir, exist_ok=True)

        if not isinstance(dataset_id, int):
            for local_id in tqdm.tqdm(dataset_id, desc='Plot', leave=False):
                self.plot(
                    dataset_id = local_id,
                    plot_type = plot_type,
                    frequency_in_x_axis = frequency_in_x_axis,
                    colormap = colormap,
                    override = override)
            return

        if plot_type == PlotType.EXPORT_RAW or plot_type == PlotType.EXPORT_PLOT:
            filename = os.path.join(output_dir,f'{dataset_id}.png')
        elif plot_type == PlotType.EXPORT_TEX:
            filename = os.path.join(output_dir,f'{dataset_id}.tex')
        else:
            filename = " "

        if os.path.exists(filename) and not override:
            return

        power, freqs, times = self._process(dataset_id)

        if frequency_in_x_axis:
            power = power.T

        if plot_type == PlotType.EXPORT_RAW:
            power = colormap(power)
            power_color = (power * 255).astype(np.uint8)
            image = PIL.Image.fromarray(power_color)
            image.save(filename)
            return

        times[0] = 0
        freqs[0] = 0

        n_ticks = 5
        time_labels = [iara_pn.get_engineering_notation(times[i], "s")
                    for i in np.linspace(0, len(times)-1, num=n_ticks, dtype=int)]

        frequency_labels = [iara_pn.get_engineering_notation(freqs[i], "Hz")
                    for i in np.linspace(0, len(freqs)-1, num=n_ticks, dtype=int)]

        time_ticks = [(x/4 * (len(times)-1)) for x in range(n_ticks)]
        frequency_ticks = [(y/4 * (len(freqs)-1)) for y in range(n_ticks)]

        plt.figure()
        plt.imshow(power, aspect='auto', origin='lower', cmap=colormap)
        plt.colorbar()

        if frequency_in_x_axis:
            plt.ylabel('Time')
            plt.xlabel('Frequency')
            plt.yticks(time_ticks)
            plt.gca().set_yticklabels(time_labels)
            plt.xticks(frequency_ticks)
            plt.gca().set_xticklabels(frequency_labels)
            plt.gca().invert_yaxis()
        else:
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.xticks(time_ticks)
            plt.gca().set_xticklabels(time_labels)
            plt.yticks(frequency_ticks)
            plt.gca().set_yticklabels(frequency_labels)

        plt.tight_layout()

        if plot_type == PlotType.SHOW_FIGURE:
            plt.show()
        elif plot_type == PlotType.EXPORT_PLOT:
            plt.savefig(filename)
            plt.close()
        elif plot_type == PlotType.EXPORT_TEX:
            tikz.save(filename)
            plt.close()


class TorchDataset(torch_data.Dataset):
    """Custom dataset abstraction to bridge between torch.utils.data.Dataset and
        pandas.DataFrame/pandas.Series.

    This class facilitates the integration of PyTorch's DataLoader with data in the form of a
        pandas.DataFrame and pandas.Series.

    The MEMORY_LIMIT attribute specifies the maximum size (in bytes) of a dataframe that can be
    loaded into memory. When a dataset exceeds this limit, the data is loaded partially as needed.
    While this approach is less efficient, it reduces the likelihood of the dataset being closed
    by the operating system due to memory constraints.
    """
    MEMORY_LIMIT = 2 * 1024 * 1024 * 1024  # 2 gigabytes

    def __init__(self,
                 dataset_processor: DatasetProcessor,
                 dataset_ids: typing.Iterable[int],
                 targets: typing.Iterable) -> None:
        """
        Args:
            - data (pd.DataFrame): Input data.
            - target (pd.Series): Target corresponding to the input data.
        """
        self.dataset_processor = dataset_processor
        self.complete_data = pd.DataFrame()
        self.targets = pd.Series()
        self.dataset_ids = dataset_ids.values
        self.limit_ids = [0]
        self.last_id = -1
        self.data = []

        total_memory = 0
        for dataset_id, target in tqdm.tqdm(list(zip(dataset_ids, targets)),
                                            desc='Loading dataset', leave=False):
            data_df = self.dataset_processor.get_data(dataset_id)
            self.limit_ids.append(self.limit_ids[-1] + len(data_df))

            replicated_targets = pd.Series([target] * len(data_df), name='Target')
            self.targets = pd.concat([self.targets, replicated_targets], ignore_index=True)

            total_memory += data_df.memory_usage(deep=True).sum()

            if total_memory > TorchDataset.MEMORY_LIMIT:
                self.complete_data = None
            else:
                self.complete_data = pd.concat([self.complete_data, data_df], ignore_index=True)

        ## Uncomment to print total memory needed by keeping a dataset in memory
        # unity = ['B', 'KB', 'MB', 'GB', 'TB']
        # cont = 0
        # while total_memory > 1024:
        #     total_memory /= 1024
        #     cont += 1
        # print('total_memory: ', total_memory, unity[cont])

        self.targets = torch.tensor(self.targets.values, dtype=torch.int64)

        if self.complete_data is not None:
            self.complete_data = torch.tensor(self.complete_data.values, dtype=torch.float32)

    def __len__(self):
        return self.limit_ids[-1]

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if self.complete_data is not None:
            return self.complete_data[index], self.targets[index]

        current_id = next(i for i, valor in enumerate(self.limit_ids) if valor > index) - 1

        if current_id != self.last_id:
            self.last_id = current_id
            self.data = self.dataset_processor.get_data(self.dataset_ids[current_id])
            self.data = torch.tensor(self.data.values, dtype=torch.float32)

        return self.data[index - self.limit_ids[current_id]], self.targets[index]
