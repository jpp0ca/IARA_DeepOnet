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
import pandas as pd
import scipy.io.wavfile as scipy_wav

import iara.processing.analysis as iara_proc


class InputType(enum.Enum):
    """Enum defining training types."""
    WINDOW = 0
    IMAGE = 1

def get_id(file:str) -> int:
    """
    Default function to extracts the ID from the given file name.

    Parameters:
        file (str): The file name without extension.

    Returns:
        int: The extracted ID.
    """
    return int(file.rsplit('-',maxsplit=1)[-1])

class DatasetProcessor():
    """ Class for handling acess to process data from a dataset. """

    def __init__(self,
                data_base_dir: str,
                dataframe_base_dir: str,
                normalization: iara_proc.Normalization,
                analysis: iara_proc.Analysis,
                n_pts: int = 1024,
                n_overlap: int = 0,
                n_mels: int = 256,
                decimation_rate: int = 1,
                extract_id: typing.Callable[[str], int] = get_id,
                input_type: InputType = InputType.WINDOW
                ) -> None:
        """
        Parameters:
            data_base_dir (str): Base directory for raw data.
            dataframe_base_dir (str): Base directory for dataframes.
            normalization (iara_proc.Normalization): Normalization object.
            analysis (iara_proc.Analysis): Analysis object.
            n_pts (int): Number of points for use in analysis.
            n_overlap (int): Number of points to overlap for use in analysis.
            n_mels (int): Number of Mel frequency bins for use in analysis.
            decimation_rate (int): Decimation rate for use in analysis when mel based.
            extract_id (Callable[[str], str]): Function to extract ID from a file name without
                extension. Default is split based on '-' em get last part of the name
            input_type (InputType): Type of input data. Default WINDOW.
        """
        self.data_base_dir = data_base_dir
        self.dataframe_base_dir = dataframe_base_dir
        self.normalization = normalization
        self.analysis = analysis
        self.n_pts = n_pts
        self.n_overlap = n_overlap
        self.n_mels = n_mels
        self.decimation_rate = decimation_rate
        self.extract_id = extract_id
        self.input_type = input_type
        self.load_data = {}

        os.makedirs(self.dataframe_base_dir, exist_ok=True)


    def find_raw_file(self, iara_id: int) -> str:
        """
        Finds the raw file associated with the given ID.

        Parameters:
            iara_id (int): The ID to search for.

        Returns:
            str: The path to the raw file.

        Raises:
            UnboundLocalError: If the file is not found.
        """
        for root, _, files in os.walk(self.data_base_dir):
            for file in files:
                filename, extension = os.path.splitext(file)
                if extension == ".wav" and self.extract_id(filename) == iara_id:
                    return os.path.join(root, file)
        raise UnboundLocalError(f'file {iara_id} not found in {self.data_base_dir}')

    def __load(self, iara_id: int):

        if self.input_type == InputType.WINDOW:

            dataset_file = os.path.join(self.dataframe_base_dir, f'{iara_id}.pkl')

            if os.path.exists(dataset_file):
                self.load_data[iara_id] = pd.read_pickle(dataset_file)
                return

            file = self.find_raw_file(iara_id = iara_id)

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

            row_list = []
            for t in range(len(times)):
                row = list(power[:,t].flatten())
                row_list.append(row)

            columns = [f'f {i}' for i in range(len(freqs))]
            df = pd.DataFrame(row_list, columns=columns)
            df.to_pickle(dataset_file)

            self.load_data[iara_id] = df

        else:
            raise NotImplementedError(f'input type {str(self.input_type)} not implemented')


    def get_data(self, iara_id: typing.Union[int, typing.List[int]]) -> pd.DataFrame:
        """
        Gets data for the given ID or list of IDs.

        Parameters:
            iara_id (Union[int, List[int]]): The ID or list of IDs to get data for.

        Returns:
            pd.DataFrame: The DataFrame containing the processed data.
        """

        if isinstance(iara_id, int):
            if not iara_id in self.load_data:
                self.__load(iara_id)
            return self.load_data[iara_id]

        result_df = pd.DataFrame()
        for local_id in iara_id if len(iara_id) == 1 else tqdm.tqdm(
                                                        iara_id, desc='Get data', leave=False):
            result_df = pd.concat([result_df, self.get_data(local_id)], ignore_index=True)
        return result_df
