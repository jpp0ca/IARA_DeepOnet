"""
IARA description Module

This module provides functionality to access the datasets of IARA.
"""
import enum
import os
import typing

import numpy as np
import pandas as pd

class Rain(enum.Enum):
    """
    Enum representing rain noise with various intensity levels
        HODGES, R. P. Underwater Acoustics Analysis, Design and Performance of SONAR.
        Reino Unido: John Wiley and Sons, Ltd, 2010.
    """
    NO_RAIN = 0
    LIGHT = 1 #(<1 mm/h)
    MODERATE = 2 #(<5 mm/h)
    HEAVY = 3 #(<10 mm/h)
    VERY_HEAVY = 4 #(<100 mm/h)

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].capitalize().replace("_", " ")

    @staticmethod
    def classify(values: np.array) -> np.array:
        """Classify a vector of rain intensity data in mm/H.

        Args:
            values (np.array): Vector of rain intensity data in mm/H.

        Returns:
            np.array: Vector of data classified according to the enum.
        """
        return np.select(
            [values == 0, values < 1, values < 5, values < 10],
            [Rain.NO_RAIN, Rain.LIGHT, Rain.MODERATE, Rain.HEAVY],
            Rain.VERY_HEAVY
        )

class SeaState(enum.Enum):
    """
    Enum representing sea state noise with different states.
        HODGES, R. P. Underwater Acoustics Analysis, Design and Performance of SONAR.
        Reino Unido: John Wiley and Sons, Ltd, 2010.
    """
    _0 = 0 # Wind < 0.75 m/s
    _1 = 1 # Wind < 2.5 m/s
    _2 = 2 # Wind < 4.4 m/s
    _3 = 3 # Wind < 6.9 m/s
    _4 = 4 # Wind < 9.8 m/s
    _5 = 5 # Wind < 12.6 m/s
    _6 = 6 # Wind < 19.3 m/s
    _7 = 7 # Wind < 26.5 m/s

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def classify_by_wind(values: np.array) -> np.array:
        """Classify a vector of wind intensity data in m/s.

        Args:
            values (np.array): Vector of wind intensity data in m/s.

        Returns:
            np.array: Vector of data classified according to the enum.
        """
        return np.select(
            [values < 0.75, values < 2.5, values < 4.4, values < 6.9, values < 9.8,
                values < 12.6, values < 19.3],
            [SeaState._0, SeaState._1, SeaState._2, SeaState._3, SeaState._4,
                SeaState._5, SeaState._6],
            SeaState._7
        )


class DatasetType(enum.Enum):
    """Enum representing the different datasets of IARA."""  
    A = 0
    OS_NEAR_CPA_IN = 0
    B = 1
    OS_NEAR_CPA_OUT = 1
    C = 2
    OS_FAR_CPA_IN = 2
    D = 3
    OS_FAR_CPA_OUT = 3
    OS_CPA_IN = 4
    OS_CPA_OUT = 5
    OS_SHIP = 6

    E = 7
    OS_BG = 7

    def __str__(self) -> str:
        if self == DatasetType.OS_CPA_IN:
            return 'with CPA'

        if self == DatasetType.OS_CPA_OUT:
            return 'without CPA'

        if self == DatasetType.OS_SHIP:
            return 'Total'

        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def _get_info_filename(self, only_sample: bool = False) -> str:
        if self.value <= DatasetType.OS_SHIP.value:
            return os.path.join(os.path.dirname(__file__), "dataset_info",
                                "os_ship.csv" if not only_sample else "os_ship_sample.csv")

        if self.value == DatasetType.E.value:
            return os.path.join(os.path.dirname(__file__), "dataset_info",
                                "os_bg.csv" if not only_sample else "os_bg_sample.csv")

        raise UnboundLocalError('info filename not specified')

    def get_selection_str(self) -> str:
        """Get string to filter the 'Dataset' column."""
        if self == DatasetType.OS_CPA_IN:
            return DatasetType.A.get_selection_str() + "|" + DatasetType.C.get_selection_str()

        if self == DatasetType.OS_CPA_OUT:
            return DatasetType.B.get_selection_str() + "|" + DatasetType.D.get_selection_str()

        if self == DatasetType.OS_SHIP:
            return DatasetType.A.get_selection_str() + "|" + DatasetType.B.get_selection_str() + \
                "|" + DatasetType.C.get_selection_str() + "|" + DatasetType.D.get_selection_str()

        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def info_to_df(self, only_sample: bool = False) -> pd.DataFrame:
        """Get information about the dataset

        Args:
            only_sample (bool, optional): If True, provides information about the sampled dataset. 
                If False, includes information about the complete dataset. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing detailed information about the dataset.
        """
        df = pd.read_csv(self._get_info_filename(only_sample=only_sample))
        return df.loc[df['Dataset'].str.contains(self.get_selection_str())]

class DatasetFilter():
    """Class representing a filter to apply on a dataset."""
    def __init__(self,
                 column: str,
                 values: typing.List[str]):
        """
        Parameters:
        - column (str): Name of the column for selection.
        - values (List[str]): List of values for selection.
        """
        self.column = column
        self.values = values

class DatasetTarget(DatasetFilter):
    """Class representing a training target on a dataset."""
    def __init__(self,
                 column: str,
                 values: typing.List[str],
                 include_others: bool = False):
        """
        Parameters:
        - column (str): Name of the column for selection.
        - values (List[str]): List of values for selection.
        - include_others (bool): Indicates whether other values should be compiled as one
            and included or discarded. Default is to discard.
        """
        super().__init__(column, values)
        self.include_others = include_others

class CustomDataset:
    """Class representing a training dataset."""

    DEFAULT_TARGET_HEADER = 'Target'

    def __init__(self,
                 dataset_type: DatasetType,
                 target: DatasetTarget,
                 filters: typing.List[DatasetFilter] = None,
                 only_sample: bool = False):
        """
        Parameters:
        - dataset_type (DatasetType): Dataset to be used.
        - target (DatasetTarget): Target selection for training.
        - filters (List[DatasetFilter], optional): List of filters to be applied.
            Default is use all dataset.
        - only_sample (bool, optional): Use only data available in sample dataset. Default is False.
        """
        self.dataset_type = dataset_type
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
        df = self.dataset_type.info_to_df(only_sample=self.only_sample)
        for filt in self.filters:
            df = df.loc[df[filt.column].isin(filt.values)]

        if not self.target.include_others:
            df = df.loc[df[self.target.column].isin(self.target.values)]

        df[self.DEFAULT_TARGET_HEADER] = df[self.target.column].map(
            {value: index for index, value in enumerate(self.target.values)})
        df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].fillna(
                                                                    len(self.target.values))
        df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].astype(int)

        return df

    def __str__(self) -> str:
        return str(self.get_dataset_info())
