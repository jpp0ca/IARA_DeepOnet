"""
IARA description Module

This module provides functionality to acess the parts of the IARA dataset.
"""
import enum
import os
import numpy as np
import pandas as pd

class Rain(enum.Enum):
    """Enum representing rain noise with various intensity levels."""   
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
    """Enum representing sea state noise with different states."""  
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

class Subdataset(enum.Enum):
    """Enum representing the different sub-datasets of IARA."""  
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

    def __get_info_filename(self, only_sample: bool = False) -> str:
        if self.value <= Subdataset.OS_SHIP.value:
            return os.path.join(os.path.dirname(__file__), "dataset_info",
                                "os_ship.csv" if not only_sample else "os_ship_sample.csv")

        if self.value == Subdataset.E.value:
            return os.path.join(os.path.dirname(__file__), "dataset_info",
                                "os_bg.csv" if not only_sample else "os_bg_sample.csv")

        raise UnboundLocalError('info filename not specified')

    def get_selection_str(self) -> str:
        """get string to filter the 'Dataset' column
        """
        if self == Subdataset.OS_CPA_IN:
            return Subdataset.A.get_selection_str() + "|" + Subdataset.C.get_selection_str()

        if self == Subdataset.OS_CPA_OUT:
            return Subdataset.B.get_selection_str() + "|" + Subdataset.D.get_selection_str()

        if self == Subdataset.OS_SHIP:
            return Subdataset.A.get_selection_str() + "|" + Subdataset.B.get_selection_str() + \
                "|" + Subdataset.C.get_selection_str() + "|" + Subdataset.D.get_selection_str()

        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def __str__(self) -> str:
        if self == Subdataset.OS_CPA_IN:
            return 'with CPA'

        if self == Subdataset.OS_CPA_OUT:
            return 'without CPA'

        if self == Subdataset.OS_SHIP:
            return 'Total'

        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def to_dataframe(self, only_sample: bool = False) -> pd.DataFrame:
        """Get information about the sub-dataset

        Args:
            only_sample (bool, optional): If True, provides information about the sampled dataset. 
                If False, includes information about the complete dataset. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing detailed information about the sub-dataset.
        """
        df = pd.read_csv(self.__get_info_filename(only_sample=only_sample))
        return df.loc[df['Dataset'].str.contains(self.get_selection_str())]
