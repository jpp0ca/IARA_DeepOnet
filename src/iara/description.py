"""IARA description Module

This module provides functionality to acess the parts of the IARA dataset.

This module defines the Enums that representing different sources of background noise:
- Rain: Enum for rain noise with various intensity levels.
- Sea: Enum for sea state noise with different states.
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
        return str(self.name).split('.')[-1].capitalize().replace("_", " ")

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

class Sea_state(enum.Enum):
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
            [values < 0.75, values < 2.5, values < 4.4, values < 6.9, values < 9.8, values < 12.6, values < 19.3],
            [Sea_state._0, Sea_state._1, Sea_state._2, Sea_state._3, Sea_state._4, Sea_state._5, Sea_state._6],
            Sea_state._7
        )

class Subdataset(enum.Enum):
    """Enum representing the different sub-datasets of IARA."""  
    A = 0
    os_near_with_cpa = 0
    B = 1
    os_near_without_cpa = 1
    C = 2
    os_far_with_cpa = 2
    D = 3
    os_far_without_cpa = 3
    os_with_cpa = 4
    os_without_cpa = 5
    os_ship = 6

    E = 7
    os_background = 7

    def __get_info_filename(self, only_sample: bool = False) -> str:

        if (self.value <= Subdataset.os_ship.value):
            return os.path.join(os.path.dirname(__file__), "dataset_info", "os_ship.csv" if not only_sample else "os_ship_sample.csv")

        if (self.value == Subdataset.E.value):
            return os.path.join(os.path.dirname(__file__), "dataset_info", "os_bg.csv" if not only_sample else "os_bg_sample.csv")

        raise UnboundLocalError('info filename not specified')

    def __get_selection_str(self) -> str:
        if (self == Subdataset.os_with_cpa):
            return Subdataset.A.__get_selection_str() + "|" + Subdataset.C.__get_selection_str()

        if (self == Subdataset.os_without_cpa):
            return Subdataset.B.__get_selection_str() + "|" + Subdataset.D.__get_selection_str()

        if (self == Subdataset.os_ship):
            return Subdataset.A.__get_selection_str() + "|" + Subdataset.B.__get_selection_str() + "|" + Subdataset.C.__get_selection_str() + "|" + Subdataset.D.__get_selection_str()

        return str(self.name).split('.')[-1]

    def __str__(self) -> str:
        if (self == Subdataset.os_with_cpa):
            return 'with CPA'

        if (self == Subdataset.os_without_cpa):
            return 'without CPA'

        if (self == Subdataset.os_ship):
            return 'Total'

        return str(self.name).split('.')[-1]

    def to_dataframe(self, only_sample: bool = False) -> pd.DataFrame:
        """Get information about the sub-dataset

        Args:
            only_sample (bool, optional): If True, provides information about the sampled dataset. 
                If False, includes information about the complete dataset. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing detailed information about the sub-dataset.
        """
        df = pd.read_csv(self.__get_info_filename(only_sample=only_sample))
        return df.loc[df['Dataset'].str.contains(self.__get_selection_str())]
