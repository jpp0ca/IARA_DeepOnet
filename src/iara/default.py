import enum
import typing

import numpy as np
import pandas as pd

import torch

import iara.ml.experiment as iara_exp
import iara.ml.models.mlp as iara_mlp
import iara.records
import iara.ml.models.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager

class Directories:
    """A structure for configuring directories for locating and storing files."""
    def __init__(self,
                 data_dir="./data/iara",
                 process_dir="./data/iara_processed",
                 training_dir="./results/trainings",
                 comparison_dir="./results/comparisons",
                 tables_dir="./results/tables"):
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.training_dir = training_dir
        self.comparison_dir = comparison_dir
        self.tables_dir = tables_dir


DEFAULT_DIRECTORIES = Directories()
DEFAULT_DEEPSHIP_DIRECTORIES = Directories(data_dir="/data/deepship",
                                           process_dir="./data/deepship_processed")

class Target(enum.Enum):
    # https://www.mdpi.com/2072-4292/11/3/353
    VERY_SMALL = 0
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    BACKGROUND = 4

    @staticmethod
    def classify(ship_length: float) -> 'Target':
        if np.isnan(ship_length):
            return Target.BACKGROUND

        if ship_length < 15:
            return Target.VERY_SMALL
        if ship_length < 50:
            return Target.SMALL
        if ship_length < 100:
            return Target.MEDIUM

        return Target.LARGE

    @staticmethod
    def classify_row(ship_length: pd.DataFrame) -> float:
        try:
            return Target.classify(float(ship_length['Length'])).value
        except ValueError:
            return np.nan

def default_iara_lofar_audio_processor(directories: Directories = DEFAULT_DIRECTORIES):
    """Method to get default AudioFileProcessor for iara."""
    return iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.MIN_MAX,
        analysis = iara_proc.SpectralAnalysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        integration_interval=2.048
    )

def default_iara_mel_audio_processor(directories: Directories = DEFAULT_DIRECTORIES,
                                     n_mels: int = 32):
    """Method to get default AudioFileProcessor for iara."""
    return iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.MIN_MAX,
        analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        n_mels=n_mels,
        integration_interval=2.048
    )

def default_collection(only_sample: bool = False):
    """Method to get default collection for iara."""
    return iara.records.CustomCollection(
            collection = iara.records.Collection.OS,
            target = iara.records.GenericTarget(
                n_targets = 5,
                function = Target.classify_row,
                include_others = False
            ),
            only_sample=only_sample
        )