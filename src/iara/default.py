import torch

import iara.ml.experiment as iara_exp
import iara.ml.models.mlp as iara_mlp
import iara.ml.models.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager

class Directories:
    """A structure for configuring directories for locating and storing files."""
    def __init__(self,
                 data_dir="./data/iara",
                 process_dir="./data/iara_processed",
                 config_dir="./results/configs",
                 training_dir="./results/trainings",
                 comparison_dir="./results/comparisons",
                 tables_dir="./results/tables"):
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.config_dir = config_dir
        self.training_dir = training_dir
        self.comparison_dir = comparison_dir
        self.tables_dir = tables_dir


DEFAULT_DIRECTORIES = Directories()
DEFAULT_DEEPSHIP_DIRECTORIES = Directories(data_dir="/data/deepship",
                                           process_dir="./data/deepship_processed")


def default_iara_audio_processor(directories: Directories = DEFAULT_DIRECTORIES):
    """Method to get default AudioFileProcessor for iara."""
    return iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3
    )

def default_iara_mel_audio_processor(directories: Directories = DEFAULT_DIRECTORIES):
    """Method to get default AudioFileProcessor for iara."""
    return iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        n_mels=64,
    )

def default_deepship_audio_processor(directories: Directories = DEFAULT_DEEPSHIP_DIRECTORIES):
    """Method to get default AudioFileProcessor for deepship."""
    return iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 2,
    )

def default_trainers(config: iara_exp.Config):
    """Get trainers for all the best models in the grid search configuration

    Args:
        config (iara_exp.Config): Training configuration
    """

    trainers = []

    for training_type in iara_trn.ModelTrainingStrategy:
        trainers.append(iara_trn.OptimizerTrainer(
                training_strategy=training_type,
                trainer_id = f'mlp_{str(training_type)}',
                n_targets = config.dataset.target.get_n_targets(),
                model_allocator=lambda input_shape, n_targets:
                        iara_mlp.MLP(input_shape=input_shape,
                            n_neurons=256,
                            n_targets=n_targets,
                            activation_hidden_layer=torch.nn.PReLU()),
                optimizer_allocator=lambda model:
                    torch.optim.Adam(model.parameters(), lr=5e-5),
                batch_size = 128,
                n_epochs = 512,
                patience=32))

    for training_type in iara_trn.ModelTrainingStrategy:
        trainers.append(iara_trn.RandomForestTrainer(
                                    training_strategy=training_type,
                                    trainer_id = f'forest_{str(training_type)}',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators = 100))

    return trainers