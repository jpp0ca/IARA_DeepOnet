
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager

class Directories:
    """A structure for configuring directories for locating and storing files."""
    def __init__(self,
                 data_dir="./data/iara",
                 process_dir="./data/iara_processed",
                 config_dir="./results/configs",
                 training_dir="./results/trainings"):
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.config_dir = config_dir
        self.training_dir = training_dir


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
        decimation_rate = 3,
        frequency_limit=5e3,
        integration_overlap=0,
        integration_interval=1.024
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
        frequency_limit=5e3,
        integration_overlap=0,
        integration_interval=1.024
    )
