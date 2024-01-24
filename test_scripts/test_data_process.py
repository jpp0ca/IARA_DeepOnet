"""
Dataset Info Tables Test Program

This script test the iara.processing.dataset.DatasetProcessor and acess part of the dataset
"""
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc

def main():
    """Main function for test acess to dataset process data."""

    dp = iara_data_proc.DatasetProcessor(
        data_base_dir = "./data/raw_dataset",
        dataframe_base_dir = "./data/lofar",
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.Analysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        n_mels = 256,
        decimation_rate = 3,
    )

    print(dp.get_data(iara_id=100))
    print(dp.get_data(iara_id=range(5,10)))
    print(dp.get_data(iara_id=[101]))

if __name__ == "__main__":
    main()
