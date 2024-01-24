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
        n_pts = 640,
        n_overlap = 0,
        decimation_rate = 3,
        frequency_limit=5e3
    )

    print(dp.get_data(dataset_id=100))
    print(dp.get_training_data(dataset_ids=range(5,10), targets=[2, 2, 1, 1, 1]))

if __name__ == "__main__":
    main()
