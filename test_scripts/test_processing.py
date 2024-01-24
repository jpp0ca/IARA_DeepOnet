"""
Test processing Test Program

This script generate as images all processed data in a subdaset of IARA for test the processing
"""
import iara.description as iara_desc
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc


def main(override: bool = False, only_sample: bool = True):
    """Main function print the processed data in a subdataset of IARA."""

    dp = iara_data_proc.DatasetProcessor(
        data_base_dir = "./data/raw_dataset",
        data_processed_base_dir = "./data/processed",
        normalization = iara_proc.Normalization.MIN_MAX,
        analysis = iara_proc.Analysis.LOFAR,
        n_pts = 640,
        n_overlap = 0,
        decimation_rate = 3,
        frequency_limit=5e3
    )

    df = iara_desc.Subdataset.OS_SHIP.to_dataframe(only_sample=only_sample)

    for plot_type in iara_data_proc.PlotType:
        dp.plot(df['ID'].head(2),
                plot_type=plot_type,
                frequency_in_x_axis=True,
                override=override)

if __name__ == "__main__":
    main()
