"""
Processing Test Program

This script generate as images all processed data in a subdaset of IARA for test the processing
"""
import iara.description as iara_desc
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc


def main(override: bool = True, only_sample: bool = False):
    """Main function print the processed data in a dataset of IARA."""

    dp = iara_data_proc.DatasetProcessor(
        data_base_dir = "./data/raw_dataset",
        data_processed_base_dir = "./data/processed",
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        frequency_limit=5e3,
        integration_overlap=0,
        integration_interval=0.5
    )

    df = iara_desc.DatasetType.OS_SHIP.info_to_df(only_sample=only_sample)
    data, _ = dp.get_training_df(df['ID'], [0, 1])
    print(data)
    print('has null: ', data.isnull().any().any())

    # for plot_type in iara_data_proc.PlotType:
    for plot_type in [iara_data_proc.PlotType.EXPORT_RAW]:
        dp.plot(df['ID'].head(2),
                plot_type=plot_type,
                frequency_in_x_axis=True,
                override=override)

if __name__ == "__main__":
    main()
