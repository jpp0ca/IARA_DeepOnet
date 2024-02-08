"""
Processor Test Program

This script generate as images all processed data in a collection of IARA for test the processing
"""
import iara.records
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager


def main(override: bool = True, only_sample: bool = False):
    """Main function print the processed data in a audio dataset."""

    dp = iara_manager.AudioFileProcessor(
        data_base_dir = "./data/raw_dataset",
        data_processed_base_dir = "./data/processed",
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        n_mels = 128,
        frequency_limit=5e3,
        integration_overlap=0,
        integration_interval=1.024
    )

    df = iara.records.Collection.OS_SHIP.to_df(only_sample=only_sample)

    data, _ = dp.get_complete_df(df['ID'].head(2), [0, 1])
    print(data)
    print('has null: ', data.isnull().any().any())


    for plot_type in iara_manager.PlotType:
    # for plot_type in [iara_data_proc.PlotType.EXPORT_RAW]:
        dp.plot(df['ID'].head(2),
                plot_type=plot_type,
                frequency_in_x_axis=True,
                override=override)

if __name__ == "__main__":
    main()
