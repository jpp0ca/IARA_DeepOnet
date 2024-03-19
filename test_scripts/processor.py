"""
Processor Test Program

This script generate as images all processed data in a collection of IARA for test the processing
"""
import os
import shutil

import iara.records
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager


def main(override: bool = True, only_sample: bool = False):
    """Main function print the processed data in a audio dataset."""

    proc_dir = "./data/processed_nan"

    if os.path.exists(proc_dir):
        shutil.rmtree(proc_dir)

    dp = iara_manager.AudioFileProcessor(
        data_base_dir = "./data/iara",
        data_processed_base_dir = proc_dir,
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts = 4096,
        n_overlap = 0,
        decimation_rate = 3,
        n_mels = 128,
        # frequency_limit=5e3,
        # integration_overlap=0,
        # integration_interval=0.256
    )

    df = iara.records.Collection.OS_SHIP.to_df(only_sample=only_sample)

    data, _ = dp.get_complete_df(df['ID'].head(1), df['Sea state'].head(1))
    print(data)
    print('has null: ', data.isnull().any().any())
    print('max: ', data.max().max())
    print('min: ', data.min().min())


    # for plot_type in iara_manager.PlotType:
    for plot_type in [iara_manager.PlotType.EXPORT_RAW]:
        dp.plot(df['ID'].head(10),
                plot_type=plot_type,
                frequency_in_x_axis=True,
                override=override)

if __name__ == "__main__":
    main()
