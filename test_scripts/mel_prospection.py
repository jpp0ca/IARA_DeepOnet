"""
Processor Test Program

This script generate as images all processed data in a collection of IARA for test the processing
"""
import os
import shutil

import iara.records
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager
import iara.default


def main(override: bool = True, only_sample: bool = False):
    """Main function print the processed data in a audio dataset."""

    proc_dir = "./data/iara_processed"

    if os.path.exists(proc_dir):
        shutil.rmtree(proc_dir)
    dp = iara.default.default_iara_mel_audio_processor()

    df = iara.records.Collection.OS_SHIP.to_df(only_sample=only_sample)

    for n_mels in [20, 25, 30, 35, 40]:
        dp.n_mels = n_mels

        for plot_type in [iara_manager.PlotType.EXPORT_PLOT]:
            dp.plot(df['ID'].head(10),
                    plot_type=plot_type,
                    frequency_in_x_axis=True,
                    override=override)

if __name__ == "__main__":
    main()
