"""
Data Visualization Teste Program

This script generate t-sne visualization for each dataset in iara.
"""
import os
import argparse
import shutil

import numpy as np

import iara.records
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager
import iara.ml.visualization as iara_vis

def main(override: bool, only_sample: bool, include_others: bool):
    """Main function for visualizing data using t-SNE."""

    output_dir = './results/plots/t-sne'

    if os.path.exists(output_dir) and override:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dp = iara_manager.AudioFileProcessor(
        data_base_dir = "./data/raw_dataset",
        data_processed_base_dir = "./data/processed",
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        frequency_limit=5e3,
        integration_overlap=0,
        integration_interval=1.024
    )

    # for colletion in iara.records.Collection:
    for colletion in [iara.records.Collection.A, iara.records.Collection.OS_SHIP]:
        if colletion.value > iara.records.Collection.OS_SHIP.value:
            break

        sufix = 'with' if include_others else 'without'
        filename=os.path.join(output_dir, f'{colletion.get_prettier_str()} {sufix} others.png')

        if os.path.exists(filename):
            continue

        values = ['Cargo', 'Tanker', 'Tug']

        dataset = iara.records.CustomCollection(
                        collection = colletion,
                        target = iara.records.Target(
                            column = 'TYPE',
                            values = values,
                            include_others = include_others
                        ),
                        only_sample=only_sample
                    )

        if include_others:
            names = np.array(values + ['Others'])
        else:
            names = np.array(values)

        df = dataset.to_df()

        data, target = dp.get_complete_df(df['ID'], df['Target'])
        target = names[target.to_numpy()]

        iara_vis.export_tsne(data=data.to_numpy(),
                             labels=target,
                             filename=filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize data as t-SNE')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--include_others', action='store_true', default=None,
                        help='Include Other than Cargo, Tanker, Tug.')

    args = parser.parse_args()

    if args.include_others is None:
        main(args.override, args.only_sample, True)
        main(args.override, args.only_sample, False)
    else:
        main(args.override, args.only_sample, args.include_others)
