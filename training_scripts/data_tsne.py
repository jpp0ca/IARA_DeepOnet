import os
import argparse
import shutil

import numpy as np

import iara.description
import iara.ml.mlp as iara_model
import iara.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc
import iara.ml.visualization as iara_vis

def main(override: bool, only_sample: bool):

    output_dir = './results/plots/t-sne'

    if os.path.exists(output_dir) and override:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

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
        integration_interval=1.024
    )

    for type in iara.description.DatasetType:
    # for type in [iara.description.DatasetType.A]:
        if type.value > iara.description.DatasetType.OS_SHIP.value:
            break

        filename=os.path.join(output_dir, f'{type.get_prettier_str()}.png')

        if os.path.exists(filename):
            continue

        values = ['Cargo', 'Tanker', 'Tug']


        dataset = iara.description.CustomDataset(
                        dataset_type = type,
                        target = iara.description.DatasetTarget(
                            column = 'TYPE',
                            values = values,
                            include_others = True
                        ),
                        only_sample=only_sample
                    )
        
        values.append('Others')

        df = dataset.get_dataset_info()

        data, target = dp.get_training_df(df['ID'], df['Target'])
        names = np.array(values)
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

    args = parser.parse_args()
    main(args.override, args.only_sample)