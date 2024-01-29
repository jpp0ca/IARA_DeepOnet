"""
Script for analyzing the impact of Closest Point of Approach (CPA) on classification.

This script generates two tests:

1. Impact of the closest point for CPA:
   - Classifier trained on OS_NEAR_CPA_IN data and evaluated on OS_FAR_CPA_IN data.
   - Classifier trained on OS_FAR_CPA_IN data and evaluated on OS_NEAR_CPA_IN data.

2. Impact of records containing CPA:
   - Classifier trained on OS_CPA_IN data and evaluated on OS_CPA_OUT data.
   - Classifier trained on OS_CPA_OUT data and evaluated on OS_CPA_IN data.
"""
import argparse
import tqdm

import iara.description
import iara.ml.mlp as iara_model
import iara.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc


def main(override: bool, only_first_fold: bool, only_sample: bool):
    """Main function for the CPA analysis script."""

    config_dir = "./results/configs"

    configs = {
        'near_cpa': iara.description.DatasetType.OS_NEAR_CPA_IN,
        'far_cpa': iara.description.DatasetType.OS_FAR_CPA_IN,
        'cpa_in': iara.description.DatasetType.OS_CPA_IN,
        'cpa_out': iara.description.DatasetType.OS_CPA_OUT
    }

    for config_name, data_type in tqdm.tqdm(configs.items(), leave=False, desc="Configs"):

        config = False
        if not override:
            try:
                config = iara_trn.TrainingConfig.load(config_dir, config_name)

            except FileNotFoundError:
                pass

        if not config:
            dataset = iara.description.CustomDataset(
                            dataset_type = data_type,
                            target = iara.description.DatasetTarget(
                                column = 'TYPE',
                                values = ['Cargo', 'Tanker', 'Tug'],
                                include_others = True
                            ),
                            only_sample=only_sample
                        )

            dp = iara_data_proc.DatasetProcessor(
                data_base_dir = "./data/raw_dataset",
                data_processed_base_dir = "./data/processed",
                normalization = iara_proc.Normalization.MIN_MAX,
                analysis = iara_proc.SpectralAnalysis.LOFAR,
                n_pts = 640,
                n_overlap = 0,
                decimation_rate = 3,
            )

            config = iara_trn.TrainingConfig(
                            name = config_name,
                            dataset = dataset,
                            dataset_processor = dp,
                            output_base_dir = "./results/trainings",
                            n_folds=10 if not only_sample else 3)

            config.save(config_dir)

        mlp_trainer = iara_trn.NNTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'MLP',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    model_allocator=lambda input_shape:
                                                iara_model.MLP(input_shape=input_shape,
                                                            n_neurons=64,
                                                            n_targets=4),
                                    batch_size = 128,
                                    n_epochs = 32)

        trainer = iara_trn.Trainer(config=config, trainer_list=[mlp_trainer])

        trainer.run(only_first_fold = only_first_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN CPA analysis')
    parser.add_argument('--override', action='store_true',
                        help='Ignore old runs')
    parser.add_argument('--only_first_fold', action='store_true',
                        help='Execute only first fold. For inspection purpose')
    parser.add_argument('--only_sample', action='store_true',
                        help='Execute only in sample_dataset. For quick training and test.')

    args = parser.parse_args()
    main(args.override, args.only_first_fold, args.only_sample)
