"""
Training Configuration Test Program

This script generates a sample training configuration for testing functionality.
In the future, this test script will be part of an application designed to create training
    configurations as described in the associated article.
"""
import argparse
import tqdm

import iara.description
import iara.ml.mlp as iara_model
import iara.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc


def main(override: bool, only_first_fold: bool, only_sample: bool):
    """Main function for the test Training Configuration."""

    config_dir = "./results/configs"
    config_name = "test_training"

    config = False
    if not override:
        try:
            config = iara_trn.TrainingConfig.load(config_dir, config_name)

        except FileNotFoundError:
            pass

    if not config:
        dataset = iara.description.CustomDataset(
                        dataset_type = iara.description.DatasetType.OS_NEAR_CPA_IN,
                        target = iara.description.DatasetTarget(
                            column = 'TYPE',
                            values = ['Cargo', 'Tanker', 'Tug'], # , 'Passenger'
                            include_others = True
                        ),
                        only_sample=only_sample
                    )
        # dataset = iara.description.CustomDataset(
        #                 dataset_type = iara.description.DatasetType.OS_CPA_IN,
        #                 target = iara.description.DatasetTarget(
        #                     column = 'DETAILED TYPE',
        #                     values = ['Bulk Carrier', 'Container Ship'],
        #                     include_others = True
        #                 ),
        #                 filters = [
        #                     iara.description.DatasetFilter(
        #                         column = 'Rain state',
        #                         values = ['No rain']),
        #                     iara.description.DatasetFilter(
        #                         column = 'TYPE',
        #                         values = ['Cargo']),
        #                 ]
        #             )

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
                        n_folds=3)

        config.save(config_dir)

    multiclass_trainer = iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                trainer_id = 'MLP',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                        iara_model.MLP(input_shape=input_shape,
                                            n_neurons=128,
                                            n_targets=n_targets),
                                batch_size = 128,
                                n_epochs = 5)

    specialist_trainer = iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                trainer_id = 'MLP',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                        iara_model.MLP(input_shape=input_shape,
                                            n_neurons=128),
                                batch_size = 128,
                                n_epochs = 5)

    trainer = iara_trn.Trainer(config=config, trainer_list=[multiclass_trainer, specialist_trainer])

    for _ in tqdm.tqdm(range(1), leave=False,
                       desc=" ########################### Training ###########################",
                       bar_format = "{desc}"):
        trainer.run(only_first_fold = only_first_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN Training test')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_first_fold', action='store_true', default=False,
                        help='Execute only first fold. For inspection purpose')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')

    args = parser.parse_args()
    main(args.override, args.only_first_fold, args.only_sample)
