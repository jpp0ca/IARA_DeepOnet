"""
Training Configuration Test Program

This script generates a sample training configuration for testing functionality.
In the future, this test script will be part of an application designed to create training
    configurations as described in the associated article.
"""
import typing
import argparse
import tqdm

import iara.records
import iara.ml.experiment as iara_exp
import iara.ml.models.mlp as iara_mlp
import iara.ml.models.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager


def main(override: bool, folds: typing.List[int], only_sample: bool):
    """Main function for the test Training Configuration."""

    config_dir = "./results/configs"
    config_name = "test_training"

    config = False
    if not override:
        try:
            config = iara_exp.Config.load(config_dir, config_name)

        except FileNotFoundError:
            pass

    if not config:
        # dataset = iara.records.CustomCollection(
        #                 collection = iara.records.Collection.OS_NEAR_CPA_IN,
        #                 target = iara.records.Target(
        #                     column = 'TYPE',
        #                     values = ['Cargo', 'Tanker', 'Tug'], # , 'Passenger'
        #                     include_others = True
        #                 ),
        #                 only_sample=only_sample
        #             )
        dataset = iara.records.CustomCollection(
                        iara.records.Collection.OS_CPA_IN,
                        iara.records.LabelTarget(
                            column = 'DETAILED TYPE',
                            values = ['Bulk Carrier', 'Container Ship'],
                            include_others = True
                        ),
                        iara.records.LabelFilter(
                                column = 'Rain state',
                                values = ['No rain']),
                        iara.records.LabelFilter(
                                column = 'TYPE',
                                values = ['Cargo']),
                        only_sample=only_sample
                    )

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

        config = iara_exp.Config(
                        name = config_name,
                        dataset = dataset,
                        dataset_processor = dp,
                        output_base_dir = "./results/trainings",
                        kfolds=3)

        config.save(config_dir)

    multiclass_trainer = iara_trn.OptimizerTrainer(
                                training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                                trainer_id = 'MLP',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                        iara_mlp.MLP(input_shape=input_shape,
                                            n_neurons=128,
                                            n_targets=n_targets),
                                batch_size = 128,
                                n_epochs = 5)

    specialist_trainer = iara_trn.OptimizerTrainer(
                                training_strategy=iara_trn.ModelTrainingStrategy.CLASS_SPECIALIST,
                                trainer_id = 'MLP',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                        iara_mlp.MLP(input_shape=input_shape,
                                            n_neurons=128),
                                batch_size = 128,
                                n_epochs = 5)

    trainer = iara_exp.Manager(config, multiclass_trainer, specialist_trainer)

    for _ in tqdm.tqdm(range(1), leave=False,
                       desc=" ########################### Training ###########################",
                       bar_format = "{desc}"):
        trainer.run(folds = folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN Training test')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--fold', type=str, default='',
                        help='Specify folds to be executed. Example: 0,4-7')

    args = parser.parse_args()

    folds_to_execute = []
    if args.fold:
        fold_ranges = args.fold.split(',')
        for fold_range in fold_ranges:
            if '-' in fold_range:
                start, end = map(int, fold_range.split('-'))
                folds_to_execute.extend(range(start, end + 1))
            else:
                folds_to_execute.append(int(fold_range))

    main(args.override, folds_to_execute, args.only_sample)
