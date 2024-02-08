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

import torch

import iara.records
import iara.ml.models.mlp as iara_mlp
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager


def main(override: bool, only_first_fold: bool, only_sample: bool, cpa_test: int):
    """Main function for the CPA analysis script."""

    config_dir = "./results/configs"

    if cpa_test == 1:
        configs = {
            'near_cpa': iara.records.Collection.OS_NEAR_CPA_IN,
            'far_cpa': iara.records.Collection.OS_FAR_CPA_IN,
        }
    elif cpa_test == 2:
        configs = {
            'cpa_in': iara.records.Collection.OS_CPA_IN,
            'cpa_out': iara.records.Collection.OS_CPA_OUT
        }
    else:
        print('Not implemented test')
        return

    config = False
    trainer_list = []
    for config_name, collection in configs.items():

        config = False
        if not override:
            try:
                config = iara_exp.Config.load(config_dir, config_name if not only_sample\
                                        else f"{config_name}_sample")

            except FileNotFoundError:
                pass

        if not config:
            custom_collection = iara.records.CustomCollection(
                            collection = collection,
                            target = iara.records.Target(
                                column = 'TYPE',
                                values = ['Cargo', 'Tanker', 'Tug'],
                                include_others = True
                            ),
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
                            dataset = custom_collection,
                            dataset_processor = dp,
                            output_base_dir = "./results/trainings/cpa_analysis" if not only_sample\
                                        else "./results/trainings/cpa_analysis_sample" ,
                            n_folds=10 if not only_sample else 3)

            config.save(config_dir)

        trainer_list = []


        trainer_list.append(iara_trn.RandomForestTrainer(
                                    training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                                    trainer_id = 'RandomForest',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=100,
                                    max_depth=None))

        trainer_list.append(iara_trn.OptimizerTrainer(
                                    training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                                    trainer_id = 'MLP',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    model_allocator=lambda input_shape, n_targets:
                                        iara_mlp.MLP(input_shape=input_shape,
                                            n_neurons=64,
                                            n_targets=n_targets),
                                    optimizer_allocator=lambda model:
                                        torch.optim.Adam(model.parameters(), lr=5e-5),
                                    batch_size = 128,
                                    n_epochs = 256,
                                    patience=8))

        trainer_list.append(iara_trn.RandomForestTrainer(
                                    training_strategy=iara_trn.ModelTrainingStrategy.CLASS_SPECIALIST,
                                    trainer_id = 'RandomForest',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=100,
                                    max_depth=None))

        trainer_list.append(iara_trn.OptimizerTrainer(
                                    training_strategy=iara_trn.ModelTrainingStrategy.CLASS_SPECIALIST,
                                    trainer_id = 'MLP',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    model_allocator=lambda input_shape, _:
                                        iara_mlp.MLP(input_shape=input_shape,
                                            n_neurons=64),
                                    batch_size = 128,
                                    n_epochs = 256,
                                    patience=None))

        trainer_list.append(iara_exp.Manager(config, *trainer_list))

        trainer_list[-1].run(only_first_fold = only_first_fold)

    comparator = iara_exp.Comparator(
                            output_dir= "./results/comparisons/cpa_analysis",
                            trainer_list=trainer_list)

    comparator.cross_compare_in_test(only_firs_fold=only_first_fold)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN CPA analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_first_fold', action='store_true', default=False,
                        help='Execute only first fold. For inspection purpose')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    test_choices=[1, 2]
    parser.add_argument('--cpa_test', type=int, choices=test_choices, default=0,
                        help='Choose test option\
                            [1. Impact of the closest point for CPA,\
                            2. Impact of records containing CPA]')

    args = parser.parse_args()
    if args.cpa_test == 0:
        for n_test in test_choices:
            main(args.override, args.only_first_fold, args.only_sample, n_test)
    else:
        main(args.override, args.only_first_fold, args.only_sample, args.cpa_test)
