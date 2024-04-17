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
import typing
import argparse

import torch

import iara.records
import iara.ml.models.mlp as iara_mlp
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES



def main(override: bool, folds: typing.List[int], only_sample: bool, cpa_test: int):
    """Main function for the CPA analysis script."""

    cpa_str = 'cpa_analysis' if not only_sample else 'cpa_analysis_sample'

    config_dir = f"{DEFAULT_DIRECTORIES.config_dir}/{cpa_str}"

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
    managers = []
    for config_name, collection in configs.items():

        config = False
        if not override:
            try:
                config = iara_exp.Config.load(config_dir, config_name)

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

            output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{cpa_str}"

            config = iara_exp.Config(
                            name = config_name,
                            dataset = custom_collection,
                            dataset_processor = iara_default.default_iara_lofar_audio_processor(),
                            output_base_dir = output_base_dir,
                            kfolds=10 if not only_sample else 3)

            config.save(config_dir)

        trainers = iara_default.default_trainers(config=config)

        managers.append(iara_exp.Manager(config, *trainers))

        managers[-1].run(folds = folds)

    output_dir = f"{DEFAULT_DIRECTORIES.comparison_dir}/{cpa_str}"

    comparator = iara_exp.Comparator(
                            output_dir= output_dir,
                            manager_list=managers)

    comparator.cross_compare_in_test(folds = folds)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN CPA analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    test_choices=[1, 2]
    parser.add_argument('--cpa_test', type=int, choices=test_choices, default=0,
                        help='Choose test option\
                            [1. Impact of the closest point for CPA,\
                            2. Impact of records containing CPA]')
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

    if args.cpa_test == 0:
        for n_test in test_choices:
            main(args.override, folds_to_execute, args.only_sample, n_test)
    else:
        main(args.override, folds_to_execute, args.only_sample, args.cpa_test)
