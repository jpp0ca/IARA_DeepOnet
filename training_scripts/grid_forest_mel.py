"""
Grid Search Application

This script performs an initial grid search to choose the MLP (Multi-Layer Perceptron) model
configuration for all training in the article. It evaluates different configurations using
cross-validation and selects the one with the best performance based on predefined evaluation
metrics.

The chosen configuration will then be used for further training and analysis in the article.
"""
import typing
import itertools
import argparse
import tqdm

import numpy as np

import iara.records
import iara.ml.experiment as iara_exp
import iara.ml.dataset as iara_dataset
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES

def main(override: bool,
        folds: typing.List[int],
        only_sample: bool,
        training_strategy: iara_trn.ModelTrainingStrategy):
    """Grid search main function"""

    grid_str = 'grid_search' if not only_sample else 'grid_search_sample'

    grid_val = iara_metrics.GridCompiler()
    grid_trn = iara_metrics.GridCompiler()


    def get_targets(row):
        try:
            length = float(row['Length'])

            if np.isnan(length):
                return 4 # background

            if length < 15:
                return 0
            if length < 50:
                return 1
            if length < 115:
                return 2

            return 3

        except ValueError:
            return np.nan

    for n_mels in tqdm.tqdm([16, 32, 64, 128], leave=False, desc="N_mels", ncols=120):

        config_name = f'forest_mel_{n_mels}_{str(training_strategy)}'

        output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

        dp = iara_manager.AudioFileProcessor(
            data_base_dir = DEFAULT_DIRECTORIES.data_dir,
            data_processed_base_dir = DEFAULT_DIRECTORIES.process_dir,
            normalization = iara_proc.Normalization.MIN_MAX,
            analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
            n_pts = 1024,
            n_overlap = 0,
            decimation_rate = 3,
            n_mels=n_mels,
            integration_interval=2.048
        )

        custom_collection = iara.records.CustomCollection(
                    collection = iara.records.Collection.OS,
                    target = iara.records.GenericTarget(
                        n_targets = 5,
                        function = get_targets,
                        include_others = False
                    ),
                    only_sample=False
                )

        config = iara_exp.Config(
                        name = config_name,
                        dataset = custom_collection,
                        dataset_processor = dp,
                        output_base_dir = output_base_dir,
                        input_type = iara_dataset.InputType.Window(),
                        exclusive_ships_on_test=False)

        grid_search = {
            'Estimators': [10, 25, 50, 75],
            'Max depth': [5, 10, 30]
        }

        mlp_trainers = []
        param_dict = {}

        combinations = list(itertools.product(*grid_search.values()))
        for combination in combinations:
            param_pack = dict(zip(grid_search.keys(), combination))
            trainer_id = f"forest_estimator[{param_pack['Estimators']}]_depth[{param_pack['Max depth']}]"

            param_dict[trainer_id] = param_pack

            mlp_trainers.append(iara_trn.RandomForestTrainer(
                                    training_strategy=training_strategy,
                                    trainer_id = trainer_id,
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators = param_pack['Estimators'],
                                    max_depth = param_pack['Max depth']))


        manager = iara_exp.Manager(config, *mlp_trainers)

        result_dict = manager.run(folds = folds, override = override)

        for trainer_id, results in result_dict.items():

            for i_fold, result in enumerate(results):

                grid_val.add(params=dict({'Number of mels': n_mels}, **param_dict[trainer_id]),
                            i_fold=i_fold,
                            target=result['Target'],
                            prediction=result['Prediction'])

        result_dict = manager.compile_results(folds = folds, dataset_id='trn', trainer_list=mlp_trainers)

        for trainer_id, results in result_dict.items():

            for i_fold, result in enumerate(results):
    
                grid_trn.add(params=dict({'Number of mels': n_mels}, **param_dict[trainer_id]),
                            i_fold=i_fold,
                            target=result['Target'],
                            prediction=result['Prediction'])

    print('grid_trn.print_cm: ' , grid_trn.print_cm())
    print('grid_val.print_cm: ' , grid_val.print_cm())
    print(grid_trn)
    print(grid_val)


if __name__ == "__main__":
    strategy_str_list = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN RandomForest grid search analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--training_strategy', type=str, choices=strategy_str_list,
                        default=None, help='Strategy for training the model')
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

    if args.training_strategy is not None:
        index = strategy_str_list.index(args.training_strategy)
        print(iara_trn.ModelTrainingStrategy(index))
        main(override = args.override,
            folds = folds_to_execute,
            only_sample = args.only_sample,
            training_strategy = iara_trn.ModelTrainingStrategy(index))

    else:
        for strategy in iara_trn.ModelTrainingStrategy:
            main(override = args.override,
                folds = folds_to_execute,
                only_sample = args.only_sample,
                training_strategy = strategy)