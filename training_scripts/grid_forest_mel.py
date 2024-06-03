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
import time

import numpy as np

import iara.records
import iara.utils
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

    result_grid = {}
    for eval_subset, eval_strategy in itertools.product(iara_trn.Subset, iara_trn.EvalStrategy):
        result_grid[eval_subset, eval_strategy] = iara_metrics.GridCompiler()

    for n_mels in tqdm.tqdm([16, 32, 64, 128, 256], leave=False, desc="N_mels", ncols=120):
    # for n_mels in tqdm.tqdm([128], leave=False, desc="N_mels", ncols=120):

        config_name = f'forest_mel_{n_mels}_{str(training_strategy)}'

        output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

        config = iara_exp.Config(
                name = config_name,
                dataset = iara_default.default_collection(only_sample=only_sample),
                dataset_processor = iara_default.default_iara_mel_audio_processor(n_mels=n_mels),
                output_base_dir = output_base_dir,
                input_type = iara_default.default_window_input())

        grid_search = {
            'Estimators': [10, 25, 50, 75, 100],
            'Max depth': [5, 10, 15, 20, 30]
        }
        # grid_search = {
        #     'Estimators': [50],
        #     'Max depth': [15]
        # }

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

        for (eval_subset, eval_strategy), grid in result_grid.items():

            for trainer_id, results in result_dict[eval_subset, eval_strategy].items():

                for i_fold, result in enumerate(results):

                    grid.add(params=dict({'Number of mels': n_mels}, **param_dict[trainer_id]),
                             i_fold=i_fold,
                             target=result['Target'],
                             prediction=result['Prediction'])

    for (eval_subset, eval_strategy), grid in result_grid.items():
        print(f'########## {eval_subset} {eval_strategy} ############')
        print(grid)


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

    start_time = time.time()

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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")
