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
import numpy as np

import torch

import iara.records
import iara.ml.models.mlp as iara_mlp
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
    for eval_subset, eval_strategy in zip(iara_trn.Subset, iara_trn.EvalStrategy):
        result_grid[eval_subset, eval_strategy] = iara_metrics.GridCompiler()

    config_name = f'mlp_lofar_{str(training_strategy)}'
    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

    config = iara_exp.Config(
                    name = config_name,
                    dataset = iara_default.default_collection(only_sample=only_sample),
                    dataset_processor = iara_default.default_iara_lofar_audio_processor(),
                    output_base_dir = output_base_dir,
                    input_type = iara_default.default_window_input())

    grid_search = {
        'Neurons': [4, 16, 64, 256, 1024],
        'Activation': ['Tanh', 'ReLU', 'PReLU'],
        'Weight decay': [0, 1e-3, 1e-5]
    }

    activation_dict = {
            'Tanh': torch.nn.Tanh(),
            'ReLU': torch.nn.ReLU(),
            'PReLU': torch.nn.PReLU()
    }

    mlp_trainers = []
    param_dict = {}

    combinations = list(itertools.product(*grid_search.values()))
    for combination in combinations:
        param_pack = dict(zip(grid_search.keys(), combination))
        weight_str = f"{param_pack['Weight decay']:.0e}" if param_pack['Weight decay'] != 0 else '0'

        trainer_id = f"mlp_{param_pack['Neurons']}_{param_pack['Activation']}_{weight_str}"

        param_dict[trainer_id] = param_pack

        mlp_trainers.append(iara_trn.OptimizerTrainer(
                training_strategy=training_strategy,
                trainer_id = trainer_id,
                n_targets = config.dataset.target.get_n_targets(),
                model_allocator=lambda input_shape, n_targets,
                    n_neurons=param_pack['Neurons'],
                    activation=activation_dict[param_pack['Activation']]:
                        iara_mlp.MLP(input_shape=input_shape,
                            n_neurons=n_neurons,
                            n_targets=n_targets,
                            activation_hidden_layer=activation),
                optimizer_allocator=lambda model, weight_decay=param_pack['Weight decay']:
                    torch.optim.Adam(model.parameters(),
                                        weight_decay=weight_decay)))

    manager = iara_exp.Manager(config, *mlp_trainers)

    result_dict = manager.run(folds = folds, override = override)

    for (eval_subset, eval_strategy), grid in result_grid.items():

        for trainer_id, results in result_dict[eval_subset, eval_strategy].items():

            for i_fold, result in enumerate(results):

                grid.add(params=param_dict[trainer_id],
                            i_fold=i_fold,
                            target=result['Target'],
                            prediction=result['Prediction'])

    for (eval_subset, eval_strategy), grid in result_grid.items():
        print(f'########## {eval_subset} {eval_strategy} ############')
        print('print_cm: ' , grid.print_cm())
        print(grid)

if __name__ == "__main__":
    strategy_str_list = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN MLP grid search analysis')
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