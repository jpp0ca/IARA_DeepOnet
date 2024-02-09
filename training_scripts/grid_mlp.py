"""
Grid Search Application

This script performs an initial grid search to choose the MLP (Multi-Layer Perceptron) model
configuration for all training in the article. It evaluates different configurations using
cross-validation and selects the one with the best performance based on predefined evaluation
metrics.

The chosen configuration will then be used for further training and analysis in the article.
"""
import os
import argparse
import itertools

import torch

import iara.records
import iara.ml.models.mlp as iara_mlp
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES


def main(override: bool,
        only_first_fold: bool,
        only_sample: bool,
        include_other: bool,
        training_strategy: iara_trn.ModelTrainingStrategy):
    """Grid search main function"""

    grid_str = 'grid_search' if not only_sample else 'grid_search_sample'

    config_dir = f"{DEFAULT_DIRECTORIES.config_dir}/{grid_str}"

    configs = {
        f'mlp_{str(training_strategy)}': iara.records.Collection.OS_SHIP
    }

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
                                include_others = include_other
                            ),
                            only_sample=only_sample
                        )

            output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

            config = iara_exp.Config(
                            name = config_name,
                            dataset = custom_collection,
                            dataset_processor = iara_default.default_iara_audio_processor(),
                            output_base_dir = output_base_dir,
                            n_folds=10 if not only_sample else 3)

            config.save(config_dir)

        mlp_trainers = []

        grid_search = {
            'Neurons': [4, 16, 64, 256],
            'Activation': ['Tanh', 'ReLU', 'PReLU']
        }

        activation_dict = {
                'Tanh': torch.nn.Tanh(),
                'ReLU': torch.nn.ReLU(),
                'PReLU': torch.nn.PReLU()
        }

        param_dict = {}

        combinations = list(itertools.product(*grid_search.values()))
        for combination in combinations:
            param_pack = dict(zip(grid_search.keys(), combination))
            trainer_id = f"mlp_{param_pack['Neurons']}_{param_pack['Activation']}"

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
                    optimizer_allocator=lambda model:
                        torch.optim.Adam(model.parameters(), lr=5e-5),
                    batch_size = 128,
                    n_epochs = 512,
                    patience=32))

        manager = iara_exp.Manager(config, *mlp_trainers)

        result_dict = manager.run(only_first_fold = only_first_fold)

        grid = iara_metrics.GridCompiler()
        for trainer_id, results in result_dict.items():

            for i_fold, result in enumerate(results):

                grid.add(params=param_dict[trainer_id],
                         i_fold=i_fold,
                         target=result['Target'],
                         prediction=result['Prediction'])

        print(grid)



if __name__ == "__main__":
    strategy_str_list = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN MLP grid search analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_first_fold', action='store_true', default=False,
                        help='Execute only first fold. For inspection purpose')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--exclude_other', action='store_true', default=False,
                        help='Include records besides than [Cargo, Tanker, Tug] in training.')
    parser.add_argument('--training_strategy', type=str, choices=strategy_str_list,
                        default=None, help='Strategy for training the model')

    args = parser.parse_args()
    if args.training_strategy is not None:
        index = strategy_str_list.index(args.training_strategy)
        main(override = args.override,
            only_first_fold = args.only_first_fold,
            only_sample = args.only_sample,
            include_other = not args.exclude_other,
            training_strategy = iara_trn.ModelTrainingStrategy(index))

    else:
        for strategy in iara_trn.ModelTrainingStrategy:
            main(override = args.override,
                only_first_fold = args.only_first_fold,
                only_sample = args.only_sample,
                include_other = not args.exclude_other,
                training_strategy = strategy)
