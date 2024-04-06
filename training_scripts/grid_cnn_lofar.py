"""
Grid Search Application

This script performs an initial grid search to choose the CNN (Convolutional Neural Network) model
configuration for all training in the article. It evaluates different configurations using
cross-validation and selects the one with the best performance based on predefined evaluation
metrics.

The chosen configuration will then be used for further training and analysis in the article.
"""
import typing
import argparse
import itertools

import torch

import iara.utils
import iara.records
import iara.ml.models.cnn as iara_cnn
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics
import iara.ml.dataset as iara_dataset

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES


def main(override: bool,
        folds: typing.List[int],
        only_sample: bool,
        include_other: bool,
        training_strategy: iara_trn.ModelTrainingStrategy):
    """Grid search main function"""

    iara.utils.print_available_device()

    grid_str = 'grid_search' if not only_sample else 'grid_search_sample'

    config_dir = f"{DEFAULT_DIRECTORIES.config_dir}/{grid_str}"

    configs = {
        f'cnn_lofar_{str(training_strategy)}': iara.records.Collection.OS_SHIP
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
                            input_type = iara_dataset.InputType.Image(16, 0.5),
                            output_base_dir = output_base_dir)

            config.save(config_dir)

        mlp_trainers = []

        # grid_search = {
        #     'conv_n_neurons': [
        #                 [16, 32],
        #                 [32, 64],
        #                 [16, 32, 64]
        #             ],
        #     'classification_n_neurons': [16, 128, 1024],
        #     'Activation': ['ReLU', 'PReLU', 'LeakyReLU'],
        #     'Weight decay': [0, 1e-3, 1e-5],
        #     'conv_pooling': ['Max', 'Avg'],
        #     'kernel': [3, 5],
        # }

        grid_search = {
            'conv_n_neurons': [
                        [16, 32],
                        [32, 64],
                    ],
            'classification_n_neurons': [128],
            'Activation': ['LeakyReLU'],
            'Weight decay': [0],
            'conv_pooling': ['Avg'],
            'kernel': [5],
        }

        activation_dict = {
                'Tanh': torch.nn.Tanh(),
                'ReLU': torch.nn.ReLU(),
                'PReLU': torch.nn.PReLU(),
                'LeakyReLU': torch.nn.LeakyReLU()
        }

        pooling_dict = {
                'Max': torch.nn.MaxPool2d(2, 2),
                'Avg': torch.nn.AvgPool2d(2, 2)
        }

        param_dict = {}

        combinations = list(itertools.product(*grid_search.values()))
        for combination in combinations:
            param_pack = dict(zip(grid_search.keys(), combination))

            weight_str = f"{param_pack['Weight decay']:.0e}" if param_pack['Weight decay'] != 0 else '0'

            trainer_id = f"cnn_{param_pack['conv_n_neurons']}_\
                    {param_pack['classification_n_neurons']}_{param_pack['Activation']}_\
                    {weight_str}_{param_pack['conv_pooling']}"

            param_dict[trainer_id] = param_pack

            mlp_trainers.append(iara_trn.OptimizerTrainer(
                    training_strategy=training_strategy,
                    trainer_id = trainer_id,
                    n_targets = config.dataset.target.get_n_targets(),
                    model_allocator=lambda input_shape, n_targets,
                        conv_neurons=param_pack['conv_n_neurons'],
                        class_neurons=param_pack['classification_n_neurons'],
                        activation=activation_dict[param_pack['Activation']],
                        pooling=pooling_dict[param_pack['conv_pooling']],
                        kernel=param_pack['kernel']:
                            iara_cnn.CNN(
                                input_shape=input_shape,
                                conv_activation = activation,
                                conv_n_neurons=conv_neurons,
                                kernel_size=kernel,
                                classification_n_neurons=class_neurons,
                                n_targets=n_targets),
                    optimizer_allocator=lambda model, weight_decay=param_pack['Weight decay']:
                        torch.optim.Adam(model.parameters(),
                                         weight_decay=weight_decay),
                    batch_size = 64))

        manager = iara_exp.Manager(config, *mlp_trainers)

        result_dict = manager.run(folds = folds)

        grid = iara_metrics.GridCompiler()
        for trainer_id, results in result_dict.items():

            for i_fold, result in enumerate(results):

                grid.add(params={'trainer_id': trainer_id},
                         i_fold=i_fold,
                         target=result['Target'],
                         prediction=result['Prediction'])

        print(grid)



if __name__ == "__main__":

    strategy_str_list = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN MLP grid search analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--exclude_other', action='store_true', default=False,
                        help='Include records besides than [Cargo, Tanker, Tug] in training.')
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
        main(override = args.override,
            folds = folds_to_execute,
            only_sample = args.only_sample,
            include_other = not args.exclude_other,
            training_strategy = iara_trn.ModelTrainingStrategy(index))

    else:
        for strategy in iara_trn.ModelTrainingStrategy:
            main(override = args.override,
                folds = folds_to_execute,
                only_sample = args.only_sample,
                include_other = not args.exclude_other,
                training_strategy = strategy)
