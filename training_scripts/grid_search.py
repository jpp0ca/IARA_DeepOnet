import enum
import argparse
import time
import sys
import typing
import itertools
import tqdm
import os

import torch

import iara.utils
import iara.default as iara_default
import iara.ml.models.mlp as iara_mlp
import iara.ml.models.cnn as iara_cnn
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager

from iara.default import DEFAULT_DIRECTORIES

class Classifier(enum.Enum):
    FOREST = 0
    MLP = 1
    CNN = 2

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

    def get_input_type(self):
        if self == Classifier.CNN:
            return iara_default.default_image_input()

        return iara_default.default_window_input()

class GridSearch():

    def __init__(self) -> None:
        self.print_headers = {
            Classifier.FOREST: ['Estimators', 'Max depth'],
            Classifier.MLP: ['Neurons', 'Activation', 'Weight decay'],
            Classifier.CNN: ['conv_n_neurons', 'classification_n_neurons', 'Activation',
                             'Weight decay', 'conv_pooling', 'kernel', 'dropout']
        }
        self.headers = {
            Classifier.FOREST: ['est', 'depth'],
            Classifier.MLP: ['neurons', 'act', 'weight'],
            Classifier.CNN: ['n_conv', 'n_mlp', 'act',
                             'weight', 'pool', 'kern', 'drop']
        }
        self.complete_grid = {
            Classifier.FOREST: {
                self.headers[Classifier.FOREST][0]: [25, 50, 100, 150, 200, 250],
                self.headers[Classifier.FOREST][1]: [5, 10, 15, 20]
            },
            Classifier.MLP: {
                self.headers[Classifier.MLP][0]: [4, 16, 64, 128, 256, 1024],
                self.headers[Classifier.MLP][1]: ['Tanh', 'ReLU', 'PReLU'],
                self.headers[Classifier.MLP][2]: [0, 1e-3, 1e-5]
            },
            Classifier.CNN: {
                self.headers[Classifier.CNN][0]: ['16, 32',
                                                  '16, 32, 64',
                                                  '16, 32, 64, 128',
                                                  '32, 64, 128, 256'],
                self.headers[Classifier.CNN][1]: [16, 32, 64, 128, 256, 512, 1024],
                self.headers[Classifier.CNN][2]: ['ReLU', 'PReLU', 'LeakyReLU'],
                self.headers[Classifier.CNN][3]: [0, 1e-3, 1e-5],
                self.headers[Classifier.CNN][4]: ['Max', 'Avg'],
                self.headers[Classifier.CNN][5]: [3, 5, 7],
                self.headers[Classifier.CNN][6]: [0.2, 0.4, 0.6]
            }
        }
        self.small_grid = {
            Classifier.FOREST: {
                self.headers[Classifier.FOREST][0]: [200],
                self.headers[Classifier.FOREST][1]: [10]
            },
            Classifier.MLP: {
                self.headers[Classifier.MLP][0]: [128],
                self.headers[Classifier.MLP][1]: ['Tanh'],
                self.headers[Classifier.MLP][2]: [0]
            },
            Classifier.CNN: {
                self.headers[Classifier.CNN][0]: ['16, 32, 64, 128'],
                self.headers[Classifier.CNN][1]: [128],
                self.headers[Classifier.CNN][2]: ['ReLU'],
                self.headers[Classifier.CNN][3]: [0],
                self.headers[Classifier.CNN][4]: ['Avg'],
                self.headers[Classifier.CNN][5]: [3],
                self.headers[Classifier.CNN][6]: [0.4]
            }
        }

    def add_grid_opt(self, arg_parser: argparse.ArgumentParser):

        classifier_choises = [str(c) for c in Classifier]

        arg_parser.add_argument('-c', '--classifier', type=str, choices=classifier_choises,
                            required=True, default='', help='classifier to execute grid')

        c_arg, _ = arg_parser.parse_known_args()

        classifier = Classifier(classifier_choises.index(c_arg.classifier))
        headers = self.print_headers[classifier]
        grid_choices=list(range(len(headers)))
        help_str = 'Choose grid parameters to vary(Example: 0,4-7): ['
        for t in grid_choices:
            help_str = f'{help_str}{t}. {headers[t]}, '
        help_str = f'{help_str[:-2]}]'

        arg_parser.add_argument('-g', '--grid', type=str, default=None, help=help_str)
        arg_parser.add_argument('-G', '--remove_grid', action='store_true', default=False)

        return classifier, grid_choices

    def get_manager(self,
                    config: iara_exp.Config,
                    classifier: Classifier,
                    training_strategy: iara_trn.ModelTrainingStrategy,
                    grids_index: typing.List[int],
                    only_eval: bool) -> typing.Tuple[iara_exp.Manager, typing.Dict]:

        grid_search = {}
        for i, header in enumerate(self.headers[classifier]):
            if i in grids_index or only_eval:
                grid_search[header] = self.complete_grid[classifier][header]
            else:
                grid_search[header] = self.small_grid[classifier][header]

        trainers = []
        param_dict = {}

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

        combinations = list(itertools.product(*grid_search.values()))
        for combination in combinations:
            param_pack = dict(zip(grid_search.keys(), combination))

            trainer_id = ""
            for param, value in param_pack.items():
                trainer_id = f'{trainer_id}_{param}[{value}]'
            trainer_id = trainer_id[1:]
            param_dict[trainer_id] = param_pack

            if classifier == Classifier.FOREST:

                trainers.append(iara_trn.RandomForestTrainer(
                        training_strategy = training_strategy,
                        trainer_id = trainer_id,
                        n_targets = config.dataset.target.get_n_targets(),
                        n_estimators = param_pack[self.headers[classifier][0]],
                        max_depth = param_pack[self.headers[classifier][1]]))

            elif classifier == Classifier.MLP:

                trainers.append(iara_trn.OptimizerTrainer(
                        training_strategy=training_strategy,
                        trainer_id = trainer_id,
                        n_targets = config.dataset.target.get_n_targets(),
                        batch_size = 4*1024,
                        model_allocator = lambda input_shape, n_targets,
                            n_neurons = param_pack[self.headers[classifier][0]],
                            activation = activation_dict[param_pack[self.headers[classifier][1]]]:
                                iara_mlp.MLP(input_shape = input_shape,
                                    n_neurons = n_neurons,
                                    n_targets = n_targets,
                                    activation_hidden_layer = activation),
                        optimizer_allocator=lambda model,
                            weight_decay = param_pack[self.headers[classifier][2]]:
                                torch.optim.Adam(model.parameters(),
                                    weight_decay=weight_decay)))

            elif classifier == Classifier.CNN:

                conv_n_neurons = [int(x.strip()) for x in param_pack[self.headers[classifier][0]].split(',')]

                trainers.append(iara_trn.OptimizerTrainer(
                        training_strategy=training_strategy,
                        trainer_id = trainer_id,
                        n_targets = config.dataset.target.get_n_targets(),
                        batch_size = 128,
                        model_allocator = lambda input_shape, n_targets,
                            conv_neurons = conv_n_neurons,
                            class_neurons = param_pack[self.headers[classifier][1]],
                            activation = activation_dict[param_pack[self.headers[classifier][2]]],
                            pooling = pooling_dict[param_pack[self.headers[classifier][4]]],
                            kernel = param_pack[self.headers[classifier][5]],
                            dropout = param_pack[self.headers[classifier][6]]:
                                iara_cnn.CNN(
                                    input_shape = input_shape,
                                    conv_activation   = activation,
                                    conv_n_neurons = conv_neurons,
                                    conv_pooling = pooling,
                                    kernel_size = kernel,
                                    classification_n_neurons = class_neurons,
                                    n_targets = n_targets,
                                    dropout_prob = dropout),
                        optimizer_allocator=lambda model,
                            weight_decay = param_pack[self.headers[classifier][3]]:
                                torch.optim.Adam(model.parameters(), weight_decay = weight_decay)))

            else:
                raise NotImplementedError(
                        f'GridSearch.get_manager not implemented for {classifier}')


        return iara_exp.Manager(config, *trainers), param_dict

class Feature(enum.Enum):
    MEL = 0
    MEL_GRID = 1
    LOFAR = 2

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

    def get_feature_loop(self, classifiers: Classifier,
                         training_strategy: iara_trn.ModelTrainingStrategy) \
        -> typing.List[typing.Tuple[str, str, iara_manager.AudioFileProcessor]]:

        loop = []
        if self == Feature.MEL:
            loop.append([f'{classifiers}_mel_{str(training_strategy)}',
                         'mel',
                        iara_default.default_iara_mel_audio_processor()])

        elif self == Feature.LOFAR:
            loop.append([f'{classifiers}_lofar_{str(training_strategy)}',
                         'lofar',
                        iara_default.default_iara_lofar_audio_processor()])
        
        elif self == Feature.MEL_GRID:
            for n_mels in [16, 32, 64, 128, 256]:
                loop.append([f'{classifiers}_mel[{n_mels}]_{str(training_strategy)}',
                             f'{n_mels}',
                            iara_default.default_iara_mel_audio_processor(n_mels=n_mels)])

        return loop


def main(classifier: Classifier,
         feature: Feature,
         grids_index: typing.List[int],
         training_strategy: iara_trn.ModelTrainingStrategy,
         folds: typing.List[int],
         only_eval: bool,
         only_sample: bool,
         override: bool):

    grid_str = 'grid_search' if not only_sample else 'grid_search_sample'
    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

    result_grid = {}
    for eval_subset, eval_strategy in itertools.product(iara_trn.Subset, iara_trn.EvalStrategy):
        result_grid[eval_subset, eval_strategy] = iara_metrics.GridCompiler()

    grid_search = GridSearch()
    feature_dict_list = feature.get_feature_loop(classifier, training_strategy)

    for config_name, feature_id, dp in feature_dict_list if len(feature_dict_list) == 1 else \
                tqdm.tqdm(feature_dict_list, leave=False, desc="Features", ncols=120):

        config = iara_exp.Config(
                name = config_name,
                dataset = iara_default.default_collection(only_sample=only_sample),
                dataset_processor = dp,
                output_base_dir = output_base_dir,
                input_type = classifier.get_input_type())

        manager, param_dict = grid_search.get_manager(config = config,
                                          classifier = classifier,
                                          training_strategy = training_strategy,
                                          grids_index = grids_index,
                                          only_eval = only_eval)

        if only_eval:
            result_dict = {}
        else:
            result_dict = manager.run(folds = folds, override = override)

        for (eval_subset, eval_strategy), grid_compiler in result_grid.items():

            if only_eval:
                result_dict[eval_subset, eval_strategy] = manager.compile_existing_results(
                        eval_subset = eval_subset,
                        eval_strategy = eval_strategy,
                        folds = folds)

            for trainer_id, results in result_dict[eval_subset, eval_strategy].items():

                for i_fold, result in enumerate(results):

                    if len(feature_dict_list) == 1:
                        params = param_dict[trainer_id]
                    else:
                        params = params=dict({'Feature': feature_id}, **param_dict[trainer_id])

                    grid_compiler.add(params = params,
                                i_fold=i_fold,
                                target=result['Target'],
                                prediction=result['Prediction'])

    for dataset_id, grid_compiler in result_grid.items():
        print(f'########## {dataset_id} ############')
        print(grid_compiler)

    if only_eval:
        compiled_dir = f'{output_base_dir}/compiled'
        os.makedirs(compiled_dir, exist_ok=True)

        for eval_strategy in iara_trn.EvalStrategy:
            filename = f'{compiled_dir}/{classifier}_{feature}_{training_strategy}_{eval_strategy}'
            result_grid[iara_trn.Subset.TEST, eval_strategy].export(f'{filename}.csv')
            # result_grid[iara_trn.Subset.TEST, eval_strategy].export(f'{filename}.tex')
            result_grid[iara_trn.Subset.TEST, eval_strategy].export(f'{filename}.pkl')

    params, cv = result_grid[iara_trn.Subset.TEST, iara_trn.EvalStrategy.BY_WINDOW].get_best()
    print('########## Best Parameters ############')
    print(params, " --- ", cv)


if __name__ == "__main__":
    start_time = time.time()

    strategy_choises = [str(i) for i in iara_trn.ModelTrainingStrategy]
    feature_choises = [str(i) for i in Feature]
    grid = GridSearch()

    parser = argparse.ArgumentParser(description='RUN GridSearch analysis', add_help=False)
    classifier, grid_choices = grid.add_grid_opt(parser)
    parser.add_argument('-f', '--feature', type=str, choices=feature_choises,
                        required=True, default='', help='feature')
    parser.add_argument('-t','--training_strategy', type=str, choices=strategy_choises,
                        default=None, help='Strategy for training the model')
    parser.add_argument('-F', '--fold', type=str, default=None,
                        help='Specify folds to be executed. Example: 0,4-7')
    parser.add_argument('--only_eval', action='store_true', default=False,
                        help='Not training, only evaluate trained models')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))
    grids_to_execute = iara.utils.str_to_list(args.grid, grid_choices)

    if args.remove_grid:
        grids_to_execute = []

    if not set(grids_to_execute).issubset(set(grid_choices)):
        print('Invalid grid options')
        parser.print_help()
        sys.exit(0)

    strategies = []
    if args.training_strategy is not None:
        index = strategy_choises.index(args.training_strategy)
        strategies.append(iara_trn.ModelTrainingStrategy(index))
    else:
        strategies = iara_trn.ModelTrainingStrategy

    for strategy in strategies:
        main(classifier = classifier,
            feature = Feature(feature_choises.index(args.feature)),
            grids_index = grids_to_execute,
            folds = folds_to_execute,
            training_strategy = strategy,
            only_eval = args.only_eval,
            only_sample = args.only_sample,
            override = args.override)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")
