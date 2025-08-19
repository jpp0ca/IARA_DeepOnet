import enum
import argparse
import time
import sys
import typing
import itertools
import tqdm
import os
import pandas as pd
import numpy as np

import torch

import iara.utils
import iara.default as iara_default
import iara.ml.models.mlp as iara_mlp
import iara.ml.models.cnn as iara_cnn
import iara.ml.models.deeponet as iara_deeponet
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc

from iara.default import DEFAULT_DIRECTORIES


class ShipsEarCollections():

    def __str__(self) -> str:
        return 'shipsear'

    def _get_info_filename(self) -> str:
        return os.path.join("./training_scripts/dataset_info/", f"{str(self)}.csv")

    def to_df(self, only_sample: bool = False) -> pd.DataFrame:
        df = pd.read_csv(self._get_info_filename(), na_values=[" - "])
        return df

    def get_id(self, file: str) -> int:
        return int(file.split('_')[0])

    def classify_row(self, df: pd.DataFrame) -> float:
        classes_by_length = ['B', 'C', 'A', 'D', 'E']
        try:
            target = (classes_by_length.index(df['Class']) - 1)
            if target < 0:
                return 0
            return target
        except ValueError:
            return np.nan

    def default_config(self,
                             config_name: str,
                             output_base_dir: str,
                             classifier: str):

        data_base_dir = "/home/joao.poca/Documents/IARA/shipsear_16e3"
        data_processed_base_dir = "./data/shipsear_processed"

        collection = iara.records.CustomCollection(
                    collection = self,
                    target = iara.records.GenericTarget(
                        n_targets = 4,
                        function = self.classify_row,
                        include_others = False
                    ),
                    only_sample=False
                )

        dataset_processor = iara_manager.AudioFileProcessor(
                data_base_dir = data_base_dir,
                data_processed_base_dir = data_processed_base_dir,
                normalization = iara_proc.Normalization.NORM_L2,
                analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
                n_pts = 1024,
                n_overlap = 0,
                decimation_rate = 1,
                n_mels=256,
                integration_interval=0.512,
                extract_id = self.get_id
            )
        
        input_type = iara_default.default_image_input() if 'cnn' in classifier else iara_default.default_window_input()

        config = iara_exp.Config(
                        name = config_name,
                        dataset = collection,
                        dataset_processor = dataset_processor,
                        output_base_dir = output_base_dir,
                        input_type = input_type)

        return config

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, class_weight):
        super().__init__()
        self.class_weight = class_weight

    def forward(self, sample, target):
        mse_loss = (sample - target) ** 2
        weighted_loss = mse_loss * self.class_weight[target.long()]
        loss = torch.mean(weighted_loss)
        return loss

class GridSearch():

    def __init__(self) -> None:
        self.headers = {
            'forest': ['Estimators',
                       'Max depth'],
            'mlp': ['Batch',
                    'Neurons',
                    'Dropout',
                    'Normalization',
                    'activation_layer',
                    'activation_output_layer',
                    'weight_decay',
                    'lr',
                    'loss'],
            'cnn': ['Batch size',
                    'conv_n_neurons',
                    'conv_activation',
                    'conv_pooling',
                    'conv_pooling_size',
                    'conv_dropout',
                    'Normalization',
                    'kernel_size',
                    'classification_n_neurons',
                    'classification_dropout',
                    'classification_norm',
                    'classification_hidden_activation',
                    'classification_output_activation',
                    'weight_decay',
                    'lr',
                    'loss'],
            'deeponet_cnn': ['Batch size',
                             'conv_n_neurons',
                             'conv_activation',
                             'conv_pooling',
                             'conv_pooling_size',
                             'conv_dropout',
                             'Normalization',
                             'kernel_size',
                             'embedding_dim',
                             'weight_decay',
                             'lr',
                             'loss'],
            'deeponet_mlp': ['Batch',
                           'Neurons',
                           'Dropout',
                           'Normalization',
                           'activation_layer',
                           'embedding_dim',
                           'weight_decay',
                           'lr',
                           'loss']
        }

        self.possible_param ={
            'forest': [
                    [2, 5, 8, 10, 50, 200],                             #Estimators
                    [5, 10, 20, 30, 50, None, 2, 8]                           #Max depth
            ],
            'mlp': [
                [512, 1024, 4096, 16*1024, 32, 64, 30],                             #Batch
                [
                    32, 64, 128, 256, 512, 1024, 2048,
                    [64, 32], [128, 64], [256, 64], 16, 8, 4, [32, 16], [16, 32],
                ],                                                      #hidden_channels
                [0, 0.2, 0.4, 0.6],                                     #dropout
                ['Batch', 'Instance', 'None'],                          #norm_layer
                ['ReLU', 'PReLU', 'LeakyReLU'],                         #activation_layer
                ['Sigmoid', 'ReLU', 'Linear'],                          #activation_output_layer
                [1e-2, 1e-3, 1e-5, 0],                                  #weight_decay
                [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],                   #lr
                ['CrossEntropyLoss', 'MSELoss', 'WeightedMSELoss']      #loss
            ],
            'cnn': [
                [32, 64, 128, 256, 512],                                #Batch size
                [   [16, 128], [32, 256], [32, 64, 256], [32, 64, 128, 256],
                    [64, 512], [256, 32], [512, 64],
                ],                                                      #conv_n_neurons
                ['ReLU', 'PReLU', 'LeakyReLU'],                         #conv_activation
                ['Avg','Max'],                                          #conv_pooling
                [[2,2], [4,2], [4,4]],                                  #conv_pooling_size
                [0, 0.2, 0.4, 0.6],                                     #conv_dropout
                ['Batch', 'Instance', 'None'],                          #batch_norm
                [3, 5, 7],                                              #kernel_size
                [32, 64, 128, 256, 512, 1024, 2048, [64, 32], [128, 64], [256, 64]], #classification_n_neurons
                [0, 0.2, 0.4, 0.6],                                     #classification_dropout
                ['Batch', 'Instance', 'None'],                          #classification_norm
                ['ReLU', 'PReLU', 'LeakyReLU'],                         #classification_hidden_activation
                ['Sigmoid', 'ReLU', 'Linear'],                          #classification_output_activation
                [1e-2, 1e-3, 1e-5, 0],                                   #weight_decay
                [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],                   #lr
                ['CrossEntropyLoss', 'MSELoss', 'WeightedMSELoss']      #loss
            ],
            'deeponet_cnn': [
                [32, 64, 128, 256, 512],
                [[16, 128], [32, 256], [32, 64, 256], [32, 64, 128, 256], [64, 512], [256, 32], [512, 64]],
                ['ReLU', 'PReLU', 'LeakyReLU'],
                ['Avg', 'Max'],
                [[2, 2], [4, 2], [4, 4]],
                [0, 0.2, 0.4, 0.6],
                ['Batch', 'Instance', 'None'],
                [3, 5, 7],
                [64, 128, 256],
                [1e-2, 1e-3, 1e-5, 0],
                [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                ['CrossEntropyLoss', 'MSELoss', 'WeightedMSELoss']
            ],
            'deeponet_mlp': [
                [512, 1024, 4096, 16*1024, 32, 64, 30],
                [32, 64, 128, 256, 512, 1024, 2048, [64, 32], [128, 64], [256, 64], 16, 8, 4, [32, 16], [16, 32]],
                [0, 0.2, 0.4, 0.6],
                ['Batch', 'Instance', 'None'],
                ['ReLU', 'PReLU', 'LeakyReLU'],
                [64, 128, 256],
                [1e-2, 1e-3, 1e-5, 0],
                [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                ['CrossEntropyLoss', 'MSELoss', 'WeightedMSELoss']
            ]
        }

        self.complete_grid = {
            Feature.MEL: {
                'forest': {
                    self.headers['forest'][0]: [0, 1, 2, 3, 4, 5],
                    self.headers['forest'][1]: [0, 1, 2, 3, 4, 5, 6, 7]
                },
                'mlp': {
                    self.headers['mlp'][0]: [0, 4, 6],
                    self.headers['mlp'][1]: [0, 1, 3, 5, 7, 9, 10, 11, 12, 13],
                    self.headers['mlp'][2]: [0, 1, 2],
                    self.headers['mlp'][3]: [0, 1, 2],
                    self.headers['mlp'][4]: [0, 1, 2],
                    self.headers['mlp'][5]: [0, 1, 2],
                    self.headers['mlp'][6]: [1, 2, 3],
                    self.headers['mlp'][7]: [0, 1, 2, 3],
                    self.headers['mlp'][8]: [0, 1, 2],
                },
                'cnn': {
                    self.headers['cnn'][0]:  [0, 2, 4],
                    self.headers['cnn'][1]:  [0, 3, 4, 5],
                    self.headers['cnn'][2]:  [0, 1, 2],
                    self.headers['cnn'][3]:  [0, 1],
                    self.headers['cnn'][4]:  [0, 1, 2],
                    self.headers['cnn'][5]:  [0, 1, 2, 3],
                    self.headers['cnn'][6]:  [0, 1, 2],
                    self.headers['cnn'][7]:  [0, 1, 2],
                    self.headers['cnn'][8]:  [0, 1, 2, 5, 7],
                    self.headers['cnn'][9]:  [0, 1, 2],
                    self.headers['cnn'][10]: [0, 1, 2],
                    self.headers['cnn'][11]: [0, 1, 2],
                    self.headers['cnn'][12]: [0, 1, 2],
                    self.headers['cnn'][13]: [0, 1, 2, 3],
                    self.headers['cnn'][14]: [1, 2, 3],
                    self.headers['cnn'][15]: [0, 1, 2],
                },
                'deeponet_cnn': {
                    self.headers['deeponet_cnn'][0]: [0, 2, 4],
                    self.headers['deeponet_cnn'][1]: [0, 3, 4, 5],
                    self.headers['deeponet_cnn'][2]: [0, 1, 2],
                    self.headers['deeponet_cnn'][3]: [0, 1],
                    self.headers['deeponet_cnn'][4]: [0, 1, 2],
                    self.headers['deeponet_cnn'][5]: [0, 1, 2, 3],
                    self.headers['deeponet_cnn'][6]: [0, 1, 2],
                    self.headers['deeponet_cnn'][7]: [0, 1, 2],
                    self.headers['deeponet_cnn'][8]: [0, 1, 2],
                    self.headers['deeponet_cnn'][9]: [0, 1, 2, 3],
                    self.headers['deeponet_cnn'][10]: [1, 2, 3],
                    self.headers['deeponet_cnn'][11]: [0, 1, 2],
                },
                'deeponet_mlp': {
                    self.headers['deeponet_mlp'][0]: [0, 4, 6],
                    self.headers['deeponet_mlp'][1]: [0, 1, 3, 5, 7, 9, 10, 11, 12, 13],
                    self.headers['deeponet_mlp'][2]: [0, 1, 2],
                    self.headers['deeponet_mlp'][3]: [0, 1, 2],
                    self.headers['deeponet_mlp'][4]: [0, 1, 2],
                    self.headers['deeponet_mlp'][5]: [0, 1, 2],
                    self.headers['deeponet_mlp'][6]: [1, 2, 3],
                    self.headers['deeponet_mlp'][7]: [0, 1, 2, 3],
                    self.headers['deeponet_mlp'][8]: [0, 1, 2],
                }
            },
            Feature.LOFAR: {
                'forest': {
                    self.headers['forest'][0]: [2, 4, 6],
                    self.headers['forest'][1]: [0, 1, 2, 3]
                },
                'mlp': {
                    self.headers['mlp'][0]: [1, 2, 3],
                    self.headers['mlp'][1]: [1, 3, 5, 7, 9],
                    self.headers['mlp'][2]: [0, 1, 2],
                    self.headers['mlp'][3]: [0, 1, 2],
                    self.headers['mlp'][4]: [0, 1, 2],
                    self.headers['mlp'][5]: [0, 1, 2],
                    self.headers['mlp'][6]: [1, 2, 3],
                    self.headers['mlp'][7]: [1, 2, 3],
                    self.headers['mlp'][8]: [0, 1, 2],
                },
                'cnn': {
                    self.headers['cnn'][0]:  [0, 1, 2],
                    self.headers['cnn'][1]:  [0, 1, 2],
                    self.headers['cnn'][2]:  [0, 1, 2],
                    self.headers['cnn'][3]:  [0, 1],
                    self.headers['cnn'][4]:  [0, 1, 2],
                    self.headers['cnn'][5]:  [1, 3],
                    self.headers['cnn'][6]:  [0, 1, 2],
                    self.headers['cnn'][7]:  [0, 1, 2],
                    self.headers['cnn'][8]:  [2, 5, 7, 9],
                    self.headers['cnn'][9]:  [0, 1, 2],
                    self.headers['cnn'][10]: [0, 1, 2],
                    self.headers['cnn'][11]: [0, 1, 2],
                    self.headers['cnn'][12]: [0, 1, 2],
                    self.headers['cnn'][13]: [0, 1, 2, 3],
                    self.headers['cnn'][14]: [1, 2],
                    self.headers['cnn'][15]: [0, 1, 2],
                },
                'deeponet_cnn': {
                    self.headers['deeponet_cnn'][0]: [0, 1, 2],
                    self.headers['deeponet_cnn'][1]: [0, 1, 2],
                    self.headers['deeponet_cnn'][2]: [0, 1, 2],
                    self.headers['deeponet_cnn'][3]: [0, 1],
                    self.headers['deeponet_cnn'][4]: [0, 1, 2],
                    self.headers['deeponet_cnn'][5]: [1, 3],
                    self.headers['deeponet_cnn'][6]: [0, 1, 2],
                    self.headers['deeponet_cnn'][7]: [0, 1, 2],
                    self.headers['deeponet_cnn'][8]: [0, 1, 2],
                    self.headers['deeponet_cnn'][9]: [0, 1, 2, 3],
                    self.headers['deeponet_cnn'][10]: [1, 2],
                    self.headers['deeponet_cnn'][11]: [0, 1, 2],
                },
                'deeponet_mlp': {
                    self.headers['deeponet_mlp'][0]: [1, 2, 3],
                    self.headers['deeponet_mlp'][1]: [1, 3, 5, 7, 9],
                    self.headers['deeponet_mlp'][2]: [0, 1, 2],
                    self.headers['deeponet_mlp'][3]: [0, 1, 2],
                    self.headers['deeponet_mlp'][4]: [0, 1, 2],
                    self.headers['deeponet_mlp'][5]: [0, 1, 2],
                    self.headers['deeponet_mlp'][6]: [1, 2, 3],
                    self.headers['deeponet_mlp'][7]: [1, 2, 3],
                    self.headers['deeponet_mlp'][8]: [0, 1, 2]
                }
            }
        }

        self.small_grid = {
            Feature.MEL: {
                'forest': {
                    self.headers['forest'][0]: [2],
                    self.headers['forest'][1]: [0]
                },
                'mlp': {
                    self.headers['mlp'][0]: [6],
                    self.headers['mlp'][1]: [1],
                    self.headers['mlp'][2]: [0],
                    self.headers['mlp'][3]: [0],
                    self.headers['mlp'][4]: [1],
                    self.headers['mlp'][5]: [0],
                    self.headers['mlp'][6]: [3],
                    self.headers['mlp'][7]: [1],
                    self.headers['mlp'][8]: [0],
                },
                'cnn': {
                    self.headers['cnn'][0]:  [0],
                    self.headers['cnn'][1]:  [5],
                    self.headers['cnn'][2]:  [0],
                    self.headers['cnn'][3]:  [1],
                    self.headers['cnn'][4]:  [0],
                    self.headers['cnn'][5]:  [0],
                    self.headers['cnn'][6]:  [0],
                    self.headers['cnn'][7]:  [2],
                    self.headers['cnn'][8]:  [7],
                    self.headers['cnn'][9]:  [0],
                    self.headers['cnn'][10]: [2],
                    self.headers['cnn'][11]: [1],
                    self.headers['cnn'][12]: [0],
                    self.headers['cnn'][13]: [1],
                    self.headers['cnn'][14]: [1],
                    self.headers['cnn'][15]: [0],
                },
                'deeponet_cnn': {
                    self.headers['deeponet_cnn'][0]: [0],
                    self.headers['deeponet_cnn'][1]: [5],
                    self.headers['deeponet_cnn'][2]: [0],
                    self.headers['deeponet_cnn'][3]: [1],
                    self.headers['deeponet_cnn'][4]: [0],
                    self.headers['deeponet_cnn'][5]: [0],
                    self.headers['deeponet_cnn'][6]: [0],
                    self.headers['deeponet_cnn'][7]: [2],
                    self.headers['deeponet_cnn'][8]: [1],
                    self.headers['deeponet_cnn'][9]: [1],
                    self.headers['deeponet_cnn'][10]: [1],
                    self.headers['deeponet_cnn'][11]: [0],
                },
                'deeponet_mlp': {
                    self.headers['deeponet_mlp'][0]: [6],
                    self.headers['deeponet_mlp'][1]: [1],
                    self.headers['deeponet_mlp'][2]: [0],
                    self.headers['deeponet_mlp'][3]: [0],
                    self.headers['deeponet_mlp'][4]: [1],
                    self.headers['deeponet_mlp'][5]: [1],
                    self.headers['deeponet_mlp'][6]: [3],
                    self.headers['deeponet_mlp'][7]: [1],
                    self.headers['deeponet_mlp'][8]: [0]
                }
            },
            Feature.LOFAR: {
                'forest': {
                    self.headers['forest'][0]: [4],
                    self.headers['forest'][1]: [2]
                },
                'mlp': {
                    self.headers['mlp'][0]: [2],
                    self.headers['mlp'][1]: [3],
                    self.headers['mlp'][2]: [2],
                    self.headers['mlp'][3]: [2],
                    self.headers['mlp'][4]: [0],
                    self.headers['mlp'][5]: [0],
                    self.headers['mlp'][6]: [1],
                    self.headers['mlp'][7]: [1],
                    self.headers['mlp'][8]: [0],
                },
                'cnn': {
                    self.headers['cnn'][0]:  [2],
                    self.headers['cnn'][1]:  [0],
                    self.headers['cnn'][2]:  [0],
                    self.headers['cnn'][3]:  [0],
                    self.headers['cnn'][4]:  [0],
                    self.headers['cnn'][5]:  [2],
                    self.headers['cnn'][6]:  [0],
                    self.headers['cnn'][7]:  [1],
                    self.headers['cnn'][8]:  [2],
                    self.headers['cnn'][9]:  [2],
                    self.headers['cnn'][10]: [2],
                    self.headers['cnn'][11]: [0],
                    self.headers['cnn'][12]: [0],
                    self.headers['cnn'][13]: [1],
                    self.headers['cnn'][14]: [3],
                    self.headers['cnn'][15]: [0],
                },
                'deeponet_cnn': {
                    self.headers['deeponet_cnn'][0]: [2],
                    self.headers['deeponet_cnn'][1]: [0],
                    self.headers['deeponet_cnn'][2]: [0],
                    self.headers['deeponet_cnn'][3]: [0],
                    self.headers['deeponet_cnn'][4]: [0],
                    self.headers['deeponet_cnn'][5]: [2],
                    self.headers['deeponet_cnn'][6]: [0],
                    self.headers['deeponet_cnn'][7]: [1],
                    self.headers['deeponet_cnn'][8]: [1],
                    self.headers['deeponet_cnn'][9]: [1],
                    self.headers['deeponet_cnn'][10]: [3],
                    self.headers['deeponet_cnn'][11]: [0],
                },
                'deeponet_mlp': {
                    self.headers['deeponet_mlp'][0]: [2],
                    self.headers['deeponet_mlp'][1]: [3],
                    self.headers['deeponet_mlp'][2]: [2],
                    self.headers['deeponet_mlp'][3]: [2],
                    self.headers['deeponet_mlp'][4]: [0],
                    self.headers['deeponet_mlp'][5]: [1],
                    self.headers['deeponet_mlp'][6]: [1],
                    self.headers['deeponet_mlp'][7]: [1],
                    self.headers['deeponet_mlp'][8]: [0]
                }
            }
        }

    def add_grid_opt(self, arg_parser: argparse.ArgumentParser):
        
        classifier_choises = list(self.headers.keys())

        arg_parser.add_argument('-c', '--classifier', type=str, choices=classifier_choises,
                            required=True, default='', help='classifier to execute grid')

        c_arg, _ = arg_parser.parse_known_args()
        classifier_str = c_arg.classifier
        
        headers = self.headers[classifier_str]
        grid_choices=list(range(len(headers)))
        help_str = 'Choose grid parameters to vary(Example: 0,4-7): ['
        for t in grid_choices:
            help_str = f'{help_str}{t}. {headers[t]}, '
        help_str = f'{help_str[:-2]}]'

        arg_parser.add_argument('-g', '--grid', type=str, default=None, help=help_str)
        arg_parser.add_argument('-G', '--remove_grid', action='store_true', default=False)

        return classifier_str, grid_choices

    def get_manager(self,
                    config: iara_exp.Config,
                    classifier: str,
                    feature,
                    training_strategy: iara_trn.ModelTrainingStrategy,
                    grids_index: typing.List[int],
                    only_eval: bool) -> typing.Tuple[iara_exp.Manager, typing.Dict]:

        feature = Feature.MEL if feature == Feature.MEL_GRID else feature

        grid_search = {}
        for i, header in enumerate(self.headers[classifier]):
            if i in grids_index:
                grid_search[i] = self.complete_grid[feature][classifier][header]
            else:
                grid_search[i] = self.small_grid[feature][classifier][header]

        trainers = []
        param_dict = {}

        activation_dict = {
                'Tanh': torch.nn.Tanh,
                'ReLU': torch.nn.ReLU,
                'PReLU': torch.nn.PReLU,
                'LeakyReLU': torch.nn.LeakyReLU,
                'Sigmoid': torch.nn.Sigmoid,
                'Linear': None
        }

        pooling_dict = {
                'Max': torch.nn.MaxPool2d,
                'Avg': torch.nn.AvgPool2d
        }

        norm_dict1d = {
                'Batch': torch.nn.BatchNorm1d,
                'Instance': torch.nn.InstanceNorm1d,
                'None': None
        }

        norm_dict2d = {
                'Batch': torch.nn.BatchNorm2d,
                'Instance': torch.nn.InstanceNorm2d,
                'None': None
        }

        def loss_allocator(class_weights, loss):
            if loss == 'CrossEntropyLoss':
                return torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            elif loss == 'MSELoss':
                return torch.nn.MSELoss()
            elif loss == 'WeightedMSELoss':
                return WeightedMSELoss(class_weights)


        combinations = list(itertools.product(*grid_search.values()))
        for combination in combinations:
            param_pack = dict(zip(grid_search.keys(), combination))

            trainer_id = ""
            for param, value in param_pack.items():
                trainer_id = f'{trainer_id}_{param}[{value}]'
            trainer_id = trainer_id[1:]

            out_param = {}
            for key, value in param_pack.items():
                out_param[self.headers[classifier][key]] = self.possible_param[classifier][key][value]
            param_dict[trainer_id] = out_param


            if classifier == 'forest':

                trainers.append(iara_trn.RandomForestTrainer(
                        training_strategy = training_strategy,
                        trainer_id = trainer_id,
                        n_targets = config.dataset.target.get_n_targets(),
                        n_estimators = self.possible_param[classifier][0][param_pack[0]],
                        max_depth = self.possible_param[classifier][1][param_pack[1]]))


            elif classifier == 'mlp':

                trainers.append(iara_trn.OptimizerTrainer(
                        training_strategy=training_strategy,
                        trainer_id = trainer_id,
                        n_targets = config.dataset.target.get_n_targets(),
                        batch_size = self.possible_param[classifier][0][param_pack[0]],
                        n_epochs = 100,
                        patience = 8,
                        model_allocator = lambda input_shape, n_targets,
                            hidden_channels = self.possible_param[classifier][1][param_pack[1]],
                            dropout = self.possible_param[classifier][2][param_pack[2]],
                            norm_layer = norm_dict1d[self.possible_param[classifier][3][param_pack[3]]],
                            activation_layer = activation_dict[self.possible_param[classifier][4][param_pack[4]]],
                            activation_output_layer = activation_dict[self.possible_param[classifier][5][param_pack[5]]]:

                                iara_mlp.MLP(input_shape = input_shape,
                                    hidden_channels = hidden_channels,
                                    n_targets = n_targets,
                                    dropout = dropout,
                                    norm_layer = norm_layer,
                                    activation_layer = activation_layer,
                                    activation_output_layer = activation_output_layer),

                        optimizer_allocator=lambda model,
                            weight_decay = self.possible_param[classifier][6][param_pack[6]],
                            lr = self.possible_param[classifier][7][param_pack[7]]:
                                torch.optim.Adam(model.parameters(), weight_decay = weight_decay, lr = lr),
                        loss_allocator = lambda class_weights,
                            loss = self.possible_param[classifier][8][param_pack[8]]:
                                loss_allocator(class_weights, loss)
                ))


            elif classifier == 'cnn':

                trainers.append(iara_trn.OptimizerTrainer(
                        training_strategy=training_strategy,
                        trainer_id = trainer_id,
                        n_targets = config.dataset.target.get_n_targets(),
                        batch_size = self.possible_param[classifier][0][param_pack[0]],
                        model_allocator = lambda input_shape, n_targets,
                            conv_n_neurons = self.possible_param[classifier][1][param_pack[1]],
                            conv_activation = activation_dict[self.possible_param[classifier][2][param_pack[2]]],
                            conv_pooling = pooling_dict[self.possible_param[classifier][3][param_pack[3]]],
                            conv_pooling_size = self.possible_param[classifier][4][param_pack[4]],
                            conv_dropout = self.possible_param[classifier][5][param_pack[5]],
                            batch_norm = norm_dict2d[self.possible_param[classifier][6][param_pack[6]]],
                            kernel_size = self.possible_param[classifier][7][param_pack[7]],

                            classification_n_neurons = self.possible_param[classifier][8][param_pack[8]],
                            classification_dropout = self.possible_param[classifier][9][param_pack[9]],
                            classification_norm = norm_dict1d[self.possible_param[classifier][10][param_pack[10]]],
                            classification_hidden_activation = activation_dict[self.possible_param[classifier][11][param_pack[11]]],
                            classification_output_activation = activation_dict[self.possible_param[classifier][12][param_pack[12]]]:

                                iara_cnn.CNN(
                                        input_shape = input_shape,
 
                                        conv_n_neurons = conv_n_neurons,
                                        conv_activation = conv_activation,
                                        conv_pooling = conv_pooling,
                                        conv_pooling_size = conv_pooling_size,
                                        conv_dropout = conv_dropout,
                                        batch_norm = batch_norm,
                                        kernel_size = kernel_size,

                                        classification_n_neurons = classification_n_neurons,
                                        n_targets = n_targets,
                                        classification_dropout = classification_dropout,
                                        classification_norm = classification_norm,
                                        classification_hidden_activation = classification_hidden_activation,
                                        classification_output_activation = classification_output_activation,
                                ),
                        optimizer_allocator=lambda model,
                            weight_decay = self.possible_param[classifier][13][param_pack[13]],
                            lr = self.possible_param[classifier][14][param_pack[14]]:
                                torch.optim.Adam(model.parameters(), weight_decay = weight_decay, lr = lr),
                        loss_allocator = lambda class_weights,
                            loss = self.possible_param[classifier][15][param_pack[15]]:
                                loss_allocator(class_weights, loss)
                        ))
            
            elif classifier == 'deeponet_cnn':
                trainers.append(iara_trn.OptimizerTrainer(
                    training_strategy=training_strategy,
                    trainer_id=trainer_id,
                    n_targets=config.dataset.target.get_n_targets(),
                    batch_size=self.possible_param[classifier][0][param_pack[0]],
                    model_allocator=lambda input_shape, n_targets,
                                           conv_n_neurons=self.possible_param[classifier][1][param_pack[1]],
                                           conv_activation=activation_dict[self.possible_param[classifier][2][param_pack[2]]],
                                           conv_pooling=pooling_dict[self.possible_param[classifier][3][param_pack[3]]],
                                           conv_pooling_size=self.possible_param[classifier][4][param_pack[4]],
                                           conv_dropout=self.possible_param[classifier][5][param_pack[5]],
                                           batch_norm=norm_dict2d[self.possible_param[classifier][6][param_pack[6]]],
                                           kernel_size=self.possible_param[classifier][7][param_pack[7]],
                                           embedding_dim=self.possible_param[classifier][8][param_pack[8]]:
                        iara_deeponet.DeepONet(
                            branch_net=iara_cnn.CNN(
                                input_shape=input_shape,
                                conv_n_neurons=conv_n_neurons,
                                conv_activation=conv_activation,
                                conv_pooling=conv_pooling,
                                conv_pooling_size=conv_pooling_size,
                                conv_dropout=conv_dropout,
                                batch_norm=batch_norm,
                                kernel_size=kernel_size,
                                classification_n_neurons=[],
                                n_targets=0,
                                classification_output_activation=None
                            ).conv_layers,
                            input_shape=input_shape,
                            n_targets=n_targets,
                            embedding_dim=embedding_dim,
                        ),
                    optimizer_allocator=lambda model,
                                             weight_decay=self.possible_param[classifier][9][param_pack[9]],
                                             lr=self.possible_param[classifier][10][param_pack[10]]:
                    torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr),
                    loss_allocator=lambda class_weights,
                                        loss=self.possible_param[classifier][11][param_pack[11]]:
                    loss_allocator(class_weights, loss)
                ))
            
            elif classifier == 'deeponet_mlp':
                trainers.append(iara_trn.OptimizerTrainer(
                    training_strategy=training_strategy,
                    trainer_id=trainer_id,
                    n_targets=config.dataset.target.get_n_targets(),
                    batch_size=self.possible_param[classifier][0][param_pack[0]],
                    n_epochs=100,
                    patience=8,
                    model_allocator=lambda input_shape, n_targets,
                                           hidden_channels=self.possible_param[classifier][1][param_pack[1]],
                                           dropout=self.possible_param[classifier][2][param_pack[2]],
                                           norm_layer=norm_dict1d[self.possible_param[classifier][3][param_pack[3]]],
                                           activation_layer=activation_dict[self.possible_param[classifier][4][param_pack[4]]],
                                           embedding_dim=self.possible_param[classifier][5][param_pack[5]]:
                        iara_deeponet.DeepONet(
                            branch_net=iara_mlp.MLP(
                                input_shape=input_shape,
                                hidden_channels=hidden_channels,
                                n_targets=embedding_dim,
                                dropout=dropout,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                activation_output_layer=None,
                            ),
                            input_shape=input_shape,
                            n_targets=n_targets,
                            embedding_dim=embedding_dim,
                        ),
                    optimizer_allocator=lambda model,
                                             weight_decay=self.possible_param[classifier][6][param_pack[6]],
                                             lr=self.possible_param[classifier][7][param_pack[7]]:
                    torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr),
                    loss_allocator=lambda class_weights,
                                        loss=self.possible_param[classifier][8][param_pack[8]]:
                    loss_allocator(class_weights, loss)
                ))


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

    def get_feature_loop(self, classifiers: str,
                         training_strategy: iara_trn.ModelTrainingStrategy) \
        -> typing.List[typing.Tuple[str, str]]:

        loop = []
        if self == Feature.MEL:
            loop.append([f'{classifiers}_mel_{str(training_strategy)}',
                         'mel'])

        elif self == Feature.LOFAR:
            loop.append([f'{classifiers}_lofar_{str(training_strategy)}',
                         'lofar'])
        
        elif self == Feature.MEL_GRID:
            for n_mels in [16, 32, 64, 128, 256]:
                loop.append([f'{classifiers}_mel[{n_mels}]_{str(training_strategy)}',
                             f'{n_mels}'])

        return loop


def main(classifier: str,
         feature: Feature,
         grids_index: typing.List[int],
         training_strategy: iara_trn.ModelTrainingStrategy,
         folds: typing.List[int],
         only_eval: bool,
         only_eval_completed: bool,
         override: bool):

    if only_eval_completed:
        only_eval = True

    grid_str = 'shipsear_grid'
    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

    result_grid = {}
    for eval_subset, eval_strategy in itertools.product(iara_trn.Subset, iara_trn.EvalStrategy):
        result_grid[eval_subset, eval_strategy] = iara_metrics.GridCompiler()

    grid_search = GridSearch()
    feature_dict_list = feature.get_feature_loop(classifier, training_strategy)

    for config_name, feature_id in feature_dict_list if len(feature_dict_list) == 1 else \
                tqdm.tqdm(feature_dict_list, leave=False, desc="Features", ncols=120):

        collection = ShipsEarCollections()
        config = collection.default_config(
                             config_name = config_name,
                             output_base_dir = output_base_dir,
                             classifier =classifier)

        manager, param_dict = grid_search.get_manager(config = config,
                                          classifier = classifier,
                                          feature=feature,
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
                        only_eval_completed = only_eval_completed,
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

    for (eval_subset, eval_strategy), grid_compiler in result_grid.items():
        if eval_subset == iara_trn.Subset.ALL:
            continue
        print(f'########## {eval_subset} - {eval_strategy} ############')
        print(grid_compiler)

    if only_eval:
        compiled_dir = f'{output_base_dir}/compiled'
        os.makedirs(compiled_dir, exist_ok=True)

        for eval_strategy in iara_trn.EvalStrategy:
            filename = f'{compiled_dir}/{classifier}_{feature}_{training_strategy}_{eval_strategy}'
            result_grid[iara_trn.Subset.TEST, eval_strategy].export(f'{filename}.csv')
            result_grid[iara_trn.Subset.TEST, eval_strategy].export(f'{filename}.pkl')

    params, cv = result_grid[iara_trn.Subset.TEST, iara_trn.EvalStrategy.BY_AUDIO].get_best()
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
    parser.add_argument('--only_eval_completed', action='store_true', default=False,
                        help='Not training, only evaluate trained models')
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
        strategies = [iara_trn.ModelTrainingStrategy.MULTICLASS]

    for strategy in strategies:
        main(classifier = classifier,
            feature = Feature(feature_choises.index(args.feature)),
            grids_index = grids_to_execute,
            folds = folds_to_execute,
            training_strategy = strategy,
            only_eval = args.only_eval,
            only_eval_completed = args.only_eval_completed,
            override = args.override)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")