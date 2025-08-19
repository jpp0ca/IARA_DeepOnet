import enum
import argparse
import time
import typing
import itertools
import os
import pandas as pd
import numpy as np
import functools

import torch

# Módulos do seu projeto IARA
import iara.utils
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
from iara.ml.models.cnn import CNN
from iara.ml.models.deeponet import DeepONet
from iara.records import CustomCollection, GenericTarget
from shipears_grid import (
    GridSearch as ShipsEarGridSearch,
)  # Importa a classe original


# --- Classe para gerenciar o dataset ShipsEar (reutilizada) ---
class ShipsEarCollectionManager:
    """
    Classe auxiliar para gerenciar o carregamento e a configuração do dataset ShipsEar.

    """

    def __str__(self) -> str:
        return "shipsear"

    def _get_info_filename(self) -> str:
        return os.path.join("./training_scripts/dataset_info/", f"{str(self)}.csv")

    def to_df(self, only_sample: bool = False) -> pd.DataFrame:
        return pd.read_csv(self._get_info_filename(), na_values=[" - "])

    def get_id(self, file: str) -> int:
        return int(file.split("_")[0])

    def classify_row(self, df: pd.DataFrame) -> float:
        classes_by_length = ["B", "C", "A", "D", "E"]
        try:
            target = classes_by_length.index(df["Class"]) - 1
            return 0 if target < 0 else target
        except ValueError:
            return np.nan


# --- Enum para escolher o tipo de Branch Net ---
class BranchType(enum.Enum):
    MLP = 0
    CNN = 1

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

    def get_input_type(self):
        return (
            iara.default.default_window_input()
            if self == BranchType.MLP
            else iara.default.default_image_input()
        )


# --- Classe de Grid Search adaptada para a DeepONet ---
class GridSearchDeepONet:
    def __init__(self, branch_type: BranchType):
        self.branch_type = branch_type
        self.original_grid = (
            ShipsEarGridSearch()
        )  # Instancia a classe do shipears_grid.py

        # Define os cabeçalhos e parâmetros possíveis para cada tipo de branch
        if self.branch_type == BranchType.MLP:
            base_classifier = iara.default.Classifier.MLP
            self.headers = self.original_grid.headers[base_classifier] + [
                "Embedding Dim"
            ]
            self.possible_param = self.original_grid.possible_param[base_classifier] + [
                [32, 64, 128, 256]
            ]
        else:  # CNN
            base_classifier = iara.default.Classifier.CNN
            self.headers = self.original_grid.headers[base_classifier] + [
                "Embedding Dim"
            ]
            self.possible_param = self.original_grid.possible_param[base_classifier] + [
                [32, 64, 128, 256]
            ]

    def get_manager(
        self, config: iara_exp.Config, grids_index: typing.List[int]
    ) -> iara_exp.Manager:

        base_classifier = (
            iara.default.Classifier.MLP
            if self.branch_type == BranchType.MLP
            else iara.default.Classifier.CNN
        )

        grid_search_params = {}
        for i, header in enumerate(self.headers):
            if i in grids_index:
                grid_search_params[i] = list(range(len(self.possible_param[i])))
            else:
                grid_search_params[i] = [0]

        trainers = []
        combinations = list(itertools.product(*grid_search_params.values()))

        activation_dict = {
            "ReLU": torch.nn.ReLU,
            "PReLU": torch.nn.PReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            "Sigmoid": torch.nn.Sigmoid,
            "Linear": None,
        }
        pooling_dict = {"Max": torch.nn.MaxPool2d, "Avg": torch.nn.AvgPool2d}
        norm_dict1d = {
            "Batch": torch.nn.BatchNorm1d,
            "Instance": torch.nn.InstanceNorm1d,
            "None": None,
        }
        norm_dict2d = {
            "Batch": torch.nn.BatchNorm2d,
            "Instance": torch.nn.InstanceNorm2d,
            "None": None,
        }

        print(f"Gerando {len(combinations)} combinações de treinamento...")
        for combination in combinations:
            p = dict(zip(grid_search_params.keys(), combination))

            trainer_id = ""
            for param_idx, value_idx in p.items():
                trainer_id += f"_{param_idx}[{value_idx}]"
            trainer_id = trainer_id[1:]

            embedding_dim = self.possible_param[len(self.headers) - 1][
                p[len(self.headers) - 1]
            ]

            def model_allocator(input_shape, n_targets, params=p):
                if self.branch_type == BranchType.MLP:
                    input_features = functools.reduce(lambda x, y: x * y, input_shape)
                    hidden_channels = self.possible_param[1][params[1]]
                    dropout = self.possible_param[2][params[2]]
                    norm_layer = norm_dict1d[self.possible_param[3][params[3]]]
                    activation_layer = activation_dict[
                        self.possible_param[4][params[4]]
                    ]

                    layers = [
                        torch.nn.Flatten(),
                        torch.nn.Linear(
                            input_features,
                            (
                                hidden_channels[0]
                                if isinstance(hidden_channels, list)
                                else hidden_channels
                            ),
                        ),
                    ]
                    if norm_layer:
                        layers.append(
                            norm_layer(
                                hidden_channels[0]
                                if isinstance(hidden_channels, list)
                                else hidden_channels
                            )
                        )
                    if activation_layer:
                        layers.append(activation_layer())
                    if dropout > 0:
                        layers.append(torch.nn.Dropout(dropout))
                    layers.append(
                        torch.nn.Linear(
                            (
                                hidden_channels[0]
                                if isinstance(hidden_channels, list)
                                else hidden_channels
                            ),
                            embedding_dim,
                        )
                    )
                    branch_net = torch.nn.Sequential(*layers)

                else:  # CNN
                    conv_n_neurons = self.possible_param[1][params[1]]
                    conv_activation = activation_dict[self.possible_param[2][params[2]]]
                    conv_pooling = pooling_dict[self.possible_param[3][params[3]]]
                    conv_pooling_size = self.possible_param[4][params[4]]
                    conv_dropout = self.possible_param[5][params[5]]
                    batch_norm = norm_dict2d[self.possible_param[6][params[6]]]
                    kernel_size = self.possible_param[7][params[7]]

                    base_cnn = CNN(
                        input_shape=input_shape,
                        conv_n_neurons=conv_n_neurons,
                        conv_activation=conv_activation,
                        conv_pooling=conv_pooling,
                        conv_pooling_size=conv_pooling_size,
                        conv_dropout=conv_dropout,
                        batch_norm=batch_norm,
                        kernel_size=kernel_size,
                        classification_n_neurons=128,
                        n_targets=n_targets,
                    )
                    feature_extractor = base_cnn.conv_layers

                    with torch.no_grad():
                        dummy_input = torch.zeros(1, *input_shape)
                        dummy_conv_out = feature_extractor(dummy_input)
                        flattened_size = dummy_conv_out.view(1, -1).shape[1]

                    branch_net = torch.nn.Sequential(
                        feature_extractor,
                        torch.nn.Flatten(),
                        torch.nn.Linear(flattened_size, embedding_dim),
                    )

                return DeepONet(
                    branch_net=branch_net,
                    n_targets=n_targets,
                    embedding_dim=embedding_dim,
                )

            trainer = iara_trn.OptimizerTrainer(
                training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                trainer_id=trainer_id,
                n_targets=config.dataset.target.get_n_targets(),
                batch_size=self.possible_param[0][p[0]],
                n_epochs=200,
                patience=25,
                model_allocator=model_allocator,
                optimizer_allocator=lambda model: torch.optim.Adam(
                    model.parameters(), lr=1e-4
                ),
                loss_allocator=lambda w: torch.nn.CrossEntropyLoss(weight=w),
            )
            trainers.append(trainer)

        return iara_exp.Manager(config, *trainers)


def get_shipsear_config(
    output_base_dir: str, input_type: iara.ml.dataset.InputType
) -> iara.ml.experiment.Config:
    shipsear_manager = ShipsEarCollectionManager()
    collection = CustomCollection(
        collection=shipsear_manager,
        target=GenericTarget(
            n_targets=4, function=shipsear_manager.classify_row, include_others=False
        ),
    )
    data_processor = iara_manager.AudioFileProcessor(
        data_base_dir="/home/joao.poca/Documents/IARA/shipsear_16e3",
        data_processed_base_dir="./data/shipsear_processed",
        normalization=iara_proc.Normalization.NORM_L2,
        analysis=iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts=1024,
        n_overlap=0,
        decimation_rate=1,
        n_mels=256,
        integration_interval=0.512,
        extract_id=shipsear_manager.get_id,
    )
    config = iara_exp.Config(
        name=f"deeponet_{input_type.type_str()}_shipsear",
        dataset=collection,
        dataset_processor=data_processor,
        output_base_dir=output_base_dir,
        input_type=input_type,
    )
    return config


def main(
    branch_type: BranchType,
    grids_index: typing.List[int],
    folds: typing.List[int],
    override: bool,
):
    output_base_dir = (
        f"{iara.default.DEFAULT_DIRECTORIES.training_dir}/deeponet_grid_{branch_type}"
    )
    config = get_shipsear_config(output_base_dir, branch_type.get_input_type())
    grid_search = GridSearchDeepONet(branch_type)
    manager = grid_search.get_manager(config, grids_index)
    print(f"--- Iniciando Grid Search para DeepONet com Branch {branch_type} ---")
    print(f"Número de combinações de hiperparâmetros: {len(manager.trainer_list)}")
    manager.run(folds=folds, override=override)
    print("--- Grid Search Concluído ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa Grid Search para a DeepONet no dataset ShipsEar."
    )
    branch_choices = [str(b) for b in BranchType]
    parser.add_argument(
        "-b",
        "--branch_type",
        type=str,
        choices=branch_choices,
        required=True,
        help="Tipo de Branch Net: mlp ou cnn",
    )

    # Leitura temporária dos argumentos para obter o tipo de branch
    temp_args, _ = parser.parse_known_args()
    branch = BranchType[temp_args.branch_type.upper()]
    grid = GridSearchDeepONet(branch)
    headers = grid.headers
    help_str = f"Escolha os parâmetros do grid para variar (Ex: 0,4-7): {list(enumerate(headers))}"

    parser.add_argument("-g", "--grid", type=str, default="", help=help_str)
    parser.add_argument(
        "-F",
        "--fold",
        type=str,
        default=None,
        help="Especifique as folds (Ex: 0,1,2). Padrão: todas.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Sobrescreve execuções antigas.",
    )

    args = parser.parse_args()

    # --- CORREÇÃO APLICADA AQUI ---
    # Processa o argumento do grid apenas se ele não for uma string vazia.
    if args.grid:
        grids_to_execute = iara.utils.str_to_list(args.grid, [])
    else:
        grids_to_execute = []

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))

    iara.utils.set_seed()
    iara.utils.print_available_device()

    main(
        branch_type=branch,
        grids_index=grids_to_execute,
        folds=folds_to_execute,
        override=args.override,
    )
