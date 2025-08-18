import os
import typing
import torch
import argparse
import pandas as pd
import numpy as np
import functools  # Import para o cálculo do tamanho

# Módulos do seu projeto IARA
import iara.utils
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.records
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc
from iara.ml.models.deeponet import DeepONet
from iara.ml.models.mlp import MLP


# A classe ShipsEarCollectionManager e a função get_shipsear_config permanecem as mesmas
class ShipsEarCollectionManager:
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


def get_shipsear_config(
    output_base_dir: str, input_type: iara.ml.dataset.InputType
) -> iara.ml.experiment.Config:
    shipsear_manager = ShipsEarCollectionManager()
    collection = iara.records.CustomCollection(
        collection=shipsear_manager,
        target=iara.records.GenericTarget(
            n_targets=4, function=shipsear_manager.classify_row, include_others=False
        ),
        only_sample=False,
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
    config = iara.ml.experiment.Config(
        name="deeponet_mlp_shipsear",
        dataset=collection,
        dataset_processor=data_processor,
        output_base_dir=output_base_dir,
        input_type=input_type,
    )
    return config


def main(folds: typing.List[int], override: bool):
    print(
        "--- Configurando experimento DeepONet com Branch MLP no dataset ShipsEar ---"
    )

    output_base_dir = (
        f"{iara.default.DEFAULT_DIRECTORIES.training_dir}/deeponet_mlp_shipsear"
    )
    input_type = iara.default.default_window_input()
    config = get_shipsear_config(output_base_dir, input_type)

    def model_allocator(input_shape: typing.List[int], n_targets: int) -> DeepONet:

        embedding_dim = 128

        # --- CORREÇÃO APLICADA AQUI ---
        # 1. Calcula o número total de features multiplicando todas as dimensões do input.
        #    Isso torna o código robusto para diferentes formatos de "janela".
        #    Ex: (1, 256) -> 256; (256,) -> 256
        input_features = functools.reduce(lambda x, y: x * y, input_shape)

        # 2. Constrói a Branch Net, adicionando uma camada Flatten no início.
        #    Isso garante que a entrada para a primeira camada Linear seja sempre um vetor 1D.
        branch_net_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),  # Garante que a entrada seja um vetor
            torch.nn.Linear(input_features, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, embedding_dim),
        )

        deeponet_model = DeepONet(
            branch_net=branch_net_mlp, n_targets=n_targets, embedding_dim=embedding_dim
        )
        return deeponet_model

    trainer = iara.ml.models.trainer.OptimizerTrainer(
        training_strategy=iara.ml.models.trainer.ModelTrainingStrategy.MULTICLASS,
        trainer_id="deeponet_mlp_v1",
        n_targets=config.dataset.target.get_n_targets(),
        batch_size=1024,
        n_epochs=200,
        patience=25,
        model_allocator=model_allocator,
        optimizer_allocator=lambda model: torch.optim.Adam(
            model.parameters(), weight_decay=1e-3, lr=1e-4
        ),
        loss_allocator=lambda class_weights: torch.nn.CrossEntropyLoss(
            weight=class_weights, reduction="mean"
        ),
    )

    manager = iara.ml.experiment.Manager(config, trainer)
    print("--- Iniciando Treinamento com Branch MLP no ShipsEar ---")
    manager.run(folds=folds, override=override)
    print("--- Treinamento Concluído ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina um modelo DeepONet com Branch MLP no dataset ShipsEar."
    )
    parser.add_argument(
        "-F",
        "--fold",
        type=str,
        default=None,
        help="Especifique as folds a serem executadas. Exemplo: 0,4-7",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Ignora e sobrescreve execuções antigas.",
    )
    args = parser.parse_args()
    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))
    iara.utils.set_seed()
    iara.utils.print_available_device()
    main(folds=folds_to_execute, override=args.override)
