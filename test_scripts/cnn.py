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
import iara.ml.dataset as iara_dataset
import iara.default as iara_default
import iara.ml.models.mlp as iara_mlp
import iara.ml.models.cnn as iara_cnn
import iara.ml.experiment as iara_exp
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn
import iara.processing.manager as iara_manager
import iara.processing.analysis as iara_proc

from iara.default import DEFAULT_DIRECTORIES

def main(folds: typing.List[int]):

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/tests"
    directories = DEFAULT_DIRECTORIES

    grid = iara_metrics.GridCompiler()

    input_type = iara_dataset.InputType.Image(n_windows=32, overlap=0.5)

    dp = iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.MIN_MAX,
        analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        n_mels=256,
        integration_interval=0.512
    )

    config = iara_exp.Config(
            name = 'cnn',
            dataset = iara_default.default_collection(),
            dataset_processor = dp,
            output_base_dir = output_base_dir,
            input_type = input_type)

    trainers = []
    trainers.append(iara_trn.OptimizerTrainer(
            training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
            trainer_id = 'cnn',
            n_targets = config.dataset.target.get_n_targets(),
            batch_size = 128,
            n_epochs = 512,
            patience = 16,
            model_allocator = lambda input_shape, n_targets:
                    iara_cnn.CNN(
                            input_shape = input_shape,

                            conv_n_neurons = [16, 32, 64, 128],
                            conv_activation = torch.nn.ReLU,
                            conv_pooling = torch.nn.MaxPool2d,
                            conv_dropout = 0.4,
                            batch_norm = True,
                            kernel_size = 5,
                            padding = None,

                            classification_n_neurons = 128,
                            n_targets = n_targets,
                            classification_dropout = 0,
                            classification_norm = False,
                            classification_hidden_activation = torch.nn.ReLU,
                            classification_output_activation = torch.nn.ReLU
                    ),
            optimizer_allocator=lambda model:
                    torch.optim.Adam(
                            model.parameters(),
                            weight_decay = 1e-3,
                            lr = 1e-6),
            loss_allocator = lambda class_weights:
                    torch.nn.CrossEntropyLoss(
                            weight=class_weights,
                            reduction='mean')
            ))

    manager = iara_exp.Manager(config, *trainers)

    result_grid = manager.run(folds = folds, override=True)

    for (eval_subset, _), result_dict in result_grid.items():

        for trainer_id, results in result_dict.items():

            for i_fold, result in enumerate(results):

                grid.add(params = {
                                'trainer_id': trainer_id,
                                'eval_subset': eval_subset,
                            },
                            i_fold=i_fold,
                            target=result['Target'],
                            prediction=result['Prediction'])

    print(grid)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='RUN GridSearch analysis')
    parser.add_argument('-F', '--fold', type=str, default=None,
                        help='Specify folds to be executed. Example: 0,4-7')

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(1)))

    main(folds = folds_to_execute)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")
