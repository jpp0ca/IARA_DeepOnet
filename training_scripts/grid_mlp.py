"""
Grid Search Application

This script performs an initial grid search to choose the MLP (Multi-Layer Perceptron) model
configuration for all training in the article. It evaluates different configurations using
cross-validation and selects the one with the best performance based on predefined evaluation
metrics.

The chosen configuration will then be used for further training and analysis in the article.
"""
import argparse

import torch

import iara.records
import iara.ml.models.mlp as iara_mlp
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager


def main(override: bool,
        only_first_fold: bool,
        only_sample: bool,
        include_other: bool,
        training_strategy: iara_trn.ModelTrainingStrategy):
    """Grid search main function"""

    if only_sample:
        config_dir = "./results/configs/sample/grid_search"
    else:
        config_dir = "./results/configs/grid_search"

    configs = {
        'mlp': iara.records.Collection.OS_SHIP
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

            dp = iara_manager.AudioFileProcessor(
                data_base_dir = "./data/raw_dataset",
                data_processed_base_dir = "./data/processed",
                normalization = iara_proc.Normalization.NORM_L2,
                analysis = iara_proc.SpectralAnalysis.LOFAR,
                n_pts = 1024,
                n_overlap = 0,
                decimation_rate = 3,
                frequency_limit=5e3,
                integration_overlap=0,
                integration_interval=1.024
            )

            config = iara_exp.Config(
                            name = config_name,
                            dataset = custom_collection,
                            dataset_processor = dp,
                            output_base_dir = "./results/trainings",
                            n_folds=10 if not only_sample else 3)

            config.save(config_dir)

        mlp_trainers = []

        activation_dict = {
                'ReLU': torch.nn.ReLU(),
                'Tanh': torch.nn.Tanh(),
                'PReLU': torch.nn.PReLU()
        }

        for n_neurons in [4, 16, 64, 256]:
            for activation_id, activation in activation_dict.items():

                mlp_trainers.append(iara_trn.OptimizerTrainer(
                        training_strategy=training_strategy,
                        trainer_id = f'MLP_{n_neurons}_{activation_id}',
                        n_targets = config.dataset.target.get_n_targets(),
                        model_allocator=lambda input_shape, n_targets,
                            n_neurons=n_neurons, activation=activation:
                                iara_mlp.MLP(input_shape=input_shape,
                                    n_neurons=n_neurons,
                                    n_targets=n_targets,
                                    activation_hidden_layer=activation),
                        optimizer_allocator=lambda model:
                            torch.optim.Adam(model.parameters(), lr=5e-5),
                        batch_size = 128,
                        n_epochs = 256,
                        patience=8))

        trainer = iara_exp.Manager(config, *mlp_trainers)

        trainer.run(only_first_fold = only_first_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN MLP grid search analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_first_fold', action='store_true', default=False,
                        help='Execute only first fold. For inspection purpose')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--exclude_other', action='store_true', default=False,
                        help='Include records besides than [Cargo, Tanker, Tug] in training.')
    parser.add_argument('--training_strategy', type=iara_trn.ModelTrainingStrategy,
                        choices=iara_trn.ModelTrainingStrategy,
                        default=None, help='Strategy for training the model')

    args = parser.parse_args()
    if args.training_strategy is not None:
        main(override = args.override,
            only_first_fold = args.only_first_fold,
            only_sample = args.only_sample,
            include_other = not args.exclude_other,
            training_strategy = args.training_strategy)
    else:
        for strategy in iara_trn.ModelTrainingStrategy:
            main(override = args.override,
                only_first_fold = args.only_first_fold,
                only_sample = args.only_sample,
                include_other = not args.exclude_other,
                training_strategy = strategy)
