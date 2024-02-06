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

import iara.description
import iara.ml.mlp as iara_model
import iara.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc


def main(override: bool, only_first_fold: bool, only_sample: bool):
    """Grid search main function"""

    config_dir = "./results/configs"

    configs = {
        'grid_search': iara.description.DatasetType.OS_SHIP
    }

    for config_name, data_type in configs.items():

        config = False
        if not override:
            try:
                config = iara_trn.TrainingConfig.load(config_dir, config_name)

            except FileNotFoundError:
                pass

        if not config:
            dataset = iara.description.CustomDataset(
                            dataset_type = data_type,
                            target = iara.description.DatasetTarget(
                                column = 'TYPE',
                                values = ['Cargo', 'Tanker', 'Tug'],
                                include_others = True
                            ),
                            only_sample=only_sample
                        )

            dp = iara_data_proc.DatasetProcessor(
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

            config = iara_trn.TrainingConfig(
                            name = config_name,
                            dataset = dataset,
                            dataset_processor = dp,
                            output_base_dir = "./results/trainings",
                            n_folds=5 if not only_sample else 3)

            config.save(config_dir)

        mlp_trainers = []

        mlp_trainers.append(iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                trainer_id = 'MLP_4',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                    iara_model.MLP(input_shape=input_shape,
                                        n_neurons=4,
                                        n_targets=n_targets),
                                optimizer_allocator=lambda model:
                                    torch.optim.Adam(model.parameters(), lr=5e-5),
                                batch_size = 128,
                                n_epochs = 256,
                                patience=8))

        mlp_trainers.append(iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                trainer_id = 'MLP_16',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                    iara_model.MLP(input_shape=input_shape,
                                        n_neurons=16,
                                        n_targets=n_targets),
                                optimizer_allocator=lambda model:
                                    torch.optim.Adam(model.parameters(), lr=5e-5),
                                batch_size = 128,
                                n_epochs = 256,
                                patience=8))

        mlp_trainers.append(iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                trainer_id = 'MLP_64',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                    iara_model.MLP(input_shape=input_shape,
                                        n_neurons=64,
                                        n_targets=n_targets),
                                optimizer_allocator=lambda model:
                                    torch.optim.Adam(model.parameters(), lr=5e-5),
                                batch_size = 128,
                                n_epochs = 256,
                                patience=8))

        mlp_trainers.append(iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                trainer_id = 'MLP_4',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                    iara_model.MLP(input_shape=input_shape,
                                        n_neurons=4),
                                optimizer_allocator=lambda model:
                                    torch.optim.Adam(model.parameters(), lr=5e-5),
                                batch_size = 128,
                                n_epochs = 256,
                                patience=8))

        mlp_trainers.append(iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                trainer_id = 'MLP_16',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                    iara_model.MLP(input_shape=input_shape,
                                        n_neurons=16),
                                optimizer_allocator=lambda model:
                                    torch.optim.Adam(model.parameters(), lr=5e-5),
                                batch_size = 128,
                                n_epochs = 256,
                                patience=8))

        mlp_trainers.append(iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                trainer_id = 'MLP_64',
                                n_targets = config.dataset.target.get_n_targets(),
                                model_allocator=lambda input_shape, n_targets:
                                    iara_model.MLP(input_shape=input_shape,
                                        n_neurons=64),
                                optimizer_allocator=lambda model:
                                    torch.optim.Adam(model.parameters(), lr=5e-5),
                                batch_size = 128,
                                n_epochs = 256,
                                patience=8))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                    trainer_id = 'Forest_25_None',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=25,
                                    max_depth=None))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                    trainer_id = 'Forest_100_None',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=100,
                                    max_depth=None))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                    trainer_id = 'Forest_250_None',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=250,
                                    max_depth=None))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_25_None',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=25,
                                    max_depth=None))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_100_None',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=100,
                                    max_depth=None))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_250_None',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=250,
                                    max_depth=None))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_25_5',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=25,
                                    max_depth=5))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_100_5',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=100,
                                    max_depth=5))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_250_5',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=250,
                                    max_depth=5))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_25_20',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=25,
                                    max_depth=20))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_100_20',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=100,
                                    max_depth=20))

        mlp_trainers.append(iara_trn.ForestTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = 'Forest_250_20',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    n_estimators=250,
                                    max_depth=20))



        trainer = iara_trn.Trainer(config=config, trainer_list=mlp_trainers)

        trainer.run(only_first_fold = only_first_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN grid search analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_first_fold', action='store_true', default=False,
                        help='Execute only first fold. For inspection purpose')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')

    args = parser.parse_args()
    main(args.override, args.only_first_fold, args.only_sample)
