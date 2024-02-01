import argparse
import tqdm

import iara.description
import iara.ml.mlp as iara_model
import iara.trainer as iara_trn
import iara.processing.analysis as iara_proc
import iara.processing.dataset as iara_data_proc


def main(override: bool, only_first_fold: bool, only_sample: bool):

    config_dir = "./results/configs"

    configs = {
        'grid_search': iara.description.DatasetType.OS_SHIP
    }

    # for config_name, data_type in tqdm.tqdm(configs.items(), leave=False, desc="Configs"):
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
        for n_neurons in [4, 16, 64]:
            mlp_trainers.append(iara_trn.NNTrainer(
                                    training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                    trainer_id = f'MLP_{n_neurons}',
                                    n_targets = config.dataset.target.get_n_targets(),
                                    model_allocator=lambda input_shape:
                                                iara_model.MLP(input_shape=input_shape,
                                                            n_neurons=n_neurons,
                                                            n_targets=dataset.target.get_n_targets()),
                                    batch_size = 128,
                                    n_epochs = 64,
                                    patience=8))

        # for n_neurons in [4, 16, 64]:
        #     mlp_trainers.append(iara_trn.NNTrainer(
        #                             training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
        #                             trainer_id = f'MLP_{n_neurons}',
        #                             n_targets = config.dataset.target.get_n_targets(),
        #                             model_allocator=lambda input_shape:
        #                                         iara_model.MLP(input_shape=input_shape,
        #                                                     n_neurons=n_neurons),
        #                             batch_size = 128,
        #                             n_epochs = 64,
        #                             patience=8))

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
