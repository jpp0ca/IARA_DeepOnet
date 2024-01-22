"""
Training Configuration Test Program

This script generates a sample training configuration for testing functionality.
In the future, this test script will be part of an application designed to create training
    configurations as described in the associated article.
"""
import iara.trainer
import iara.description
import iara.processing.analysis as iara_proc


def main(override: bool = True):
    """Main function for the test Training Configuration."""

    config_dir = "./results/configs"
    config_name = "test_training"

    config = False
    if not override:
        try:
            config = iara.trainer.TrainingConfig.load(config_dir, config_name)

        except FileNotFoundError:
            pass

    if not config:
        # dataset = iara.trainer.TrainingDataset(
        #                 dataset_base_dir = "./data/raw_dataset",
        #                 dataset = iara.description.Subdataset.OS_SHIP,
        #                 target = iara.trainer.DatasetSelection(
        #                     column = 'TYPE',
        #                     values = ['Cargo', 'Tanker', 'Tug', 'Passenger'],
        #                     include_others = False
        #                 )
        #             )
        dataset = iara.trainer.TrainingDataset(
                        dataset_base_dir = "./data/raw_dataset",
                        dataset = iara.description.Subdataset.OS_CPA_IN,
                        target = iara.trainer.DatasetSelection(
                            column = 'DETAILED TYPE',
                            values = ['Bulk Carrier', 'Container Ship'],
                            include_others = True
                        ),
                        filters = [
                            iara.trainer.DatasetSelection(
                                column = 'Rain state',
                                values = ['No rain']),
                            iara.trainer.DatasetSelection(
                                column = 'TYPE',
                                values = ['Cargo']),
                        ]
                    )

        config = iara.trainer.TrainingConfig(
                        name = config_name,
                        dataset = dataset,
                        analysis = iara_proc.Analysis.LOFAR,
                        analysis_parameters= {
                            'n_pts': 1024,
                            'n_overlap': 0,
                            'decimation_rate': 1
                        },
                        output_base_dir = "./results",
                        training_type= iara.trainer.TrainingType.WINDOW)

        config.save(config_dir)

    print(config)

if __name__ == "__main__":
    main()
