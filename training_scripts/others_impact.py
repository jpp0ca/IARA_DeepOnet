"""
Grid Search Application

This script performs an initial grid search to choose the MLP (Multi-Layer Perceptron) model
configuration for all training in the article. It evaluates different configurations using
cross-validation and selects the one with the best performance based on predefined evaluation
metrics.

The chosen configuration will then be used for further training and analysis in the article.
"""
import typing
import argparse

import iara.records
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES


def main(override: bool,
        folds: typing.List[int],
        only_sample: bool):
    """Grid search main function"""

    impact_str = 'others_impact' if not only_sample else 'others_impact_sample'

    config_dir = f"{DEFAULT_DIRECTORIES.config_dir}/{impact_str}"

    configs = {
        'with': iara.records.Collection.OS_SHIP,
        'without': iara.records.Collection.OS_SHIP
    }


    grid = iara_metrics.GridCompiler()

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
                            target = iara.records.LabelTarget(
                                column = 'TYPE',
                                values = ['Cargo', 'Tanker', 'Tug'],
                                include_others = True if config_name == 'with' else False
                            ),
                            only_sample=only_sample
                        )

            output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{impact_str}"

            config = iara_exp.Config(
                            name = config_name,
                            dataset = custom_collection,
                            dataset_processor = iara_default.default_iara_lofar_audio_processor(),
                            output_base_dir = output_base_dir,
                            kfolds=10 if not only_sample else 3)

            config.save(config_dir)

        trainers = iara_default.default_trainers(config=config)

        manager = iara_exp.Manager(config, *trainers)

        result_dict = manager.run(folds = folds)

        for trainer_id, results in result_dict.items():

            for i_fold, result in enumerate(results):

                grid.add(params={
                            'Others': config_name,
                            'Model': trainer_id,
                        },
                        i_fold=i_fold,
                        target=result['Target'],
                        prediction=result['Prediction'])

    print(grid)



if __name__ == "__main__":
    strategy_str_list = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN others impact analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
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

    main(override = args.override,
        folds = folds_to_execute,
        only_sample = args.only_sample)
