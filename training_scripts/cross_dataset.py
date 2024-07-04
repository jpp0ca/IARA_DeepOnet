import enum
import os
import pandas as pd
import itertools
import shutil
import time
import argparse
import numpy as np
import typing

import iara.records
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics
import iara.ml.dataset as iara_dataset
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES


class OtherCollections(enum.Enum):
    SHIPSEAR = 0
    DEEPSHIP = 1

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

    def _get_info_filename(self) -> str:
        return os.path.join("./training_scripts/dataset_info", f"{str(self)}.csv")

    def to_df(self, only_sample: bool = False) -> pd.DataFrame:
        df = pd.read_csv(self._get_info_filename(), na_values=[" - "])
        return df

    def get_id(self, file: str) -> int:
        if self == OtherCollections.SHIPSEAR:
            return int(file.split('_')[0])
        elif self == OtherCollections.DEEPSHIP:
            return int(file.split('.')[0])
        raise NotImplementedError(f'get_id not implemented for {self}')

    def classify_row(self, df: pd.DataFrame) -> float:
        if self == OtherCollections.SHIPSEAR:
            classes_by_length = ['B', 'C', 'A', 'D', 'E']
            try:
                target = (classes_by_length.index(df['Class']) - 1)
                if target < 0:
                    return 0
                return target
            except ValueError:
                return np.nan
            
        elif self == OtherCollections.DEEPSHIP:
            return iara_default.Target.classify_row(df)

        raise NotImplementedError(f'classify_row not implemented for {self}')

    def default_mel_managers(self,
                         output_base_dir: str,
                         classifiers: typing.List[iara_default.Classifier],
                         training_strategy: iara_trn.ModelTrainingStrategy = iara_trn.ModelTrainingStrategy.MULTICLASS):
        if self != OtherCollections.SHIPSEAR:
            raise NotImplementedError(f'default_mel_managers not implemented for {self}')

        shipsear_name = f'shipsear'
        data_base_dir = "./data/shipsear_16e3"
        data_processed_base_dir = "./data/shipsear_processed"

        collection = iara.records.CustomCollection(
                    collection = OtherCollections.SHIPSEAR,
                    target = iara.records.GenericTarget(
                        n_targets = 4,
                        function = OtherCollections.SHIPSEAR.classify_row,
                        include_others = False
                    ),
                    only_sample=False
                )

        dataset_processor = iara_manager.AudioFileProcessor(
                data_base_dir = data_base_dir,
                data_processed_base_dir = data_processed_base_dir,
                normalization = iara_proc.Normalization.MIN_MAX,
                analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
                n_pts = 1024,
                n_overlap = 0,
                decimation_rate = 1,
                n_mels=256,
                integration_interval=0.512,
                extract_id = OtherCollections.SHIPSEAR.get_id
            )

        return iara_default.default_mel_managers(config_name = shipsear_name,
                            output_base_dir = output_base_dir,
                            classifiers = classifiers,
                            collection = collection,
                            data_processor = dataset_processor,
                            training_strategy = training_strategy)


def main(
         training_strategy: iara_trn.ModelTrainingStrategy,
         folds: typing.List[int],
         only_eval: bool,
         override: bool):


    # classifiers = [iara_default.Classifier.FOREST, iara_default.Classifier.CNN]
    classifiers = [iara_default.Classifier.FOREST]
    eval_subsets = [iara_trn.Subset.TEST, iara_trn.Subset.ALL]

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/cross_dataset"
    comparison_dir = f'{output_base_dir}/comparison'
    deep_comparison_dir = f'{output_base_dir}/deep_comparison'
    grids_dir = f'{output_base_dir}/grids'

    os.makedirs(grids_dir, exist_ok=True)

    cross_grids = {}
    cross_incomplete = False
    deep_grids = {}
    deep_incomplete = False
    for classifier in classifiers:
        for subset in eval_subsets:
            cross_grids[subset, classifier] = {
                'filename': f'{grids_dir}/{classifier}_{subset}.pkl',
                'cv': None
            }
            if os.path.exists(cross_grids[subset, classifier]['filename']):
                cross_grids[subset, classifier]['cv'] = iara_metrics.GridCompiler.load(cross_grids[subset, classifier]['filename'])
            else:
                cross_incomplete = True

        for eval_strategy in [iara_trn.EvalStrategy.BY_WINDOW, iara_trn.EvalStrategy.BY_AUDIO]:

            deep_grids[classifier, eval_strategy] = {
                'filename': f'{grids_dir}/deepship_{classifier}_{eval_strategy}.pkl',
                'cv': None
            }
            if os.path.exists(deep_grids[classifier, eval_strategy]['filename']):
                deep_grids[classifier, eval_strategy]['cv'] = iara_metrics.GridCompiler.load(deep_grids[classifier, eval_strategy]['filename'])
            else:
                deep_incomplete = True


    iara_name = f'iara'

    manager_dict_iara = iara_default.default_mel_managers(config_name = iara_name,
                        output_base_dir = output_base_dir,
                        classifiers = classifiers,
                        collection = iara_default.default_collection(),
                        data_processor = iara_default.default_iara_mel_audio_processor(),
                        training_strategy = training_strategy)

    manager_dict_shipsear = OtherCollections.SHIPSEAR.default_mel_managers(
                        output_base_dir = output_base_dir,
                        classifiers = classifiers,
                        training_strategy = training_strategy)

    print("############ IARA ############")
    id_listA = manager_dict_iara[classifiers[-1]].config.split_datasets()
    manager_dict_iara[classifiers[-1]].print_dataset_details(id_listA)
    print("############ Shipsear ############")
    id_listB = manager_dict_shipsear[classifiers[-1]].config.split_datasets()
    manager_dict_shipsear[classifiers[-1]].print_dataset_details(id_listB)


    if deep_incomplete or cross_incomplete:

        if not only_eval:

            for _, manager in manager_dict_iara.items():
                manager.run(folds = folds, override = override, without_ret = True)

            for _, manager in manager_dict_shipsear.items():
                manager.run(folds = folds, override = override, without_ret = True)

        if cross_incomplete:
            for eval_subsets in eval_subsets:
                for classifier in classifiers:
                    comparator = iara_exp.CrossComparator(comparator_eval_dir = comparison_dir,
                                                        manager_a = manager_dict_iara[classifier],
                                                        manager_b = manager_dict_shipsear[classifier])

                    cross_grids[eval_subsets, classifier]['cv'] = comparator.cross_compare(
                                            eval_strategy = iara_trn.EvalStrategy.BY_WINDOW,
                                            folds = folds,
                                            eval_subset=eval_subsets)
                    
                    cross_grids[eval_subsets, classifier]['cv'].export(cross_grids[eval_subsets, classifier]['filename'])


        if deep_incomplete:

            data_base_dir = "./data/deepship"
            data_processed_base_dir = "./data/deepship_processed"

            collection = iara.records.CustomCollection(
                        collection = OtherCollections.DEEPSHIP,
                        target = iara.records.GenericTarget(
                            n_targets = 4,
                            function = OtherCollections.DEEPSHIP.classify_row,
                            include_others = False
                        ),
                        only_sample=False
                    )

            dataset_processor = iara_manager.AudioFileProcessor(
                    data_base_dir = data_base_dir,
                    data_processed_base_dir = data_processed_base_dir,
                    normalization = iara_proc.Normalization.MIN_MAX,
                    analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
                    n_pts = 1024,
                    n_overlap = 0,
                    decimation_rate = 2,
                    n_mels=256,
                    integration_interval=0.512,
                    extract_id = OtherCollections.DEEPSHIP.get_id
                )

            df = collection.to_df()
            print(collection.to_compiled_df())

            loader = iara_dataset.ExperimentDataLoader(
                            processor = dataset_processor,
                            file_ids = df['ID'].to_list(),
                            targets = df['Target'].to_list())

            for eval_strategy in [iara_trn.EvalStrategy.BY_WINDOW, iara_trn.EvalStrategy.BY_AUDIO]:
                for classifier in classifiers:

                    dataset = iara_dataset.AudioDataset(
                                loader = loader,
                                input_type = classifier.get_input_type(),
                                file_ids = df['ID'].to_list())

                    comparator = iara_exp.CrossComparator(comparator_eval_dir = deep_comparison_dir,
                                                        manager_a = manager_dict_iara[classifier],
                                                        manager_b = manager_dict_shipsear[classifier])

                    deep_grids[classifier, eval_strategy]['cv'] = comparator.cross_compare_outsource(dataset = dataset,
                                    eval_strategy = eval_strategy,
                                    folds = folds)

                    deep_grids[classifier, eval_strategy]['cv'].export(deep_grids[classifier, eval_strategy]['filename'])


    print("############### Cross comparison ###############")
    for (subset, classifier), cv_dict in cross_grids.items():
        print(f"--------- {subset} - {classifier} ---------")
        print(cv_dict['cv'])


    print("############### Deepship classification ###############")
    for (classifier, eval_strategy), cv_dict in deep_grids.items():
        if eval_strategy != iara_trn.EvalStrategy.BY_AUDIO:
            continue
        print(f"--------- {eval_strategy} - {classifier} ---------")
        print(cv_dict['cv'])
        for hash, dict in cv_dict['cv'].cv_dict.items():
            dict['cv'].print_cm(relative=False)


if __name__ == "__main__":
    start_time = time.time()

    strategy_choises = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN GridSearch analysis', add_help=False)
    parser.add_argument('-F', '--fold', type=str, default=None,
                        help='Specify folds to be executed. Example: 0,4-7')
    parser.add_argument('-t','--training_strategy', type=str, choices=strategy_choises,
                        default=None, help='Strategy for training the model')
    parser.add_argument('--only_eval', action='store_true', default=False,
                        help='Not training, only evaluate trained models')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))

    strategies = []
    if args.training_strategy is not None:
        index = strategy_choises.index(args.training_strategy)
        strategies.append(iara_trn.ModelTrainingStrategy(index))
    else:
        strategies = iara_trn.ModelTrainingStrategy

    for strategy in strategies:
        main(training_strategy = strategy,
             folds = folds_to_execute,
             only_eval = args.only_eval,
             override = args.override)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")
