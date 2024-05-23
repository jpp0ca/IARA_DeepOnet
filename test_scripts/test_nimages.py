import os
import tqdm
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.utils.data as torch_data

import iara.utils
import iara.records
import iara.ml.models.cnn as iara_cnn
import iara.ml.models.mlp as iara_mlp
import iara.ml.experiment as iara_exp
import iara.ml.dataset as iara_dataset

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES

custom_collection = iara_default.default_collection()

output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/test_image_loader"

config = iara_exp.Config(
                name = 'image_loader',
                dataset = custom_collection,
                dataset_processor = iara_default.default_iara_lofar_audio_processor(),
                # dataset_processor = iara_default.default_iara_mel_audio_processor(),
                # input_type = iara_dataset.InputType.Image(32, 0.5),
                input_type = iara_dataset.InputType.Window(),
                output_base_dir = output_base_dir,
                exclusive_ships_on_test=True,
                test_ratio = 0.25)


id_list = config.split_datasets()
trn_set, val_set, test_set = id_list[0]

df = config.dataset.to_compiled_df()
df = df.rename(columns={'Qty': 'Total'})

for i_fold, (trn_set, val_set, test_set) in enumerate(id_list):

    df_trn = config.dataset.to_compiled_df(trn_set)
    df_val = config.dataset.to_compiled_df(val_set)
    df_test = config.dataset.to_compiled_df(test_set)

    df_trn = df_trn.rename(columns={'Qty': f'Trn_{i_fold}'})
    df_val = df_val.rename(columns={'Qty': f'Val_{i_fold}'})
    df_test = df_test.rename(columns={'Qty': f'Test_{i_fold}'})

    df = pd.merge(df, df_trn, on=config.dataset.target.grouped_column())
    df = pd.merge(df, df_val, on=config.dataset.target.grouped_column())
    df = pd.merge(df, df_test, on=config.dataset.target.grouped_column())
    # break

print(f'--- Dataset with {len(id_list)} n_folds ---')
print(df)


# df = config.dataset.to_df()
# exp_loader = iara_dataset.ExperimentDataLoader(config.dataset_processor,
#                                 df['ID'].to_list(),
#                                 df['Target'].to_list(),
#                                 df['CPA time'].to_list()
#                                 )

# exp_loader.pre_load(trn_set['ID'].to_list())
# exp_loader.pre_load(val_set['ID'].to_list())
# # exp_loader.pre_load(test_set['ID'].to_list())

# trn_dataset = iara_dataset.AudioDataset(exp_loader, config.input_type, trn_set['ID'].to_list())
# val_dataset = iara_dataset.AudioDataset(exp_loader, config.input_type, val_set['ID'].to_list())
# # test_dataset = iara_dataset.AudioDataset(exp_loader, config.input_type, test_set['ID'].to_list())

# print(f'--- Details ---')
# print('trn_dataset: ', trn_dataset)
# print('val_dataset: ', val_dataset)
# # print('test_dataset: ', test_dataset)
