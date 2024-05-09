import time
import functools

import tqdm

import torch
import torch.utils.data as torch_data

import iara.utils
import iara.records
import iara.ml.experiment as iara_exp
import iara.ml.dataset as iara_dataset

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES

iara.utils.print_available_device()

class MLP(torch.nn.Module):
    def __init__(self, input_size, n_neurons, n_targets):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, n_neurons)
        self.fc2 = torch.nn.Linear(n_neurons, n_targets)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

custom_collection = iara.records.CustomCollection(
                collection = iara.records.Collection.A,
                filters = iara.records.LabelFilter(
                    column='TYPE',
                    values=['Cargo']
                ),
                target = iara.records.LabelTarget(
                    column = 'TYPE',
                    values = ['Cargo', 'Tanker', 'Tug'],
                    include_others = True
                ),
                only_sample=True,
            )

output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/performance"

config = iara_exp.Config(
                name = 'performance',
                dataset = custom_collection,
                dataset_processor = iara_default.default_iara_lofar_audio_processor(),
                input_type = iara_dataset.InputType.Window(),
                output_base_dir = output_base_dir,
                exclusive_ships_on_test=False)

list = config.split_datasets()
batch_size = 512
n_epochs = 5
device = iara.utils.get_available_device()


start_time = time.time()

df = custom_collection.to_df()

experiment_loader = iara_dataset.ExperimentDataLoader(config.dataset_processor,
                                df['ID'].to_list(),
                                df['Target'].to_list())


for n_neurons in tqdm.tqdm([16], desc='Trainers', leave=False):

    for i_fold, (trn_set, val_set, test_set) in \
            enumerate(tqdm.tqdm(list, desc='Fold', leave=False)):

        trn_dataset = iara_dataset.AudioDataset(experiment_loader, config.input_type, trn_set['ID'].to_list())

        val_dataset = iara_dataset.AudioDataset(experiment_loader, config.input_type, val_set['ID'].to_list())

        trn_loader = torch_data.DataLoader(trn_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        val_loader = torch_data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

        sample, target = trn_dataset[0]
        input_dim = functools.reduce(lambda x, y: x * y, sample.shape)

        net = MLP(input_size=input_dim,
                  n_neurons=n_neurons,
                  n_targets=custom_collection.target.get_n_targets()).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        for epoch in tqdm.tqdm(range(n_epochs), leave=False, desc="Epoch"):
            for i, (samples, targets) in enumerate(tqdm.tqdm(trn_loader, leave=False, desc="Batch")):

                samples, targets = samples.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(samples)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()


end_time = time.time()



def format_time(seconds):
    if seconds < 1:
        formatted_time = f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        formatted_time = f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        formatted_time = f"{int(minutes)} min {seconds:.2f} s"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        formatted_time = f"{int(hours)} h {int(minutes)} min {seconds:.2f} s"

    return formatted_time

execution_time = end_time - start_time
print("Execution time: ", format_time(execution_time))