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
                collection = iara.records.Collection.OS_SHIP,
                filters = iara.records.LabelFilter(
                    column='TYPE',
                    values=['Cargo']
                ),
                target = iara.records.Target(
                    column = 'DETAILED TYPE',
                    values = ['Bulk Carrier', 'Container Ship', 'General Cargo'],
                    include_others = False
                ),
                only_sample=True,
            )

output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/performance"

config = iara_exp.Config(
                name = 'performance',
                dataset = custom_collection,
                dataset_processor = iara_default.default_iara_audio_processor(),
                output_base_dir = output_base_dir,
                n_folds=4,
                excludent_ship_id=False)

list = config.split_datasets2()
batch_size = 32
n_epochs = 5
device = iara.utils.get_available_device()


start_time = time.time()

for n_neurons in tqdm.tqdm([4, 16, 64, 256], desc='Trainers', leave=False):

    for i_fold, (trn_set, val_set, test_set) in enumerate(tqdm.tqdm(list, desc='Fold', leave=False)):

        trn_dataset = iara_dataset.AudioDataset(config.dataset_processor,
                                                trn_set['ID'],
                                                trn_set['Target'])

        val_dataset = iara_dataset.AudioDataset(config.dataset_processor,
                                                val_set['ID'],
                                                val_set['Target'])

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