"""
Training MLP on an Unbalanced Dataset Test Program

This script generates a sample training configuration for the MNIST dataset by unbalancing it and
training a Multilayer Perceptron (MLP) for each iara.trainer.TrainingStrategy.
"""

# Seu código aqui...

import os
import argparse
import shutil

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics

import torch.utils.data as torch_data
import torchvision

import iara.ml.mlp as iara_model
import iara.trainer as iara_trn


class Dataset(torch_data.Dataset):

    def __init__(self, indexes, dataset) -> None:
        self.indexes = indexes
        self.data = dataset.data[indexes].float()/255
        self.targets = dataset.targets[indexes]
        self.transform = torchvision.transforms.Normalize((0.5,), (0.5,))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.transform(self.data[index].unsqueeze(0)), self.targets[index]


def main(override: bool):
    """Main function for the Training MLP Test."""
    output_dir = "./results/mnist/unbalanced/"
    model_dir = os.path.join(output_dir, 'model')
    eval_dir = os.path.join(output_dir, 'eval')
    n_neurons = 64
    dropout = 0.2

    if os.path.exists(output_dir) and override:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5,), (0.5,))])
    base_trn_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                             transform=transform, download=True)

    qtds = [15, 20, 50, 100, 800, 800, 800, 800, 800, 800]

    indexes = []
    for i, qtd in enumerate(qtds):
        class_index = np.where(np.array(base_trn_dataset.targets) == i)[0]
        indexes.extend(class_index[:qtd])

    trn_dataset = Dataset(indexes, base_trn_dataset)

    print('qtd: ', len(trn_dataset), '/', len(base_trn_dataset.targets))

    val_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                             transform=transform, download=True)

    trn_multiclass = iara_trn.NNTrainer(training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                trainer_id='MLP',
                                n_targets=10,
                                model_allocator=lambda input_shape:
                                            iara_model.MLP(input_shape=input_shape,
                                                           n_neurons=n_neurons,
                                                           n_targets=10,
                                                           dropout=dropout),
                                batch_size=64,
                                n_epochs=32,
                                patience=5)

    trn_multiclass.fit(model_base_dir=model_dir,
                       trn_dataset=trn_dataset,
                       val_dataset=val_dataset)

    multiclass_result = trn_multiclass.eval(dataset_id='val',
                                            model_base_dir=model_dir,
                                            eval_base_dir=eval_dir,
                                            dataset=val_dataset)

    print('################ MULTICLASS ################')
    accuracy = sk_metrics.accuracy_score(multiclass_result['Target'],
                                         multiclass_result['Prediction'])
    print("Accuracy:", accuracy)

    f1 = sk_metrics.f1_score(multiclass_result['Target'],
                             multiclass_result['Prediction'],
                             average='weighted')
    print("F1-score:", f1)

    conf_matrix = sk_metrics.confusion_matrix(multiclass_result['Target'],
                                              multiclass_result['Prediction'])
    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix))

    trn_specialist = iara_trn.NNTrainer(
                                training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                                trainer_id='MLP',
                                n_targets=10,
                                model_allocator=lambda input_shape:
                                            iara_model.MLP(input_shape=input_shape,
                                                           n_neurons=n_neurons,
                                                           dropout=dropout),
                                batch_size=64,
                                n_epochs=5)

    trn_specialist.fit(model_base_dir=model_dir,
                       trn_dataset=trn_dataset,
                       val_dataset=val_dataset)

    specialist_result = trn_specialist.eval(dataset_id='val',
                                            model_base_dir=model_dir,
                                            eval_base_dir=eval_dir,
                                            dataset=val_dataset)

    print('################ CLASS_SPECIALIST ################')
    accuracy = sk_metrics.accuracy_score(specialist_result['Target'],
                                         specialist_result['Prediction'])
    print("Accuracy:", accuracy)

    f1 = sk_metrics.f1_score(specialist_result['Target'],
                             specialist_result['Prediction'],
                             average='weighted')
    print("F1-score:", f1)

    conf_matrix = sk_metrics.confusion_matrix(specialist_result['Target'],
                                              specialist_result['Prediction'])
    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN CPA analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')

    args = parser.parse_args()
    main(args.override)
