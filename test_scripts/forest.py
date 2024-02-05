"""
Training RandomForest Test Program

This script generates a sample training configuration for MNIST dataset and traine a RandomForest
for each iara.trainer.TrainingStrategy.
"""
import os
import argparse
import shutil

import pandas as pd
import numpy as np
import torchvision
import sklearn.metrics as sk_metrics

import torch.utils.data as torch_data

import iara.ml.forest as iara_model
import iara.trainer as iara_trn

class Dataset(torch_data.Dataset):
    """Simple dataset to subset MNIST keeping interface for training."""

    def __init__(self, dataset) -> None:
        self.transform = torchvision.transforms.Normalize((0.5,), (0.5,))
        self.data = dataset.data
        self.data = (dataset.data.float()/255)
        self.data = self.transform(self.data.unsqueeze(1))
        self.targets = dataset.targets
        self.classes = range(10)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def main(override: bool):
    """Main function for the Training RandomForest Test."""
    output_dir = "./results/trainings/mnist/forest"
    model_dir = os.path.join(output_dir, 'model')
    eval_dir = os.path.join(output_dir, 'eval')

    if os.path.exists(output_dir) and override:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.5,), (0.5,))])
    trn_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                             transform=transform, download=True)
    val_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                             transform=transform, download=True)

    trn_dataset = Dataset(trn_dataset)
    val_dataset = Dataset(val_dataset)

    trn_multiclass = iara_trn.ForestTrainer(
                        training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                        trainer_id = 'forest',
                        n_targets=10,
                        n_estimators=100,
                        max_depth=None)

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

    trn_specialist = iara_trn.ForestTrainer(
                        training_strategy=iara_trn.TrainingStrategy.CLASS_SPECIALIST,
                        trainer_id = 'forest',
                        n_targets=10,
                        n_estimators=50,
                        max_depth=None)

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
