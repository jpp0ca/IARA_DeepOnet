"""
Training MLP Test Program

This script generates a sample training configuration for MNIST dataset and traine a MLP for each
iara.trainer.TrainingStrategy.
"""
import os
import pandas as pd
import torchvision
import sklearn.metrics as sk_metrics

import iara.ml.mlp as iara_model
import iara.trainer as iara_trn

def main():
    """Main function for the Training MLP Test."""
    output_dir = "./results/mnist/"
    model_dir = os.path.join(output_dir, 'model')
    eval_dir = os.path.join(output_dir, 'eval')
    n_neurons = 128
    dropout = 0.2

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5,), (0.5,))])
    trn_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                             transform=transform, download=True)
    val_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                             transform=transform, download=True)

    trn_multiclass = iara_trn.NNTrainer(training_strategy=iara_trn.TrainingStrategy.MULTICLASS,
                                trainer_id = 'MLP',
                                n_targets = 10,
                                model_allocator=lambda input_shape:
                                            iara_model.MLP(input_shape=input_shape,
                                                           n_neurons=n_neurons,
                                                           n_targets=10,
                                                           dropout=dropout),
                                batch_size = 64,
                                n_epochs = 5)

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
                                trainer_id = 'MLP',
                                n_targets = 10,
                                model_allocator=lambda input_shape:
                                            iara_model.MLP(input_shape=input_shape,
                                                           n_neurons=n_neurons,
                                                           dropout=dropout),
                                batch_size = 64,
                                n_epochs = 5)

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
    main()
