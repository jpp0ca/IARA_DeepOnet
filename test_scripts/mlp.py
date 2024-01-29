"""
Training MLP Test Program

This script generates a sample training configuration for MNIST dataset and traine a MLP for each
iara.trainer.TrainingStrategy.
"""
from torchvision import datasets, transforms

import iara.ml.mlp as iara_model
import iara.trainer as iara_trn

def main():
    """Main function for the Training MLP Test."""
    output_dir = "./results/mnist/"
    n_neurons = 128
    dropout = 0.2

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trn_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

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

    trn_multiclass.fit(model_base_dir=output_dir,
                       trn_dataset=trn_dataset,
                       val_dataset=val_dataset)

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

    trn_specialist.fit(model_base_dir=output_dir,
                       trn_dataset=trn_dataset,
                       val_dataset=val_dataset)


if __name__ == "__main__":
    main()
