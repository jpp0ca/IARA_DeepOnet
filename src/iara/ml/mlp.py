"""
Module containing a Multi-Layer Perceptron (MLP) model.

Classes:
    - MLP: Multi-Layer Perceptron model for regression tasks.
"""
import functools
import typing
import torch

import iara.ml.base_model as ml_model

class MLP(ml_model.BaseModel):
    """Multi-Layer Perceptron (MLP) model."""

    def __init__(self,
                 input_shape: typing.Union[int, typing.Iterable[int]],
                 n_neurons: int,
                 n_targets: int = 1,
                 regularization: torch.nn.Module = torch.nn.Dropout(0.2),
                 activation_hidden_layer: torch.nn.Module = torch.nn.ReLU(),
                 activation_output_layer: torch.nn.Module = torch.nn.Sigmoid()):
        """
        Parameters:
            - input_shape (Union[int, List[int]]): Shape of the input data, one or more dimensions.
            - n_neurons (int): Number of neurons in the hidden layer.
            - n_targets (int): Number of target classes. Default 1. If 1 class specialist
                for other values max probability MLP
            - dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
            - activation_hidden_layer (torch.nn.Module, optional): Activation function for the
                hidden layer. Defaults to torch.nn.ReLU().
            - activation_output_layer (torch.nn.Module, optional): Activation function for the
                output layer. Defaults to torch.nn.Sigmoid().
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.n_targets = n_targets
        self.regularization = regularization

        input_dim = functools.reduce(lambda x, y: x * y, input_shape)

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear(input_dim, self.n_neurons),
            activation_hidden_layer,
        )

        self.activation = torch.nn.Sequential(
            torch.nn.Linear(self.n_neurons, self.n_targets),
            activation_output_layer
        )

        if self.n_targets > 1:
            self.softmax = torch.nn.Softmax(dim=-1)
        else:
            self.softmax = None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): Teste input data

        Returns:
            torch.Tensor: prediction
        """
        data = self.model(data)
        if self.regularization is not None:
            data = self.regularization(data)
        prediction = self.activation(data)

        if self.n_targets > 1:
            prediction = self.softmax(prediction)
        else:
            prediction = prediction.squeeze(1)

        return prediction
