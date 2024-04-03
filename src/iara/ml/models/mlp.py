"""
Module containing a Multi-Layer Perceptron (MLP) based models.
"""
import functools
import typing
import torch

import iara.ml.models.base_model as iara_model

class MLP(iara_model.BaseModel):
    """Multi-Layer Perceptron (MLP) model."""

    def __init__(self,
                 input_shape: typing.Union[int, typing.Iterable[int]],
                 n_neurons: int,
                 n_targets: int = 1,
                 activation_hidden_layer: torch.nn.Module = torch.nn.ReLU(),
                 activation_output_layer: torch.nn.Module = torch.nn.Sigmoid()):
        """
        Parameters:
            - input_shape (Union[int, List[int]]): Shape of the input data, one or more dimensions.
            - n_neurons (int): Number of neurons in the hidden layer.
            - n_targets (int): Number of target classes. Default 1. If 1 class specialist
                for other values max probability MLP
            - activation_hidden_layer (torch.nn.Module, optional): Activation function for the
                hidden layer. Defaults to torch.nn.ReLU().
            - activation_output_layer (torch.nn.Module, optional): Activation function for the
                output layer. Defaults to torch.nn.Sigmoid().
        """
        super().__init__()
        self.n_targets = n_targets


        input_dim = functools.reduce(lambda x, y: x * y, input_shape)

        layers = [
            torch.nn.Flatten(1),
            torch.nn.Linear(input_dim, n_neurons),
            activation_hidden_layer,
            torch.nn.Linear(n_neurons, n_targets),
        ]

        if activation_output_layer is not None:
            layers.append(activation_output_layer)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): Teste input data

        Returns:
            torch.Tensor: prediction
        """
        prediction = self.model(data)

        if self.n_targets > 1:
            if not self.training:
                prediction = torch.softmax(prediction, dim=-1)
        else:
            prediction = prediction.squeeze(1)

        return prediction

    def __str__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            str: A string containing the name of the model class.
        """
        return f'{super().__str__()} ------- \n' + f'{str(self.model)}'