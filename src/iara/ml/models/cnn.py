import math

import torch

import iara.ml.models.base_model as ml_model

class CNN(ml_model.Base):
    def __init__(self, n_channels: int, feature_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.n_channels = n_channels
        self.feature_dim = feature_dim

        final_layer_size = 8

        num_layers = int(round(math.log2(feature_dim)-math.log2(final_layer_size)-1)) # reduzir feature_dim/2 -> 4 - considerando seguidas divisões por 2

        # input is batch_size x (n_channels x feature_dim x feature_dim)  - batch x imagem
        layers = [
            torch.nn.Conv2d(self.n_channels, self.feature_dim, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(negative_slope, inplace=True)
        ]

        # state size - (batch_size) x (feature_dim/2 x feature_dim/2)
        for i in range(num_layers):
            layers.extend([
                torch.nn.Conv2d(self.feature_dim * (2**i), self.feature_dim * (2**(i+1)), 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(self.feature_dim * (2**(i+1))),
                torch.nn.LeakyReLU(negative_slope, inplace=True)
            ])

        # state size - (batch_size) x (final_layer_size x final_layer_size)
        layers.extend([
            torch.nn.Conv2d(self.feature_dim * (2**num_layers), 1, 4, 1, 0, bias=False),
        ])

        # state size - (batch_size) x (1 x 1)
        self.model = torch.nn.Sequential(*layers)

        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear((final_layer_size-3)**2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(self.model(x))