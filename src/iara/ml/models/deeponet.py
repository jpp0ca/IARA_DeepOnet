import typing
import torch
import functools

import iara.ml.models.base_model as iara_model

class DeepONet(iara_model.BaseModel):
    """
    Implementação da Deep Operator Network (DeepONet) para classificação.
    Esta versão é flexível e aceita qualquer módulo PyTorch como Branch Net.
    """
    def __init__(self,
                 branch_net: torch.nn.Module,
                 input_shape: typing.List[int], # Argumento adicionado
                 n_targets: int,
                 embedding_dim: int = 128,
                 use_bias: bool = True):
        super().__init__()

        self.branch_net = branch_net

        # Determina o tamanho da saída da branch_net dinamicamente
        with torch.no_grad():
            # Cria uma entrada fictícia com o formato correto (adicionando a dimensão do batch)
            dummy_input = torch.randn(1, *input_shape)
            branch_output = self.branch_net(dummy_input)
            
            # Achata a saída para obter o número total de features
            branch_output_size = branch_output.view(branch_output.size(0), -1).shape[1]

        # Adiciona uma camada de projeção para garantir a compatibilidade de dimensões
        self.projection = torch.nn.Linear(branch_output_size, embedding_dim)
        self.trunk_net = torch.nn.Embedding(num_embeddings=n_targets, embedding_dim=embedding_dim)

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_targets))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        branch_output = self.branch_net(data)
        
        # Achata a saída da branch net antes de passá-la para a camada de projeção
        branch_output_flat = branch_output.view(branch_output.size(0), -1)
        projected_branch_output = self.projection(branch_output_flat)
        
        trunk_prototypes = self.trunk_net.weight
        logits = torch.matmul(projected_branch_output, trunk_prototypes.t())

        if self.use_bias:
            logits = logits + self.bias
            
        return logits