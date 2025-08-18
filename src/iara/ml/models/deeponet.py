import typing
import torch

import iara.ml.models.base_model as iara_model

class DeepONet(iara_model.BaseModel):
    """
    Implementação da Deep Operator Network (DeepONet) para classificação.
    Esta versão é flexível e aceita qualquer módulo PyTorch como Branch Net.
    """
    def __init__(self,
                 branch_net: torch.nn.Module, # <-- MUDANÇA: Recebe a branch_net pronta
                 n_targets: int,
                 embedding_dim: int = 128,
                 use_bias: bool = True):
        super().__init__()

        # --- 1. Branch Net ---
        # A Branch Net agora é o módulo que foi passado diretamente.
        self.branch_net = branch_net

        # --- 2. Trunk Net ---
        # A Trunk Net continua sendo uma camada de Embedding para aprender os protótipos de classe.
        self.trunk_net = torch.nn.Embedding(num_embeddings=n_targets, embedding_dim=embedding_dim)

        # --- 3. Bias (Opcional) ---
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_targets))


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # branch_output -> [batch_size, embedding_dim]
        branch_output = self.branch_net(data)

        # trunk_prototypes -> [n_targets, embedding_dim]
        trunk_prototypes = self.trunk_net.weight

        # Produto escalar via multiplicação de matrizes
        logits = torch.matmul(branch_output, trunk_prototypes.t())

        if self.use_bias:
            logits = logits + self.bias
            
        return logits