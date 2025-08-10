import typing
import torch

import iara.ml.models.base_model as iara_model
import iara.ml.models.cnn as iara_cnn 

class DeepONet(iara_model.BaseModel):
    """
    Implementação da Deep Operator Network (DeepONet) para classificação,
    reutilizando a classe CNN existente como base para a Branch Net.
    """

    def __init__(self,
                 # O __init__ agora recebe a própria classe CNN e seus parâmetros
                 branch_cnn_model: iara_cnn.CNN,
                 n_targets: int,
                 embedding_dim: int = 128,
                 use_bias: bool = True):
        super().__init__()

        # --- 1. Definição da Branch Net ---
        # A Branch Net agora é composta pelo extrator de características da CNN
        # e uma camada linear final para projetar no espaço de embedding.
        
        # Extrai o extrator de características convolucionais do modelo CNN passado
        feature_extractor = branch_cnn_model.conv_layers
        
        # Calcula o tamanho da saída do extrator para a camada linear
        with torch.no_grad():
            dummy_input = torch.zeros(1, *branch_cnn_model.input_shape)
            flattened_size = feature_extractor(dummy_input).shape[1]

        # A Branch Net completa é o extrator seguido pela camada linear
        self.branch_net = torch.nn.Sequential(
            feature_extractor,
            torch.nn.Linear(flattened_size, embedding_dim)
        )

        # --- 2. Definição da Trunk Net ---
        # A Trunk Net continua sendo uma camada de Embedding para aprender os protótipos de classe.
        self.trunk_net = torch.nn.Embedding(num_embeddings=n_targets, embedding_dim=embedding_dim)

        # --- 3. Definição do Bias (opcional) ---
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_targets))


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Calcula os logits para cada classe através do produto escalar entre
        a saída da Branch Net e os protótipos da Trunk Net.
        """
        # branch_output -> [batch_size, embedding_dim]
        branch_output = self.branch_net(data)

        # trunk_prototypes -> [n_targets, embedding_dim]
        trunk_prototypes = self.trunk_net.weight

        # Produto escalar via multiplicação de matrizes
        logits = torch.matmul(branch_output, trunk_prototypes.t())

        if self.use_bias:
            logits = logits + self.bias
            
        return logits