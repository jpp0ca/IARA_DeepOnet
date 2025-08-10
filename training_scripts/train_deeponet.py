import typing
import torch
import argparse

# Módulos do seu projeto IARA
import iara.utils
import iara.default as iara_default
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
from iara.ml.models.cnn import CNN
from iara.ml.models.deeponet import DeepONet

def main(folds: typing.List[int], override: bool, only_sample: bool):
    """
    Função principal para configurar e executar o treinamento do modelo DeepONet.
    """
    print("--- Configurando o experimento DeepONet ---")

    # --- 1. Configurações Gerais ---
    exp_str = 'deeponet_experiment' if not only_sample else 'deeponet_experiment_sample'
    output_base_dir = f"{iara_default.DEFAULT_DIRECTORIES.training_dir}/{exp_str}"
    
    # A DeepONet com uma Branch Net CNN requer uma entrada em formato de imagem
    input_type = iara_default.default_image_input()

    # Use o processador de dados padrão para espectrogramas MEL
    data_processor = iara_default.default_iara_mel_audio_processor()

    # Use a coleção de dados padrão (OS - Oceanic-Shallow)
    collection = iara_default.default_collection(only_sample=only_sample)

    # Configuração do experimento
    config = iara_exp.Config(
            name='deeponet_mel_classification',
            dataset=collection,
            dataset_processor=data_processor,
            output_base_dir=output_base_dir,
            input_type=input_type
    )

    # --- 2. Alocador de Modelo (Model Allocator) ---
    # Esta função será chamada pelo Trainer para criar o modelo para cada fold.
    def model_allocator(input_shape: typing.List[int], n_targets: int) -> DeepONet:
        
        # Primeiro, crie uma instância da sua CNN base.
        # Os parâmetros aqui podem ser ajustados para o seu grid search.
        # Note que `classification_n_neurons` e `n_targets` na CNN base
        # não serão usados, pois a DeepONet substitui o classificador MLP.
        base_cnn = CNN(
            input_shape=input_shape,
            conv_n_neurons=[1024, 128],
            conv_activation=torch.nn.LeakyReLU,
            kernel_size=5,
            padding=2, # Padding = (kernel_size - 1) / 2 para manter as dimensões
            conv_pooling=torch.nn.MaxPool2d,
            conv_pooling_size=[2,2],
            conv_dropout=0.2,
            batch_norm=torch.nn.BatchNorm2d,
            classification_n_neurons=128, # Não utilizado pela DeepONet
            n_targets=n_targets # Não utilizado pela DeepONet
        )

        # Em seguida, passe a instância da CNN para o construtor da DeepONet.
        deeponet_model = DeepONet(
            branch_cnn_model=base_cnn,
            n_targets=n_targets,
            embedding_dim=128 # Dimensão do espaço latente
        )
        
        return deeponet_model

    # --- 3. Configuração do Trainer ---
    trainer = iara_trn.OptimizerTrainer(
            training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
            trainer_id='deeponet_v1',
            n_targets=config.dataset.target.get_n_targets(),
            batch_size=32,
            n_epochs=150, # Ajuste conforme necessário
            patience=20,  # Aumente a paciência para modelos mais complexos
            model_allocator=model_allocator,
            optimizer_allocator=lambda model: torch.optim.Adam(
                model.parameters(),
                weight_decay=1e-3,
                lr=1e-5 # Uma taxa de aprendizado menor pode ser benéfica
            ),
            loss_allocator=lambda class_weights: torch.nn.CrossEntropyLoss(
                weight=class_weights, reduction='mean'
            )
    )

    # --- 4. Gerenciador e Execução ---
    manager = iara_exp.Manager(config, trainer)

    print("--- Iniciando Treinamento da DeepONet ---")
    manager.run(folds=folds, override=override)
    print("--- Treinamento Concluído ---")


if __name__ == "__main__":
    # Adiciona argumentos de linha de comando para flexibilidade
    parser = argparse.ArgumentParser(description='Treina um modelo DeepONet no dataset IARA.')
    parser.add_argument('-F', '--fold', type=str, default=None,
                        help='Especifique as folds a serem executadas. Exemplo: 0,4-7')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignora e sobrescreve execuções antigas.')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Executa o treinamento apenas em uma amostra do dataset.')
    
    args = parser.parse_args()

    # Define as folds a serem executadas (todas as 10 por padrão)
    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))

    # Define a semente para reprodutibilidade
    iara.utils.set_seed()
    iara.utils.print_available_device()

    # Executa a função principal
    main(folds=folds_to_execute, override=args.override, only_sample=args.only_sample)