import os
import argparse
import pandas as pd

# Módulos do seu projeto IARA
import iara.utils
import iara.ml.metrics as iara_metrics
import iara.ml.models.trainer as iara_trn

def compile_results_from_directory(experiment_dir: str, trainer_id: str) -> iara_metrics.GridCompiler:
    """
    Carrega os resultados de avaliação de um diretório de experimento e os compila.
   
    """
    print(f"--- Compilando resultados para o trainer '{trainer_id}' ---")
    
    grid_compiler = iara_metrics.GridCompiler()
    eval_base_dir = os.path.join(experiment_dir, 'eval')
    
    # Assumimos que os resultados são do conjunto de teste e avaliados por áudio
    eval_subset = iara_trn.Subset.TEST
    eval_strategy = iara_trn.EvalStrategy.BY_AUDIO

    # Itera sobre todas as pastas de fold (ex: fold_0, fold_1, ...)
    for fold_dir in sorted(os.listdir(eval_base_dir)):
        if not fold_dir.startswith('fold_'):
            continue
        
        i_fold = int(fold_dir.split('_')[-1])
        
        # Constrói o nome do arquivo de resultados esperado
        # Ex: deeponet_v1_multiclass_test.csv
        results_filename = f"{trainer_id}_{iara_trn.ModelTrainingStrategy.MULTICLASS.to_str()}_{eval_subset}.csv"
        results_filepath = os.path.join(eval_base_dir, fold_dir, results_filename)
        
        if not os.path.exists(results_filepath):
            print(f"Aviso: Arquivo de resultado não encontrado para a fold {i_fold}: {results_filepath}")
            continue

        # Carrega os resultados e os adiciona ao compilador
        df = pd.read_csv(results_filepath)
        
        # Para avaliação por áudio, precisamos agregar os resultados primeiro
        if eval_strategy == iara_trn.EvalStrategy.BY_AUDIO:
             df = df.groupby('File').agg(lambda x: x.value_counts().index[0]).reset_index()

        grid_compiler.add(
            params={'Trainer': trainer_id},
            i_fold=i_fold,
            target=df['Target'],
            prediction=df['Prediction']
        )

    return grid_compiler

def main(experiment_dir: str, trainer_id: str):
    """
    Função principal para analisar e visualizar os resultados de um experimento.
    """
    if not os.path.exists(experiment_dir):
        print(f"Erro: O diretório do experimento '{experiment_dir}' não foi encontrado.")
        return

    # 1. Compila os resultados a partir dos arquivos de avaliação
    grid = compile_results_from_directory(experiment_dir, trainer_id)

    if grid.is_empty():
        print("Nenhum resultado encontrado para compilar. Verifique o diretório e o ID do trainer.")
        return

    # 2. Exibe a tabela de métricas
    print("\n--- Tabela de Métricas (Média ± Desvio Padrão entre Folds) ---")
    print(grid)

    # 3. Encontra e exibe o melhor resultado (neste caso, o único resultado)
    params, best_cv = grid.get_best(metric=iara_metrics.Metric.SP_INDEX)
    print("\n--- Resultado Compilado ---")
    print(f"Parâmetros: {params}")
    print(f"Métricas: {best_cv}")

    # 4. Cria um diretório de visualizações dentro da pasta do experimento
    viz_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    print(f"\n--- Gerando Visualizações em: {viz_dir} ---")

    # 5. Gera e salva a Matriz de Confusão Relativa (em porcentagem)
    cm_relative_path = os.path.join(viz_dir, f"{trainer_id}_confusion_matrix_relative.png")
    print(f"Salvando Matriz de Confusão Relativa em: {cm_relative_path}")
    best_cv.print_cm(filename=cm_relative_path, relative=True)
    
    # 6. Gera e salva a Matriz de Confusão Absoluta (contagem)
    cm_absolute_path = os.path.join(viz_dir, f"{trainer_id}_confusion_matrix_absolute.png")
    print(f"Salvando Matriz de Confusão Absoluta em: {cm_absolute_path}")
    best_cv.print_cm(filename=cm_absolute_path, relative=False)
    
    print("\n--- Análise Concluída ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analisa e visualiza os resultados de um experimento de treinamento IARA.')
    parser.add_argument(
        'experiment_dir', 
        type=str, 
        help='O caminho para o diretório raiz do experimento (ex: ./results/trainings/deeponet_shipsear/deeponet_shipsear_classification)'
    )
    parser.add_argument(
        '--trainer_id',
        type=str,
        default='deeponet_shipsear_v1',
        help='O ID do trainer usado no treinamento (ex: deeponet_shipsear_v1)'
    )
    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir, trainer_id=args.trainer_id)