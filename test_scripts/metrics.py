"""
Metrics Test Program

This script generates fake grid search results and uses the metrics module to evaluate them.
"""
import os
import itertools
import numpy as np

import iara.utils
import iara.ml.metrics as iara_metrics

def main():
    """Main function for the metrics test program."""

    iara.utils.set_seed()

    os.makedirs('./results/metrics', exist_ok=True)

    n_classes = 4
    n_samples = 1024
    n_folds = 5


    grid_search = {
        'Neurons': [4, 16, 64, 256],
        'Activation': ['Tanh', 'ReLU', 'PReLU']
    }

    grid = iara_metrics.GridCompiler()

    combinations = list(itertools.product(*grid_search.values()))
    for combination in combinations:
        param_pack = dict(zip(grid_search.keys(), combination))

        for i in range(n_folds):

            target_multiclass = np.random.randint(0, n_classes, size=n_samples)
            predict_multiclass = np.random.randint(0, n_classes, size=n_samples)

            grid.add(params=param_pack,
                    i_fold=i,
                    target=target_multiclass,
                    prediction=predict_multiclass)

    print(grid)

if __name__ == "__main__":
    main()
