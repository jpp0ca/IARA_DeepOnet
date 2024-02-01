"""
Metrics Test Program

This script generates fake grid search results and uses the metrics module to evaluate them.
"""
import os
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

    compiler = iara_metrics.GridCompiler()

    for i in range(n_folds):

        target_multiclass = np.random.randint(0, n_classes, size=n_samples)
        predict_multiclass = np.random.randint(0, n_classes, size=n_samples)

        compiler.add(grid_id='Random 1',
                    i_fold=i,
                    target=target_multiclass,
                    prediction=predict_multiclass)

        target_multiclass = np.random.randint(0, n_classes, size=n_samples)
        predict_multiclass = np.random.randint(0, n_classes, size=n_samples)

        compiler.add(grid_id='Random 2',
                    i_fold=i,
                    target=target_multiclass,
                    prediction=predict_multiclass)

    print(compiler.as_str())

if __name__ == "__main__":
    main()
