import os
import numpy as np

import scipy.stats as scipy

import iara.default
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics

from iara.default import DEFAULT_DIRECTORIES

import grid_search

threshold_confidence = 0.9

training_strategy = iara_trn.ModelTrainingStrategy.MULTICLASS
grid_str = 'grid_search'
output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

eval_strategy = iara_trn.EvalStrategy.BY_AUDIO
compiled_dir = f'{output_base_dir}/compiled'

dict = {}
labels = []

for feature in [grid_search.Feature.MEL, grid_search.Feature.LOFAR]:
    for classifier in iara.default.Classifier:
        filename = f'{compiled_dir}/{classifier}_{feature}_{training_strategy}_{eval_strategy}.pkl'
        if os.path.exists(filename):
            dict[feature, classifier] = filename
            labels.append(f'{feature} {classifier}')

keys = list(dict.keys())


confidence_matrix = np.zeros((len(dict), len(dict)))

for i in range(len(dict)):

    gci = iara_metrics.GridCompiler.load(dict[keys[i]])
    parami, cvi = gci.get_best()
    spi = np.array(cvi.get(iara_metrics.Metric.SP_INDEX))

    for j in range(i+1, len(dict)):

        gcj = iara_metrics.GridCompiler.load(dict[keys[j]])
        paramj, cvj = gcj.get_best()
        spj = np.array(cvj.get(iara_metrics.Metric.SP_INDEX))

        _, confidence = iara_metrics.Test.f_test_5x2(spi, spj, threshold_confidence)
        confidence_matrix[i,j] = confidence
        confidence_matrix[j,i] = confidence

print(confidence_matrix)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = [(0.2, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0.5, 0)]  # branco a vermelho, verde claro a verde escuro
values = [0, threshold_confidence, threshold_confidence, 1]

cmap_name = 'custom_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, list(zip(values, colors)))


plt.figure(figsize=(10, 8))
plt.imshow(confidence_matrix, cmap=cm, vmin=0, vmax=1)
plt.colorbar()
plt.title('Confidence Matrix')
plt.xlabel('Feature-Classifier Combination')
plt.ylabel('Feature-Classifier Combination')
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.savefig(f'{output_base_dir}/grid_ftest.png')