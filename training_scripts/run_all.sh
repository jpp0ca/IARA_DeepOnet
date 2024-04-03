#!/bin/bash

python training_scripts/grid_forest_mel.py --training_strategy multiclass
python training_scripts/grid_forest_lofar.py --training_strategy multiclass
python training_scripts/grid_mlp_lofar.py --training_strategy multiclass