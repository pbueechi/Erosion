import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = '~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/cfactor'

comps = ['calibration_analysis_mlp_strat_noley/nn_predictions_per_pixel.csv', 'calibration_analysis_twobeta_noley/calibration_results_stratified_per_pixel']

df = pd.read_csv(os.path.join(path, comps[0]), encoding='latin-1', delimiter=',')
print(df.tillage.unique())