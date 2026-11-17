import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_lese_results():
    path = '~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/cfactor'
    
    comps = ['calibration_analysis_mlp_strat_noley/nn_predictions_per_pixel.csv', 'calibration_analysis_twobeta_noley/calibration_results_stratified_per_pixel.csv']
    
    df_nn = pd.read_csv(os.path.join(path, comps[0]), encoding='latin-1', delimiter=',')
    df_beta = pd.read_csv(os.path.join(path, comps[1]), encoding='latin-1', delimiter=',')
    print(df_nn.shape, df_beta.shape)
    
    df = df_nn.copy()
    df.rename(columns={'C_pred': 'MLP_pred'}, inplace=True)
    df['Beta_pred'] = df_beta['C_predicted']
    
    # Convert from wide to long format
    df_long = df.melt(
        id_vars='tillage',
        value_vars=['C_ref', 'MLP_pred', 'Beta_pred'],
        var_name='Type',
        value_name='C'
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_long,
        x='tillage',
        y='C',
        hue='Type',
    )
    
    plt.xlabel('Tillage class')
    plt.ylabel('C')
    plt.title('Comparison of C_ref and C_pred by Tillage Class')
    plt.legend(title='')
    plt.tight_layout()
    plt.show()


