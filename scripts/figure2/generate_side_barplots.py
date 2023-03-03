import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv(os.path.join('inputs','mean_rAUC_per_variables_and_models.tsv'),sep='\t',index_col=0)

"""
Iterate one by one
"""

color_palette = ['#5FCAAA','#77A1CD','#FF8964']

for var in ['Atrophy_36m','Fibrosis_36m','Atrophy|Fibrosis_36m']:

    # creating subplots
    ax = plt.subplots()
    subdf = df[df['experiment'] == var]
    #generate sum of groups
    groupsum_dict = subdf.groupby('Variables').median().to_dict()
    groupsum = [groupsum_dict['rAUROC'][x] for x in subdf['Variables']]
    subdf['groupsum'] = groupsum
    ssubdf = subdf.sort_values(by='groupsum',ascending=False)
    ax = sns.barplot(data= ssubdf, x= "Variables", y= "rAUROC", hue= 'Method', palette= color_palette, hue_order=['rf','xgboost','svm'])
    
    ax.set_yticklabels(labels=[-0.025, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
    plt.yticks([-0.025, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
    plt.savefig(os.path.join('figures','figure2',f"{var.replace('|','_')}_mean_rAUC_per_group_of_features.pdf"))  



