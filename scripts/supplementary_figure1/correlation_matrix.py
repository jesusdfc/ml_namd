import pandas as pd
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


#load dataset
d = pd.read_csv(os.getcwd()+'/inputs/FIS_dataset.txt',sep='\t')

#add healthy_ill
d.insert(d.shape[1],'healthy_ill',d['atrophy_36m']|d['fibrosis_36m'])

#re order
d = d.loc[:,['age', 'ETDRS_b', 'ETDRS_V4', 'ETDRS_12m', 
         'sex', 'tabaquism', 'hypertension',  'vitamins', 'hypercholesterolemia',
         'foveal_thickness_b', 'foveal_thickness_V4', 
         'neovascular_membrane_b', 'neovascular_membrane_V4',
         'cataract_b', 'cataract_V4', 'cataract_12m',
         'intraretinal_fluid_b', 'subretinal_fluid_b', 'intraretinal_fluid_V4', 'subretinal_fluid_V4', 
         'ARMS2', 'CFI', 'VEGFR', 'SMAD7', 
         'atrophy_b', 'fibrosis_b', 'atrophy_V4', 'fibrosis_V4',
         'atrophy_36m', 'fibrosis_36m', 'healthy_ill']]

d.columns = ['Age', 'ETDRS V1', 'ETDRS V4', 'ETDRS 12m', 
             'Sex', 'Tabaquism', 'Hypertension', 'Vitamin supplements', 'Hypercholesterolemia',
             'Foveal thickness V1', 'Foveal thickness V4',
             'Neovascular membrane V1', 'Neovascular membrane V4',
             'Cataract V1', 'Cataract V4', 'Cataract 12m',
             'Intraretinal fluid V1', 'Subretinal fluid V1', 'Intraretinal fluid V4', 'Subretinal fluid V4', 
             'ARMS2', 'CFI', 'VEGFR', 'SMAD7', 
             'Atrophy V1', 'Fibrosis V1', 'Atrophy V4', 'Fibrosis V4',
             'Atrophy 36m', 'Fibrosis 36m', '(Atrophy|Fibrosis)_36m']

# Compute the correlation matrix
corr = d.corr(method = 'spearman')

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 26))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
ax1 = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax1.set_xticklabels(ax1.get_xmajorticklabels(), fontsize = 16)
ax1.set_yticklabels(ax1.get_ymajorticklabels(), fontsize = 16)

cax = ax1.figure.axes[-1]
cax.tick_params(labelsize=20)

plt.savefig('figures/corr_mat.pdf')