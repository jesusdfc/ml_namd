import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv(os.path.join('inputs','mean_rAUC_per_group_of_features.tsv'),sep='\t')
df.columns = ['Group'] + df.columns[1:].values.tolist()
df = pd.melt(df,id_vars=['Group'], value_vars=['Atrophy_36m|Fibrosis_36m','Fibrosis_36m','Atrophy_36m'])
df.columns = ['Group','Disease','Mean rAUC']

##define groupal order
order_group = df.groupby('Group')['Mean rAUC'].aggregate(np.sum).reset_index().sort_values('Mean rAUC', ascending=False)


"""
Iterate one by one
"""

var_dict = {'Atrophy_36m|Fibrosis_36m':0, 'Fibrosis_36m':1, 'Atrophy_36m':2}
color_palette = ['#917C62','#3175A2','#E1812C']

var = 'Fibrosis_36m'
#read df
df = pd.read_csv(os.path.join('inputs','mean_rAUC_per_group_of_features.tsv'),sep='\t')
df = df.iloc[:,[0,var_dict[var]+1]]
df.columns = ['Group'] + df.columns[1:].values.tolist()
df = pd.melt(df,id_vars=['Group'], value_vars=['Atrophy_36m|Fibrosis_36m','Fibrosis_36m','Atrophy_36m'][var_dict[var]])
df.columns = ['Group','Disease','Mean rAUC']



fig=plt.figure(figsize=(15,10))
ax = sns.barplot(data=df, x="Group", y="Mean rAUC", order = order_group['Group'],
                 palette = [color_palette[var_dict[var]]])


# ax.legend(labels=("Female", "Male"),)
ax.set_ylim([-0.01,0.14])
# ax.set_xticklabels(labels=[-100, -50, -25, 0, 50, 100])
ax.set_yticklabels(labels=[-0.01, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
plt.yticks([-0.01, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])


plt.savefig(os.path.join('figures',f"{var.replace('|','_')}_mean_rAUC_per_group_of_features.pdf"))  



