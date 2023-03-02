import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['pdf.fonttype'] = 42

"""
ATROPHY
"""
df_atrophy = pd.read_csv(os.path.join('inputs','atrophy_36m_alleles_frequency.tsv'),sep='\t')

#generate a grouped sorted 
df_allele_order = df_atrophy[['Alelle Frequency (%)','Minor Allele']].copy()
df_allele_order['Alelle Frequency (%)'] = df_allele_order['Alelle Frequency (%)'].abs()
allele_group_order = df_allele_order.sort_values(by = ['Minor Allele', 'Alelle Frequency (%)'], ascending= False).index

#order
df_atrophy = df_atrophy.loc[allele_group_order,:]

fig=plt.figure(figsize=(15,10))
ax=sns.barplot(data = df_atrophy, x='Alelle Frequency (%)',y='SNP',
                hue='Type', orient='horizontal',
                dodge = False)
# ax.legend(labels=("Female", "Male"),)
ax.set_xlim([-100,100])
# ax.set_xticklabels(labels=[-100, -50, -25, 0, 50, 100])
# ax.set_yticklabels(labels=hr_data['Position'].value_counts().index,fontdict={"fontsize":13})
# ax.set_ylabel('Position',size=20, rotation=0)
ax.yaxis.set_label_coords(-.1, 1)

# h, l = ax.get_legend_handles_labels()
# labels=["Non-Atrophy 36m", "Atrophy 36m"]
# ax.legend(h, labels, title="Type",loc="upper right")

plt.savefig(os.path.join('figures','atrophy_36m_allele_frequency_barplot.pdf'))  


"""
FIBROSIS
"""

df_fibrosis = pd.read_csv(os.path.join('inputs','fibrosis_36m_alleles_frequency.tsv'),sep='\t')

#generate a grouped sorted 
df_allele_order = df_fibrosis[['Alelle Frequency (%)','Minor Allele']].copy()
df_allele_order['Alelle Frequency (%)'] = df_allele_order['Alelle Frequency (%)'].abs()
allele_group_order = df_allele_order.sort_values(by = ['Minor Allele', 'Alelle Frequency (%)'], ascending= False).index

#order
df_fibrosis = df_fibrosis.loc[allele_group_order,:]

fig=plt.figure(figsize=(15,10))
ax=sns.barplot(data = df_fibrosis, x='Alelle Frequency (%)',y='SNP',
                hue='Type', orient='horizontal',
                dodge = False)
# ax.legend(labels=("Female", "Male"),)
ax.set_xlim([-100,100])
# ax.set_xticklabels(labels=[-100, -50, -25, 0, 50, 100])
# ax.set_yticklabels(labels=hr_data['Position'].value_counts().index,fontdict={"fontsize":13})
# ax.set_ylabel('Position',size=20, rotation=0)
ax.yaxis.set_label_coords(-.1, 1)

# h, l = ax.get_legend_handles_labels()
# labels=["Non-fibrosis 36m", "fibrosis 36m"]
# ax.legend(h, labels, title="Type",loc="upper right")

plt.savefig(os.path.join('figures','fibrosis_36m_allele_frequency_barplot.pdf'))



