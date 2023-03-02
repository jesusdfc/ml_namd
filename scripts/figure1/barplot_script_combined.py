import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['pdf.fonttype'] = 42


def reorder_to_match(df_total):

    order_l = ['Atrophy 36m Alleles MAC','NonAtrophy 36m Alleles MAC',
               'Fibrosis 36m Alleles MAC','NonFibrosis 36m Alleles MAC']

    for i in range(df_total.shape[0]//4):

        df_chunk = df_total.iloc[(i*4):(i+1)*4,:].copy()
        empty_df = pd.DataFrame([])

        for order_str in order_l:

            empty_df = pd.concat([empty_df,df_chunk[df_chunk['Type'] == order_str]])

        df_total.iloc[(i*4):(i+1)*4,:] = empty_df

    df_total.reset_index(drop=True,inplace=True)

    return df_total

            

"""
ATROPHY and FIBROSIS
"""
df_atrophy = pd.read_csv('atrophy_36m_alleles_frequency.tsv',sep='\t')
df_fibrosis = pd.read_csv('fibrosis_36m_alleles_frequency.tsv',sep='\t')

#combine
df_total = pd.concat([df_atrophy,df_fibrosis])
df_total.reset_index(drop=True,inplace=True)

#negate nonatrophy
nonatrophy_idx = df_total[df_total['Type'] == 'NonAtrophy 36m Alleles MAC'].index
df_total.loc[nonatrophy_idx,'Alelle Frequency (%)'] = -df_total.loc[nonatrophy_idx,'Alelle Frequency (%)']

#negate fibrosis
fibrosis_idx = df_total[df_total['Type'] == 'Fibrosis 36m Alleles MAC'].index
df_total.loc[fibrosis_idx,'Alelle Frequency (%)'] = -df_total.loc[fibrosis_idx,'Alelle Frequency (%)']

#generate a grouped sorted 
df_allele_order = df_total[['Alelle Frequency (%)','Minor Allele']].copy()
df_allele_order['Alelle Frequency (%)'] = df_allele_order['Alelle Frequency (%)'].abs()
allele_group_order = df_allele_order.sort_values(by = ['Minor Allele', 'Alelle Frequency (%)'], ascending= False).index

#order
df_total = df_total.loc[allele_group_order,:]
df_total.reset_index(drop=True,inplace=True)
df_total = reorder_to_match(df_total)

fig=plt.figure(figsize=(15,10))
ax=sns.barplot(data = df_total, x='Alelle Frequency (%)',y='SNP',
                hue='Type', orient='horizontal',
                dodge = True, palette= ["#e1812c", "#e5a370","#3274a1", "#5ab9ed",])
# ax.legend(labels=("Female", "Male"),)
ax.set_xlim([-70,70])
# ax.set_xticklabels(labels=[-100, -50, -25, 0, 50, 100])
# ax.set_yticklabels(labels=hr_data['Position'].value_counts().index,fontdict={"fontsize":13})
# ax.set_ylabel('Position',size=20, rotation=0)
ax.yaxis.set_label_coords(-.1, 1)

# h, l = ax.get_legend_handles_labels()
# labels=["Non-fibrosis 36m", "fibrosis 36m"]
# ax.legend(h, labels, title="Type",loc="upper right")

plt.savefig(os.path.join('figures','combined_36m_allele_frequency_barplot_raw.pdf'))



