import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#read table
df = pd.read_csv(os.path.join('inputs','best_combination_metrics.tsv'),index_col=0, sep='\t')

#dropout comb variable
df.drop('Comb',axis=1,inplace=True)
df['model'] = ['atrophy','fibrosis','both']

#get the order ['Atrophy|Fibrosis','Fibrosis','Atrophy]
df = df.iloc[[2,1,0],:]
df.reset_index(drop=True,inplace=True)

#melt
dfm = pd.melt(df, id_vars=['model'])

#barplot
sns.barplot(x = "variable", y = "value", hue = 'model', data = dfm)
plt.xticks(rotation=45)
plt.ylim([0.4, 1.025])
plt.savefig(os.path.join('figures','figure3','best_comb_metrics.pdf'))
plt.clf()
plt.cla()