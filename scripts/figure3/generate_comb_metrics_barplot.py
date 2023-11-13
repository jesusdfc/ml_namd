import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#read table
df = pd.read_csv(os.path.join('inputs','best_combination_metrics.tsv'),index_col=0, sep='\t')

#dropout comb variable
df.columns = df.columns[:-1].tolist() + ['model']
df['model'] = df['model'].replace(pd.unique(df['model']), ['both','fibrosis','atrophy'])

#melt
dfm = pd.melt(df, id_vars=['model'])

#barplot
plt.figure(figsize=(10,10))
sns.barplot(x = "variable", y = "value", hue = 'model', data = dfm, estimator = np.mean)
plt.xticks(rotation=45)
plt.ylim([0.4, 1.025])
plt.savefig(os.path.join('figures','figure3','best_comb_metrics_ci.pdf'))
plt.clf()
plt.cla()