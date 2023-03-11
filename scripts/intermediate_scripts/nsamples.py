import pandas as pd
import os

df = pd.read_csv(os.path.join('inputs','FIS_dataset.txt'), sep='\t')

##
prev_atrophy = df['atrophy_b']|df['atrophy_V4']
prev_atrophy_idx = prev_atrophy[prev_atrophy == 0].index
prev_fibrosis = df['fibrosis_b']|df['fibrosis_V4']
prev_fibrosis_idx = prev_fibrosis[prev_fibrosis == 0].index
kept_idx = list(set(prev_atrophy_idx).intersection(set(prev_fibrosis_idx)))

print(f"Atrophy experiment {len(prev_atrophy_idx)} samples")
print(f"Fibrosis experiment {len(prev_fibrosis_idx)} samples")
print(f"Atrophy or Fibrosis experiment {len(kept_idx)} samples")