import numpy as np
import pandas as pd
import os
import pickle
import sys
from itertools import permutations
# pd.set_option('display.max_rows', 500)
pd.set_option("display.max_rows", 500, "display.max_columns", 500)
pd.set_option('display.max_colwidth', -1)


dataset_s = sys.argv[1]
best_df = pd.DataFrame()

for folder in ['svm','xgboost','rf']:
    print(folder)
    for file in os.listdir(os.getcwd()+f'/outputs/nested_cv_6_6/{dataset_s}/'+folder):

        #Read pickles
        folder_file_d = pd.read_pickle(os.getcwd()+f'/outputs/nested_cv_6_6/{dataset_s}/'+folder+'/'+file)
        folder_file_d.columns = list(map(lambda x: x.lstrip(), folder_file_d.columns))

        #insert filename and method
        folder_file_d.insert(loc=0, column='FileName', value=file)
        folder_file_d.insert(loc=0, column='Model', value=folder_file_d.shape[0] * [folder]) 
        
        #test_balacc,test_area_balacc,test_area_reject_rate
        best_df = pd.concat((best_df,folder_file_d),axis=0)

#print("Saving all combinations as .tsv")
#best_df.to_csv(os.getcwd()+f'/best_conf/{dataset_s}_best_conf_6_6.tsv',sep='\t',index=False)