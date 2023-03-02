import numpy as np
from itertools import product as p
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import pandas as pd
import random


def generate_swarm_and_violin(method='Atrophy_36m', third_plot = False, plot_var = 'Val rAUROC', save_var = 'rAUROC'):

    def choose_variables(variables_p):

        variables_p = variables_p[:-7]
        #RETURN, FROM THE PATH, THE ACTUAL VARIABLES THAT WE HAVE TO SELECT TO USE LOAD_DATA():
        #create dict
        variables_d = {}
        #define var names
        var_names = ['extravars','liquids','foveolar','vascular','cristalin','ETDRS','SNPs']
        #fill it with True
        for x in var_names:
            variables_d[x] = True

        variables_split = variables_p.split('_')
        variables_d = dict(zip(variables_split[0:len(variables_split):2],variables_split[1:len(variables_split):2]))

        vars = []
        for x in variables_d:
            if variables_d[x] =='False': #this mean it has not been dropped, so it has been included
                vars.append(x)
        return sorted(vars)

    def reorder_df_from_list(df,orderlist,var = 'Variables'):

        #init df 
        dfconcat = []

        #concat according to orderlist 
        for ordervar in orderlist:
            dfconcat.append(df[df[var] == ordervar])

        return pd.concat(dfconcat)

    def reduce_points(df,th=0.05,sp=0.05, feature='Val rAUROC'):

        newdf = []
        accumulated = []

        for method in ['rf','svm','xgboost']:
            for i,row in enumerate(df[df.Method == method].sort_values(by=feature,ascending=False).values):

                value = row[0]
                var = row[1]
                #append values
                accumulated+=[[value,var,method]]

                if i==0:
                    ref = value    
                    newdf+=[[value,var,method]]
                    accumulated = []
                else:
                    # print(ref - value)
                    if abs(ref - value) > th:
                        ref = value
                        newdf+= random.sample(accumulated,int(len(accumulated)*sp))
                        accumulated = []

        return pd.DataFrame(newdf, columns = df.columns)

    def dict_value(unparsed_df,var):

        ##
        valdd = dict()
        for row_i in range(unparsed_df.shape[0]):

            if not row_i%25000:
                print(row_i)

            method = unparsed_df.loc[row_i,'Model']
            specs = unparsed_df.loc[row_i,'Comb']
            key = method.upper()+'_'+'_'.join(unparsed_df.loc[row_i,'FileName'])+'_'+specs
            value = unparsed_df.loc[row_i,var]

            #store
            valdd[key] = value

        return valdd

    def compute_variable_aportation(df, valdd, var='Val rAUROC', savevar='rAUROC'):

        file_name_splitted = []

        for row_i in range(df.shape[0]):
            ##
            if not row_i % 25000:
                print(row_i)
            #
            score = df.loc[row_i, var]
            method = df.loc[row_i,'Model']
            specs = df.loc[row_i,'Comb']
            variables = df.loc[row_i,'FileName']
            #unbalanced first
            if len(variables) > 1:
                for feature in variables:
                    old_score = valdd[method.upper() +'_'+ '_'.join(sorted(set(variables).difference(set([feature])))) + '_' + specs]
                    file_name_splitted.append([score-old_score,feature,method]) ##add the difference

        return pd.DataFrame(file_name_splitted,columns=[savevar,'Variables','Method'])

    def generate_violin(df,x,y,name, method):

        fig, ax = plt.subplots(figsize=(12,3))
        #palett = sns.color_palette("tab10",nvars)
        #ax = sns.violinplot(x=x,y=y,data=df,ax=ax, scale='width', inner=None, palette = palett)
        ax = sns.violinplot(x=x,y=y,data=df,ax=ax, scale='width', inner=None)
        #xlabels = pd.DataFrame(variables_imp_d).mean(axis=0).sort_values(ascending=False).index
        #
        #ax.set_xticklabels(xlabels, rotation=45, ha='right',rotation_mode='anchor')
        #
        plt.savefig(os.path.join('figures',f'{method}/violin_{name}.pdf'))

    def generate_swarmplot_on_top_of_violin(df,x,y,col,name, method, scale=True):

        fig, ax = plt.subplots(figsize=(24,6))
        if scale:
            ax = sns.violinplot(x=x, y=y, data=df, ax=ax, scale='width', inner=None, color = 'white')
        else:
            ax = sns.violinplot(x=x, y=y, data=df, ax=ax, inner=None, color = 'white')
        for coll in ax.collections:
            coll.set_edgecolor('#404040')
        ax = sns.swarmplot(x=x, y=y, data=df, edgecolor="black",size=1.5, hue=col, palette = sns.color_palette("Set2"))
        plt.legend(loc = 'upper right')
        ax.grid(alpha=0.4, linestyle='--',linewidth=0.1)
        plt.savefig(os.path.join('figures',f'{method}/swarm_{name}.pdf'))

    #Select method
    excel_df = pd.read_csv(os.path.join('inputs',method+'_best_conf_6_6.tsv'), sep='\t')

    #--------------------------------------PLOT 1, SCORE AS A FUNCTION OF THE METHOD (RF, SVM, XGBOOST)---------------------------------------------#
    #plot1 = excel_df[[plot_var,'Model']]
    #generate_violin(plot1,'Model',plot_var,f'{method}_method_score')

    #--------------------------------------PLOT 2, SCORE AS A FUNCTION OF THE VARIABLE (RF, SVM, XGBOOST)-------------------------------------------#
    plot2 = excel_df[[plot_var,'FileName','Model','Comb']]
    file_name_parsed = [choose_variables(x) for x in plot2['FileName']]
    plot2['FileName'] = file_name_parsed

    #store in dict
    valdd = dict_value(plot2, plot_var)

    ##Parse it
    plot2_parsed = compute_variable_aportation(plot2, valdd, var = plot_var, savevar = save_var)

    #Filter AUROC IN A DYNAMIC WAY
    if save_var == 'AUROC':

        #Compute the mean by variable so we can sort it
        plot2_parsed['AUROC'] = plot2_parsed['AUROC'].astype(float)
        plot2_parsed_group_mean = plot2_parsed.groupby('Variables')['AUROC'].mean().sort_values(ascending=False)
        print(plot2_parsed_group_mean)
        #reorder
        plot2_parsed = reorder_df_from_list(plot2_parsed, list(plot2_parsed_group_mean.index))

    plot2_parsed[save_var] = plot2_parsed[save_var].astype('float')
    generate_violin(plot2_parsed,'Variables', save_var, f'{method}_variable_{save_var}', method.lower())
    reducedplot2 = reduce_points(plot2_parsed, th=0.01, sp=0.005, feature = save_var)

    if save_var == 'AUROC':
        #reorder
        reducedplot2 = reorder_df_from_list(reducedplot2, list(plot2_parsed_group_mean.index))

    #add color
    generate_swarmplot_on_top_of_violin(reducedplot2, 'Variables', save_var, 'Method', name=f'{method}_variable_{save_var}', method = method.lower())

    #Generate the other way around

    #--------------------------------------PLOT 3, BALANCED ACCURACY AS A FUNCTION OF THE BALANCED ACCURACY (RF, SVM, XGBOOST)-------------------------------------------#

    if third_plot:
        plot3 = excel_df[['Test BalAcc','FileName','Model','Comb']]
        file_name_parsed = [choose_variables(x) for x in plot3['FileName']]
        plot3['FileName'] = file_name_parsed

        #store in dict
        valdd = dict_value(plot3,'Test BalAcc')

        ##Parse it
        plot3_parsed = compute_variable_aportation(plot3,valdd,var='Test BalAcc',savevar='Test Balanced Accuracy')
        plot3_parsed['Test Balanced Accuracy'] = plot3_parsed['Test Balanced Accuracy'].astype('float')
        generate_violin(plot3_parsed,'Variables','Test Balanced Accuracy',f'{method}_variable_balanced_accuracy')
        reducedplot3 = reduce_points(plot3_parsed,th=0.01,sp=0.005, feature = 'Test Balanced Accuracy')

        #add color
        generate_swarmplot_on_top_of_violin(reducedplot3,'Variables','Test Balanced Accuracy','Method',name=f'{method}_variable_balanced_accuracy')

        #Generate the other way around
        generate_swarmplot_on_top_of_violin(reducedplot3,'Method','Test Balanced Accuracy','Variables',name=f'{method}_variable_balanced_accuracy_switched', scale=False)


   

# Get the method
method = sys.argv[1]

print(f"Generating probabilities distributions for {method}")

## GENERATE FIGURES
generate_swarm_and_violin(method=method, plot_var = 'Val AUROC', save_var = 'AUROC')