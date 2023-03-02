import os
import pandas as pd

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

def dict_value(unparsed_df,var):

        ##
        valdd = dict()
        for row_i in range(unparsed_df.shape[0]):

            # if not row_i%25000:
            #     print(row_i)

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
            # if not row_i % 25000:
            #     print(row_i)
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

def generate_rAUC():

    exp_l = []
    for experiment in ['Atrophy_36m','Fibrosis_36m','healthy_ill']:

        #load best_conf
        best_conf_df = pd.read_csv(os.path.join('inputs',f'{experiment}_best_conf_6_6.tsv'),sep='\t')

        fbest_conf_df = best_conf_df[['Val AUROC','FileName','Model','Comb']]
        file_name_parsed = [choose_variables(x) for x in fbest_conf_df['FileName']]
        fbest_conf_df['FileName'] = file_name_parsed

        #store in dict
        valdd = dict_value(fbest_conf_df, 'Val AUROC')

        ##Parse it
        rauc_df = compute_variable_aportation(fbest_conf_df, valdd, var = 'Val AUROC', savevar = 'rAUROC')
        rauc_df['experiment'] = experiment if experiment in ['Atrophy_36m','Fibrosis_36m'] else 'Atrophy|Fibrosis_36m'
        exp_l.append(rauc_df)

    #concat them
    rauc_exp_df = pd.concat(exp_l)

    #generate mean by Variables
    #by group of features (gof)
    rauc_gof = rauc_exp_df.groupby(['experiment','Variables']).mean().reset_index()
    rauc_gof.to_csv(os.path.join('inputs','mean_rAUC_per_group_of_features.tsv'),sep='\t')
    #by group of features & model
    rauc_m = rauc_exp_df.groupby(['experiment','Variables','Method']).mean().reset_index()
    rauc_m.to_csv(os.path.join('inputs','mean_rAUC_per_variables_and_models.tsv'),sep='\t')