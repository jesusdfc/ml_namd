import numpy as np
from itertools import product as p
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import linear_model
from  sklearn.linear_model import LogisticRegression as LR
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from random import random
from functools import reduce as r
import copy
import sys

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_feature_distribution(yvar='atrophy_36m', N_SPLITS_INNER=6, N_SPLITS_OUTER=6, metrics_only=False):

    #RUN FROM oftal_gridding DIRECTORY
    def load_data(drop_previous_ill=True, drop_extravars=False, drop_liquids=False,
              drop_foveolar=False, drop_vascular=False, drop_cristalin=False,
              drop_ETDRS = False, drop_SNPs=False, y_var='healthy_ill'):
        #
        clinic = pd.read_csv('inputs/FIS_dataset.txt',sep='\t')
        
        #Make group of variables to drop
        extravars = ['age','tabaquism', 'sex', 'hypertension', 'vitamins', 'hypercholesterolemia']
        liquids = ['intraretinal_fluid_b', 'subretinal_fluid_b', 'intraretinal_fluid_V4', 'subretinal_fluid_V4']
        foveolar_vars = ['foveal_thickness_b','foveal_thickness_V4']
        vascular_vars =['neovascular_membrane_b','neovascular_membrane_V4']
        cristalin_vars = ['cataract_b','cataract_V4','cataract_12m']
        ETDRs_vars = ['ETDRS_b','ETDRS_V4','ETDRS_12m']
        SNPs_vars = ['ARMS2', 'CFI', 'VEGFR', 'SMAD7']
        PrevAtrophFibr = ['atrophy_b','fibrosis_b','atrophy_V4','fibrosis_V4'] #This is always dropped
        Predicted = ['atrophy_36m','fibrosis_36m']
        #danger_vars = ['INYECCIONES_36m']

        #Get Y
        Y = clinic[Predicted]

        #Add the healthy/iill and the softmax problem
        Y.insert(2,'healthy_ill',Y['atrophy_36m']|Y['fibrosis_36m'])
        Y.insert(2,'softmax', Y['atrophy_36m']+Y['fibrosis_36m'])

        #Define drop vars
        drop_vars = []
        #
        #Atrophy_V4 in case we need it
        Atrophy_V4 = clinic['atrophy_V4']
        Fibrosis_V4 = clinic['fibrosis_V4']
        healthy_ill_drop = Atrophy_V4|Fibrosis_V4

        #Define the drop dictionary
        drop_dict = {'atrophy_36m': Atrophy_V4,
                    'fibrosis_36m': Fibrosis_V4,
                    'healthy_ill': healthy_ill_drop}

        if drop_extravars:
            drop_vars += extravars
        if drop_liquids:
            drop_vars += liquids
        if drop_foveolar:
            drop_vars += foveolar_vars
        if drop_vascular:
            drop_vars += vascular_vars
        if drop_cristalin:
            drop_vars += cristalin_vars
        if drop_ETDRS:
            drop_vars += ETDRs_vars
        if drop_SNPs:
            drop_vars += SNPs_vars
        # if drop_danger:
        #     drop_vars += danger_vars

        ##Drop also Atrophy and Fibrosis from previous time and current time (predicted variables).##
        drop_vars += PrevAtrophFibr
        drop_vars += Predicted

        #Drop variables and reset index
        clinic.drop(drop_vars,axis=1,inplace=True)
        clinic.reset_index(drop=True,inplace=True)
        #
        #Drop samples that previously had depending on the predicted variabel (ATROPHY,FIBROSIS,HEALTHY-ILL)
        if drop_previous_ill:
            prev_v4 = list(np.where(drop_dict[y_var]==1)[0])
            X = clinic.drop(prev_v4,axis=0)
            Y = Y.drop(prev_v4,axis=0)  

        #reset index
        X.reset_index(drop=True,inplace=True)
        Y.reset_index(drop=True,inplace=True)

        return X,Y[y_var]

    def choose_variables(variables_p):

        #convert
        variables_p = variables_p.split('.')[0].split('_')

        #create dict
        variables_d = {}

       #define var names
        var_names = ['extravars','liquids','foveolar_vars','vascular_vars','cristalin_vars','ETDRS_vars','SNPs_vars']

        #define only true
        onlyTrueNames = ''

        #fill it with True
        for i,x in enumerate(var_names):
            variables_d[x] = True if variables_p[2*i+1] =='True' else False

            if not variables_d[x]:
                onlyTrueNames += x.split('_')[0] + '_'

        return variables_d, onlyTrueNames[:-1]

    def choose_model(model_params, model_type):

        comb = model_params.split('_')
        #print(f'Before -> {comb}')
        #transform to int and 
        for i,_ in enumerate(comb):
            #check numeric
            if comb[i].replace('.','').isnumeric():
                #float
                if '.' in comb[i]:
                    comb[i] = float(comb[i])
                #int
                else:
                    comb[i] = int(comb[i])
            #check boolean
            elif 'True' == comb[i] or 'False' == comb[i]:
                comb[i] = bool(comb[i])

        #print(f'After -> {comb}')
        if model_type=='rf':
            #Generate model
            model_orig = RandomForestClassifier(n_estimators=comb[0],max_features=comb[1],max_depth=comb[2],
                                        min_samples_split=comb[3],min_samples_leaf=comb[4],bootstrap=comb[5],
                                        random_state=1)
        elif model_type=='svm':
            #Generate model
            model_orig = svm.SVC(C=comb[0], gamma=comb[1], kernel=comb[2], class_weight=comb[3], probability=True,
                                random_state=1)

        elif model_type=='xgboost':
            #Generate model
            model_orig = xgb.XGBClassifier(use_label_encoder=False,verbosity=0,nthread=1,
                                            min_child_weight=comb[0], gamma=comb[1], subsample=comb[2], 
                                            colsample_bytree=comb[3], max_depth=comb[4], 
                                            learning_rate=comb[5], n_estimators=comb[6], reg_alpha=comb[7], reg_lambda=comb[8],
                                            random_state = 1)

        return model_orig,comb

    def ncv_metrics(model_orig, N_SPLITS_INNER=6, N_SPLITS_OUTER=6, random_state = 42):

        def computingAUAC_AURC(real_np, probs):

            def selectMax(v):
                probs = []

                for zeroProb,onePro in v:
                    if zeroProb>onePro:
                        probs.append(0)
                    else:
                        probs.append(1)

                return probs
            
            #INIT THE AREA UNDER THE ACCURACY CURVE VECTOR
            auac = []

            #INIT THE AREA UNDER THE REJECTION CURVE VECTOR
            aurc = []

            #Init number of samples and max probabilities
            n = real_np.shape[0]
            max_probs = np.max(probs,axis=1)

            for th in range(10,20,1):

                #from 0.5 to 1
                dec_th = th/20

                #get index of true
                ind = [i for i,x in enumerate(max_probs) if x >= dec_th]

                if len(ind):
                    auac.append(accuracy_score(real_np[ind], selectMax(probs[ind])))
                    aurc.append(1-(len(ind)/n))
                else:
                    auac.append(0)
                    aurc.append(1)

            #Compute Area
            auacArea = np.mean(auac)
            aurcArea = np.mean(aurc)

            return auacArea, aurcArea

        #Select Max
        def selectMax(v):
            probs = []

            for zeroProb,onePro in v:
                if zeroProb>onePro:
                    probs.append(0)
                else:
                    probs.append(1)

            return probs
   
        n_splits_inner,n_splits_outer=N_SPLITS_INNER,N_SPLITS_OUTER
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)

        #results df
        results_df = pd.DataFrame()

        ## TEST ARRAYS
        test_real = []
        test_pred = []
        test_pred_prob = []
        test_bac = []
        test_ac = []
        test_auroc = []
    
        ## -------------------- OUTER LOOP, TRAIN AND TEST ---------------------
        for train_index_outer, test_index_outer in outer_cv.split(X,y):

            #Get train and test
            X_outer_train, X_outer_test = X.iloc[train_index_outer,:].to_numpy(), X.iloc[test_index_outer,:].to_numpy()
            y_outer_train, y_outer_test = y[train_index_outer].to_numpy(), y[test_index_outer].to_numpy()

            #Append the TEST
            test_real.append(y_outer_test)

            #save model outer
            model_outer = copy.deepcopy(model_orig)

            #Fit the model
            _ = model_outer.fit(X_outer_train,y_outer_train)

            #Predict the test, and predict the probabilities also
            test_pred_prob.append(model_outer.predict_proba(X_outer_test))
            test_pred.append(selectMax(test_pred_prob[-1]))

            #BALANCED ACCURACY TEST
            test_bac.append(balanced_accuracy_score(np.hstack(test_real), np.hstack(test_pred)))
            test_ac.append(accuracy_score(np.hstack(test_real), np.hstack(test_pred)))

            #TEST AUROC
            test_auroc.append(roc_auc_score(y_outer_test, model_outer.predict_proba(X_outer_test)[:,1]))

            ## VALIDATION ARRAYS
            val_real = []
            val_auac = []
            val_aurc = []
            val_bac = []
            val_auroc = []
            val_ac = []

            ## TRAIN ARRAYS
            train_bac = []
            train_acc = []
            train_auroc = []

            ## -------------------- INNER LOOP, TRAIN AND VALIDATION ---------------------
            for train_index_inner, test_index_inner in inner_cv.split(X_outer_train,y_outer_train):
    
                #Get the model
                model_inner = copy.deepcopy(model_orig)

                #Get Train and Validation
                X_inner_train, X_inner_validation = X_outer_train[train_index_inner,:], X_outer_train[test_index_inner,:]
                y_inner_train, y_inner_validation = y_outer_train[train_index_inner], y_outer_train[test_index_inner]

                #APPEND THE VALIDATION
                val_real.append(y_inner_validation)

                #Fit the model
                _ = model_inner.fit(X_inner_train,y_inner_train)

                #AREA UNDER THE ACCURACY AND REJECTION RATE CURVE
                auacArea, aurcArea = computingAUAC_AURC(y_inner_validation,model_inner.predict_proba(X_inner_validation))
                val_auac.append(auacArea)
                val_aurc.append(aurcArea)

                #AREA UNDER THE ROC CURVE
                val_auroc.append(roc_auc_score(y_inner_validation,model_inner.predict_proba(X_inner_validation)[:,1]))
                train_auroc.append(roc_auc_score(y_inner_train,model_inner.predict_proba(X_inner_train)[:,1]))

                #BALANCED ACCURACY (TRAIN and VALIDATION)
                val_bac.append(balanced_accuracy_score(y_inner_validation,model_inner.predict(X_inner_validation)))
                val_ac.append(accuracy_score(y_inner_validation,model_inner.predict(X_inner_validation)))
                train_bac.append(balanced_accuracy_score(y_inner_train,model_inner.predict(X_inner_train)))
                train_acc.append(accuracy_score(y_inner_train, model_inner.predict(X_inner_train)))

        results_row = [np.mean(train_bac), np.mean(val_bac), np.mean(test_bac), 
                       np.mean(train_acc), np.mean(val_ac), np.mean(test_ac),
                       np.mean(val_auac), np.mean(val_aurc), 
                       np.mean(train_auroc), np.mean(val_auroc), np.mean(test_auroc),
                       '_'.join(list(map(str,comb)))]

        #Make dataframe
        comb_df = pd.DataFrame(results_row, index= ['Train BalAcc', 'Val BalAcc', 'Test BalAcc',
                                                    'Train Accuracy', 'Val Accuracy', 'Test Accuracy', 
                                                    'Val AUAC', 'Val AURC', 
                                                    'Train AUROC', 'Val AUROC',  'Test AUROC',
                                                    'Comb']).T

        #concat
        results_df = pd.concat((results_df,comb_df))

        return results_df

    #Load All combinations
    all_combs = pd.read_csv(os.getcwd()+f'/inputs/{yvar}_best_conf_{N_SPLITS_INNER}_{N_SPLITS_OUTER}.tsv', sep='\t')

    #SORT BY -> MAXIMUM VALAUROC
    bestConfValAUROC = all_combs.sort_values(by='Val AUROC',ascending=False).iloc[0,:]
   
    #Get total vars and only trues
    variables_d, variables_p = choose_variables(bestConfValAUROC['FileName'])

    #Retrieve the data
    X,y = load_data(drop_extravars = variables_d['extravars'], drop_liquids = variables_d['liquids'],
                    drop_foveolar = variables_d['foveolar_vars'], drop_vascular = variables_d['vascular_vars'],
                    drop_cristalin = variables_d['cristalin_vars'], drop_ETDRS = variables_d['ETDRS_vars'],
                    drop_SNPs = variables_d['SNPs_vars'], y_var = yvar)
    
    print(f'Selected variables are {variables_p}')

    #Select the configuration
    model_params, model = bestConfValAUROC['Comb'], bestConfValAUROC['Model']

    #retrieve the model
    selected_model, comb = choose_model(model_params,model)

    #get metrics
    metrics = []
    for seed in [42, 7, 11, 128, 314]:
        metrics.append(ncv_metrics(selected_model, random_state=seed))

    return pd.concat(metrics)
    
    
#generate table
atrophy = plot_feature_distribution(yvar='atrophy_36m', metrics_only=True)
fibrosis = plot_feature_distribution(yvar='fibrosis_36m', metrics_only=True)
healthy_ill = plot_feature_distribution(yvar='healthy_ill', metrics_only=True)

#save table
dftable = pd.concat([healthy_ill, fibrosis, atrophy])
dftable = dftable.drop(['Val AUAC','Val AURC'],axis=1)
dftable.reset_index(drop=True,inplace=True)
dftable.to_csv(os.path.join('inputs','best_combination_metrics.tsv'),sep='\t')
