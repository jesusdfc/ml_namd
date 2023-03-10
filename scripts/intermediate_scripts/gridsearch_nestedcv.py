import pandas as pd
import numpy as np
from collections import Counter 
import seaborn as sns
import os
import copy
import sklearn
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import linear_model
from  sklearn.linear_model import LogisticRegression as LR
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import re
import functools as f
import pickle
import tqdm
import multiprocessing as mlp
from itertools import chain
import matplotlib
from matplotlib import pyplot as plt
from itertools import combinations as c
from itertools import product as p
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer,balanced_accuracy_score
from tqdm import tqdm
import sys
from collections import defaultdict
from sklearn.preprocessing import binarize
import warnings
from functools import reduce as r
warnings.filterwarnings('ignore') 
#pd.set_option('display.height', 500)
#pd.set_option('display.max_rows', 500)

WORKING_VAR = sys.argv[1]
print(f'Gridding over {WORKING_VAR}')

N_SPLITS_INNER,N_SPLITS_OUTER=6,6

#COMPUTE AREA UNDER THE ACURACCY CURVE AND REJECTION RATE CURVE
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

#RUN FROM oftal_gridding DIRECTORY
def load_data(drop_previous_ill=True, drop_extravars=False, drop_liquids=False,
              drop_foveolar=False, drop_vascular=False, drop_cristalin=False,
              drop_ETDRS = False, drop_SNPs=False, y_var='healthy_ill'):
    #
    clinic = pd.read_csv(os.getcwd()+'/inputs/FIS_dataset.txt',sep='\t')
    
    #Make group of variables to drop
    extravars = ['Edad','Tabaquismo', 'Sexo_femenino', 'Hipertension', 'Suplementos_vitaminicos', 'Hipercolesterolemia']
    liquids = ['Intraretinal_fluid_b', 'Subretinal_fluid_b', 'Intraretinal_fluid_V4', 'Subretinal_fluid_V4']
    foveolar_vars = ['grosor_foveolar_b','grosor_foveolar_V4']
    vascular_vars =['membrana_neovascular_b','membrana_neovascular_V4']
    cristalin_vars = ['estado_cristalino_b','estado_cristalino_V4','estado_cristalino_12m']
    ETDRs_vars = ['ETDRS_b','ETDRS_V4','ETDRS_12m']
    SNPs_vars = ['ARMS2', 'CFI', 'VEGFR', 'SMAD7']
    PrevAtrophFibr = ['Atrophy_b','Fibrosis_b','Atrophy_V4','Fibrosis_V4'] #This is always dropped
    Predicted = ['Atrophy_36m','Fibrosis_36m']
    #danger_vars = ['INYECCIONES_36m']

    #Get Y
    Y = clinic[Predicted]

    #Add the healthy/iill and the softmax problem
    Y.insert(2,'healthy_ill',Y['Atrophy_36m']|Y['Fibrosis_36m'])
    Y.insert(2,'softmax', Y['Atrophy_36m']+Y['Fibrosis_36m'])

    #Define drop vars
    drop_vars = []
    #
    #Atrophy_V4 in case we need it
    Atrophy_V4 = clinic['Atrophy_V4']
    Fibrosis_V4 = clinic['Fibrosis_V4']
    healthy_ill_drop = Atrophy_V4|Fibrosis_V4

    #Define the drop dictionary
    drop_dict = {'Atrophy_36m': Atrophy_V4,
                'Fibrosis_36m': Fibrosis_V4,
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

def nested_gridsearch(X,y,model_type='rf'):

    #Choose the model
    def choose_model(model_type):

        if model_type=='rf':
            ##-------------------RF-----------------------##
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 5)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 50, num = 3)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 5]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            return p(n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf,bootstrap)

        elif model_type=='svm':
            ##-------------------------SVM-----------------------##
            #Define the grid
            C = [0.1, 0.5, 1]
            gamma =  [1, 0.01]
            kernel = ['rbf', 'linear','sigmoid']
            class_weight = [None,'balanced']

            return p(C,gamma,kernel,class_weight)

        elif model_type=='xgb':

            ##----------------------------XGB----------------------##
            min_child_weight = [0.1, 5]
            gamma = [0.5, 1.5]
            subsample = [0.6, 1.0]
            colsample_bytree = [0.6, 1.0]
            max_depth = [int(x) for x in np.linspace(10, 50, num = 3)]
            learning_rate = [0.0001, 0.1]
            n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 5)]
            reg_alpha = [0.0001, 0.1 ]
            reg_lambda = [0.0001, 0.1]

            return p(min_child_weight, gamma, subsample, colsample_bytree, max_depth, learning_rate, n_estimators, reg_alpha, reg_lambda)

    #Select Max
    def selectMax(v):
        probs = []

        for zeroProb,onePro in v:
            if zeroProb>onePro:
                probs.append(0)
            else:
                probs.append(1)

        return probs
    
    #SPlit and KFOLD
    n_splits_inner,n_splits_outer=N_SPLITS_INNER,N_SPLITS_OUTER
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

    #results df
    results_df = pd.DataFrame()
    #
    for comb in tqdm(list(choose_model(model_type))):
   
        if model_type=='rf':
            #Generate model
            model_orig = RandomForestClassifier(n_estimators=comb[0],max_features=comb[1],max_depth=comb[2],
                                        min_samples_split=comb[3],min_samples_leaf=comb[4],bootstrap=comb[5],
                                        random_state=1)
        elif model_type=='svm':
            #Generate model
            model_orig = svm.SVC(C=comb[0], gamma=comb[1], kernel=comb[2], class_weight=comb[3], probability=True,
                                random_state=1)

        elif model_type=='xgb':
            #Generate model
            model_orig = xgb.XGBClassifier(use_label_encoder=False,verbosity=0,nthread=1,
                                            min_child_weight=comb[0], gamma=comb[1], subsample=comb[2], 
                                            colsample_bytree=comb[3], max_depth=comb[4], 
                                            learning_rate=comb[5], n_estimators=comb[6], reg_alpha=comb[7], reg_lambda=comb[8],
                                            random_state = 1)

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

        results_row = [np.mean(train_bac), np.mean(val_bac), np.mean(test_bac), 
                       np.mean(val_ac), np.mean(test_ac),
                       np.mean(val_auac), np.mean(val_aurc), 
                       np.mean(train_auroc), np.mean(val_auroc), np.mean(test_auroc),
                       '_'.join(list(map(str,comb)))]

        #Make dataframe
        comb_df = pd.DataFrame(results_row, index= ['Train BalAcc', 'Val BalAcc', 'Test BalAcc',
                                                    'Val Accuracy', 'Test Accuracy', 
                                                    'Val AUAC', 'Val AURC', 
                                                    'Train AUROC', 'Val AUROC',  'Test AUROC',
                                                    'Comb']).T

        #concat
        results_df = pd.concat((results_df,comb_df))

    return results_df

##----------------------DATA-----------------------##
def dataset_comb_computation(extravars, liquids, foveolar_vars, vascular_vars, cristalin_vars, ETDRs_vars, SNPs_vars):
    #
    #extravars,liquids,foveolar_vars,vascular_vars,cristalin_vars,ETDRs_vars = comb_v
    #
    save_p = '_'.join([x+'_'+str(y) for x,y in zip(['extravars','liquids','foveolar','vascular','cristalin','ETDRS','SNPs'],
                                                    [extravars,liquids,foveolar_vars,vascular_vars,cristalin_vars,ETDRs_vars,SNPs_vars])])
    #print(extravars,liquids,foveolar_vars,ETDRs_vars)
    #
    X,y = load_data(drop_extravars = extravars, drop_liquids = liquids,
            drop_foveolar = foveolar_vars, drop_vascular = vascular_vars, drop_cristalin = cristalin_vars,
            drop_ETDRS = ETDRs_vars, drop_SNPs = SNPs_vars, y_var = WORKING_VAR)
    #check if it exists
    if not os.path.isfile(os.getcwd()+f'/outputs/nested_cv_{N_SPLITS_INNER}_{N_SPLITS_OUTER}/{WORKING_VAR}/rf/'+save_p+'.pickle'):
        print('rf')
        #Do the RF grid
        best_params_rf = nested_gridsearch(X,y,model_type='rf')
        #
        #RF
        with open(os.getcwd()+f'/outputs/nested_cv_{N_SPLITS_INNER}_{N_SPLITS_OUTER}/{WORKING_VAR}/rf/'+save_p+'.pickle', 'wb') as handle:
            pickle.dump(best_params_rf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    
    if not os.path.isfile(os.getcwd()+f'/outputs/nested_cv_{N_SPLITS_INNER}_{N_SPLITS_OUTER}/{WORKING_VAR}/xgboost/'+save_p+'.pickle'):
        print('xgb')
        #Do the XGB grid
        best_params_xgb = nested_gridsearch(X,y,model_type='xgb')
        #
        #XGB
        with open(os.getcwd()+f'/outputs/nested_cv_{N_SPLITS_INNER}_{N_SPLITS_OUTER}/{WORKING_VAR}/xgboost/'+save_p+'.pickle', 'wb') as handle:
            pickle.dump(best_params_xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    if not os.path.isfile(os.getcwd()+f'/outputs/nested_cv_{N_SPLITS_INNER}_{N_SPLITS_OUTER}/{WORKING_VAR}/svm/'+save_p+'.pickle'):
        print('svm')
        #Do the SVM grid
        best_params_svm = nested_gridsearch(X,y,model_type='svm')
        #
        #SVM
        with open(os.getcwd()+f'/outputs/nested_cv_{N_SPLITS_INNER}_{N_SPLITS_OUTER}/{WORKING_VAR}/svm/'+save_p+'.pickle', 'wb') as handle:
            pickle.dump(best_params_svm, handle, protocol=pickle.HIGHEST_PROTOCOL)

def grid_models_data():
    #
    #generate combination of these 4
    extravars_c,liquids_c,foveolar_vars_c,vascular_vars_c,cristalin_vars_c,ETDRs_vars_c,SNPs_vars_c = [False,True],[False,True],[False,True],[False,True],[False,True],[False,True],[False,True]
    #Compute all the combinations
    tot_combs = list(map(len,[extravars_c,liquids_c,foveolar_vars_c,vascular_vars_c,cristalin_vars_c,ETDRs_vars_c,SNPs_vars_c]))
    tot_combs_n = r(lambda a,b:a*b,tot_combs)
    print(f'Generating {tot_combs_n} combinations')
    #generate all the combinations (DATA)
    comb_v = p(extravars_c,liquids_c,foveolar_vars_c,vascular_vars_c,cristalin_vars_c,ETDRs_vars_c,SNPs_vars_c)
    #drop the first one (empty dataset)
    comb_v_l = list(comb_v)[:-1]

    #number of cores (from 6 to 80% of the maximum)
    ncores = max(6,int(mlp.cpu_count()*(0.8)))
    
    #Multicore!
    with mlp.Pool(processes=ncores) as pool:
        _ = pool.starmap(dataset_comb_computation, comb_v_l)

#MAIN
#grid_models_data()
