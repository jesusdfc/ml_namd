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


yvar = sys.argv[1]

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

    def ncv_metrics(model_orig, N_SPLITS_INNER=6, N_SPLITS_OUTER=6):

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
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

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

    #PLOT DISTRIBUTION OF PREDICTIONS 
    def plotPredDistributions(ytest, ypred, title, conf_mat):

        #order by ypred
        sortedIndex = np.argsort(ypred)[::-1]
        ytest_ordered = ytest[sortedIndex]
        ypred_ordered = ypred[sortedIndex]

        #compute XOR to see the differences (and negate to have 1-> RIGHT, 0 -> WRONG)
        nxor = [int(not int(x)^int(y)) for x,y in zip(ytest_ordered,np.round(ypred_ordered))]

        print(f'Plotting {title}')
        #Plot
        fig, ax = plt.subplots(figsize=(15,15))
        ax = sns.scatterplot(x = range(len(ypred_ordered)), y = ypred_ordered, hue = nxor ,ax=ax, palette = 'deep')
        handles, labels  =  ax.get_legend_handles_labels()
        ax.legend(handles, ['Wrong', 'Right'], loc='upper right')
        ax.table(conf_mat,loc='upper right',colWidths = [0.02, 0.02], bbox = [0.8, 0.925, 0.05, 0.055])
        plt.plot([0, len(ypred_ordered)],[0.5, 0.5], 'k--', color = 'black')
        plt.savefig(f'figures/{yvar.lower()}_{variables_p}_{title}_pred_distribution_{yvar}_{variables_p}.pdf')
        
    def accByThreshold(real, predprobs, verbose=False):
        
        for th in [0, 0.05,0.15,0.35,0.45]:
            thr = []
            thp = []
            for x,y in zip(real,predprobs):
                if y>(0.5+th) or y<(0.5-th):
                    thr.append(x)
                    thp.append(y)
            if len(thr) and verbose:
                print(f"Using threshold {th}, the balanced accuracy is {bal_accuracy(thr,thp)} for nº of samples {len(thr)}")
            else:
                break

    def applyModel(X, y, selected_model, model, comb, no_feature_imp=False, dropout = False):

        def singleCV(Xsub,ysub,orig_model, no_feature_imp):

            #Initialize variables vector
            variables_imp = []

            #init ytest and ypred v
            ytest_v = []
            ypred_v = []
            ypredProb_v = []
            
            #TRAIN (This fold would be the Train/Validation, as we have already selected the model)
            CV = StratifiedKFold(n_splits=6, shuffle=True, random_state = 43)

            for train_index, test_index in CV.split(Xsub,ysub):
                
                Xtrain, Xtest = Xsub.iloc[train_index,:].to_numpy(), Xsub.iloc[test_index,:].to_numpy()
                ytrain, ytest = ysub[train_index].to_numpy(), ysub[test_index].to_numpy()
                
                #FIT
                _ = orig_model.fit(Xtrain,ytrain)

                if model=='svm':
                    if comb[2] == 'linear':
                        variables_imp.append(orig_model.coef_.flatten().tolist())
                    else:
                        print('No linear kernel, so no plot of feature importance')
                        no_feature_imp = True
                else:
                    variables_imp.append(orig_model.feature_importances_.tolist())

                #PREDICT
                ytest_v.append(ytest)
                ypredProb_v.append([1-x if x>y else y for x,y in orig_model.predict_proba(Xtest)])
                ypred_v.append(orig_model.predict(Xtest))

            accByThreshold(np.hstack(ytest_v), np.hstack(ypredProb_v))

            return np.hstack(ytest_v), np.hstack(ypredProb_v), np.hstack(ypred_v), no_feature_imp, variables_imp

        #init copy of the model
        orig_model = copy.deepcopy(selected_model)
        Xval, yval = copy.deepcopy(X), copy.deepcopy(y)
        keepSamplesIndex = []

        if dropout:
            for _ in range(5):
                Xval, yval = copy.deepcopy(X), copy.deepcopy(y)
                xindex, yindex = Xval.index, yval.index
                numIter = 3
                for i in range(numIter):
                    #print(f"\nNumIter: {i}")
                    ytest, ypredProb, ypred, no_feature_imp, variables_imp = singleCV(Xval, yval, orig_model, no_feature_imp)
                    nxor = [int(not int(x)^int(y)) for x,y in zip(ytest, ypred)]
                    Xval.index, yval.index = xindex, yindex
                    Xval, yval = Xval.loc[np.array(nxor)==1,], yval.loc[np.array(nxor)==1,]
                    xindex, yindex = Xval.index, yval.index
                    Xval.reset_index(inplace=True,drop=True)
                    yval.reset_index(inplace=True,drop=True)

                keepSamplesIndex.append(xindex.tolist())

            #keep the intersection
            keepSamplesIndexIntersect = list(r(lambda x,y: set(x).intersection(set(y)), keepSamplesIndex))
            print(f"Total intersection {keepSamplesIndexIntersect}")
            return keepSamplesIndexIntersect

        else:
            ytest, ypredProb, ypred, no_feature_imp, variables_imp = singleCV(Xval, yval, orig_model, no_feature_imp)
            return ytest, ypredProb, ypred, no_feature_imp, variables_imp

    #generate new balanced accuracy
    def bal_accuracy(ytest, ypred):
        return balanced_accuracy_score(ytest,np.round(ypred))

    def confmat(ytest, ypred):
        return confusion_matrix(ytest,np.round(ypred))

    #Load All combinations
    all_combs = pd.read_csv(os.getcwd()+f'/inputs/{yvar}_best_conf_{N_SPLITS_INNER}_{N_SPLITS_OUTER}.tsv', sep='\t')

    #SORT BY -> MAXIMUM VALAUROC
    bestConfValAUROC = all_combs.sort_values(by='Val AUROC',ascending=False).iloc[0,:]
   
    title = 'MaxVal AUROC'
    #Get total vars and only trues
    variables_d, variables_p = choose_variables(bestConfValAUROC['FileName'])

    #init variables importance
    variables_imp_d = {}

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
    if metrics_only:
        metrics_df = ncv_metrics(selected_model)
        return metrics_df
    
    #apply the model
    ytest_v, ypred_v, _, _, variables_imp = applyModel(X,y, selected_model, model, comb)

    #Plot distribution
    plotPredDistributions(np.hstack(ytest_v), np.hstack(ypred_v), title=title, conf_mat = confmat(np.hstack(ytest_v), np.hstack(ypred_v)))

    for i,var in enumerate(X.columns):
        variables_imp_d[var] = [x[i] for x in variables_imp]
    #
    #print('Generating Figure')
    #Plot
    fig, ax = plt.subplots(figsize=(15,15))
    _ = plt.xticks(rotation=90)
    ax = sns.boxplot(data = pd.DataFrame(variables_imp_d)[pd.DataFrame(variables_imp_d).mean(axis=0).sort_values(ascending=False).index],ax=ax)
    ax = sns.stripplot(data = pd.DataFrame(variables_imp_d)[pd.DataFrame(variables_imp_d).mean(axis=0).sort_values(ascending=False).index], ax = ax, color = "black")
    _ = plt.savefig(f'figures/{yvar.lower()}_{variables_p}_{title}_feature_imp_{yvar.lower()}_{variables_p}.pdf')

    
#Generate for a variable, the 3 best models according to ValBalAcc, ValAUROC and ValScore
plot_feature_distribution(yvar, metrics_only=True)

#generate table
atrophy = plot_feature_distribution(yvar='atrophy_36m', metrics_only=True)
fibrosis = plot_feature_distribution(yvar='fibrosis_36m', metrics_only=True)
healthy_ill = plot_feature_distribution(yvar='healthy_ill', metrics_only=True)

#save table
dftable = pd.concat([atrophy,fibrosis,healthy_ill])
dftable = dftable.drop(['Val AUAC','Val AURC'],axis=1)
dftable.reset_index(drop=True,inplace=True)
dftable.to_csv(os.path.join('inputs','best_combination_metrics.tsv'),sep='\t')
