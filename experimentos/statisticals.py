# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:53:44 2022

@author: 34695
"""


def statisticals(tlabels, plabels, samples= None,lstm = None ):
    """

    Parameters
    ----------
    tlabels : matrix
        True labels.
    plabels : matrix
        Predicted labels.

    Returns
    -------
    coonfusion matrix.

    """
    from sklearn import metrics
    
    tlabels =  np.array(tlabels, dtype=bool)*1 
    plabels =  np.array(plabels, dtype=bool)*1
    
    diff = tlabels - plabels
    
    false_positives =  np.where(diff > 0, 0, diff)*-1
    true_positives = np.where((2*tlabels - plabels) != 1,0,2*tlabels - plabels)
    false_negatives  =np.where((2*tlabels - plabels) != 2,0,2*tlabels - plabels)/2
    true_negatives = np.where((tlabels + plabels+1) != 1,0,tlabels - plabels+1)
    #precision is the ratio tp / (tp + fp)
    
    precision = np.sum(true_positives)/(np.sum(true_positives)+np.sum(false_positives))
    
    # The recall is the ratio tp / (tp + fn)
    recall = np.sum(true_positives)/(np.sum(true_positives)+np.sum(false_negatives))
    
    # F score harmonic mean of precision and recall
    f_score = 2*precision*recall/(precision+recall)
    
    # AUC
    
    fpr, tpr, thresholds = metrics.roc_curve(tlabels.reshape(-1), plabels.reshape(-1))
    auc= metrics.auc(fpr, tpr)
    
    
    # AUPR 
    
    aupr = metrics.average_precision_score(tlabels.reshape(-1), plabels.reshape(-1))
    
    # The false positive rate is  fpr = fp / (fp+tn)
    
    fpr = np.sum(false_positives) /(np.sum(false_positives) + np.sum(true_negatives))
    
    # The false negative rate is fnr = fn / (fn + tp) 
    
    fnr = np.sum(false_negatives)  / (np.sum(false_negatives) + np.sum(true_positives))
    
    # Dictionary with all the results
    results =	{"precision" : precision, "recall" : recall, "f_score" : f_score,
               "auc" : auc ,"aupr" : aupr, "fpr" : fpr, "fnr" : fnr, "samples" : samples,"lstm" : lstm}
  
    return(results)



#res = statisticals(ground_truths, val_mat_nn_1_3)

res_lstm = statisticals(ground_truths, val_mat_nn_1_3)


# %% TFM 


"""

We want to find if there is a correlation between the number of LSTM neurons and the size of the data

What we are going to do ?  

We are going to test for a different numbers of neurons [2,5,8,12,18,25,40,50,70] 
and for different N_samples :[100,300,500,700,9000,1200,1500] . We will save all o the statistics
and with them we will be able to find if the is some relations.

"""


N_LSTM = [[20,10], [100,5], [30,10], [20,2], [100,50,20,5]] 

N_samples = [50,100,200,300,400,500]

res = []


K = 20
for sam in N_samples:
    for lstm in N_LSTM:

        for k in range(K):
            print(f"calculating: sample:  {sam}, LST : {lstm} iteration :{k}")

            val_matrix = (nn_cause2(experiments[k,:sam,:],maxlag = 5,Dense_neurons =lstm , sens = 0.2))

            res.append(statisticals(ground_truths, val_matrix, samples = sam, lstm = lstm))





#%% step 2

##############

# now we should do the average of the K measures of each combination of N_lstm and N_samples

###############




# empty data frame
df_means = pd.DataFrame(df, index = [] )



# or maybe this works : 
    
df = pd.DataFrame(res)
 
# lista que lo almacenara
prob = []

df = df.dropna()
for i in df.index:
    df["lstm"][i]= str(df["lstm"][i])
    
# this should works
for sam in N_samples:
    for lstm in N_LSTM:
        
        mean_values = df[(df["samples"] == sam) & (df["lstm"] == str(lstm))]
        
        prob.append(np.mean(mean_values,axis = 0 ))
        
 
 #    
df_means2 = pd.DataFrame(prob)
    
df_means2 = df_means2.dropna()

df.to_csv("datos_mlp_.csv", index= False)

import pandas as pd
df = pd.read_csv('datos_mlp_.csv')


import numpy as np
import matplotlib.pyplot as plt



import seaborn as sns

# for auc : espero obtener mejores resultadoscundo tenga los datos correctamente 
neurons = []

for j in range(6):
    for i in N_LSTM:
        neurons.append(str(i))

df_means2["neurons"] = neurons



n=len(df_means2.columns)
fig,ax = plt.subplots(n-2,1, figsize=(6,n*2), sharex=True)

for i, col in enumerate(df.columns):
    if not (col ==  'lstm' or col == 'sample'):
        plt.sca(ax[i])
        g = sns.scatterplot(data=df_means2, x='neurons', y=col, hue='samples',  palette=['r', 'g', 'b','y','k','orange'])
        g.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), ncol=1)
