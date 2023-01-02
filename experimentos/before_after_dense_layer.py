# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:29:30 2022

@author: 34695
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd



# %% start
train_df = experiments[4,:,:]

T, N = train_df.shape

dropout = 0.1

patience = 8

lstm_neurons = 10

maxlag = 25

epochs = 200

train_df=  pd.DataFrame(train_df)

batch_size = 16

j = 5

# build the network
model = keras.Sequential()
model.add(Dense(N,activation='linear'))
model.add(LSTM(lstm_neurons, return_sequences=False,
                recurrent_activation="tanh", dropout=dropout))

model.add(Dense(1,activation = 'linear'))


# define our window
window = WindowGenerator(train_df=train_df,batch_size = batch_size,
    input_width=maxlag, label_width=1, shift=1,label_columns = [j])
       

        
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                 patience=patience,
                                                  mode='min',
                                                  restore_best_weights=False)



model.compile(loss=tf.losses.MeanSquaredError(),
               optimizer=tf.optimizers.Adam(),
               metrics=[tf.metrics.MeanAbsoluteError()])


model.fit(window.train, epochs=epochs,verbose = 0, callbacks =[early_stopping])




# obtenemos graficamente como salen las series temporales despues de pasarlas por la capa densa


# para llegar aquí tenemos que entrenar un modelo para la regresion de una de las columnas



from keras import backend as K

get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])

layer_output = []
datos_norm = []
for element in window.train: 
    
    layer_output.append(get_layer_output([element[0]])[0])
    datos_norm.append(element[0].numpy())


n = 7
fig, ax = plt.subplots(n,figsize=(14, 14), dpi=100)
fig.suptitle('Time series',fontsize = 20)
fig.legend()

for i in range(0,n):
    s1 = layer_output[0][0,:,i]
    
    s2 = datos_norm[0][0,:,i]
    
    
    print(i)
    #plt.plot(ax[i])
    ax[i].plot(s1, "black", label = 'after dense layer')
    #ax[i].plot(s2, "red", label = 'before dense layer')
ax[n-3].legend(loc = 'center right',bbox_to_anchor=(1.1, 0.5), fontsize = 12)
#plt.subplots_adjust(right=0.9)


# %% Matriz de pesos 

fig, ax = plt.subplots(1,1, figsize=(8, 8))
fig.suptitle('Matrix of dense layer weights',fontsize = 15)
img = plt.imshow(a)
fig.colorbar(img)

for i in range(10):
    for j in range(10):
        text = ax.text(j, i, np.round(a[i, j],2),
                       ha="center", va="center", color="w")
fig.tight_layout()
plt.show()



# %% Statistics 

def statisticals(tlabels, plabels, dense = None ):
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
    p1labels = np.array(np.round(plabels,2))
    tlabels =  np.array(tlabels,dtype=bool) *1
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
    
    fpr, tpr, thresholds = metrics.roc_curve(tlabels.reshape(-1), p1labels.reshape(-1))
    auc= metrics.auc(fpr, tpr)
    
    
    # AUPR 
    
    aupr = metrics.average_precision_score(tlabels.reshape(-1), p1labels.reshape(-1))
    
    # The false positive rate is  fpr = fp / (fp+tn)
    
    fpr = np.sum(false_positives) /(np.sum(false_positives) + np.sum(true_negatives))
    
    # The false negative rate is fnr = fn / (fn + tp) 
    
    fnr = np.sum(false_negatives)  / (np.sum(false_negatives) + np.sum(true_positives))
    
    # Dictionary with all the results
    results =	{"precision" : precision, "recall" : recall, "f_score" : f_score,
               "auc" : auc ,"aupr" : aupr, "fpr" : fpr, "fnr" : fnr, "Dense" : dense}
  
    return(results)



#%% AUC and     



for i in range(20):
    val_matrix_dense = lstm_cause(experiments[i,:,:],maxlag = 6, graph = False, sens = 0.)
    res.append(statisticals(ground_truths, val_matrix_dense, dense ="SI"))



df1 =pd.DataFrame(res)
df2 = pd.DataFrame(res2)


df1.to_csv("stats_dense_vs_nodense.csv", index= False)



#%%  Sacamos la comparativa
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("stats_dense_vs_nodense.csv")


df_dense = df[df['Dense'] == "SI"]
df_no_dense = df[df['Dense'] == "NO"]

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

df['Medidas'] = x 
n = 2
fig,ax = plt.subplots(n,1, figsize=(7,n*3), sharex=True)
fig.suptitle('')

i=0
for  col in df.columns:
    
    if (col == 'auc' or col == 'aupr'):
        plt.sca(ax[i])
        g = sns.scatterplot(data=df, x='Medidas', y=col, hue='Dense')
        g.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), ncol=1, title = 'Capa Densa')
        i = i+1
        



