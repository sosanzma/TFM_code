# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import networkx as nx


# %% window generator

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None,
               label_columns=None,
               batch_size = None):
    # Store the raw data.
    self.train_df = train_df
    
    self.batch_size = batch_size

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}
    
    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])



# this function will split the data attending the window defined above

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


# we need a time series format foor feed the lstm
def make_dataset(self, data,batch_size ):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=batch_size,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df, self.batch_size)




WindowGenerator.train = train




# %% main function 

def lstm_cause(train_df,
               sens : float = 0.0,
               dropout : float = 0.1,
               maxlag: int = 1,
               batch_size : int = 64 ,
               lstm_neurons = None,
               epochs : int = 100,
               loss : str = "mse",
               graph : bool = False,
               patience : int = 8,
               noise : float = 0.05): 
    
    assert maxlag > 0
  
    T, N = train_df.shape
    train_df =pd.DataFrame(train_df)
    
    if lstm_neurons is None:
        lstm_neurons = int(round(maxlag+N/2,0))
    
  
    val_matrix = np.zeros((N, N))
    
    for j in range (N):   
             
        print(f"Calculating causality for serie {j} ... ")
        
        # build the network
        model = keras.Sequential()
        model.add(Dense(N,activation='sigmoid'))
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
        



   
        # now we compute the gradients for substract the importance of each variable in the forecasing
        grad = []
        for element in window.train:
            x = tf.constant(element[0])
            # we compute de gradient of y with  x
            with tf.GradientTape() as t:   
                t.watch(x)
                y = model(x)
            
            
            dy_dx = t.gradient(y, x)
            # mean of the absolute value of all the gradients of the 62 batch of 8 samples
            dy_dx = np.sum(np.abs(dy_dx.numpy()),axis=0)/len(dy_dx)
            # if the mean of all the batches is lower than sens value -> 0
            dy_dx[dy_dx < noise] =0  
            #sum of each lag contribution :
            dy_dx = np.sum(dy_dx,axis = 0) 
        
            grad.append(dy_dx)
            
        val_matrix[:,j] =  np.mean(grad,axis = 0)
        print( np.mean(grad,axis = 0))

    # if the contribution of each gradient is lower than sens value -> 0      
    #val_matrix[val_matrix < sens] = 0 
    val_matrix = normalize(val_matrix, norm='max', axis=0)
    #print(lstm_model.summary())
    
    if graph :

        # we take a round value of the val_matrix  
        G = np.round(val_matrix-noise,1)
        G
        G =  np.array(G, dtype=bool)*1
        G = nx.from_numpy_matrix(G, create_using=nx.DiGraph)


        # Draw the graph using , we  want node labels.
        nx.draw(G,with_labels=True)
        plt.title(f"Graph causal relationships with {lstm_neurons} LSTM neurons")
        plt.show()
    
    return val_matrix




def statisticals(tlabels, plabels, var= None,lstm = None ):
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
    tlabels =  np.array(tlabels) 
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
               "auc" : auc ,"aupr" : aupr, "fpr" : fpr, "fnr" : fnr, "variables" : var,"lstm" : lstm}
  
    return(results)




#%% Calculos 

res = []
df = []
ground_truth = []

var = [5,10,15,20,30,40]
lstm = [2,5,10,15,20,40]

# hasta variables : 50 , LSTM = 4. // tenemos que redefinir lstm


# terminal 1

#lstm = [2,4,6,8,10,12,14]

# terminal 2. Tengo que compilarlo todo otra vez.. 




# Ahora vamos a --> AQUI


#a = pd.read_csv(f"Datos_maxlag5maxp4var5.csv")



# Leemos los datos : 

for v in var: 
    df.append(pd.read_csv(f"Datos_maxlag5maxp4var{v}.csv"))
    ground_truth.append(pd.read_csv(f"Ground_truth_maxlag5maxp4var{v}.txt",sep = ",",header=None))




# corregimos el los gorund truths sumandole la identidad

for i in  range(len(var)):
    F,C = ground_truth[i].shape
    ground_truth[i] = ground_truth[i] + np.identity(F)


for k in range(len(var)):
 
    #print(f"calculating: variables:  {var[k]}, LST : {i} ")º
    df[k].columns = np.arange(0,var[k]).tolist()
    df[k] = df[k].to_numpy()


## AQUÍ
for i in lstm:
    for k in range(len(var)):
     
        print(f"calculating: padres:  {padres[k]}, LST : {i} ")

        val_matrix= lstm_cause(df[k],maxlag = 5, lstm_neurons = i)
    
        res.append(statisticals(ground_truth[k], val_matrix, var = var[k], lstm = i))


#%% save it 

df = pd.DataFrame(res)


df.to_csv("resultados_exp_lstm_norm.csv", index= False)


# %% plot results 

df = pd.read_csv('resultados_exp_lstm_norm.csv')


# import old results to compare 

df_1 = pd.read_csv('resultados4_10.csv')
df_2 = pd.read_csv('resultados1_10.csv')


df_T2 = pd.concat([df_2,df_1])

import seaborn as sns


df_T2.rename({'Variables': 'variables'}, axis=1, inplace=True)

extra = []
for i in range(len(df_T2)):
    extra.append("No Normalized")
    
df_T2['extra'] = extra 

df_T2

df_T = df_T2


df_T = pd.concat([df,df_T2])




df_T.rename({'extra': 'type'}, axis=1, inplace=True)



n=len(df_T.columns)



n = 4
fig,ax = plt.subplots(n,1, figsize=(7,n*3), sharex=True)

i=0
for  col in df_T.columns:
    
    if (col ==  'f_score' or col == 'auc' or col == 'aupr' or col =='fpr'):
        plt.sca(ax[i])
        g = sns.scatterplot(data=df_T, x='lstm', y=col, hue='variables', style = 'type',palette=['r', 'g', 'b','y','k','brown'])
        g.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), ncol=1, title = 'Variables')
        i = i+1
        



