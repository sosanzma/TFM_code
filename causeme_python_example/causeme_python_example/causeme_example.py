"""
This script can be used to iterate over the datasets of a particular experiment.
Below you import your function "my_method" stored in the module causeme_my_method.

Importantly, you need to first register your method on CauseMe.
Then CauseMe will return a hash code that you use below to identify which method
you used. Of course, we cannot check how you generated your results, but we can
validate a result if you upload code. Users can filter the Ranking table to only
show validated results.
"""

# Imports
import numpy as np
import json
import zipfile
import bz2
import time

#from causeme_my_method import my_method

from lstm_method2 import lstm_cause

from  elasnet_method import var_elasnet

from nn_method import nn_cause2

# Setup a python dictionary to store method hash, parameter values, and results
results = {}

################################################
# Identify method and used parameters
################################################

# Method name just for file saving
method_name = 'NN'

# Insert method hash obtained from CauseMe after method registration
# nn
results['method_sha'] = "52e90c3d119e4c4eb2921d58d3466e57"

# LSTM 
#results['method_sha'] = "37a62b74b4b3444590949a53fda400c6"

# Elasnet 
#results['method_sha'] = "8bf2e0fa3c144e18858a25b0bdf8ef09"

# The only parameter here is the maximum time lag
maxlags =2
lstm_neurons = 100
epochs = 100




# Parameter values: These are essential to validate your results
# provided that you also uploaded code
results['parameter_values'] = f"maxlags={maxlags} ,epochs = {epochs}" #, lstm_neurons = {lstm_neurons}, epochs = {epochs}"


#################################################
# Experiment details
#################################################
# Choose model and experiment as downloaded from causeme

results['model'] = 'TestCLIMnoise'
#results['model'] = 'nonlinear-VAR'
results['model'] = 'TestCLIMnonstat'
results['model'] = 'nonlinear-VAR'

# Here we choose the setup with N=3 variables and time series length T=150
experimental_setup = 'N-40_T-100'
experimental_setup = 'N-40_T-250'
experimental_setup = 'N-5_T-600'
#experimental_setup = 'N-10_T-300'
results['experiment'] = results['model'] + '_' + experimental_setup

# Adjust save name if needed
save_name = '{}_{}_{}'.format(method_name,
                              results['parameter_values'],
                              results['experiment'])

# Setup directories (adjust to your needs)
experiment_zip = 'experiments/%s.zip' % results['experiment']
results_file = 'results/%s.json.bz2' % (save_name)

#################################################

# Start of script
scores = []
pvalues = []
lags = []
runtimes = []

# (Note that runtimes on causeme are only shown for validated results, this is more for
# your own assessment here)

# Loop over all datasets within an experiment
# Important note: The datasets need to be stored in the order of their filename
# extensions, hence they are sorted here
print("Load data")
with zipfile.ZipFile(experiment_zip, "r") as zip_ref:
    for name in sorted(zip_ref.namelist()):

        print("Run {} on {}".format(method_name, name))
        data = np.loadtxt(zip_ref.open(name))

        # Runtimes for your own assessment
        start_time = time.time()

        # Run your method (adapt parameters if needed)

        val_matrix  = nn_cause2(data, maxlags = maxlags, epochs = epochs)
                                # lstm_neurons = lstm_neurons, epochs = epochs)
        runtimes.append(time.time() - start_time)

        # Now we convert the matrices to the required format
        # and write the results file
        scores.append(val_matrix.flatten())

        # pvalues and lags are recommended for a more comprehensive method evaluation,
        # but not required. Then you can leave the dictionary field empty          
        #if p_matrix is not None: pvalues.append(p_matrix.flatten())
        #if lag_matrix is not None: lags.append(lag_matrix.flatten())

# Store arrays as lists for json
results['scores'] = np.array(scores).tolist()
if len(pvalues) > 0: results['pvalues'] = np.array(pvalues).tolist()
if len(lags) > 0: results['lags'] = np.array(lags).tolist()
results['runtimes'] = np.array(runtimes).tolist()



# Save data
print('Writing results ...')
results_json = bytes(json.dumps(results), encoding='latin1')
with bz2.BZ2File(results_file, 'w') as mybz2:
    mybz2.write(results_json)
    

