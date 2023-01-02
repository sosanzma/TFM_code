"""
This file must contain a function called my_method that triggers all the steps 
required in order to obtain

 *val_matrix: mandatory, (N, N) matrix of scores for links
 *p_matrix: optional, (N, N) matrix of p-values for links; if not available, 
            None must be returned
 *lag_matrix: optional, (N, N) matrix of time lags for links; if not available, 
              None must be returned

Zip this file (together with other necessary files if you have further handmade 
packages) to upload as a code.zip. You do NOT need to upload files for packages 
that can be imported via pip or conda repositories. Once you upload your code, 
we are able to validate results including runtime estimates on the same machine.
These results are then marked as "Validated" and users can use filters to only 
show validated results.

Shown here is a vector-autoregressive model estimator as a simple method.
"""

import numpy as np
import statsmodels.tsa.api as tsa

# Your method must be called 'my_method'
# Describe all parameters (except for 'data') in the method registration on CauseMe
def my_method(data, maxlags=1, correct_pvalues=True):

    # Input data is of shape (time, variables)
    T, N = data.shape

    # Standardize data
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Fit VAR model and get coefficients and p-values
    tsamodel = tsa.var.var_model.VAR(data)
    results = tsamodel.fit(maxlags=maxlags,  trend='nc')
    pvalues = results.pvalues
    values = results.coefs

    # CauseMe requires to upload a score matrix and
    # optionally a matrix of p-values and time lags where
    # the links occur

    # In val_matrix an entry [i, j] denotes the score for the link i --> j and
    # must be a non-negative real number with higher values denoting a higher
    # confidence for a link.
    # Fitting a VAR model results in several lagged coefficients for a
    # dependency of j on i.
    # Here we pick the absolute value of the coefficient corresponding to the
    # lag with the smallest p-value.
    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    for j in range(N):
        for i in range(N):

            # Store only values at lag with minimum p-value
            tau_min_pval = np.argmin(pvalues[
                                    (np.arange(1, maxlags+1)-1)*N + i , j]) + 1
            p_matrix[i, j] = pvalues[(tau_min_pval-1)*N + i , j]

            # Store absolute coefficient value as score
            val_matrix[i, j] = np.abs(values[tau_min_pval-1, j, i])

            # Store lag
            lag_matrix[i, j] = tau_min_pval

    # Optionally adjust p-values since we took the minimum over all lags 
    # [1..maxlags] for each i-->j; should lead to an expected false positive
    # rate of 0.05 when thresholding the (N, N) p-value matrix at alpha=0.05
    # You can, of course, use different ways or none. This will only affect
    # evaluation metrics that are based on the p-values, see Details on CauseMe
    if correct_pvalues:
        p_matrix *= float(maxlags)
        p_matrix[p_matrix > 1.] = 1.

    return val_matrix, p_matrix, lag_matrix


if __name__ == '__main__':

    # Simple example: Generate some random data
    data = np.random.randn(100000, 3)

    # Create a causal link 0 --> 1 at lag 2
    data[2:, 1] -= 0.5*data[:-2, 0]

    # Estimate VAR model
    vals, pvals, lags = my_method(data, maxlags=3)

    # Score is just absolute coefficient value, significant p-value is at entry 
    # (0, 1) and corresponding lag is 2
    print(vals.round(2))
    print(pvals.round(3))
    print(pvals < 0.0001)
    print(lags)
