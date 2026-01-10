
"""
SVA functions.
"""

import numpy as np


def get_residuals(Y, X):
    """
    Remove effect of X from Y.
    
    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Expression matrix
    X : array (n_samples, n_covariates)
        Primary model matrix
    
    Returns
    -------
    residuals : array (n_samples, n_genes)
        Y with X effects removed
    """
    beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
    Y_hat = X @ beta_hat
    residuals = Y - Y_hat
    
    return residuals

