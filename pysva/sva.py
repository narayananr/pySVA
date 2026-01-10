
"""
SVA functions.
"""

import numpy as np

from scipy.linalg import svd


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



def extract_svs(residuals, n_sv):
    """
    Extract surrogate variables from residuals using SVD.
    
    Parameters
    ----------
    residuals : array (n_samples, n_genes)
        Residuals from get_residuals()
    n_sv : int
        Number of surrogate variables to extract
    
    Returns
    -------
    sv : array (n_samples, n_sv)
        Surrogate variables
    """
    U, S, Vt = svd(residuals, full_matrices=False)
    sv = U[:, :n_sv]
    
    return sv
