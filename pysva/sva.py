
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



def estimate_n_sv(residuals, n_perm=20):
    """
    Estimate number of surrogate variables using permutation.
    
    Parameters
    ----------
    residuals : array (n_samples, n_genes)
        Residuals from get_residuals()
    n_perm : int
        Number of permutations for null distribution
    
    Returns
    -------
    n_sv : int
        Estimated number of surrogate variables
    """
    n_samples, n_genes = residuals.shape
    
    # Get observed singular values
    U, S_obs, Vt = svd(residuals, full_matrices=False)
    
    # Build null distribution by permuting each column
    S_null = np.zeros((n_perm, len(S_obs)))
    for p in range(n_perm):
        residuals_perm = residuals.copy()
        for j in range(n_genes):
            residuals_perm[:, j] = np.random.permutation(residuals_perm[:, j])
        _, S_perm, _ = svd(residuals_perm, full_matrices=False)
        S_null[p, :] = S_perm
    
    # Count how many observed SVs exceed 95th percentile of null
    threshold = np.percentile(S_null, 95, axis=0)
    n_sv = int(np.sum(S_obs > threshold))
    
    return n_sv

from scipy import stats


def identify_null_genes(Y, X, sv, alpha=0.25):
    """
    Identify genes not affected by primary variable.
    
    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Expression matrix
    X : array (n_samples, n_covariates)
        Primary model matrix
    sv : array (n_samples, n_sv)
        Current SV estimate
    alpha : float
        P-value threshold (genes with p > alpha are null)
    
    Returns
    -------
    null_genes : array of bool (n_genes,)
        True if gene is likely null
    """
    n_samples, n_genes = Y.shape
    
    # Combine X and SV
    X_full = np.column_stack([X, sv])
    n_params = X_full.shape[1]
    
    pvals = np.zeros(n_genes)
    
    for j in range(n_genes):
        y = Y[:, j]
        
        # Fit full model
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
        resid = y - X_full @ beta
        
        # Standard error for treatment effect (column 1)
        mse = np.sum(resid ** 2) / (n_samples - n_params)
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
        se = np.sqrt(mse * XtX_inv[1, 1])
        
        # t-test
        t_stat = beta[1] / se
        pvals[j] = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - n_params))
    
    null_genes = pvals > alpha
    
    return null_genes

def sva(Y, X, n_sv=None):
    """
    Surrogate Variable Analysis.
    
    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Expression matrix
    X : array (n_samples, n_covariates)
        Primary model matrix
    n_sv : int, optional
        Number of surrogate variables. If None, estimate automatically.
    
    Returns
    -------
    sv : array (n_samples, n_sv)
        Estimated surrogate variables
    """
    # Step 1: Get residuals
    residuals = get_residuals(Y, X)
    
    # Step 2: Estimate n_sv if not provided
    if n_sv is None:
        n_sv = estimate_n_sv(residuals)
    
    # Step 3: Extract surrogate variables
    sv = extract_svs(residuals, n_sv)
    
    return sv

