
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


def center_and_scale(Y, scale=False):
    """
    Center and optionally scale data.
    
    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Data matrix
    scale : bool
        If True, scale to unit variance.
        If False (default), only center.
        
    Returns
    -------
    Y_processed : array
    """
    means = np.mean(Y, axis=0)
    centered = Y - means
    
    if scale:
        stds = np.std(Y, axis=0, ddof=1)
        stds[stds == 0] = 1.0 # Avoid div/0
        return centered / stds
        
    return centered



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



    return n_sv

def estimate_n_sv_bic(residuals, max_sv=10):
    """
    Estimate number of surrogate variables using BIC.
    
    Parameters
    ----------
    residuals : array (n_samples, n_genes)
        Residuals
    max_sv : int
        Maximum number of SVs to test
        
    Returns
    -------
    n_sv : int
        Estimated number of SVs
    """
    n_samples, n_genes = residuals.shape
    
    # SVD
    U, S, Vt = svd(residuals, full_matrices=False)
    
    # Total variance
    total_var = np.sum(residuals**2)
    
    bics = []
    
    # Test k from 0 to max_sv
    limit = min(max_sv, min(n_samples, n_genes) - 1)
    
    for k in range(limit + 1):
        # Variance explained by k components
        if k == 0:
            rss = total_var
        else:
            # RSS = Total Var - Variance Explained by top k components
            # Var explained = sum(s_i^2)
            rss = total_var - np.sum(S[:k]**2)
        
        # Avoid log(0)
        if rss <= 0:
            rss = 1e-10
            
        # Log Likelihood approximation (Gaussian)
        # LL = - (N*G)/2 * ln(RSS / (N*G))
        # BIC = -2*LL + k_params * ln(N_data)
        
        sigma2 = rss / (n_samples * n_genes)
        log_lik = -0.5 * n_samples * n_genes * np.log(sigma2)
        
        # Number of parameters:
        # k vectors of size N + k vectors of size G - rotation adjustment
        # Approximation: k * (N + G)
        n_params = k * (n_samples + n_genes)
        
        # Sample size for BIC: N*G
        n_data = n_samples * n_genes
        
        bic = -2 * log_lik + n_params * np.log(n_data)
        bics.append(bic)
    
    # Find k that minimizes BIC
    n_sv = np.argmin(bics)
    return int(n_sv)


def estimate_n_sv(residuals, method="permutation", n_perm=20):
    """
    Estimate number of surrogate variables.
    
    Parameters
    ----------
    residuals : array (n_samples, n_genes)
        Residuals from get_residuals()
    method : str
        'permutation' or 'bic'
    n_perm : int
        Number of permutations (for method='permutation')
    
    Returns
    -------
    n_sv : int
        Estimated number of surrogate variables
    """
    if method == "bic":
        return estimate_n_sv_bic(residuals)
        
    # Default: Permutation
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



def sva_iterative(Y, X, n_sv=None, n_iter=5, alpha=0.25):
    """
    SVA with iterative refinement.

    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Expression matrix
    X : array (n_samples, n_covariates)
        Primary model matrix
    n_sv : int, optional
        Number of surrogate variables. If None, estimate automatically.
    n_iter : int
        Number of iterations
    alpha : float
        P-value threshold for null genes
    
    Returns
    -------
    sv : array (n_samples, n_sv)
        Estimated surrogate variables
    """
 
    # Step 0: Center and Scale Raw Data
    Y = center_and_scale(Y)

    # Step 1: Get residuals
    residuals = get_residuals(Y, X)
    
    # Step 2: Estimate n_sv if not provided:
    if n_sv is None: 
       n_sv = estimate_n_sv(residuals)
       if n_sv == 0:
          raise ValueError("Ni Significant Surrogate Variables Found")
    
    # Step 3: Get Initial estimates of SVs from all genes
    sv = extract_svs(residuals, n_sv) 
   
    # Step 4. Iterate
    for i in range(n_iter):
        # Find null genes
        null_genes = identify_null_genes(Y, X, sv, alpha)
        
        # Check if we have enough null genes
        if np.sum(null_genes) <= n_sv:
           break # Stop, return Current SVs  

        # Re-estimate SV from null genes only
        # Y is already scaled, so we just take the subset
        Y_null = Y[:, null_genes]
        sv_new = extract_svs(Y_null, n_sv)

        # Check stability comparing the first SV's correlation
        # correlation of sv_new[:,0] with sv[:,0]
        corr = np.corrcoef(sv[:, 0], sv_new[:, 0])[0,1]
        if abs(corr) > 0.99:
           sv = sv_new
           break
        
        # update and continue
        sv = sv_new
    
    return sv
 

def sva_iterative_residual(Y, X, n_sv=None, n_iter=5, alpha=0.25):
    """
    SVA with iterative refinement (Original/Legacy Version).
    Uses residuals for null genes. 
    WARNING: Fails to capture confounded SVs (see sva_iterative for fix).

    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Expression matrix
    X : array (n_samples, n_covariates)
        Primary model matrix
    n_sv : int, optional
        Number of surrogate variables. If None, estimate automatically.
    n_iter : int
        Number of iterations
    alpha : float
        P-value threshold for null genes
    
    Returns
    -------
    sv : array (n_samples, n_sv)
        Estimated surrogate variables
    """
 
    # Step 0: Center and Scale Raw Data (for fair comparison)
    Y = center_and_scale(Y)

    # Step 1: Get residuals
    residuals = get_residuals(Y, X)
    
    # Step 2: Estimate n_sv if not provided:
    if n_sv is None: 
       n_sv = estimate_n_sv(residuals)
       if n_sv == 0:
          raise ValueError("No Significant Surrogate Variables Found")
    
    # Step 3: Get Initial estimates of SVs from all genes
    sv = extract_svs(residuals, n_sv) 
   
    # Step 4. Iterate
    for i in range(n_iter):
        # Find null genes
        null_genes = identify_null_genes(Y, X, sv, alpha)
        
        # Check if we have enough null genes
        if np.sum(null_genes) <= n_sv:
           break # Stop, return Current SVs  

    	# Re-estimate SV from null genes only
        # LEGACY BEHAVIOR: Use residuals
        residuals_null = residuals[:, null_genes]
        sv_new = extract_svs(residuals_null, n_sv)

        # Check stability comparing the first SV's correlation
        # correlation of sv_new[:,0] with sv[:,0]
        corr = np.corrcoef(sv[:, 0], sv_new[:, 0])[0,1]
        if abs(corr) > 0.99:
           sv = sv_new
           break
        
        sv = sv_new
    
    return sv
 

def calculate_null_probabilities(Y, X, sv, alpha=0.25, n_boot=20):
    """
    Bootstrap to estimate probability that each gene is Null.
    """
    n_samples, n_genes = Y.shape
    null_counts = np.zeros(n_genes)
    
    for i in range(n_boot):
        # Bootstrap Resample (rows)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        Y_b = Y[indices]
        X_b = X[indices]
        sv_b = sv[indices]
        
        # Check Null
        # Use existing function on resampled data
        is_null = identify_null_genes(Y_b, X_b, sv_b, alpha)
        null_counts += is_null
        
    probabilities = null_counts / n_boot
    return probabilities


def sva_iterative_weighted(Y, X, n_sv=None, n_iter=5, alpha=0.25, n_boot=20):
    """
    SVA with Weighted Null Genes via Bootstrapping.
    
    Robust variant that uses probabilistic weights for SVD instead of binary selection.

    Parameters
    ----------
    n_boot : int
        Number of bootstrap iterations for weight estimation
    (Other params same as sva_iterative)
    """
 
    # Step 0: Center and Scale Raw Data
    Y = center_and_scale(Y)

    # Step 1: Get residuals
    residuals = get_residuals(Y, X)
    
    # Step 2: Estimate n_sv if not provided:
    if n_sv is None: 
       n_sv = estimate_n_sv(residuals)
       if n_sv == 0:
          raise ValueError("No Significant Surrogate Variables Found")
    
    # Step 3: Get Initial estimates of SVs from all genes
    sv = extract_svs(residuals, n_sv) 
   
    # Step 4. Iterate
    for i in range(n_iter):
        # 1. Bootstrap to get weights (Prob of being Null)
        weights = calculate_null_probabilities(Y, X, sv, alpha=alpha, n_boot=n_boot)
        
        # 2. Weighted SVD logic
        # Scaling factor = sqrt(probability) for variance-based weighting
        scaling_factors = np.sqrt(weights)
        
        # Broadcast multiplication: (n_samples, n_genes) * (n_genes,)
        Y_weighted = Y * scaling_factors[np.newaxis, :]
        
        # 3. Extract SV from weighted matrix
        sv_new = extract_svs(Y_weighted, n_sv)

        # Check stability
        corr = np.corrcoef(sv[:, 0], sv_new[:, 0])[0,1]
        if abs(corr) > 0.99:
           sv = sv_new
           break
        
        sv = sv_new
    
    return sv

def svaseq(Y, X, n_sv=None, method="weighted", vfilter=None, constant=1.0, **kwargs):
    """
    Wrapper for RNA-seq count data.
    
    1. Log-transforms Y: log(Y + constant)
    2. (Optional) Filters to top 'vfilter' genes by variance for SV estimation.
    3. Runs SVA (default: sva_iterative_weighted).
    
    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Raw count matrix (must be non-negative)
    X : array
        Primary variables
    n_sv : int
        Number of SVs (optional)
    method : str
        'weighted' (default), 'iterative', or 'standard'
    vfilter : int
        Number of most variable genes to use for SV estimation.
    constant : float
        Constant added before log transform (default 1.0)
    **kwargs : dict
        Arguments passed to the underlying SVA function
    
    Returns
    -------
    sv : array (n_samples, n_sv)
        Estimated surrogate variables
    """
    # 1. Validation
    if np.any(Y < 0):
        raise ValueError("Y must be non-negative counts")
        
    # 2. Log Transform
    Y_trans = np.log(Y + constant)
    
    # 3. Variance Filter
    Y_run = Y_trans
    if vfilter is not None and vfilter < Y.shape[1]:
        variances = np.var(Y_trans, axis=0)
        # Get indices of top 'vfilter' variance genes
        # np.argsort sorts ascending, so take last vfilter elements
        top_indices = np.argsort(variances)[-vfilter:]
        Y_run = Y_trans[:, top_indices]
        
    # 4. Dispatch
    if method == "weighted":
        return sva_iterative_weighted(Y_run, X, n_sv=n_sv, **kwargs)
    elif method == "iterative":
        return sva_iterative(Y_run, X, n_sv=n_sv, **kwargs)
    else:
        return sva(Y_run, X, n_sv=n_sv)
