
"""
SVA functions.
"""

import numpy as np

from scipy import stats
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
    if n_sv <= 0:
        raise ValueError(f"n_sv must be positive, got {n_sv}")

    U, S, Vt = svd(residuals, full_matrices=False)
    max_available = min(residuals.shape)

    if n_sv > max_available:
        import warnings
        warnings.warn(
            f"Requested n_sv={n_sv} exceeds maximum available singular values "
            f"({max_available}). Returning {max_available} components.",
            UserWarning
        )
        n_sv = max_available

    sv = U[:, :n_sv]

    return sv


def estimate_n_sv_bic(residuals, max_sv=10, criterion="bic"):
    """
    Estimate n_sv using Information Criterion (BIC or AIC).
    
    Parameters
    ----------
    residuals : array (n_samples, n_genes)
        Residuals
    max_sv : int
        Maximum number of SVs to test
    criterion : str
        'bic' (Bayesian) or 'aic' (Akaike)
        
    Returns
    -------
    n_sv : int
        Estimated number of SVs
    """
    n_samples, n_genes = residuals.shape
    
    # SVD
    U, S, Vt = svd(residuals, full_matrices=False)
    
    # Total variance
    # We essentially fit a low rank approximation: M = U S V.T
    # Residuals of this approx: E = Residuals - U_k S_k V_k.T
    # RSS = sum(E**2) = sum(S[k:]**2)
    
    total_rss = np.sum(S**2)
    
    ics = []
    
    # Limit max_sv to number of available singular values
    max_sv_available = min(max_sv, len(S) - 1)

    # Range 0 to max_sv_available
    for k in range(max_sv_available + 1):
        # RSS for k factors
        # Sum of squared singular values excluded
        rss = np.sum(S[k:]**2)

        # Log Likelihood (Gaussian approximation)
        # log L ~ - (N*G)/2 * log(RSS / (N*G))
        # ignoring constants
        n_data = n_samples * n_genes
        variance = rss / n_data

        # Handle numerical precision: if RSS/variance is extremely small relative to total variance,
        # we're at or beyond the numerical rank. Cap the log-likelihood to avoid numerical issues.
        # Use a more conservative threshold: 1e-10 * mean(S^2)
        total_mean_var = np.mean(S**2)
        if rss < 1e-10 * total_mean_var:
            # At numerical rank: use a large but finite log-likelihood
            # This allows BIC penalty to dominate and select the smallest sufficient k
            log_lik = 0.5 * n_data * 23  # log(1e-10) ≈ -23
        else:
            log_lik = -0.5 * n_data * np.log(variance)

        # Number of parameters
        # U (N*k) + V (G*k) - Rotation (k*k) ?
        # Degrees of freedom for rank k SVD: k*(N+G - k)
        n_params = k * (n_samples + n_genes - k)

        if criterion == "bic":
            ic = -2 * log_lik + n_params * np.log(n_data)
        elif criterion == "aic":
            ic = -2 * log_lik + 2 * n_params
        else:
            raise ValueError("Unknown criterion")

        ics.append(ic)
    
    # Find k that minimizes IC
    n_sv = np.argmin(ics)
    return int(n_sv)


def estimate_n_sv(residuals, method="bic", n_perm=20):
    """
    Estimate number of surrogate variables.
    
    Parameters
    ----------
    residuals : array (n_samples, n_genes)
        Residuals from get_residuals()
    method : str
        'bic' (default, robust) or 'permutation' (aggressive)
    n_perm : int
        Number of permutations (for method='permutation')
    
    Returns
    -------
    n_sv : int
        Estimated number of surrogate variables
    """
    if method == "bic":
        return estimate_n_sv_bic(residuals)
        
    n_samples, n_genes = residuals.shape
    
    # SVD
    # full_matrices=False gives min(n_samples, n_genes) singular values
    U, S, Vt = svd(residuals, full_matrices=False)
    
    # Permutation Test (Emulates algorithms like Leek/Buja-Eyuboglu)
    # We maintain column-wise structure but break row-wise correlation.
    # Note: R's 'be' method permutes within rows (features), breaking sample-sample correlation driven by biology/tech.
    
    S_sq = S**2
    total_var = np.sum(S_sq)
    obs_prop_var = S_sq / total_var

    # Null distribution
    null_prop_var = np.zeros((n_perm, len(S)))
    
    # Optimized Permutation (Vectorized Shuffle)
    for p in range(n_perm):
        # Create random indices for shuffling each column independently
        idx = np.random.rand(n_samples, n_genes).argsort(axis=0)
        res_perm = np.take_along_axis(residuals, idx, axis=0)
            
        _, S_perm, _ = svd(res_perm, full_matrices=False)
        S_perm_sq = S_perm**2
        null_prop_var[p, :] = S_perm_sq / np.sum(S_perm_sq)
        
    p_vals = np.zeros(len(S))
    for k in range(len(S)):
        # Count how many null singular values (at rank k) are greater than observed
        p_vals[k] = np.sum(null_prop_var[:, k] > obs_prop_var[k]) / n_perm
        
    n_sv = np.sum(p_vals < 0.05)
    return int(n_sv)


def identify_null_genes(Y, X, sv, alpha=0.25):
    """
    Identify genes not affected by primary variable.

    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Expression matrix
    X : array (n_samples, n_covariates)
        Primary model matrix. IMPORTANT: The primary variable of interest
        must be at index 1 (X[:, 1]). Typically, X[:, 0] is the intercept
        and X[:, 1] is the treatment/condition variable.
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
    
    # Precompute (X'X)^-1 once for efficiency and to check for singularity
    XtX = X_full.T @ X_full
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix (e.g., collinear covariates)
        XtX_inv = np.linalg.pinv(XtX)

    for j in range(n_genes):
        y = Y[:, j]

        # Fit full model
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
        resid = y - X_full @ beta

        # Standard error for treatment effect (column 1)
        mse = np.sum(resid ** 2) / (n_samples - n_params)
        se = np.sqrt(mse * XtX_inv[1, 1])

        # t-test
        t_stat = beta[1] / se
        pvals[j] = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - n_params))

    null_genes = pvals > alpha

    return null_genes

def identify_null_genes_vectorized(Y, X, sv, alpha=0.25):
    """
    Identify null genes using vectorized matrix operations (Fast).

    Parameters
    ----------
    Y : array (n_samples, n_genes)
        Expression matrix
    X : array (n_samples, n_covariates)
        Primary model matrix. IMPORTANT: The primary variable of interest
        must be at index 1 (X[:, 1]). Typically, X[:, 0] is the intercept
        and X[:, 1] is the treatment/condition variable.
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
    
    # 1. Precompute (X'X)^-1
    # This is small: (params x params)
    XtX = X_full.T @ X_full
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix
        XtX_inv = np.linalg.pinv(XtX)
        
    # 2. Beta = (X'X)^-1 X' Y
    # (params x n_samples) @ (n_samples x n_genes) -> (params x n_genes)
    Beta = XtX_inv @ X_full.T @ Y
    
    # 3. Residuals = Y - X Beta
    # X Beta -> (n_samples x params) @ (params x n_genes) -> (n_samples x n_genes)
    Y_hat = X_full @ Beta
    Residuals = Y - Y_hat
    
    # 4. MSE = sum(res^2) / dof
    # Sum over samples (axis=0) -> (n_genes,)
    RSS = np.sum(Residuals**2, axis=0)
    MSE = RSS / (n_samples - n_params)
    
    # 5. Standard Error
    # var(beta_j) = MSE * (XtX_inv)_jj
    # We care about treatment effect, usually index 1 (Intercept is 0, Treatment is 1)
    # Check if we assume treatment is index 1. Yes, per slow function code: `beta[1] / se`
    scaling_factor = XtX_inv[1, 1]
    SE = np.sqrt(MSE * scaling_factor)
    
    # 6. t-stat
    t_stats = Beta[1, :] / SE
    
    # 7. p-values
    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_samples - n_params))
    
    return pvals > alpha

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
        Estimated surrogate variables. If n_sv=0, returns empty array
        with shape (n_samples, 0).
    """

    # Step 1: Get residuals
    residuals = get_residuals(Y, X)

    # Step 2: Estimate n_sv if not provided
    if n_sv is None:
        n_sv = estimate_n_sv(residuals)

    # Step 3: Handle case where no SVs are needed
    if n_sv == 0:
        return np.empty((Y.shape[0], 0))

    # Step 4: Extract surrogate variables
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
          raise ValueError("No Significant Surrogate Variables Found")
    
    # Step 3: Get Initial estimates of SVs from all genes
    sv = extract_svs(residuals, n_sv) 
   
    # Step 4. Iterate
    for i in range(n_iter):
        # Find null genes
        null_genes = identify_null_genes_vectorized(Y, X, sv, alpha)
        
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
        is_null = identify_null_genes_vectorized(Y_b, X_b, sv_b, alpha)
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
        # Use vectorized logic inside helper if possible? 
        # Actually calculate_null_probabilities calls identify_null_genes.
        # We should update calculate_null_probabilities to use the fast version.
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
