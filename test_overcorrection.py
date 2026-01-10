
import numpy as np
import matplotlib.pyplot as plt
from pysva.sva import sva, sva_iterative
from scipy import stats

# ============================================
# SCENARIO: UNCONFOUNDED (The "Safe" Case)
# ============================================
# We want to see if the NEW iterative method accidentally 
# "learns" the treatment signal and removes it.

np.random.seed(100) # Fixed seed

n_samples = 50
n_genes = 1000  # More genes to get better stats
n_de_genes = 100 # 10% DE

# Primary variable: treatment (Balanced)
treatment = np.array([0]*25 + [1]*25)
X = np.column_stack([np.ones(n_samples), treatment])

# True Hidden Factor (SV) - INDEPENDENT of treatment
# This is the crucial difference from the previous test.
true_sv = np.random.randn(n_samples) 

print("Data simulated (UNCONFOUNDED):")
print(f"  SV-treatment correlation: {np.corrcoef(true_sv, treatment)[0, 1]:.3f}")

# Gene effects
baseline = np.random.randn(n_genes) * 2
treatment_effect = np.zeros(n_genes)
# First 100 genes are True Positives
treatment_effect[:n_de_genes] = 1.5  # Strong effect to ensure detectability
sv_effect = np.random.randn(n_genes) * 1.0
noise = np.random.randn(n_samples, n_genes) * 0.5

Y = np.zeros((n_samples, n_genes))
for i in range(n_samples):
    Y[i, :] = baseline + treatment[i] * treatment_effect + noise[i, :] + true_sv[i] * sv_effect

# ============================================
# RUN METHODS
# ============================================

print("\nRunning SVA (Basic - like 'Residuals' method)...")
sv_basic = sva(Y, X, n_sv=1)

print("Running SVA (Iterative - New 'Raw Data' method)...")
sv_iter = sva_iterative(Y, X, n_sv=1, n_iter=5)

# ============================================
# METRIC 1: FALSE CORRELATION
# ============================================
# Does the estimated SV correlate with treatment? (It shouldn't!)
corr_basic_trt = np.corrcoef(sv_basic[:,0], treatment)[0,1]
corr_iter_trt = np.corrcoef(sv_iter[:,0], treatment)[0,1]

print("\n--- CHECK 1: PHANTOM CONFOUNDING ---")
print("Does the estimated SV look like Treatment?")
print(f"Basic SVA Corr(SV, Trt):     {abs(corr_basic_trt):.4f}")
print(f"Iterative SVA Corr(SV, Trt): {abs(corr_iter_trt):.4f}")
print("(High values here mean the SV is 'absorbing' the treatment signal)")

# ============================================
# METRIC 2: POWER (Signal Preservation)
# ============================================
# Fit the final model: Y ~ X + SV
# Check how many of the 100 DE genes we recover (p < 0.05)

def get_power(sv_est, name):
    X_full = np.column_stack([X, sv_est])
    pvals = []
    
    for j in range(n_genes):
        y = Y[:, j]
        # OLS
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
        resid = y - X_full @ beta
        # Stats
        n_params = X_full.shape[1]
        mse = np.sum(resid ** 2) / (n_samples - n_params)
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
        se = np.sqrt(mse * XtX_inv[1, 1])
        t_stat = beta[1] / se
        p = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - n_params))
        pvals.append(p)
    
    pvals = np.array(pvals)
    # True Positives (indices 0 to 99)
    tp = np.sum(pvals[:n_de_genes] < 0.05)
    return tp

tp_basic = get_power(sv_basic, "Basic")
tp_iter = get_power(sv_iter, "Iterative")

print("\n--- CHECK 2: STATISTICAL POWER ---")
print(f"True DE Genes: {n_de_genes}")
print(f"Recovered by Basic SVA:     {tp_basic}")
print(f"Recovered by Iterative SVA: {tp_iter}")

if tp_iter < tp_basic * 0.95:
    print("\n[WARNING] Iterative method lost significant power! Over-correction detected.")
else:
    print("\n[PASS] Iterative method preserved signal.")
