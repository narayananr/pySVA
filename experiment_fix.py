
import numpy as np
import matplotlib.pyplot as plt
from pysva.sva import sva, sva_iterative, get_residuals, extract_svs, identify_null_genes, estimate_n_sv

# Define the improved function
def sva_iterative_improved(Y, X, n_sv=None, n_iter=5, alpha=0.25):
    # Step 1: Get residuals (for initial estimate)
    residuals = get_residuals(Y, X)
    
    # Step 2: Estimate n_sv if not provided
    if n_sv is None: 
       n_sv = estimate_n_sv(residuals)
       if n_sv == 0:
          raise ValueError("No Significant Surrogate Variables Found")
    
    # Step 3: Get Initial estimates of SVs from all genes (using residuals)
    sv = extract_svs(residuals, n_sv) 
   
    # Step 4. Iterate
    for i in range(n_iter):
        # Find null genes
        null_genes = identify_null_genes(Y, X, sv, alpha)
        
        # Check if we have enough null genes
        if np.sum(null_genes) <= n_sv:
           break 

        # Re-estimate SV from null genes ONLY
        # MODIFIED: Use Y (centered) instead of residuals
        # Ideally, we should regress out only the INTERCEPT, not the full X
        # Assuming X[:,0] is intercept.
        
        # For null genes, Y = Intercept + SV + E. Treatment effect is 0.
        # So we just center the data (remove intercept).
        Y_null = Y[:, null_genes]
        Y_null_centered = Y_null - np.mean(Y_null, axis=0)
        
        sv_new = extract_svs(Y_null_centered, n_sv)

        # Check stability
        corr = np.corrcoef(sv[:, 0], sv_new[:, 0])[0,1]
        if abs(corr) > 0.99:
           sv = sv_new
           break
        
        sv = sv_new
    
    return sv

# Copy the simulation setup from test_confounded.py
np.random.seed(42)
n_samples = 50
n_genes = 200
treatment = np.array([0]*25 + [1]*25)
X = np.column_stack([np.ones(n_samples), treatment])
true_sv = treatment * 0.5 + np.random.randn(n_samples) * 0.5

# Gene effects
baseline = np.random.randn(n_genes) * 2
treatment_effect = np.zeros(n_genes)
treatment_effect[:20] = np.random.randn(20)  # 20 DE genes
sv_effect = np.random.randn(n_genes) * 0.8
noise = np.random.randn(n_samples, n_genes) * 0.5

Y_clean = np.zeros((n_samples, n_genes))
Y = np.zeros((n_samples, n_genes))
for i in range(n_samples):
    Y_clean[i, :] = baseline + treatment[i] * treatment_effect + noise[i, :]
    Y[i, :] = Y_clean[i, :] + true_sv[i] * sv_effect

# Run Comparison
print("Running SVA (Basic)...")
sv_basic = sva(Y, X, n_sv=1)

print("Running SVA (Original Iterative)...")
sv_iter = sva_iterative(Y, X, n_sv=1, n_iter=5)

print("Running SVA (Improved Iterative)...")
sv_improved = sva_iterative_improved(Y, X, n_sv=1, n_iter=5)

# Metrics
corr_basic = np.abs(np.corrcoef(true_sv, sv_basic[:, 0])[0, 1])
corr_iter = np.abs(np.corrcoef(true_sv, sv_iter[:, 0])[0, 1])
corr_improved = np.abs(np.corrcoef(true_sv, sv_improved[:, 0])[0, 1])

print("\n" + "="*40)
print("RESULTS COMPARED")
print("="*40)
print(f"Basic SVA:        {corr_basic:.4f}")
print(f"Iterative (Old):  {corr_iter:.4f}")
print(f"Iterative (New):  {corr_improved:.4f}")
print("="*40)
