"""
Test SVA when SV is confounded with treatment.
This is the hard case where iterative SVA should help.
"""

import numpy as np
import matplotlib.pyplot as plt
from pysva.sva import sva, sva_iterative

np.random.seed(42)

# ============================================
# SIMULATE CONFOUNDED DATA
# ============================================

n_samples = 50
n_genes = 200

# Primary variable: treatment (0 or 1)
treatment = np.array([0]*25 + [1]*25)
X = np.column_stack([np.ones(n_samples), treatment])

# True hidden confounder - CORRELATED with treatment!
true_sv = treatment * 0.5 + np.random.randn(n_samples) * 0.5

corr_sv_treatment = np.corrcoef(true_sv, treatment)[0, 1]

print("Data simulated (CONFOUNDED):")
print(f"  Samples: {n_samples}")
print(f"  Genes: {n_genes}")
print(f"  DE genes: 20")
print(f"  SV-treatment correlation: {corr_sv_treatment:.3f}")

# Gene effects
baseline = np.random.randn(n_genes) * 2
treatment_effect = np.zeros(n_genes)
treatment_effect[:20] = np.random.randn(20)  # 20 DE genes
sv_effect = np.random.randn(n_genes) * 0.8
noise = np.random.randn(n_samples, n_genes) * 0.5

# Generate Y_clean (no SV) and Y (with SV)
Y_clean = np.zeros((n_samples, n_genes))
Y = np.zeros((n_samples, n_genes))
for i in range(n_samples):
    Y_clean[i, :] = baseline + treatment[i] * treatment_effect + noise[i, :]
    Y[i, :] = Y_clean[i, :] + true_sv[i] * sv_effect

# ============================================
# RUN BOTH METHODS
# ============================================

print("\nRunning SVA (basic)...")
sv_basic = sva(Y, X, n_sv=1)

print("Running SVA (iterative)...")
sv_iter = sva_iterative(Y, X, n_sv=1, n_iter=5)

# ============================================
# COMPARE
# ============================================

corr_basic = np.corrcoef(true_sv, sv_basic[:, 0])[0, 1]
corr_iter = np.corrcoef(true_sv, sv_iter[:, 0])[0, 1]

print("\n" + "="*40)
print("RESULTS (CONFOUNDED CASE)")
print("="*40)
print(f"Basic SVA correlation with true SV:     {abs(corr_basic):.4f}")
print(f"Iterative SVA correlation with true SV: {abs(corr_iter):.4f}")
print("="*40)

if abs(corr_iter) > abs(corr_basic):
    print("Winner: Iterative SVA")
else:
    print("Winner: Basic SVA")

# ============================================
# VISUALIZE
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Sort samples by treatment
sample_order = np.argsort(treatment)

# DE genes first
gene_order = np.concatenate([np.arange(20), np.arange(20, 200)])

# Helper: sort and center by gene
def sort_and_center(data):
    sorted_data = data[sample_order, :][:, gene_order].T
    centered = sorted_data - sorted_data.mean(axis=1, keepdims=True)
    return centered

# Panel 1: Clean data (no SV)
ax = axes[0, 0]
im = ax.imshow(sort_and_center(Y_clean), aspect='auto', cmap='RdBu_r')
ax.axvline(x=24.5, color='black', linewidth=2, linestyle='--')
ax.axhline(y=19.5, color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Genes (DE: 0-19)')
ax.set_xlabel('Samples')
ax.set_title('Y without SV')
plt.colorbar(im, ax=ax)

# Panel 2: SV effect
ax = axes[0, 1]
sv_contribution = np.outer(true_sv, sv_effect)
im = ax.imshow(sort_and_center(sv_contribution), aspect='auto', cmap='RdBu_r')
ax.axvline(x=24.5, color='black', linewidth=2, linestyle='--')
ax.axhline(y=19.5, color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Genes')
ax.set_xlabel('Samples')
ax.set_title(f'SV Effect (corr with trt: {corr_sv_treatment:.2f})')
plt.colorbar(im, ax=ax)

# Panel 3: Raw data (with SV)
ax = axes[0, 2]
im = ax.imshow(sort_and_center(Y), aspect='auto', cmap='RdBu_r')
ax.axvline(x=24.5, color='black', linewidth=2, linestyle='--')
ax.axhline(y=19.5, color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Genes')
ax.set_xlabel('Samples')
ax.set_title('Y with SV (observed)')
plt.colorbar(im, ax=ax)

# Panel 4: Basic SVA recovery
ax = axes[1, 0]
ax.scatter(true_sv, sv_basic[:, 0], alpha=0.6)
ax.set_xlabel('True SV')
ax.set_ylabel('Estimated SV')
ax.set_title(f'Basic SVA (r = {abs(corr_basic):.3f})')

# Panel 5: Iterative SVA recovery
ax = axes[1, 1]
ax.scatter(true_sv, sv_iter[:, 0], alpha=0.6)
ax.set_xlabel('True SV')
ax.set_ylabel('Estimated SV')
ax.set_title(f'Iterative SVA (r = {abs(corr_iter):.3f})')

# Panel 6: Comparison bar
ax = axes[1, 2]
bars = ax.bar(['Basic', 'Iterative'], [abs(corr_basic), abs(corr_iter)], 
              color=['steelblue', 'darkorange'])
ax.set_ylabel('Correlation with True SV')
ax.set_title('Comparison (Confounded)')
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('sva_confounded.png', dpi=150)
plt.show()

print("\nPlot saved to sva_confounded.png")
