"""
Test SVA on simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pysva.sva import sva

np.random.seed(42)

# ============================================
# SIMULATE DATA
# ============================================

n_samples = 50
n_genes = 200

# Primary variable: treatment (0 or 1)
treatment = np.array([0]*25 + [1]*25)
X = np.column_stack([np.ones(n_samples), treatment])

# True hidden confounder (this is what SVA should find)
true_sv = np.random.randn(n_samples)

# Gene effects
baseline = np.random.randn(n_genes) * 2
treatment_effect = np.zeros(n_genes)
treatment_effect[:20] = np.random.randn(20)
sv_effect = np.random.randn(n_genes) * 0.8

# Generate Y
Y = np.zeros((n_samples, n_genes))
for i in range(n_samples):
    Y[i, :] = (baseline 
               + treatment[i] * treatment_effect 
               + true_sv[i] * sv_effect 
               + np.random.randn(n_genes) * 0.5)

print("Data simulated:")
print(f"  Y shape: {Y.shape}")

# ============================================
# RUN SVA
# ============================================

print("\nRunning SVA...")
sv_estimated = sva(Y, X, n_sv=1)

correlation = np.corrcoef(true_sv, sv_estimated[:, 0])[0, 1]
print(f"Correlation between true SV and estimated SV: {correlation:.3f}")

# ============================================
# VISUALIZE
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Sort samples by true SV for better visualization
order = np.argsort(true_sv)

# Panel 1: Raw data Y
ax = axes[0, 0]
im = ax.imshow(Y[order, :], aspect='auto', cmap='RdBu_r')
ax.set_xlabel('Genes')
ax.set_ylabel('Samples (sorted by true SV)')
ax.set_title('Raw Data (Y)')
plt.colorbar(im, ax=ax)

# Panel 2: True SV vs Estimated SV
ax = axes[0, 1]
ax.scatter(true_sv, sv_estimated[:, 0], alpha=0.6)
ax.set_xlabel('True SV')
ax.set_ylabel('Estimated SV')
ax.set_title(f'SV Recovery (r = {correlation:.3f})')

# Panel 3: True SV effect pattern
ax = axes[1, 0]
sv_contribution = np.outer(true_sv, sv_effect)
im = ax.imshow(sv_contribution[order, :], aspect='auto', cmap='RdBu_r')
ax.set_xlabel('Genes')
ax.set_ylabel('Samples (sorted by true SV)')
ax.set_title('True SV Effect')
plt.colorbar(im, ax=ax)

# Panel 4: Treatment effect pattern  
ax = axes[1, 1]
trt_contribution = np.outer(treatment, treatment_effect)
im = ax.imshow(trt_contribution[order, :], aspect='auto', cmap='RdBu_r')
ax.set_xlabel('Genes')
ax.set_ylabel('Samples (sorted by true SV)')
ax.set_title('Treatment Effect (first 20 genes)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('sva_simulation.png', dpi=150)
plt.show()

print("\nPlot saved to sva_simulation.png")
