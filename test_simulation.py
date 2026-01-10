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

# True hidden confounder
true_sv = np.random.randn(n_samples)

# Gene effects
baseline = np.random.randn(n_genes) * 2
treatment_effect = np.zeros(n_genes)
treatment_effect[:20] = np.random.randn(20)
sv_effect = np.random.randn(n_genes) * 0.8
noise = np.random.randn(n_samples, n_genes) * 0.5

# Generate Y_clean (no SV) and Y (with SV)
Y_clean = np.zeros((n_samples, n_genes))
Y = np.zeros((n_samples, n_genes))
for i in range(n_samples):
    Y_clean[i, :] = baseline + treatment[i] * treatment_effect + noise[i, :]
    Y[i, :] = Y_clean[i, :] + true_sv[i] * sv_effect

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

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Sort samples by treatment
sample_order = np.argsort(treatment)

# DE genes first
gene_order = np.concatenate([np.arange(20), np.arange(20, 200)])

# Helper: sort and center by gene (remove baseline)
def sort_and_center(data):
    sorted_data = data[sample_order, :][:, gene_order].T
    centered = sorted_data - sorted_data.mean(axis=1, keepdims=True)
    return centered

# Panel 1: Clean data (centered)
ax = axes[0, 0]
im = ax.imshow(sort_and_center(Y_clean), aspect='auto', cmap='RdBu_r')
ax.axvline(x=24.5, color='black', linewidth=2, linestyle='--')
ax.axhline(y=19.5, color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Genes (DE: 0-19)')
ax.set_xlabel('Samples')
ax.set_title('Y without SV')
plt.colorbar(im, ax=ax)

# Panel 2: SV effect only
ax = axes[0, 1]
sv_contribution = np.outer(true_sv, sv_effect)
im = ax.imshow(sort_and_center(sv_contribution), aspect='auto', cmap='RdBu_r')
ax.axvline(x=24.5, color='black', linewidth=2, linestyle='--')
ax.axhline(y=19.5, color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Genes')
ax.set_xlabel('Samples')
ax.set_title('SV Effect (confounder)')
plt.colorbar(im, ax=ax)

# Panel 3: Raw data (centered)
ax = axes[0, 2]
im = ax.imshow(sort_and_center(Y), aspect='auto', cmap='RdBu_r')
ax.axvline(x=24.5, color='black', linewidth=2, linestyle='--')
ax.axhline(y=19.5, color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Genes')
ax.set_xlabel('Samples')
ax.set_title('Y with SV (observed)')
plt.colorbar(im, ax=ax)

# Panel 4: Treatment effect only
ax = axes[1, 0]
trt_contribution = np.outer(treatment, treatment_effect)
im = ax.imshow(sort_and_center(trt_contribution), aspect='auto', cmap='RdBu_r')
ax.axvline(x=24.5, color='black', linewidth=2, linestyle='--')
ax.axhline(y=19.5, color='black', linewidth=2, linestyle='--')
ax.set_ylabel('Genes')
ax.set_xlabel('Samples')
ax.set_title('Treatment Effect (truth)')
plt.colorbar(im, ax=ax)

# Panel 1: Clean data (centered)

# Panel 2: SV effect only

# Panel 3: Raw data (centered)

# Panel 4: Treatment effect only
ax.set_title('Treatment Effect (biology)')

# Panel 5: SV recovery
ax = axes[1, 1]
ax.scatter(true_sv, sv_estimated[:, 0], alpha=0.6)
ax.set_xlabel('True SV')
ax.set_ylabel('Estimated SV')
ax.set_title(f'SV Recovery (r = {correlation:.3f})')

# Panel 6: Summary
ax = axes[1, 2]
ax.axis('off')
ax.text(0.5, 0.5, 
        'Summary:\n\n'
        f'Samples: {n_samples}\n'
        f'Genes: {n_genes}\n'
        f'DE genes: 20\n'
        f'SV correlation: {correlation:.3f}\n\n'
        'All heatmaps centered\n(baseline removed)',
        ha='center', va='center', fontsize=12,
        transform=ax.transAxes)


plt.tight_layout()
plt.savefig('sva_simulation.png', dpi=150)
plt.show()

print("\nPlot saved to sva_simulation.png")
