# pySVA

Surrogate Variable Analysis in Python.

## What is SVA?

SVA identifies hidden sources of variation (batch effects, confounders) in high-dimensional data like gene expression.

**The problem:** Your data contains:
- Signal you care about (e.g., treatment effect)
- Hidden confounders you don't know about (e.g., batch effects)
- Noise

**SVA finds the hidden confounders** so you can adjust for them.

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/pySVA.git
cd pySVA
pip install -e .
```

## Quick Start
```python
import numpy as np
from pysva.sva import sva_iterative

# Your data
Y = np.random.randn(50, 200)  # 50 samples, 200 genes

# Primary variables (intercept + treatment)
treatment = np.array([0]*25 + [1]*25)
X = np.column_stack([np.ones(50), treatment])

# Find surrogate variables
sv = sva_iterative(Y, X, n_sv=2)

# Include sv in your downstream analysis
```

## Functions

| Function | Description |
|----------|-------------|
| `sva(Y, X, n_sv)` | Basic SVA (one-shot) |
| `sva_iterative(Y, X, n_sv, n_iter, alpha)` | SVA with iterative refinement |
| `get_residuals(Y, X)` | Remove primary variable effects |
| `extract_svs(residuals, n_sv)` | Extract SVs via SVD |
| `estimate_n_sv(residuals)` | Estimate number of SVs (permutation test) |
| `identify_null_genes(Y, X, sv, alpha)` | Find genes not affected by primary variable |

## How SVA Works

1. **Remove primary variable effect** → residuals
2. **SVD on residuals** → initial SV estimate
3. **Iterate** (for `sva_iterative`):
   - Find "null genes" (not affected by treatment)
   - Re-estimate SVs using only null genes
   - Repeat until stable

## Parameters

- `Y`: Expression matrix (n_samples × n_genes)
- `X`: Primary model matrix (n_samples × n_covariates)
- `n_sv`: Number of surrogate variables (None = auto-estimate)
- `n_iter`: Max iterations (default: 5)
- `alpha`: P-value threshold for null genes (default: 0.25)

## References

Leek & Storey (2007, 2008). "Capturing Heterogeneity in Gene Expression Studies by Surrogate Variable Analysis"
