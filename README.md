# pySVA

**Surrogate Variable Analysis (SVA) for Python.**

Identify and remove hidden sources of variation (batch effects, confounders) in high-dimensional gene expression data. 
This package provides a robust implementation of the SVA algorithm, including modern improvements for confounded data and RNA-seq integration.

## Features

- **Robust Confounding Recovery**: Solves the classic SVA failure case where hidden factors are correlated with treatment.
- **RNA-Seq Support**: Dedicated `svaseq` wrapper with log-transformation and variance filtering.
- **Improved Estimation**: BIC-based method to accurately estimate the number of hidden factors (avoiding false positives in noise).


## Installation

```bash
git clone https://github.com/YOUR_USERNAME/pySVA.git
cd pySVA
pip install -e .
```

## Quick Start

### 1. Standard Usage (Continuous Data / Microarrays)

```python
import numpy as np
from pysva.sva import sva_iterative_weighted

# Y: Expression Matrix (Samples x Genes)
# X: Design Matrix (Samples x Covariates, e.g. Treatment)

# Use the 'Weighted-Bootstrap' method (Recommended Gradient)
sv = sva_iterative_weighted(Y, X, n_sv=2)

# ...or let SVA estimate n_sv automatically
sv_auto = sva_iterative_weighted(Y, X)
```

### 2. RNA-Seq Usage (Count Data)

Use the `svaseq` wrapper which handles log-transformation and filtering high-variance genes.

```python
from pysva.sva import svaseq

# Y: Raw Counts (Must be non-negative)
# vfilter: Number of most variable genes to use (e.g. 5000)

sv = svaseq(Y, X, vfilter=5000)
```

### 3. Estimating Number of SVs

Standard methods often overestimate the number of factors in noisy data. We provide a **BIC** (Bayesian Information Criterion) estimator for robust detection.

```python
from pysva.sva import estimate_n_sv, get_residuals

residuals = get_residuals(Y, X)

# Robust / Conservative (Recommended)
n = estimate_n_sv(residuals, method="bic") 

# Sensitive / Aggressive (Standard Permutation)
n = estimate_n_sv(residuals, method="permutation")
```


## How It Works

SVA assumes gene expression $Y$ is composed of:
1.  **Primary Signal**: Effects of terms in your model $X$ (e.g., Treatment).
2.  **Surrogate Variables (SVs)**: Hidden factors influencing expression (e.g., Batch, Cell Type).
3.  **Noise**: Random variation.

$$ Y = X\beta + SV\gamma + \epsilon $$

### The Algorithm (Iterative Weighted SVA)

1.  **Initial Estimate**: We fit the model $Y \sim X$ and calculate residuals. SVD on residuals gives the initial Surrogage Variables.
2.  **Null Gene Identification**: We identify "Null Genes" — genes that are *not* significantly affected by the primary variable $X$ (p-value > 0.25). These genes should purely reflect the hidden factors.
3.  **Weighting**: Instead of hard filtering, we calculate a probability weight for each gene ($P_{null}$) using bootstrapping.
4.  **Re-Estimation**: We perform a **Weighted SVD** on the original data, prioritizing genes with high $P_{null}$. This captures the hidden factor structure from genes that don't have biological signal, recovering the SVs even if they are confounded with the treatment.

## Options and Parameters

### `sva_iterative_weighted(Y, X, ...)`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_sv` | `int` | `None` | Number of hidden factors. If `None`, estimated automatically. |
| `n_iter` | `int` | `5` | Number of refinement iterations. Usually converges quickly. |
| `alpha` | `float` | `0.25` | P-value threshold for identifying null genes. |
| `n_boot` | `int` | `20` | Bootstrap iterations for weighting. Higher = more precise but slower. |

### `svaseq(Y, X, ...)`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vfilter` | `int` | `None` | If set (e.g. 5000), uses only the top $V$ most variable genes. Recommended for RNA-seq. |
| `constant` | `float` | `1.0` | Constant added before log-transform: $\log(Y + c)$. |

### `estimate_n_sv(residuals, ...)`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `method` | `str` | `"permutation"` | `"bic"` (Conservative, good for large N) or `"permutation"` (Sensitive). |

### Technical Implementation

*   **Center-Only**: By default, `pySVA` performs **Mean Centering** but **No Scaling** (variance is preserved), matching the logic of the original R `sva` package. This allows highly variable genes to drive the SV estimation.
*   **Confounding Fix**: Unlike legacy implementations that used residuals (which removed the signal of interest), our iterative methods use the **Raw Expression** of null genes to recover confounded batch effects.

## References

1. Leek & Storey (2007), [Capturing Heterogeneity in Gene Expression Studies by Surrogate Variable Analysis](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0030161).
2. Leek (2014). [svaseq: removing batch effects and other unwanted noise from sequencing data](https://academic.oup.com/nar/article/42/21/e161/2903156).
