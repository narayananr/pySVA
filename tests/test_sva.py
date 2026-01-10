"""
Tests for SVA functions.
"""

import numpy as np
from pysva.sva import get_residuals, extract_svs

from pysva.sva import get_residuals, extract_svs, estimate_n_sv

def test_get_residuals_shape():
    """Output should have same shape as input Y."""
    Y = np.random.randn(10, 5)
    X = np.column_stack([np.ones(10), [0]*5 + [1]*5])
    
    residuals = get_residuals(Y, X)
    
    assert residuals.shape == Y.shape




def test_extract_svs_shape():
    """Output should be (n_samples, n_sv)."""
    residuals = np.random.randn(10, 50)
    n_sv = 3
    
    sv = extract_svs(residuals, n_sv)
    
    assert sv.shape == (10, 3)




def test_estimate_n_sv_returns_integer():
    """Should return a non-negative integer."""
    residuals = np.random.randn(20, 100)
    
    n_sv = estimate_n_sv(residuals)
    
    assert isinstance(n_sv, int)
    assert n_sv >= 0
