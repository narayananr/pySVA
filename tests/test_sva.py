"""
Tests for SVA functions.
"""

import numpy as np

from pysva.sva import get_residuals
from pysva.sva import extract_svs
from pysva.sva import estimate_n_sv
from pysva.sva import sva
from pysva.sva import identify_null_genes
from pysva.sva import sva_iterative


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


def test_sva_returns_correct_shape():
    """SVA should return (n_samples, n_sv) array."""
    np.random.seed(42)
    
    Y = np.random.randn(20, 100)
    X = np.column_stack([np.ones(20), [0]*10 + [1]*10])
    
    sv = sva(Y, X, n_sv=2)
    
    assert sv.shape == (20, 2)




def test_identify_null_genes_returns_boolean_array():
    """Should return boolean array of length n_genes."""
    np.random.seed(42)
    
    Y = np.random.randn(20, 100)
    X = np.column_stack([np.ones(20), [0]*10 + [1]*10])
    sv = np.random.randn(20, 1)
    
    null_genes = identify_null_genes(Y, X, sv)
    
    assert null_genes.dtype == bool
    assert len(null_genes) == 100


def test_sva_iterative_returns_correct_shape():
    """Should return (n_samples, n_sv) array."""
    np.random.seed(42)
    
    Y = np.random.randn(20, 100)
    X = np.column_stack([np.ones(20), [0]*10 + [1]*10])
    
    sv = sva_iterative(Y, X, n_sv=2, n_iter=3)
    
    assert sv.shape == (20, 2)
