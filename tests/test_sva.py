"""
Tests for SVA functions.
"""

import numpy as np
from pysva.sva import get_residuals


def test_get_residuals_shape():
    """Output should have same shape as input Y."""
    Y = np.random.randn(10, 5)
    X = np.column_stack([np.ones(10), [0]*5 + [1]*5])
    
    residuals = get_residuals(Y, X)
    
    assert residuals.shape == Y.shape
