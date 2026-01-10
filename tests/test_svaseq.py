
import numpy as np
import pytest
from pysva.sva import svaseq

def test_svaseq_validation():
    """Test that svaseq rejects negative inputs."""
    Y = np.array([[-1, 10], [5, 5]])
    X = np.ones((2, 1))
    with pytest.raises(ValueError, match="must be non-negative"):
        svaseq(Y, X)

def test_svaseq_runs_basic():
    """Test basic execution on count data."""
    # Simulate count data (Negative Binomial-ish)
    np.random.seed(42)
    n_samples = 20
    n_genes = 100
    # Mean counts
    mu = np.random.uniform(10, 100, n_genes)
    Y = np.random.poisson(lam=mu, size=(n_samples, n_genes))
    
    # Primary variable
    X = np.ones((n_samples, 1))
    X[:,0] = np.random.choice([0, 1], n_samples)
    
    # Run
    sv = svaseq(Y, X, n_sv=1, method="weighted")
    
    assert sv.shape == (n_samples, 1)
    assert not np.isnan(sv).any()

def test_svaseq_variance_filtering():
    """Test that vfilter substantially changes results or at least runs."""
    np.random.seed(99)
    n_samples = 30
    n_genes = 200
    
    # Create 100 "Noise" genes (low variance) and 100 "Signal" genes (high variance)
    Y_noise = np.random.poisson(5, size=(n_samples, 100))
    Y_signal = np.random.poisson(500, size=(n_samples, 100)) # High mean = High var in Poisson
    
    Y = np.hstack([Y_noise, Y_signal])
    X = np.random.randn(n_samples, 2)
    
    # Run with filter=50 (should pick from Y_signal)
    sv = svaseq(Y, X, n_sv=2, vfilter=50, method="weighted")
    
    assert sv.shape == (n_samples, 2)
    
    # If we filter to 1 gene, it should run but might be weird. 
    # Just checking it doesn't crash.
    sv_small = svaseq(Y, X, n_sv=1, vfilter=10, method="weighted")
    assert sv_small.shape == (n_samples, 1)

if __name__ == "__main__":
    test_svaseq_validation()
    test_svaseq_runs_basic()
    test_svaseq_variance_filtering()
    print("All svaseq tests passed!")
