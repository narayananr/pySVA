
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysva.sva import estimate_n_sv

# Settings
N_SIMS = 10
N_SAMPLES = 50
N_GENES = 500

def simulate_residuals(n_true_sv, noise_level=1.0):
    # Create pure noise
    noise = np.random.randn(N_SAMPLES, N_GENES) * noise_level
    
    if n_true_sv == 0:
        return noise
        
    # Create True SVs
    # Random orthogonal factors
    U = np.linalg.qr(np.random.randn(N_SAMPLES, n_true_sv))[0]
    V = np.linalg.qr(np.random.randn(N_GENES, n_true_sv))[0]
    # Noise floor (approx largest singular value) is sqrt(N) + sqrt(G) ~ 7 + 22 = 29
    # We need signal > 29. Let's use 100.
    S = np.array([100]*n_true_sv) # Strong signal (S >> Noise)
    
    signal = U @ np.diag(S) @ V.T
    
    return signal + noise

def run_benchmark():
    results = []
    true_sv_counts = [0, 1, 3, 5]
    
    print(f"Running n_sv Benchmark ({N_SIMS} sims per setting)...")
    
    for n_true in true_sv_counts:
        print(f"  Testing True n_sv = {n_true}")
        for i in range(N_SIMS):
            # Generate data
            residuals = simulate_residuals(n_true)
            
            # Method 1: Permutation
            est_perm = estimate_n_sv(residuals, method="permutation", n_perm=20)
            
            # Method 2: BIC
            est_bic = estimate_n_sv(residuals, method="bic")
            
            results.append({"True": n_true, "Method": "Permutation", "Estimate": est_perm})
            results.append({"True": n_true, "Method": "BIC", "Estimate": est_bic})
            
    df = pd.DataFrame(results)
    
    # Summary
    print("\n" + "="*60)
    print("ESTIMATION ACCURACY SUMMARY (Mean Estimate)")
    print("="*60)
    summary = df.groupby(["True", "Method"])["Estimate"].agg(['mean', 'std'])
    print(summary)
    
    # Check Accuracy (Exact Hits)
    df['Correct'] = df['Estimate'] == df['True']
    accuracy = df.groupby(["True", "Method"])['Correct'].mean()
    print("\n" + "="*60)
    print("ACCURACY (% Exact Matches)")
    print("="*60)
    print(accuracy)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Offset positions for clarity
    methods = ["Permutation", "BIC"]
    colors = ["salmon", "skyblue"]
    
    for i, method in enumerate(methods):
        subset = df[df["Method"] == method]
        # Jitter x for visibility
        x = subset["True"] + (i * 0.2 - 0.1)
        y = subset["Estimate"]
        ax.scatter(x, y, label=method, color=colors[i], alpha=0.6, s=50)
        
    ax.plot([-1, 6], [-1, 6], 'k--', alpha=0.5, label="Perfect")
    ax.set_xlabel("True Number of SVs")
    ax.set_ylabel("Estimated Number of SVs")
    ax.set_title("n_sv Estimation: Permutation vs BIC")
    ax.legend()
    ax.set_xticks(true_sv_counts)
    
    plt.savefig("benchmark_nsv.png")
    print("\nPlot saved to benchmark_nsv.png")

if __name__ == "__main__":
    run_benchmark()
