
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pysva.sva import sva_iterative, sva_iterative_residual, sva_iterative_weighted

# Settings
N_SIMS = 20
N_SAMPLES = 50
N_GENES = 500
N_DE = 50

def simulate_data(confounded=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # Primary Variable (Treatment)
    treatment = np.array([0]*(N_SAMPLES//2) + [1]*(N_SAMPLES//2))
    X = np.column_stack([np.ones(N_SAMPLES), treatment])
    
    # Hidden Factor (SV)
    if confounded:
        # Correlated with treatment
        true_sv = treatment * 0.5 + np.random.randn(N_SAMPLES) * 0.5
    else:
        # Independent
        true_sv = np.random.randn(N_SAMPLES)
        
    # Effects
    baseline = np.random.randn(N_GENES) * 2
    treatment_effect = np.zeros(N_GENES)
    treatment_effect[:N_DE] = 1.0 # True Positives
    sv_effect = np.random.randn(N_GENES) * 0.8
    noise = np.random.randn(N_SAMPLES, N_GENES) * 0.5
    
    # Combine
    Y = np.zeros((N_SAMPLES, N_GENES))
    for i in range(N_SAMPLES):
        Y[i, :] = baseline + treatment[i] * treatment_effect + noise[i, :] + true_sv[i] * sv_effect
        
    return Y, X, true_sv, treatment

def get_power(Y, X, sv_est):
    """Calculate proportion of DE genes recovered."""
    X_full = np.column_stack([X, sv_est])
    n_params = X_full.shape[1]
    pvals = []
    
    for j in range(N_GENES):
        y = Y[:, j]
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
        resid = y - X_full @ beta
        mse = np.sum(resid ** 2) / (N_SAMPLES - n_params)
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
        se = np.sqrt(mse * XtX_inv[1, 1])
        t_stat = beta[1] / se
        p = 2 * (1 - stats.t.cdf(np.abs(t_stat), N_SAMPLES - n_params))
        pvals.append(p)
    
    # Simple p < 0.05 cutoff (not adjusting for multiple testing for simplicity of benchmark)
    tp = np.sum(np.array(pvals)[:N_DE] < 0.05)
    return tp / N_DE

def run_benchmark():
    results = []
    
    print(f"Running Benchmark ({N_SIMS} sims per scenario)...")
    
    for scenario in ["Confounded", "Unconfounded"]:
        is_confounded = (scenario == "Confounded")
        
        for i in range(N_SIMS):
            Y, X, true_sv, treatment = simulate_data(confounded=is_confounded, seed=i*100)
            
            # Method 1: New (Raw Data)
            sv_new = sva_iterative(Y, X, n_sv=1, n_iter=5)
            
            # Method 2: Old (Residuals)
            sv_old = sva_iterative_residual(Y, X, n_sv=1, n_iter=5)
            
            # Method 3: Weighted (Bootstrap n=20)
            sv_weighted = sva_iterative_weighted(Y, X, n_sv=1, n_iter=5, n_boot=20)

            # Method 4: Weighted (Bootstrap n=100)
            sv_weighted_100 = sva_iterative_weighted(Y, X, n_sv=1, n_iter=5, n_boot=100)
            
            # Collect Metrics
            methods_to_test = [
                ("Null-Raw-Data", sv_new), 
                ("Null-Residuals", sv_old),
                ("Weighted-Boot-20", sv_weighted),
                ("Weighted-Boot-100", sv_weighted_100)
            ]
            
            for method, sv_est in methods_to_test:
                # 1. SV Recovery (Correlation with Truth)
                acc = np.abs(np.corrcoef(true_sv, sv_est[:,0])[0,1])
                
                # 2. Safety (Correlation with Treatment - should be low for Unconfounded)
                # Note: In Confounded case, this will naturally be high because SV IS confounded.
                safety = np.abs(np.corrcoef(treatment, sv_est[:,0])[0,1])
                
                # 3. Power (Signal Recovery)
                power = get_power(Y, X, sv_est)
                
                results.append({
                    "Scenario": scenario,
                    "Method": method,
                    "SV Accuracy": acc,
                    "Trt Correlation": safety,
                    "Power": power
                })
    
    df = pd.DataFrame(results)
    
    # Summary Table
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    summary = df.groupby(["Scenario", "Method"]).mean()
    print(summary)
    print("="*60)
    
    # Save Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ["SV Accuracy", "Trt Correlation", "Power"]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Use simple boxplot
        data_to_plot = []
        labels = []
        
        # Order: Conf-New, Conf-Old, Unconf-New, Unconf-Old
        groups = [
            ("Confounded", "New (Raw)"),
            ("Confounded", "Old (Resid)"),
            ("Unconfounded", "New (Raw)"),
            ("Unconfounded", "Old (Resid)")
        ]
        
        for scen, meth in groups:
            subset = df[(df["Scenario"] == scen) & (df["Method"] == meth)][metric]
            data_to_plot.append(subset)
            labels.append(f"{scen}\n{meth}")
            
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nPlot saved to benchmark_results.png")

if __name__ == "__main__":
    run_benchmark()
