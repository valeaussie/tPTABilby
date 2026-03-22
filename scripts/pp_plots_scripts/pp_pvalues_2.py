import json
import numpy as np
import bilby
import matplotlib.pyplot as plt
from scipy.stats import kstest
import os

# Paths and constants
base_results_dir = "/fred/oz005/users/vdimarco/tBilby/results/300_sims_tight_uniforms"
base_sim_dir = "/fred/oz005/users/vdimarco/tBilby/sims/300_sims_tight_uniforms"
num_simulations = 300
psrname = "J0437-4715"
parameter_translation = {
    'efac': f'{psrname}_efac',
    'equad': f'{psrname}_log10_tnequad',
    'ecorr': f'{psrname}_log10_ecorr',
    'red_amp': f'{psrname}_red_noise_log10_A',
    'red_gamma': f'{psrname}_red_noise_gamma',
    'dm_amp': f'{psrname}_dm_gp_log10_A',
    'dm_gamma': f'{psrname}_dm_gp_gamma',
    'ch_amp': f'{psrname}_chrom_gp_log10_A',
    'ch_gamma': f'{psrname}_chrom_gp_gamma',
    # 'sw_amp': f'{psrname}_gp_sw_log10_A',   # only if solar wind is used
    # 'sw_gamma': f'{psrname}_gp_sw_gamma'
}

all_p_values = []
def compute_parameter_p_values(result_file, injected_params_file):
    """
    Computes posterior quantiles (p-values) for injected parameters based on Bilby result.
    """
    try:
        result = bilby.result.read_in_result(filename=result_file)
    except Exception as e:
        print(f"Could not load result file {result_file}: {e}")
        return None

    injected_params = {}
    try:
        with open(injected_params_file, "r") as f:
            for line in f:
                key, value = line.strip().split(": ", 1)
                injected_params[key] = float(value)
    except Exception as e:
        print(f"Could not read injected parameter file {injected_params_file}: {e}")
        return None

    result = bilby.result.read_in_result(filename=result_file)

    p_values = {}
    for injected_key, true_val in injected_params.items():
        bilby_key = parameter_translation.get(injected_key)
        if not bilby_key or bilby_key not in result.posterior.columns:
            print(f"Skipping parameter {injected_key}: not found in Bilby result.")
            continue
        samples = result.posterior[bilby_key].values
        quantile = np.mean(samples < true_val)
        p_values[bilby_key] = quantile
    return p_values

def plot_rank_histograms(p_value_list, parameter_name):
    # Extract p-values for the given parameter
    p_vals = [sim[parameter_name] for sim in p_value_list if parameter_name in sim]

    # Perform KS test against uniform distribution
    ks_stat, ks_pval = kstest(p_vals, 'uniform')

    # Create the histogram plot
    plt.figure(figsize=(6, 4))
    plt.hist(p_vals, bins=10, range=(0,1), edgecolor='black')
    plt.title(f"P-val hist for {parameter_name}", fontsize=16)
    plt.xlabel("Posterior quantile", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add horizontal line for expected uniform frequency
    expected = len(p_vals)/10
    plt.axhline(expected, color='red', linestyle='--')
                #label=f'Expected (Uniform)\nKS stat = {ks_stat:.3f}, p = {ks_pval:.3f}')
    
    # Add legend and layout
    #plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the plot
    save_path = f"pvalue_histogram_{parameter_name}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved plot to {save_path}")

    plt.close()


def ks_test(p_vals):
    stat, p = kstest(p_vals, 'uniform')
    print(f"KS test: D = {stat:.3f}, p = {p:.3f}")

# Loop over all simulations
for i in range(num_simulations):
    result_dir = os.path.join(base_results_dir)
    result_file = os.path.join(
        result_dir,
        f"test_sim_{i}-model_full_{psrname}_5000_livepoints",
        f"sims_test_sim_{i}-model_full_{psrname}_5000_livepoints_result.json"
    )
    sim_dir = os.path.join(base_sim_dir, f"test_sim_{i}")
    injected_file = os.path.join(sim_dir, "parameters.txt")

    if not os.path.exists(result_file) or not os.path.exists(injected_file):
        print(f"Skipping simulation {i}: result or parameter file not found.")
        continue

    pvals = compute_parameter_p_values(result_file, injected_file)
    if pvals:
        all_p_values.append(pvals)

# Gather all parameter keys
all_params = set()
for sim in all_p_values:
    all_params.update(sim.keys())

# Plot histograms and KS tests
for param in sorted(all_params):
    print("here")
    print(f"\n--- {param} ---")
    plot_rank_histograms(all_p_values, param)
