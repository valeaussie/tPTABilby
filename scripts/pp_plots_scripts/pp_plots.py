import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Define folder path
folder_path = '/fred/oz005/users/vdimarco/tBilby/300_sims_tight_uniforms_2/p_values'
# Prepare storage for the 32 models
model_pvalues = {i: [] for i in range(0, 32)}

# Load and process each file
for i in range(0, 299):
    file_path = os.path.join(folder_path, f'p_values_model_{i}.json')
    try:
        with open(file_path, 'r') as f:
            d = json.load(f)
            injected = d["injected"]
            pval = d.get(str(injected), 0.0)  # Get p-value or 0.0 if missing
            model_pvalues[injected].append(pval)
            #print(f"Model {injected} p-value: {pval}")
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping.")
        continue

#print(model_pvalues)
os.makedirs("pval_histograms", exist_ok=True)

# Combine all p-values from all models into a single list
all_pvals = []
for pvals in model_pvalues.values():
    all_pvals.extend(pvals)

# Sort the p-values
sorted_pvals = np.sort(all_pvals)

# Empirical CDF
n = len(sorted_pvals)
empirical_cdf = np.arange(1, n + 1) / n

# Expected CDF under uniform distribution
expected_cdf = sorted_pvals  # since uniform(0,1)

# Plot the PP plot
plt.figure(figsize=(8, 6))
plt.plot(expected_cdf, empirical_cdf, marker='o', linestyle='-', color='blue', label='Empirical CDF')
plt.plot([0, 1], [0, 1], 'k--', label='Ideal (Uniform)')  # reference line

plt.xlabel('p(M_true | y)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('PP plot of p-values', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('pval_histograms/pp_plot_tight_uniforms_2.pdf', bbox_inches="tight")
plt.close()


# Define bins
bins = np.linspace(0, 1, 11)

# # Plot histograms for each model
# for model_to_plot in range(0, 32):
#     pvals = model_pvalues.get(model_to_plot, [])

#     counts, bins = np.histogram(pvals, bins=bins)
#     proportions = counts / sum(counts)  # Normalize by total number of p-values

#     # Plot normalized histogram
#     plt.figure()
#     plt.bar(bins[:-1], proportions, width=0.1, align='edge', edgecolor='black')

#     # Label and title
#     plt.xlabel('p-value')
#     plt.ylabel('Density')
#     plt.title(f'Histogram of p-values for model {model_to_plot}')

#     # Grid and save
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(f'pval_histograms/p_values_histogram_model_{model_to_plot}.png', dpi=300)
#     plt.close()

# Plot the histogram of all p-values (not normalized)
plt.figure(figsize=(8, 6))
plt.hist(all_pvals, bins=bins, edgecolor='black', color='steelblue')

# Label and title
plt.xlabel('p(M_true | y)')
plt.ylabel('Count')
plt.title('Histogram of p-values for all models')

# Grid and save
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('pval_histograms/p_values_histogram_all_models_combined_300_sims_tight_uniforms_2.png', dpi=300)
plt.close()

