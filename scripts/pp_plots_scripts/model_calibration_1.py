import os
import json
import numpy as np
import matplotlib.pyplot as plt

pvalue_dir = "/fred/oz005/users/vdimarco/tBilby/sims/300_sims_tight_uniforms_2/p_values"
num_bins = 10
bin_edges = np.linspace(0, 1, num_bins + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

print("printing bin edges:", bin_edges)
print("printing bin centres:", bin_centres)

# Store all model probabilities and true model identities
all_probs = []
all_truths = []

# Load all simulation results
for i in range(300):
    filename = os.path.join(pvalue_dir, f"p_values_model_{i}.json")
    if not os.path.exists(filename):
        continue
    with open(filename, 'r') as f:
        data = json.load(f)
    injected_model = int(data["injected"])
    for model_str, prob in data.items():
        if model_str == "injected":
            continue
        model_int = int(model_str)
        all_probs.append(float(prob))
        all_truths.append(1 if model_int == injected_model else 0)
        

# Convert to arrays
all_probs = np.array(all_probs) # these are the preditced probabilities for each (simulation, model) pair
all_truths = np.array(all_truths) # these are binary labels: 1 = true model, 0 = else

# Bin and compute empirical frequencies
counts_per_bin = np.zeros(num_bins)
correct_per_bin = np.zeros(num_bins)

# Loop over each predicted probability and its corresponding truth label
for prob, is_correct in zip(all_probs, all_truths):
    # Determine which bin this probability belongs to
    bin_index = np.digitize(prob, bin_edges) - 1
    # Only process if the probability falls within the defined bins
    if 0 <= bin_index < num_bins:
        # Increment the total count for this bin
        counts_per_bin[bin_index] += 1
        # Increment the "correct" count if this prediction was actually correct
        correct_per_bin[bin_index] += is_correct

# Compute the fraction of correct predictions per bin
fraction_correct = np.divide(correct_per_bin, counts_per_bin, out=np.zeros_like(correct_per_bin), where=counts_per_bin > 0)

#print(f"Fraction correct per bin: {fraction_correct}")

# Plot the calibration curve
plt.figure(figsize=(6, 5))
plt.plot(bin_centres, fraction_correct, marker='o', label='Empirical frequency')
plt.plot([0, 1], [0, 1], 'r--', label='y=x')
plt.xlabel('Posterior probability assigned to a model')
plt.ylabel('Fraction of times that model was correct')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("posterior_model_calibration_1.png")

# # Histogram of posterior probabilities assigned to the correct model
# injected_model_probs = []

# # Re-load only the injected model probabilities
# for i in range(300):
#     filename = os.path.join(pvalue_dir, f"p_values_model_{i}.json")
#     if not os.path.exists(filename):
#         continue
#     with open(filename, 'r') as f:
#         data = json.load(f)
#     injected_model = str(data["injected"])
#     if injected_model in data:
#         injected_model_probs.append(float(data[injected_model]))

# # Plot histogram
# plt.figure(figsize=(6, 5))
# plt.hist(injected_model_probs, bins=10, range=(0, 1), edgecolor='black')
# plt.xlabel('Posterior probability assigned to true model')
# plt.ylabel('Number of simulations')
# plt.tight_layout()
# plt.savefig("posterior_mass_histogram_3.png")
