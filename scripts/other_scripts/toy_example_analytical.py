import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# True parameters
mu_true = 0.0
sigma_true = 1.0

# Competing models
sigma_model1 = 1.0  # matches truth
sigma_model2 = 2.0  # wrong noise model

# Priors for mean (mu) in both models: N(0, prior_sigma)
prior_mu_mean = 0.0
prior_mu_sigma = 5.0

n_sims = 2000 # Number of simulations
n_points = 10 # Number of data points per simulation

param_quantiles = []
true_model_posteriors = []

for _ in range(n_sims):
    # Generate data from true model
    data = np.random.normal(mu_true, sigma_true, size=n_points)
    
    # --- Model 1 posterior for mu (analytic Gaussian update) ---
    post_var1 = 1 / (1/prior_mu_sigma**2 + n_points/sigma_model1**2)
    post_mean1 = post_var1 * (prior_mu_mean/prior_mu_sigma**2 + data.sum()/sigma_model1**2)
    
    # --- Model 2 posterior for mu ---
    post_var2 = 1 / (1/prior_mu_sigma**2 + n_points/sigma_model2**2)
    post_mean2 = post_var2 * (prior_mu_mean/prior_mu_sigma**2 + data.sum()/sigma_model2**2)
    
    # Evidence for model 1 and model 2 (integrating over mu)
    like_var1 = sigma_model1**2
    evidence1 = norm.pdf(data, loc=prior_mu_mean, scale=np.sqrt(like_var1 + prior_mu_sigma**2)).prod()
    
    like_var2 = sigma_model2**2
    evidence2 = norm.pdf(data, loc=prior_mu_mean, scale=np.sqrt(like_var2 + prior_mu_sigma**2)).prod()
    
    # Model posterior for the true model (model 1 here)
    p_true_model = evidence1 / (evidence1 + evidence2)
    true_model_posteriors.append(p_true_model)
    
    # Parameter quantile for true mu in true model's posterior
    cdf_val = norm.cdf(mu_true, loc=post_mean1, scale=np.sqrt(post_var1))
    param_quantiles.append(cdf_val)


# -----------------
# Plotting results
# -----------------

# Convert to numpy arrays
param_quantiles = np.array(param_quantiles)
true_model_posteriors = np.array(true_model_posteriors)

sorted_q = np.sort(param_quantiles)
pp_theory = np.linspace(0, 1, len(sorted_q))

# Prepare 3-panel figure
plt.figure(figsize=(18, 5))  # wide for 3 plots

# P-P plot for parameter quantiles
plt.subplot(1, 3, 1)
plt.plot(pp_theory, sorted_q, marker='o', linestyle='', alpha=0.3)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Theoretical quantile')
plt.ylabel('Empirical quantile')
plt.title('P–P plot for μ (true model)')
plt.grid(True, linestyle='--', alpha=0.6)

# Histogram of true model posterior probabilities
plt.subplot(1, 3, 2)
plt.hist(true_model_posteriors, bins=20, range=(0, 1), alpha=0.7, edgecolor='k')
plt.xlabel('Posterior probability for true model')
plt.ylabel('Count')
plt.title('True model posterior probabilities')
plt.grid(True, linestyle='--', alpha=0.6)

# Calibration plot for model posterior probabilities
num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
counts = np.zeros(num_bins)
correct_counts = np.zeros(num_bins)

# true model is always correct = 1, so correct_counts = counts here but keeping form for generality
for p in true_model_posteriors:
    bin_idx = np.digitize(p, bins) - 1
    if 0 <= bin_idx < num_bins:
        counts[bin_idx] += 1
        correct_counts[bin_idx] += 1

fraction_correct = np.divide(correct_counts, counts, out=np.zeros_like(correct_counts), where=counts>0)

plt.subplot(1, 3, 3)
plt.plot(bin_centers, fraction_correct, marker='o', linestyle='-', label='Empirical freq.')
plt.plot([0,1], [0,1], 'r--', label='Perfect calibration')
plt.xlabel('Predicted posterior probability for true model')
plt.ylabel('Fraction of times true model is correct')
plt.title('Calibration plot for model posterior probabilities')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('pp_and_model_posteriors.png', dpi=150)
plt.show()
