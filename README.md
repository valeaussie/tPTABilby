Transdimensional Pulsar Timing Noise Modelling (Scripts)
Overview

This repository contains scripts for transdimensional noise modelling of pulsar timing data using a Bilby / tBilby + Enterprise framework.

The core idea is to perform model selection and parameter estimation simultaneously by allowing different noise components to be switched on and off during sampling. This is implemented using a transdimensional approach with binary indicators.

These scripts are designed for single-pulsar analyses and validation against standard Enterprise workflows.

Key Features
Transdimensional noise modelling (automatic component selection)
Joint model selection + parameter estimation
Built on:
Bilby
tBilby
Enterprise
Modular scripts for:
model setup
sampling
post-processing
Repository Structure (Relevant Only)
scripts/
│
├── tbilby_scripts/        # Core transdimensional modelling logic
├── post_pred_scripts/     # Posterior predictive analysis
├── pp_plots_scripts/      # Plotting for posterior predictive checks
├── ...                    # Other helper / execution scripts

Requirements
Python 3.8+
bilby
enterprise
enterprise_extensions (if used)
numpy
scipy
matplotlib

Install dependencies (example):

pip install bilby enterprise numpy scipy matplotlib
Typical Workflow
1. Run Transdimensional Analysis

Navigate to the core scripts:

cd scripts/tbilby_scripts/

Run the main analysis script:

python run_tbilby_analysis.py

Adjust script names or arguments depending on your configuration.

2. Posterior Predictive Checks (Optional)

Run posterior predictive simulations:

cd ../post_pred_scripts/
python run_post_pred.py
3. Generate Diagnostic Plots
cd ../pp_plots_scripts/
python make_plots.py
Method Summary

This framework uses a transdimensional Bayesian approach where:

Each noise component (e.g. red noise, DM noise) is assigned a binary indicator
The sampler explores:
whether a component is included
the parameters of that component (if active)

This allows:

model selection and parameter estimation in a single run
direct inference on which noise processes are supported by the data
Outputs

Typical outputs include:

Posterior samples (Bilby result objects)
Noise parameter estimates
Model indicator probabilities (component inclusion)
Posterior predictive diagnostics and plots
Reproducibility

To reproduce results:

Use the same pulsar data and priors defined in the scripts
Keep sampler settings consistent (e.g. number of live points)
Ensure dependency versions are compatible
Citation

If you use this work, please cite the associated paper.
