# Transdimensional Pulsar Timing Noise Modelling (Scripts)

## Overview

This repository contains scripts for transdimensional noise modelling of pulsar timing data using a transdimentional framework.

The core idea is to perform **model selection and parameter estimation simultaneously** by allowing different noise components to be switched on and off during sampling. This is implemented using a transdimensional approach with binary indicators.

These scripts are designed for single-pulsar analyses and validation against standard Enterprise workflows.

---

## Key Features

- Transdimensional noise modelling (automatic component selection)
- Joint model selection + parameter estimation
- Modular scripts for:
  - model setup
  - sampling
  - post-processing

---

## Repository Structure

```
scripts/
│
├── tbilby_scripts/        # Core transdimensional modelling logic
├── post_pred_scripts/     # Posterior predictive analysis
├── pp_plots_scripts/      # Plotting for posterior predictive checks
├── ...                    # Other helper / execution scripts
```

---

## Requirements (to run tPTABilby)

- Bilby (https://git.ligo.org/lscsoft/bilby)
- tBilby (https://github.com/tBilby/tBilby)
- Enterprise (https://github.com/nanograv/enterprise)
- Enterprise_extensions (https://github.com/nanograv/enterprise_extensions)
- Enterprise_warp (https://enterprise-warp.readthedocs.io/en/latest/)

---

## Typical Workflow

### 1. Run Transdimensional Analysis

Navigate to the core scripts:

```
cd scripts/tbilby_scripts/
```

Run the main analysis script:

```
python run_tbilby_analysis.py
```

> The models are built with Enterprise language and are implemented in the noise_definition.py file.

---

### 2. Posterior Predictive Checks (to validate the method against other methods)

```
cd ../post_pred_scripts/
python run_post_pred.py
```

---

### 3. Generate Diagnostic Plots

```
cd ../pp_plots_scripts/
python make_plots.py
```

---

## Method Summary

This framework uses a **transdimensional Bayesian approach** where:

- Each noise component (e.g. red noise, DM noise) is assigned a binary indicator
- The sampler explores:
  - whether a component is included
  - the parameters of that component (if active)

This allows:

- model selection and parameter estimation in a single run
- direct inference on which noise processes are supported by the data


## Citation

If you use this work, please cite the associated paper.

(BibTeX to be added)

---

## Acknowledgements

This work builds on:

- Bilby: https://git.ligo.org/lscsoft/bilby
- tBilby: https://github.com/tBilby/tBilby
- Enterprise: https://github.com/nanograv/enterprise
