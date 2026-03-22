#!/usr/bin/env python
import bilby
import numpy as np
import pandas as pd
import math, os
import matplotlib.pyplot as plt
import json
import scipy
from scipy.stats import ks_2samp


# ----------- Paths -----------
Matt_json = "/fred/oz002/users/mmiles/MPTA_DR3/analysis/enterprise_runs/out_pbilby/thirdpass/J1713+0747/J1713+0747_CHROMCIDX_SGWB/CHROMCIDX_SGWB_final_res.json"
CHAIN_FILE = "/fred/oz005/users/vdimarco/tBilby/results/chains/singlePsrNoise/J1713+0747_tm+ef+ec+eq+chrom+gw_1/chain_1.txt"

# Load chain: whitespace-separated, no header
chain = np.loadtxt(CHAIN_FILE)
print("Matt chain shape:", chain.shape)

# drop burn-in first 25% samples
burn_frac = 0.25
nburn     = int(burn_frac * chain.shape[0])
chain     = chain[nburn:, :]
print(f"Using {chain.shape[0]} samples after dropping first {nburn} as burn-in.")

# Model
mod = "Checked data with matt CH only"
# tBilby model key (like "1-0-1"), to select which model to compare:

with open(Matt_json) as f:
    outer = json.load(f)
data_Matt = json.loads(outer)


# ----------- Config -----------
PULSAR = "J1713+0747"


# ----------- Main code -----------
# Random number generator
rng = np.random.default_rng()


# =========================
# Load Matt's preferred-model samples
# =========================
# Map columns -> parameters
# g_RN  = chain[:, 5]
# A_RN  = chain[:, 6]
# g_DM  = chain[:, 2]
# A_DM  = chain[:, 3]
A_CH  = chain[:, 1]
g_CH  = chain[:, 0]
A_GW  = chain[:, 2]
g_GW  = np.full_like(A_GW, 13.0/3.0)  # fixed GW slope


# Matt params dict (same structure as params_tBilby)
params_Matt = {
    # "A_RN": A_RN,
    # "g_RN": g_RN,
    # "A_DM": A_DM,
    # "g_DM": g_DM,
    "A_CH": A_CH,
    "g_CH": g_CH,
    "A_GW": A_GW,
    "g_GW": g_GW,
}


# =========================
# Load Matt's preferred-model samples
# =========================
df_pref = pd.DataFrame(data_Matt)
print("Loaded Matt's samples with columns:", df_pref.columns)

pref = {c: df_pref[c].to_numpy() for c in df_pref.columns}
A_DM_p = pref.get(f"{PULSAR}_dm_gp_log10_A")
g_DM_p = pref.get(f"{PULSAR}_dm_gp_gamma")
A_CH_p = pref.get(f"{PULSAR}_chromcidx_gp_log10_A", pref.get(f"{PULSAR}_chrom_gp_log10_A"))
g_CH_p = pref.get(f"{PULSAR}_chromcidx_gp_gamma",   pref.get(f"{PULSAR}_chrom_gp_gamma"))
A_SW_p = pref.get(f"{PULSAR}_sw_gp_log10_A")
g_SW_p = pref.get(f"{PULSAR}_sw_gp_gamma")
A_GW_p = pref.get("log10_A_gw")
g_GW_p = np.full_like(A_GW_p, 13.0/3.0) if A_GW_p is not None else None
A_RN_p = pref.get(f"{PULSAR}_red_noise_log10_A")
g_RN_p = pref.get(f"{PULSAR}_red_noise_gamma")

# Matt params dict
params_Matt = {
    "A_RN": A_RN_p,
    "g_RN": g_RN_p,
    "A_DM": A_DM_p,
    "g_DM": g_DM_p,
    "A_CH": A_CH_p,
    "g_CH": g_CH_p,
    "A_SW": A_SW_p,
    "g_SW": g_SW_p,
    "A_GW": A_GW_p,
    "g_GW": g_GW_p,
}


# =========================
# KS test between Enterprise and Matt preferred-model samples
# =========================

print("\n--- KS test between Enterprise and Matt preferred-model samples ---")
for name, arr_en, arr_mt in [
    ("CH log10_A", A_CH, A_CH_p),
    ("CH gamma",   g_CH, g_CH_p),
    ("GW log10_A", A_GW, A_GW_p),
]:
    if arr_en is None or arr_mt is None:
        print(f"Skipping {name} (missing in one of the sets).")
        continue
    ks_res = ks_2samp(arr_en, arr_mt)
    print(f"{name}: KS stat = {ks_res.statistic:.4f}, p-value = {ks_res.pvalue:.4f}")

# plotting histograms of A_CH
name = "CH log10_A"
arr_en = A_CH
arr_mt = A_CH_p
plt.hist(arr_mt, bins=30, alpha=0.5, label='Matt preferred-model', density=True)
plt.hist(arr_en, bins=30, alpha=0.5, label='Enterprise model', density=True)
plt.xlabel(name)
plt.ylabel('Density')
plt.title(f'Histogram of {name}')
plt.legend()
plt.savefig("density_A_CH_Enterprise_Matt.png")
plt.close()

# plotting histograms of g_CH
name = "CH gamma"
arr_en = g_CH
arr_mt = g_CH_p
plt.hist(arr_mt, bins=30, alpha=0.5, label='Matt preferred-model', density=True)
plt.hist(arr_en, bins=30, alpha=0.5, label='Enterprise model', density=True)
plt.xlabel(name)
plt.ylabel('Density')
plt.title(f'Histogram of {name}')
plt.legend()
plt.savefig("density_g_CH_Enterprise_Matt.png")
plt.close()

# plotting histograms of A_GW
name = "GW log10_A"
arr_en = A_GW
arr_mt = A_GW_p
plt.hist(arr_mt, bins=30, alpha=0.5, label='Matt preferred-model', density=True)
plt.hist(arr_en, bins=30, alpha=0.5, label='Enterprise model', density=True)
plt.xlabel(name)
plt.ylabel('Density')
plt.title(f'Histogram of {name}')
plt.legend()
plt.savefig("density_A_GW_Enterprise_Matt.png")
plt.close()

