#!/usr/bin/env python
import bilby
import numpy as np
import pandas as pd
import math, os
import matplotlib.pyplot as plt
import json
import scipy
from scipy.stats import ks_2samp


MATT_KEY = "0-0-1"
print("MATT_KEY =", MATT_KEY)

# ----------- Paths -----------
RESFILE_TB = "/fred/oz005/users/vdimarco/tBilby/results/Meerkat_J1713+0747_5000_livepoints_apples_with_apples_4/Meerkat-J1713+0747_5000_livepoints_result.json"
Matt_json = "/fred/oz002/users/mmiles/MPTA_DR3/analysis/enterprise_runs/out_pbilby/thirdpass/J1713+0747/J1713+0747_CHROMCIDX_SGWB/_CHROMCIDX_SGWB_final_res.json"
# Model
mod = "Checked data with matt CH only"
# tBilby model key (like "1-0-1"), to select which model to compare:

with open(Matt_json) as f:
    outer = json.load(f)
data_Matt = json.loads(outer)


# ----------- Config -----------
mod = "with_enterprise_complete_model"
PULSAR = "J1713+0747"
DIR_SAVE = "/fred/oz005/users/vdimarco/tBilby/plots/posterior_predictive_checks"
os.makedirs(DIR_SAVE, exist_ok=True)
FIG = os.path.join(DIR_SAVE, f"{PULSAR}_ppd_prefModel_compare_{mod}_SGWB.png")
N_HAIR, N_BAND, N_PREF = 250, 2000, 2000
USE_CHAR_STRAIN = False    # False -> residual PSD units (s^2/Hz)
GWB_FID_LOG10_A, GWB_FID_GAMMA = -14.35, 13.0/3.0

# ----------- Frequency grid -----------
YR = 365.25*24*3600.0
DAY = 24*3600.0
T_YRS, CAD_D = 4.5, 14
T = T_YRS * YR
cad = CAD_D * DAY
n_max = int(np.floor(T/cad))
f = (np.arange(1, n_max+1)) / T
fref = 1.0/YR

# ----------- Functions -----------
# Function that computes power spectrum from amplitude and spectral index
def psd_pow(f, log10_A, gamma):
    A = 10.0 ** log10_A
    alpha = (3.0 - gamma)/2.0
    h_c = A * (f/fref)**alpha
    if USE_CHAR_STRAIN:
        return h_c
    else:
        return (h_c**2) / (12.0 * math.pi**2 * f**3)

# Function that computes total spectrum from a given posterior row
def total_spec_from_row(i, A_RN, g_RN, A_DM, g_DM, A_CH, g_CH, A_SW, g_SW, A_GW=None, g_GW=None):
    S = np.zeros_like(f)

    def add_if_finite(S, A, g):
        if A is None or g is None:
            return S
        ai = A[i]; gi = g[i]
        if np.isfinite(ai) and np.isfinite(gi):
            S += psd_pow(f, ai, gi)
        return S

    S = add_if_finite(S, A_RN, g_RN)
    S = add_if_finite(S, A_DM, g_DM)
    S = add_if_finite(S, A_CH, g_CH)
    S = add_if_finite(S, A_SW, g_SW)
    S = add_if_finite(S, A_GW, g_GW)
    return S

# Function to find indicator columns
def find_ind_cols(columns):
    return sorted([c for c in columns if c.startswith("n") and c[1:].isdigit()],
                  key=lambda x: int(x[1:]))

# Function to create model key from indicator bits
def key_from_bits(bits):
    return "-".join(str(int(b)) for b in bits)

# Function to find first existing column from candidates in given dataframes    
def first_existing(cands, frames):
    for c in cands:
        for df in frames:
            if c in df.columns:
                return c
    return None

# wrapper function to compute spectrum from params dict
def spec_from_params(j, pars):
    return total_spec_from_row(
        j,
        pars["A_RN"], pars["g_RN"],
        pars["A_DM"], pars["g_DM"],
        pars["A_CH"], pars["g_CH"],
        pars["A_SW"], pars["g_SW"],
        pars["A_GW"], pars["g_GW"]
    )

# ----------- Main code -----------
# Random number generator
rng = np.random.default_rng()

# =========================
# Load tBilby result
# =========================
res_tb = bilby.result.read_in_result(RESFILE_TB)
post_tb = res_tb.posterior

# =========================
print("search params:", res_tb.search_parameter_keys)
print("fixed  params:", res_tb.fixed_parameter_keys)
nest_tb = res_tb.nested_samples


# Indicators & per-model probabilities (from nested weights)
ind_cols = find_ind_cols(list(nest_tb.columns))
print("Indicator columns found:", ind_cols)
if not ind_cols:
    raise RuntimeError("No indicator columns in nested_samples.")
# Extract indicator columns (e.g. n2, n3, n4) as an integer NumPy array
I = nest_tb[ind_cols].to_numpy(int)
# Extract weights
w = nest_tb["weights"].to_numpy()
# Ensure no negative weights (can happen due to numerical issues)
w = np.maximum(w, 0.0)
# Normalize weights
w /= w.sum()

# Find all unique model indicator combinations (rows in I) 
# and map each sample to its corresponding unique model index
uniq, inv = np.unique(I, axis=0, return_inverse=True)
# Sum the normalised weights of samples belonging to each unique model
# gives the posterior probability for each model
P = np.bincount(inv, weights=w, minlength=len(uniq))
# Normalize model probabilities
P /= P.sum()
# Create model keys like "1-0-1" from unique indicator rows
keys = [key_from_bits(u) for u in uniq]

# indices per model in nested samples
idx_per_model_nest = {k: np.where(inv == i)[0] for i, k in enumerate(keys)}
print("indices per model in nested samples:", {k: len(v) for k,v in idx_per_model_nest.items()})

# Map posterior rows to models

ind_cols_post = ind_cols
print("Indicator columns used for posterior model mapping:", ind_cols_post)


Ipost = post_tb[ind_cols_post].to_numpy(int)
model_mask = {}
for u, k in zip(uniq, keys):
    model_mask[k] = np.all(Ipost == u[None, :], axis=1)

idx_per_model_post = {k: np.nonzero(mask)[0] for k, mask in model_mask.items()}

# =========================
# Detect the tBilby model chosen
# =========================

def getp(names):
    for c in names:
        if c in post_tb:
            return c  # return the column name
    return None
RN_A = getp([f"{PULSAR}_red_noise_log10_A"])
RN_G = getp([f"{PULSAR}_red_noise_gamma"])
DM_A = getp([f"{PULSAR}_dm_gp_log10_A"])
DM_G = getp([f"{PULSAR}_dm_gp_gamma"])
CH_A = getp([f"{PULSAR}_chromcidx_gp_log10_A", f"{PULSAR}_chrom_gp_log10_A"])
CH_G = getp([f"{PULSAR}_chromcidx_gp_gamma",   f"{PULSAR}_chrom_gp_gamma"])
SW_A = getp([f"{PULSAR}_sw_gp_log10_A"])
SW_G = getp([f"{PULSAR}_sw_gp_gamma"])
GW_A = getp([f"{PULSAR}_gwb_log10_A", "log10_A_gw"])
GW_G = getp([f"{PULSAR}_gwb_gamma",   "gamma_gw"])

print("Detected tBilby parameter columns:")
print(RN_A, RN_G, DM_A, DM_G, CH_A, CH_G, SW_A, SW_G, GW_A, GW_G)

matt_key_tb = MATT_KEY
print(f"Using tBilby model matching Matt's: {matt_key_tb}")

idx_m_post = idx_per_model_post[matt_key_tb]
print(f"tBilby model {matt_key_tb} has {len(idx_m_post)} posterior rows.")
if len(idx_m_post) == 0:
    raise RuntimeError("Selected tBilby model has zero posterior rows.")

print(f"Using tBilby model matching Matt's: {matt_key_tb}")

print("\n=== Model indicator patterns (full n0–n5) ===")
for k, u in zip(keys, uniq):
    pattern = {col: int(bit) for col, bit in zip(ind_cols, u)}
    print(f"Model {k}: {pattern}")

# Restrict to that single model (no marginalisation)
kidx = keys.index(matt_key_tb)
print("Unique model keys:", kidx, keys)

keys = [matt_key_tb]
print("tBilby model keys now:", keys)
print(
    f"tBilby model {matt_key_tb} has posterior weight {P[kidx]:.4f} "
    f"with {len(idx_per_model_post[matt_key_tb])} posterior rows "
    f"and {len(idx_per_model_nest[matt_key_tb])} nested rows."
)

P = np.array([P[kidx]], dtype=float)
uniq = np.array([uniq[kidx]])

if len(idx_per_model_post[matt_key_tb]) == 0:
    raise RuntimeError("Selected tBilby model has zero posterior rows.")

# =========================
# Extract tBilby posterior columns (arrays)
# =========================

def get_arr_tb(name):
    return post_tb[name].to_numpy() if (name and name in post_tb.columns) else None


# Extract parameter arrays
A_RN = get_arr_tb(RN_A); g_RN = get_arr_tb(RN_G)
A_DM = get_arr_tb(DM_A); g_DM = get_arr_tb(DM_G)
A_CH = get_arr_tb(CH_A); g_CH = get_arr_tb(CH_G)
A_SW = get_arr_tb(SW_A); g_SW = get_arr_tb(SW_G)
A_GW = get_arr_tb(GW_A)
g_GW = np.full_like(A_GW, 13.0/3.0) if A_GW is not None else None

A_RN = A_RN[idx_m_post] if A_RN is not None else None
g_RN = g_RN[idx_m_post] if g_RN is not None else None
A_DM = A_DM[idx_m_post] if A_DM is not None else None
g_DM = g_DM[idx_m_post] if g_DM is not None else None
A_CH = A_CH[idx_m_post] if A_CH is not None else None
g_CH = g_CH[idx_m_post] if g_CH is not None else None
A_SW = A_SW[idx_m_post] if A_SW is not None else None
g_SW = g_SW[idx_m_post] if g_SW is not None else None
A_GW = A_GW[idx_m_post] if A_GW is not None else None
g_GW = g_GW[idx_m_post] if g_GW is not None else None

# Build the dictionary of tBilby parameters
params_tBilby = {
    "A_RN": A_RN,
    "g_RN": g_RN,
    "A_DM": A_DM,
    "g_DM": g_DM,
    "A_CH": A_CH,
    "g_CH": g_CH,
    "A_SW": A_SW,
    "g_SW": g_SW,
    "A_GW": A_GW,
    "g_GW": g_GW,
}


# =========================
# Load Matt's preferred-model samples
# =========================
df_pref = pd.DataFrame(data_Matt)
#df_pref = correct_model
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
# KS test between tBilby and Matt preferred-model samples
# =========================

print("\n--- KS test between tBilby and Matt preferred-model samples ---")
for name, arr_tb, arr_mt in [
    ("CH log10_A", A_CH, A_CH_p),
    ("CH gamma",   g_CH, g_CH_p),
    ("GW log10_A", A_GW, A_GW_p),
]:
    if arr_tb is None or arr_mt is None:
        print(f"Skipping {name} (missing in one of the sets).")
        continue
    ks_res = ks_2samp(arr_tb, arr_mt)
    print(f"{name}: KS stat = {ks_res.statistic:.4f}, p-value = {ks_res.pvalue:.4f}")

# plotting histograms of A_CH
name = "CH log10_A"
arr_tb = A_CH
arr_mt = A_CH_p
plt.hist(arr_mt, bins=30, alpha=0.5, label='Matt preferred-model', density=True)
plt.hist(arr_tb, bins=30, alpha=0.5, label='tBilby model', density=True)
plt.xlabel(name)
plt.ylabel('Density')
plt.title(f'Histogram of {name}')
plt.legend()
plt.savefig("density_A_CH_tBilby_Matt.png")
plt.close()

# plotting histograms of g_CH
name = "CH gamma"
arr_tb = g_CH
arr_mt = g_CH_p
plt.hist(arr_mt, bins=30, alpha=0.5, label='Matt preferred-model', density=True)
plt.hist(arr_tb, bins=30, alpha=0.5, label='tBilby model', density=True)
plt.xlabel(name)
plt.ylabel('Density')
plt.title(f'Histogram of {name}')
plt.legend()
plt.savefig("density_g_CH_tBilby_Matt.png")
plt.close()

# plotting histograms of A_GW
name = "GW log10_A"
arr_tb = A_GW
arr_mt = A_GW_p
plt.hist(arr_mt, bins=30, alpha=0.5, label='Matt preferred-model', density=True)
plt.hist(arr_tb, bins=30, alpha=0.5, label='tBilby model', density=True)
plt.xlabel(name)
plt.ylabel('Density')
plt.title(f'Histogram of {name}')
plt.legend()
plt.savefig("density_A_GW_tBilby_Matt.png")
plt.close()

