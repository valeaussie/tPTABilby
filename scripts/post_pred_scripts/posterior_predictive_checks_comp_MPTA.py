#!/usr/bin/env python
import bilby
import numpy as np
import pandas as pd
import math, os
import matplotlib.pyplot as plt
import json


# ----------- Paths -----------
RESFILE_TB = "/fred/oz005/users/vdimarco/tBilby/results/Meerkat_J1713+0747_5000_livepoints_apples_with_apples_4/Meerkat-J1713+0747_5000_livepoints_result.json"
Matt_json = "/fred/oz002/users/mmiles/MPTA_DR3/analysis/enterprise_runs/out_pbilby/thirdpass/J1713+0747/J1713+0747_CHROMCIDX_SGWB/CHROMCIDX_SGWB_final_res.json"

# Model
mod = "Matt_pref_model"
# tBilby model key (like "1-0-1"), to select which model to compare:
MATT_KEY = "0-0-1"
print("MATT_KEY =", MATT_KEY)

with open(Matt_json) as f:
    outer = json.load(f)
data_Matt = json.loads(outer)
# Inspect keys
print("Matt's data keys:")
print(list(data_Matt.keys())[:20])

# ----------- Config -----------
PULSAR = "J1713+0747"
DIR_SAVE = "/fred/oz005/users/vdimarco/tBilby/plots/posterior_predictive_checks"
os.makedirs(DIR_SAVE, exist_ok=True)
FIG = os.path.join(DIR_SAVE, f"ppc_tPTABilby_MPTA_{PULSAR}.pdf")
N_HAIR, N_BAND, N_PREF = 250, 3000, 3000
USE_CHAR_STRAIN = False    # False -> residual PSD units (s^2/Hz)
GWB_FID_LOG10_A, GWB_FID_GAMMA = -14.35, 13.0/3.0

# ----------- Frequency grid -----------
YR = 365.25*24*3600.0
DAY = 24*3600.0
T_YRS, CAD_D = 1.9, 14
T = T_YRS * YR
cad = CAD_D * DAY
n_max = int(np.floor(T/cad))
f = (np.arange(1, n_max+1)) / T
fref = 1.0/YR
fref_CH = 1400


# ----------- Functions -----------
# Function that computes power spectrum from amplitude and spectral index
def psd_pow(f, log10_A, gamma):
    A = 10.0 ** log10_A
    alpha = (3.0 - gamma)/2.0
    h_c = A * (f/fref)**alpha
    return (h_c**2) / (12.0 * math.pi**2 * f**3)

# def powerlaw(f, log10_A, gamma):
#     return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * fref**(gamma-3) * f**(-gamma) * f[0])

# Function that computes total spectrum from a given posterior row
# def total_spec_from_row(i, A_RN, g_RN, A_DM, g_DM, A_CH, g_CH, A_SW, g_SW, A_GW=None, g_GW=None):
#     S = np.zeros_like(f)

#     def add_if_finite(S, A, g, kind="generic"):
#         if A is None or g is None:
#             return S
#         ai = A[i]; gi = g[i]
#         if not (np.isfinite(ai) and np.isfinite(gi)):
#             return S

#         if kind == "CH":
#             S += chrom_delay(f, ai, gi, nu_ref_MHz=fref_CH)
#         else:
#             S += powerlaw(f, ai, gi)

#         return S


#     S = add_if_finite(S, A_RN, g_RN, kind="generic")
#     S = add_if_finite(S, A_DM, g_DM, kind="generic")
#     S = add_if_finite(S, A_CH, g_CH, kind="generic")
#     S = add_if_finite(S, A_SW, g_SW, kind="generic")
#     S = add_if_finite(S, A_GW, g_GW, kind="generic")
#     return S

def total_spec_from_row(i, A_RN, g_RN, A_DM, g_DM, A_CH, g_CH, A_SW, g_SW, A_GW=None, g_GW=None):
    S = np.zeros_like(f)

    def add_if_finite(S, A, g):
        if A is None or g is None:
            return S
        ai = A[i]; gi = g[i]
        if np.isfinite(ai) and np.isfinite(gi):
            # S += powerlaw(f, ai, gi)
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

# CH delay amplitude at nu = fref_CH (so chromatic index factor = 1)
def chrom_delay(fid_freqs, log10_A, gamma, nu_ref_MHz=1400.0):
    # PSD in s^2/Hz (enterprise-style chromatic GP, at nu = nu_ref so no kappa factor)
    S = psd_pow_CH(fid_freqs, fref, log10_A, gamma)  # note: uses fref=1/yr inside scaling
    # Convert PSD to "delay amplitude" per Fourier bin (seconds)
    # Δf ~ 1/T, and you already used f[0] as Δf proxy in powerlaw()
    return np.sqrt(S * fid_freqs[0])


# ----------- Main code -----------
# Random number generator
rng = np.random.default_rng(12)

# =========================
# Load tBilby result
# =========================
res_tb = bilby.result.read_in_result(RESFILE_TB)
post_tb = res_tb.posterior

# =========================
print("search params:", res_tb.search_parameter_keys)
print("fixed  params:", res_tb.fixed_parameter_keys)
nest_tb = res_tb.nested_samples

print("All columns in nested_samples:")
print(list(nest_tb.columns))

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

# Set some components to None if not present in Matt's model
params_tBilby["A_RN"] = None
params_tBilby["g_RN"] = None
params_tBilby["A_DM"] = None
params_tBilby["g_DM"] = None
# params_tBilby["A_CH"] = None
# params_tBilby["g_CH"] = None
params_tBilby["A_SW"] = None
params_tBilby["g_SW"] = None
# params_tBilby["A_GW"] = None
# params_tBilby["g_GW"] = None


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

# Set some components to None if not present in Matt's model
params_Matt["A_RN"] = None
params_Matt["g_RN"] = None
params_Matt["A_DM"] = None
params_Matt["g_DM"] = None
# params_Matt["A_CH"] = None
# params_Matt["g_CH"] = None
params_Matt["A_SW"] = None
params_Matt["g_SW"] = None
# params_Matt["A_GW"] = None
# params_Matt["g_GW"] = None

print("printed Matt's params keys:", list(params_Matt.keys()))
print("printed tbilby's params values:", {k: v is not None for k,v in params_tBilby.items()})


# =========================
# Plot
# =========================
plt.figure(figsize=(7.2,5.2), dpi=140)

n_tb = len(A_CH)

# Hair from tBilby for Matt's preferred model ONLY
for _ in range(N_HAIR):
    j = rng.integers(0, n_tb)  # uniform over posterior samples of that model
    S = spec_from_params(j, params_tBilby)
    plt.loglog(f, S, color="grey", alpha=0.10, lw=0.8)


# 68% band (tBilby, Matt's preferred model ONLY)
specs = []
for _ in range(N_BAND):
    j = rng.integers(0, n_tb)
    specs.append(spec_from_params(j, params_tBilby))
specs = np.array(specs)
med_tb = np.median(specs, 0)
lo_tb  = np.percentile(specs, 16, 0)
hi_tb  = np.percentile(specs, 84, 0)
plt.fill_between(f, lo_tb, hi_tb, alpha=0.25, label="tPTABilby 68%")
plt.loglog(f, med_tb, color="black", lw=2.0, label="tPTABilby median")

# Matt's parameter arrays: compute median + 68% interval
n_mt = len(A_CH_p)  # or len(A_GW_p), they should match
take = rng.choice(n_mt, size=min(N_PREF, n_mt), replace=False)
#take = rng.choice(len(df_pref), size=min(N_PREF, len(df_pref)), replace=False)
pref_specs = np.array([spec_from_params(j, params_Matt) for j in take])

# median + 68% interval
med_pref = np.median(pref_specs, 0)
lo_pref  = np.percentile(pref_specs, 16, 0)
hi_pref  = np.percentile(pref_specs, 84, 0)

plt.fill_between(f, lo_pref, hi_pref, alpha=0.25, label="MPTA 68%")
plt.loglog(f, med_pref, color="C1", lw=1.0, label="MPTA median")

# Final plot settings
plt.figure(1)
# ----- Final PPC plot settings and save -----
plt.xlabel("Frequency [Hz] ", fontsize=14)
#plt.ylabel("h_c(f)" if USE_CHAR_STRAIN else "P(f) [s$^2$/Hz]")
plt.ylabel('Delay [s]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#gwb_fid = psd_pow(f, GWB_FID_LOG10_A, GWB_FID_GAMMA)
#gwb_fid = powerlaw(f, GWB_FID_LOG10_A, GWB_FID_GAMMA)
gwb_fid = psd_pow(f, GWB_FID_LOG10_A, GWB_FID_GAMMA)


plt.loglog(f, gwb_fid, ls="--", lw=2.0)
           #label=f"Fiducial SGWB (γ=13/3, A=10^{GWB_FID_LOG10_A})")

plt.legend(loc="best", fontsize=14)
plt.tight_layout()
plt.savefig(FIG, bbox_inches="tight")
print(f"Saved PPC plot → {FIG}")
plt.close()   # optional but nice to keep figures separate

# # ---------------------------
# # Sanity check: delay at 1/T for a few log10A_gw values
# # ---------------------------
# f_check = np.array([f[0]])
# #f_check = np.array([1e-9]) 
# gamma_gw_fixed = 13.0/3.0

# for logA in [-18, -15, -12]:
#     delay_1nHz = powerlaw(f_check, logA, gamma_gw_fixed)[0]
#     print(f"log10A_gw={logA:>3} delay at 1/T = {delay_1nHz:.6e} s "
#           f"= {delay_1nHz*1e9:.3f} ns")

# T_years = T / YR
# f_1_over_T = 1.0 / T

# print(f"Dataset timespan T = {T_years:.3f} years")
# print(f"Lowest Fourier frequency 1/T = {f_1_over_T:.3e} Hz "
#       f"= {f_1_over_T*1e9:.3f} nHz")
