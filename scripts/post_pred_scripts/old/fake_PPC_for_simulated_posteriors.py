#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import corner
import math

# ------------------------------
# 1. Make fake posterior samples
# ------------------------------
rng = np.random.default_rng(12345)
N_SAMPLES = 50000

# --- chromatic amplitude log10_A: sharp peak near -14.7 + long tail down to -20
# mixture: 75% narrow Gaussian, 25% broad uniform tail
mix = rng.uniform(0, 1, size=N_SAMPLES)
log10A_ch = np.empty(N_SAMPLES)

# core peak
core_mask = mix < 0.75
log10A_ch[core_mask] = rng.normal(loc=-14.7, scale=0.15, size=core_mask.sum())

# tail component (low amplitude)
tail_mask = ~core_mask
log10A_ch[tail_mask] = rng.uniform(-20.0, -15.2, size=tail_mask.sum())

# clip to prior range
log10A_ch = np.clip(log10A_ch, -20.0, -11.0)

# --- chromatic gamma: pile-up near 0 with long tail up to ~7
# use a mixture of an exponential near 0 and a uniform tail
mix = rng.uniform(0, 1, size=N_SAMPLES)
gamma_ch = np.empty(N_SAMPLES)

core_mask = mix < 0.8
# exponential concentrated near 0
gamma_ch[core_mask] = rng.exponential(scale=0.5, size=core_mask.sum())
# tail
tail_mask = ~core_mask
gamma_ch[tail_mask] = rng.uniform(1.5, 7.0, size=tail_mask.sum())
gamma_ch = np.clip(gamma_ch, 0.0, 7.0)

# --- GW amplitude log10_A: broad-ish, skewed distribution
mix = rng.uniform(0, 1, size=N_SAMPLES)
log10A_gw = np.empty(N_SAMPLES)
core_mask = mix < 0.7
log10A_gw[core_mask] = rng.normal(loc=-15.6, scale=0.35, size=core_mask.sum())
tail_mask = ~core_mask
log10A_gw[tail_mask] = rng.uniform(-17.5, -14.0, size=tail_mask.sum())
log10A_gw = np.clip(log10A_gw, -18.0, -13.0)

# stack into (N,3) array for corner
#samples = np.vstack([gamma_ch, log10A_ch, log10A_gw]).T
samples = np.vstack([gamma_ch, log10A_ch]).T
labels  = [r"$\gamma_{\rm CH}$",
           r"$\log_{10} A_{\rm CH}$",
           r"$\log_{10} A_{\rm GW}$"]

print("Fake CH log10_A: median, [2.5,97.5] =",
      np.median(log10A_ch),
      np.percentile(log10A_ch, [2.5, 97.5]))
print("Fake CH gamma:   median, [2.5,97.5] =",
      np.median(gamma_ch),
      np.percentile(gamma_ch, [2.5, 97.5]))

# ------------------------------
# 2. Corner plot of fake posteriors
# ------------------------------
fig = corner.corner(
    samples,
    labels=labels,
    bins=50,
    quantiles=[0.025, 0.5, 0.975],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)
fig.savefig("fake_posteriors_corner.png", bbox_inches="tight")
plt.close(fig)
print("Saved corner plot → fake_posteriors_corner.png")

# ------------------------------
# 3. PSD hairs from these posteriors
# ------------------------------

# Frequency grid (copying your PPC setup)
YR  = 365.25 * 24 * 3600.0
DAY = 24 * 3600.0
T_YRS, CAD_D = 4.5, 14.0
T   = T_YRS * YR
cad = CAD_D * DAY
n_max = int(np.floor(T / cad))
f = (np.arange(1, n_max + 1)) / T
fref = 1.0 / YR

USE_CHAR_STRAIN = False

def psd_pow(f_arr, log10_A, gamma):
    """Power spectrum from amplitude and spectral index."""
    A = 10.0 ** log10_A
    alpha = (3.0 - gamma) / 2.0
    h_c = A * (f_arr / fref) ** alpha
    if USE_CHAR_STRAIN:
        return h_c
    else:
        return (h_c ** 2) / (12.0 * math.pi**2 * f_arr**3)

# draw hairs from the *actual* posterior samples
N_HAIR = 250

plt.figure(figsize=(7.2, 5.2), dpi=140)

for _ in range(N_HAIR):
    j = rng.integers(0, N_SAMPLES)
    S_ch = psd_pow(f, log10A_ch[j], gamma_ch[j])
    plt.loglog(f, S_ch, color="grey", alpha=0.08, lw=0.8)

# also compute and plot median + 68% band for these fake samples
N_BAND = 2000
idxs = rng.integers(0, N_SAMPLES, size=N_BAND)
specs = np.array([psd_pow(f, log10A_ch[j], gamma_ch[j]) for j in idxs])
med_psd = np.median(specs, axis=0)
lo_psd  = np.percentile(specs, 16, axis=0)
hi_psd  = np.percentile(specs, 84, axis=0)

plt.fill_between(f, lo_psd, hi_psd, alpha=0.25, label="Fake CH 68% band")
plt.loglog(f, med_psd, color="black", lw=2.0, label="Fake CH median")

# axis ranges roughly matching your real PPC
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-8, 1e-6)
#plt.ylim(1e-22, 1e-2)

plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$P_{\rm CH}(f)$ [s$^2$/Hz]")
plt.title("PSD hairs from fake CH posteriors")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("fake_CH_psd_hairs.png", bbox_inches="tight")

plt.close()
print("Saved PSD-hair plot → fake_CH_psd_hairs.png")
