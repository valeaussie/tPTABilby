#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt

# Hi values
log10_A_low = -19.00
log10_A_high = -14.00
gamma_low   = 4
gamma_high   = 7

YR  = 365.25 * 24 * 3600.0
DAY = 24 * 3600.0
fref = 1.0 / YR

def psd_pow(f, log10_A, gamma):
    A = 10.0 ** log10_A
    alpha = (3.0 - gamma) / 2.0
    h_c = A * (f / fref) ** alpha
    return (h_c**2) / (12.0 * math.pi**2 * f**3)

T_YRS, CAD_D = 4.5, 14
T   = T_YRS * YR
cad = CAD_D * DAY
n_max = int(np.floor(T / cad))
f = (np.arange(1, n_max + 1)) / T

P_A = psd_pow(f, log10_A_low, gamma_low)
P_B = psd_pow(f, log10_A_high, gamma_high)
P_C = psd_pow(f, log10_A_low, gamma_high)
P_D = psd_pow(f, log10_A_high, gamma_low)


# plots
plt.figure(figsize=(6,4), dpi=140)
plt.loglog(f, P_A, label="P_A (low A, low gamma)", lw=2)
plt.loglog(f, P_B, label="P_B (high A, high gamma)", lw=2)
plt.loglog(f, P_C, label="P_C (low A, high gamma)", lw=2)
plt.loglog(f, P_D, label="P_D (high A, low gamma)", lw=2)

plt.xlabel("Frequency [Hz]")
plt.ylabel("P(f) [s$^2$/Hz]")
plt.title("Simple CH-only Hc comparison")

plt.xlim(f[0], f[-1])
plt.ylim(1e-20, 1e-2)

plt.legend()
plt.tight_layout()
plt.savefig("plotting_extreme_PSD_from_posteriors.png", dpi=200)