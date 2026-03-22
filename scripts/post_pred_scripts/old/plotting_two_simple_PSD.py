#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt

# tBilby CH medians
log10_A_tb = -14.69
gamma_tb   = 0.71

# Matt CH medians
log10_A_mt = -15.01
gamma_mt   = 1.09

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

P_tb = psd_pow(f, log10_A_tb, gamma_tb)
P_mt = psd_pow(f, log10_A_mt, gamma_mt)
# plots
plt.figure(figsize=(6,4), dpi=140)
plt.loglog(f, P_tb, label="tBilby CH (median)", lw=2)
plt.loglog(f, P_mt, label="Matt CH (median)", lw=2)

plt.xlabel("Frequency [Hz]")
plt.ylabel("P(f) [s$^2$/Hz]")
plt.title("Simple CH-only Hc comparison")

plt.xlim(f[0], f[-1])
plt.ylim(1e-20, 1e-2)

plt.legend()
plt.tight_layout()
plt.savefig("plotting_two_simple_Hc.png", dpi=200)