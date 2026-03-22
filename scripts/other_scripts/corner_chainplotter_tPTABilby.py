#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from chainconsumer import ChainConsumer
import bilby

RESFILE_TB = "/fred/oz005/users/vdimarco/tPTABilby/results/" \
                "Meerkat_J1713+0747_5000_livepoints_apples_with_apples_4/" \
                "Meerkat-J1713+0747_5000_livepoints_result.json"
MODEL_KEY  = "1-1-0-0-1-1"  # select this model

PULSAR = "J1713+0747"

def find_ind_cols(columns):
    """Indicator columns look like n0, n1, n2, ..."""
    return sorted([c for c in columns if c.startswith("n") and c[1:].isdigit()],
                  key=lambda x: int(x[1:]))

# -----------------------------
# Load result
# -----------------------------
res = bilby.result.read_in_result(RESFILE_TB)
post = res.posterior.copy()

# -----------------------------
# Select the single model (0-0-1)
# -----------------------------
ind_cols = find_ind_cols(post.columns)
if not ind_cols:
    raise RuntimeError("No indicator columns (n0, n1, ...) found in posterior.")

bits = np.array([int(b) for b in MODEL_KEY.split("-")], dtype=int)
if len(bits) != len(ind_cols):
    raise ValueError(f"MODEL_KEY has {len(bits)} bits but found {len(ind_cols)} indicator columns: {ind_cols}")

mask_model = np.all(post[ind_cols].to_numpy(int) == bits[None, :], axis=1)
post_m = post.loc[mask_model].copy()

if len(post_m) == 0:
    raise RuntimeError(f"No posterior rows found for model {MODEL_KEY}.")

print(f"Using tPTABilby model {MODEL_KEY} with {len(post_m)} posterior samples.")

# -----------------------------
# Pick CH + GW parameters
# -----------------------------
# Handle both chrom_gp and chromcidx_gp naming
ch_cols = [c for c in post_m.columns if "chrom" in c.lower()]
gw_cols = [c for c in post_m.columns if ("gwb" in c.lower()) or (c.lower() == "log10_a_gw")]

# Keep just what we want (typical set)
keep = []

# CH gamma/log10A
for cand in [f"{PULSAR}_chromcidx_gp_gamma", f"{PULSAR}_chrom_gp_gamma"]:
    if cand in post_m.columns:
        keep.append(cand)
        break

for cand in [f"{PULSAR}_chromcidx_gp_log10_A", f"{PULSAR}_chrom_gp_log10_A"]:
    if cand in post_m.columns:
        keep.append(cand)
        break

# GW log10A
for cand in [f"{PULSAR}_gwb_log10_A", "log10_A_gw"]:
    if cand in post_m.columns:
        keep.append(cand)
        break

if len(keep) != 3:
    raise RuntimeError(f"Could not find the required CH+GW columns. Found keep={keep}")

data = post_m[keep].to_numpy()

print("Selected parameters:", keep)
print("Data shape:", data.shape)

# -----------------------------
# Pretty labels (match keep order)
# -----------------------------
label_map = {
    f"{PULSAR}_chrom_gp_gamma":        r"$\gamma_{\mathrm{CH}}$",
    f"{PULSAR}_chromcidx_gp_gamma":    r"$\gamma_{\mathrm{CH}}$",
    f"{PULSAR}_chrom_gp_log10_A":      r"$\log_{10}A_{\mathrm{CH}}$",
    f"{PULSAR}_chromcidx_gp_log10_A":  r"$\log_{10}A_{\mathrm{CH}}$",
    f"{PULSAR}_gwb_log10_A":           r"$\log_{10}A_{\mathrm{GW}}$",
    "log10_A_gw":                      r"$\log_{10}A_{\mathrm{GW}}$",
}
labels = [label_map.get(p, p) for p in keep]

# -----------------------------
# ChainConsumer plot (same style as Enterprise)
# -----------------------------
n = data.shape[1]
q16, q50, q84 = np.percentile(data, [16, 50, 84], axis=0)
mu = np.mean(data, axis=0)

cc = ChainConsumer()
cc.add_chain(data, parameters=labels, name="tPTABilby")

cc.configure(
    sigmas=[1, 2],
    shade=True,
    shade_alpha=0.25,
    bar_shade=False,
    colors=[mcolors.CSS4_COLORS["mediumblue"]],
    summary=False,
    tick_font_size=16,
    label_font_size=20,
)

fig = cc.plotter.plot(figsize=(9, 9))

# --- robustly grab the first n*n axes as the corner grid ---
axes_list = fig.get_axes()
if len(axes_list) < n * n:
    raise RuntimeError(f"Expected at least {n*n} axes, got {len(axes_list)}")

axes = np.array(axes_list[: n * n]).reshape(n, n)

# put median +68% on diagonal panels
for i in range(n):
    med = q50[i]
    lo = q50[i] - q16[i]
    hi = q84[i] - q50[i]
    axes[i, i].set_title(
        f"{med:.2f}" + r"$^{+" + f"{hi:.2f}" + r"}_{-" + f"{lo:.2f}" + r"}$",
        fontsize=18
    )

plt.savefig("tPTABilby_corner_plot.pdf", bbox_inches="tight")

print("tPTABilby_corner_plot.pdf")
