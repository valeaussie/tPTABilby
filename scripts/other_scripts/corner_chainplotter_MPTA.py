#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from chainconsumer import ChainConsumer

PULSAR = "J1713+0747"

MATT_JSON = (
    "/fred/oz002/users/mmiles/MPTA_DR3/analysis/enterprise_runs/out_pbilby/thirdpass/"
    "J1713+0747/J1713+0747_CHROMCIDX_SGWB/CHROMCIDX_SGWB_final_res.json"
)

def load_matt_json(path):
    """
    Matt's file is often a JSON string wrapped inside a JSON file.
    This loader handles both:
      - normal JSON list/dict
      - JSON-encoded string
    """
    with open(path, "r") as f:
        outer = json.load(f)
    if isinstance(outer, str):
        return json.loads(outer)
    return outer

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -----------------------------
# Load Matt samples
# -----------------------------
data = load_matt_json(MATT_JSON)
df = pd.DataFrame(data)

print("Matt columns (first 30):", list(df.columns)[:30])

# -----------------------------
# Find CH + GW columns (robust to naming)
# -----------------------------
CH_G = first_existing(df, [f"{PULSAR}_chromcidx_gp_gamma", f"{PULSAR}_chrom_gp_gamma"])
CH_A = first_existing(df, [f"{PULSAR}_chromcidx_gp_log10_A", f"{PULSAR}_chrom_gp_log10_A"])
GW_A = first_existing(df, ["log10_A_gw", f"{PULSAR}_gwb_log10_A"])

keep = [CH_G, CH_A, GW_A]
if any(k is None for k in keep):
    raise RuntimeError(f"Could not find required columns. Found: CH_G={CH_G}, CH_A={CH_A}, GW_A={GW_A}")

# Drop NaNs if present
df_sel = df[keep].dropna()
data_arr = df_sel.to_numpy()

print("Selected parameters:", keep)
print("Data shape:", data_arr.shape)

# -----------------------------
# Labels (match order in keep)
# -----------------------------
labels = [
    r"$\gamma_{\mathrm{CH}}$",
    r"$\log_{10}A_{\mathrm{CH}}$",
    r"$\log_{10}A_{\mathrm{GW}}$",
]

# -----------------------------
# ChainConsumer plot (same styling as before)
# -----------------------------

n = data_arr.shape[1]
q16, q50, q84 = np.percentile(data_arr, [16, 50, 84], axis=0)
mu = np.mean(data_arr, axis=0)

cc = ChainConsumer()
cc.add_chain(data_arr, parameters=labels, name="Meerkat (Matt)")

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

plt.savefig("MPTA_corner_plot.pdf", bbox_inches="tight")

