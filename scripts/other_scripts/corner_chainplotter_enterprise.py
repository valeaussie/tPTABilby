import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from chainconsumer import ChainConsumer

# --------------------------------------------------
# Load chain and parameter names
# --------------------------------------------------

chain = np.load(
    "/fred/oz005/users/vdimarco/tPTABilby/results/chains/"
    "singlePsrNoise/J1713+0747_tm+ef+ec+eq+chrom+gw_1/chain.npy"
)

pars = np.loadtxt(
    "/fred/oz005/users/vdimarco/tPTABilby/results/chains/"
    "singlePsrNoise/J1713+0747_tm+ef+ec+eq+chrom+gw_1/pars.txt",
    dtype=str
)

print("Raw chain shape:", chain.shape)
print("Number of parameters:", len(pars))

# --------------------------------------------------
# Drop last 4 bookkeeping columns
# --------------------------------------------------

data = chain[:, :-4]   # now matches pars length

# --------------------------------------------------
# Select chrom + gwb
# --------------------------------------------------

mask = np.array([
    ("chrom" in p.lower()) or ("gwb" in p.lower())
    for p in pars
])

data = data[:, mask]
selected_pars = pars[mask]

print("Selected parameters:", selected_pars)

# --------------------------------------------------
# Labels (must match selected_pars order)
# --------------------------------------------------

label_map = {
    "J1713+0747_chrom_gp_gamma": r"$\gamma_{\mathrm{CH}}$",
    "J1713+0747_chrom_gp_log10_A": r"$\log_{10}A_{\mathrm{CH}}$",
    "J1713+0747_gwb_log10_A": r"$\log_{10}A_{\mathrm{GW}}$",
}

labels = [label_map[p] for p in selected_pars]

# --------------------------------------------------
# Plot
# --------------------------------------------------

n = data.shape[1]
q16, q50, q84 = np.percentile(data, [16, 50, 84], axis=0)
mu = np.mean(data, axis=0)

cc = ChainConsumer()
cc.add_chain(data, parameters=labels, name="Enterprise")

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

plt.savefig("Enterprise_corner_plot.pdf", bbox_inches="tight")
