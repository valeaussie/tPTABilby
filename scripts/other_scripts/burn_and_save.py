#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Directory containing chain and pars.txt
dir = "/fred/oz005/users/vdimarco/tPTABilby/results/chains/singlePsrNoise/J1713+0747_tm+ef+ec+eq+chrom+gw_1"

chainfile = os.path.join(dir, "chain_1.txt")
parsfile  = os.path.join(dir, "pars.txt")

# ------------------------------------------------------------------
# Check files exist before loading
# ------------------------------------------------------------------

if not os.path.exists(chainfile):
    raise FileNotFoundError(f"Chain file not found: {chainfile}")

if os.path.getsize(chainfile) == 0:
    raise ValueError(f"Chain file is empty: {chainfile}")

# ------------------------------------------------------------------
# Load chain and parameter names
# ------------------------------------------------------------------

chain = np.loadtxt(chainfile)
pars  = np.loadtxt(parsfile, dtype=str)

# ------------------------------------------------------------------
# Remove burn-in (max of 25% or 25000 samples)
# ------------------------------------------------------------------

burn = int(max(0.25 * chain.shape[0], 25000))
chain = chain[burn:, :]

if chain.size == 0:
    raise ValueError("Chain is empty after burn-in removal.")

print(f"Final chain length after burn-in: {len(chain)}")

# ------------------------------------------------------------------
# Save processed chain
# ------------------------------------------------------------------

outfile = os.path.join(dir, "chain.npy")
np.save(outfile, chain)

print(f"Saved processed chain to: {outfile}")
