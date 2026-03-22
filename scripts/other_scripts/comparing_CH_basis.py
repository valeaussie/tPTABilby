#!/usr/bin/env python

import numpy as np
from enterprise.pulsar import Pulsar
from enterprise.signals import gp_bases
import matplotlib.pyplot as plt

parfile = "/fred/oz002/users/mmiles/MPTA_DR3/partim/production/third_pass/32ch/J1713+0747.par"
timfile = "/fred/oz002/users/mmiles/MPTA_DR3/partim/production/third_pass/32ch/J1713+0747.tim"

p = Pulsar(parfile, timfile, ephem="DE440")
toas = p.toas
freqs = p.freqs  # ← THIS IS REQUIRED

Tspan = toas.max() - toas.min()
cadence_days = 14
nmodes = int(Tspan / (cadence_days * 86400))
idx = 4

# Correct call for your enterprise version
F_tb, freqs_tb = gp_bases.createfourierdesignmatrix_chromatic(
    toas,
    freqs,     # MUST be 2nd arg
    nmodes=nmodes,
    Tspan=Tspan,
    idx=idx
)

F_matt, freqs_matt = gp_bases.createfourierdesignmatrix_chromatic(
    toas,
    freqs,     # MUST be 2nd arg
    nmodes=nmodes,
    Tspan=Tspan,
    idx=idx
)

print("Difference:", np.max(np.abs(F_tb - F_matt)))
