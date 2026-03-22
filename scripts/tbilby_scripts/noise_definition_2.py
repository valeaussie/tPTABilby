#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import json
import numpy as np
import subprocess
import itertools

from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import gp_priors
from enterprise.signals import utils
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
from enterprise_extensions.blocks import common_red_noise_block
from enterprise_extensions import hypermodel
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm

from utils import *

parfile = datadir + '/' + psrname + '.par'
print(parfile)
timfile = datadir + '/' + psrname + '.tim'

psr = Pulsar(parfile, timfile, ephem=ephem)
p = psr

tmin = p.toas.min()
tmax = p.toas.max()
Tspan = np.max(tmax) - np.min(tmin)

selection = selections.Selection(selections.by_backend)
ecorr_selection = selections.Selection(selections.by_backend)

"""
Define timing model
"""
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

"""
Define efac model
"""
efac = parameter.Uniform(0.5, 1.5)
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

"""
Define ecorr model
"""
ecorr = parameter.Uniform(-6.5, -6)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)

"""
Define equad model
"""
equad = parameter.Uniform(-6.5, -6)
eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)

"""
Define red noise model
"""
log10_A_red = parameter.Uniform(-14.2, -13.8)
gamma_red = parameter.Uniform(1.5, 3.0)
max_cadence_rn = 240.0  # days
red_components = int(Tspan / (max_cadence_rn*86400))
pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=red_components, Tspan=Tspan)

"""
Define DM noise model (dm)
"""
log10_A_dm = parameter.Uniform(-13.6, -13.2)
gamma_dm = parameter.Uniform(3.5, 4.5)
max_cadence_dm = 240  # days
dm_components = int(Tspan / (max_cadence_dm*86400))
pl = utils.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
dm_basis = utils.createfourierdesignmatrix_dm(nmodes = dm_components, Tspan=Tspan)
dm = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')

"""
Define chromatic noise model (chrom)
"""
log10_A_chrom_prior = parameter.Uniform(-14.8, -14.4)
gamma_chrom_prior = parameter.Uniform(3.5, 4.5)
#chrom_gp_idx = parameter.Uniform(0,7)
chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
#idx = chrom_gp_idx
idx = 4
max_cadence_ch = 240  # days
chrom_components = int(Tspan / (max_cadence_ch*86400))
chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=chrom_components, idx=idx)
chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')