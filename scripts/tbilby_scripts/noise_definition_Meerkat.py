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

max_cadence = 14
psr_components = int(Tspan / (max_cadence*86400))
selection = selections.Selection(selections.by_backend)
ecorr_selection = selections.Selection(selections.by_backend)

def dm_noise(log10_A,gamma,Tspan,components=120,option="powerlaw"):
    """
    A term to account for stochastic variations in DM. It is based on spin
    noise model, with Fourier amplitudes depending on radio frequency nu
    as ~ 1/nu^2.
    """
    nfreqs = 30
    if option=="powerlaw":
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    dm_basis = utils.createfourierdesignmatrix_dm(nmodes = components, Tspan=Tspan)
    dmn = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')

    return dmn


"""
Define timing model
"""
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

"""
Define efac model
"""
#efac = parameter.Uniform(0.1,5)
efac = parameter.Constant(1.0591568714177502)
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

"""
Define ecorr model
"""
#ecorr = parameter.Uniform(-10,-1)
ecorr = parameter.Constant(-6.902010050251256) 
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection, method="sherman-morrison")

"""
Define equad model
"""
#equad = parameter.Uniform(-10,-1)
equad = parameter.Constant(-7.535175879396984)
eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)

"""
Define red noise model
"""
log10_A_red = parameter.Uniform(-20, -11)
gamma_red = parameter.Uniform(0, 7)
pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=psr_components, Tspan=Tspan)

"""
Define DM noise model (dm)
"""
log10_A_dm = parameter.Uniform(-20, -11)
gamma_dm = parameter.Uniform(0, 7)
dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=psr_components,option="powerlaw")

"""
Define chromatic noise model (chrom)
"""
log10_A_chrom_prior = parameter.Uniform(-20, -11)
gamma_chrom_prior = parameter.Uniform(0, 7)
#chrom_gp_idx = parameter.Uniform(0, 7)
chrom_gp_idx = parameter.Constant(4)
chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
idx = chrom_gp_idx
components = 120
chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=psr_components, idx=idx)
chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')

"""
Define solar wind (sw)
"""
log10_A_sw = parameter.Uniform(-10, 1)
gamma_sw = parameter.Uniform(-4, 4)
sw_components = 120
sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
sw_basis = createfourierdesignmatrix_solar_dm(nmodes=psr_components, Tspan=Tspan)
sw = gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

"""
Define GW model
"""
log10_A_gw = parameter.Uniform(-18, -12)
gamma_gw = parameter.Constant(13/3)
gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')