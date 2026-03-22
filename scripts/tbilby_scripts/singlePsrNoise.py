#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import parameter
from enterprise.signals import utils
from enterprise.signals import selections
from enterprise.signals import gp_priors
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm
# from enterprise_extensions import blocks
from enterprise_extensions import hypermodel
# from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from ppta_dr3_utils import *
from ppta_dr3_models import *
from utils import *
    
parfile = datadir + '/' + psrname + '.par'
print(parfile)
timfile = datadir + '/' + psrname + '.tim'

psr = Pulsar(parfile, timfile, ephem=ephem)
p = psr

tmin = p.toas.min()
tmax = p.toas.max()
Tspan = np.max(tmax) - np.min(tmin)


"""
Choose whether to marginalise or sample the timing model
"""
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

"""
Define white noise model
"""
# EFAC "MeasurementNoise" can add equad, but only t2equad - we want the tnequad
efac_prior = parameter.Uniform(0.01, 10.0)
ef = white_signals.MeasurementNoise(
    efac=efac_prior,
    selection=by_backend)

# EQUAD - TempoNest definition: sigma = sqrt((efac*sigma_0)**2 + (tnequad)**2)
log10_equad_prior = parameter.Uniform(-10, -5)
eq = white_signals.TNEquadNoise(
    log10_tnequad=log10_equad_prior,
    selection=by_backend)

# ECORR - we will swap to "white_signals.EcorrKernelNoise" later
# if not dir=='ppta15':
#     log10_ecorr_prior = parameter.Uniform(-10, -5)
#     wn += gp_signals.EcorrBasisModel(
#         log10_ecorr=log10_ecorr_prior,
#         selection=ecorr_selection)
#     if psr.name in psr_groupecorr_dict_dict[dir].keys():
#         wn += gp_signals.EcorrBasisModel(log10_ecorr = log10_ecorr_prior,
#                                          selection = by_group_ecorr_dict[psr.name],
#                                          name='basis_ecorr_group')
#     if dir == 'uwl' or dir == 'dr2':
#         wn += gp_signals.EcorrBasisModel(
#             log10_ecorr=log10_ecorr_prior,
#             selection=no_selection, name='basis_ecorr_all')
#     elif dir == 'all':
log10_ecorr_prior = parameter.Uniform(-10, -5)
ec = gp_signals.EcorrBasisModel(
    log10_ecorr=log10_ecorr_prior,
    selection=global_ecorr_selection, name='basis_ecorr_all')

"""
Define red noise model
"""
log10_A_prior = parameter.Uniform(-20, -11)
gamma_prior = parameter.Uniform(0, 7)
# # powerlaw
rn_model = gp_priors.powerlaw(log10_A=log10_A_prior,
                              gamma=gamma_prior)
Tspan = psr.toas.max() - psr.toas.min()  # seconds
max_cadence = 240  # days
red_components = int(Tspan / (max_cadence*86400))
print("Using {} red noise components".format(red_components))
rn = gp_signals.FourierBasisGP(rn_model, components=red_components,
                               selection=no_selection, name='red_noise')

gwb_model = gp_priors.powerlaw(log10_A=log10_A_prior,
                              gamma=4.33)
print("Using {} gwb components".format(red_components))
gw = gp_signals.FourierBasisGP(gwb_model, components=red_components,
                               selection=no_selection, name='gwb')


"""
Define DM noise model
"""
log10_A_dm_prior = parameter.Uniform(-20, -11)
gamma_dm_prior = parameter.Uniform(0, 7)
dm_model = gp_priors.powerlaw(log10_A=log10_A_dm_prior,
                              gamma=gamma_dm_prior)
Tspan = psr.toas.max() - psr.toas.min()  # seconds
max_cadence = 60  # days
dm_components = int(Tspan / (max_cadence*86400))
print("Using {} DM components".format(dm_components))
dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=dm_components)
dm = gp_signals.BasisGP(dm_model, dm_basis, name='dm_gp')


"""
Define chromatic noise model
"""
log10_A_chrom_prior = parameter.Uniform(-20, -11)
gamma_chrom_prior = parameter.Uniform(0, 7)
chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                             gamma=gamma_chrom_prior)
idx = 4  # Define freq^-idx scaling
max_cadence = 240  # days
chrom_components = int(Tspan / (max_cadence*86400))
print("Using {} Chrom components".format(chrom_components))
chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=chrom_components,
                                                           idx=idx)
chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')


"""
SOLAR WIND. By default only models 30 components, regarless of Tspan
"""
n_earth = parameter.Uniform(0, 20)('n_earth')
deter_sw = solar_wind(n_earth=n_earth)
mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

Tspan = psr.toas.max() - psr.toas.min()
max_cadence = 60
sw_components = int(Tspan / (max_cadence*86400))

log10_A_sw = parameter.Uniform(-10, 1)
gamma_sw = parameter.Uniform(-4, 4)
sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)

sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')
