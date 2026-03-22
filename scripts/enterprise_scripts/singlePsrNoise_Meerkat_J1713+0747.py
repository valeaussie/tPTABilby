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
from enterprise.signals import selections
from enterprise.signals import gp_priors
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
# from enterprise_extensions import blocks
from enterprise_extensions import hypermodel
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals import utils


# Read in pulsar name, data directory, and chain number
psrname = "J1713+0747"
chainnum = "1"
datadir = "/fred/oz002/users/mmiles/MPTA_DR3/partim/production/third_pass/32ch"
dir = "/fred/oz005/users/vdimarco/tBilby/results"

# load in pulsar data
parfile = datadir + '/' + psrname + '.par'
print(parfile)
timfile = datadir + '/' + psrname + '.tim'
psr = Pulsar(parfile, timfile, ephem='DE421')
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
Define GW model
"""
log10_A_gw = parameter.Uniform(-18, -12)
gamma_gw = parameter.Constant(13/3)
gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')

"""
Define total model by summing all components
"""
# define model
s = tm+ef+ec+eq+rn+dm+chrom+gw
model_print = "tm+ef+ec+eq+rn+dm+chrom+gw"
print(model_print)

"""
Set up your PTA likelihood and your sampler
"""
# set up PTA
nmodels = 1
pta = dict.fromkeys(np.arange(nmodels))
pta[0] = signal_base.PTA(s(psr))
hyper_model = hypermodel.HyperModel(pta)

# set initial parameters drawn from prior
x0 = hyper_model.initial_sample()
ndim = len(x0)

# sampler for N steps
N = int(1e6)

# output directory:
outdir = dir + '/chains/singlePsrNoise/' + psrname + "_" + model_print + "_" + chainnum

# Use PTMCMC sampler. We can easily update this to use e.g. Bilby instead
sampler = hyper_model.setup_sampler(outdir=outdir, resume=False)

# Print parameter names and write to a file
print(hyper_model.param_names)
filename = outdir + "/pars.txt"
if os.path.exists(filename):
    os.remove(filename)
with open(filename, "a") as f:
    for par in hyper_model.param_names:
        f.write(par + '\n')

# Sample! The sampler parameters can be left as default.
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
