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

components_dict = {key: [] for key in ['red', 'dm', 'band', 'chrom', 'hf']}

psrnames = ['J0030+0451',
            'J0125-2327',
            'J0437-4715',
            'J0613-0200',
            'J0614-3329',
            'J0711-6830', 
            'J0900-3144',
            'J1017-7156',
            'J1022+1001',
            'J1024-0719', 
            'J1045-4509',
            'J1125-6014',
            'J1446-4701', 
            'J1545-4550',
            'J1600-3053',
            'J1603-7202',
            'J1643-1224',
            'J1713+0747',
            'J1730-2304',
            #'J1741+1351', #trash
            'J1744-1134',
            #'J1824-2452A', #also trash
            'J1857+0943',
            'J1832-0836',
            'J1902-5105',
            'J1909-3744',
            'J1933-6211',
            'J1939+2134',
            'J2124-3358',
            'J2129-5721',
            'J2145-0750',
            'J2241-5236']

"""
Define timing model
"""
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

"""
Define efac model
"""
selection = selections.Selection(selections.by_backend)
efac = parameter.Uniform(0.5, 1.5)
#efac = parameter.Uniform(0.999, 1.001)
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

"""
Define ecorr model
"""
#log10_ecorr_prior = parameter.Uniform(-7.001,-6.999)
log10_ecorr_prior = parameter.Uniform(-10,-1)
ec = white_signals.EcorrKernelNoise(log10_ecorr=log10_ecorr_prior, selection=selection)
#ec = gp_signals.EcorrBasisModel(log10_ecorr=log10_ecorr_prior, selection=selection)

"""
Define equad model
"""
log10_equad_prior = parameter.Uniform(-10,-1)
#log10_equad_prior = parameter.Uniform(-7.001,-6.999)
eq = white_signals.TNEquadNoise(log10_tnequad=log10_equad_prior, selection=selection)

"""
Define red noise model
"""
priors = {}
rn_model, rn_lgA_prior, rn_gam_prior, priors = get_rednoise_priors(psr, 'red_noise', {}, {}, priors)

Tspan = psr.toas.max() - psr.toas.min()  # seconds
max_cadence_rn = 240.0  # days
red_components = int(Tspan / (max_cadence_rn*86400))
components_dict['red'].append(red_components)
print("Using {} red noise components".format(red_components))
rn = gp_signals.FourierBasisGP(rn_model, components=red_components, name='red_noise')

"""
Define DM noise model (dm)
"""
priors = {}
dm_model, dm_lgA_prior, dm_gam_prior, priors = get_rednoise_priors(psr, 'dm_gp', {}, {}, priors)

Tspan = psr.toas.max() - psr.toas.min()  # seconds
max_cadence_dm = 240  # days
dm_components = int(Tspan / (max_cadence_dm*86400))
components_dict['dm'].append(dm_components)
print("Using {} DM components".format(dm_components))
dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=dm_components)
dm = gp_signals.BasisGP(dm_model, dm_basis, name='dm_gp')

"""
Define chromatic noise model (chrom)
"""
chrom_model, chrom_lgA_prior, chrom_gam_prior, priors = get_rednoise_priors(psr, 'chrom_gp', {}, {}, priors)

idx = 4  # Define freq^-idx scaling (chromatic index)
max_cadence_ch = 240  # days
chrom_components = int(Tspan / (max_cadence_ch*86400))
components_dict['chrom'].append(chrom_components)
print("Using {} Chrom components".format(chrom_components))
chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=chrom_components,
                                                               idx=idx)
chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')

"""
Define solar wind (sw)
"""
def get_informed_nearth_priors(psr, priors, n_earth_only=False, n_earth_fixed=False):

    if psr.name[1:5] in ['0030', '0125', '0437', '0613', '1022', '1024', '1545', '1600', '1643', '1713', '1730', '1744', '1824', '1832', '1909', '2145', '2241'] and not n_earth_fixed:
        key_ = psr.name + '_n_earth'
        print('getting prior for {}'.format(key_))
        n_earth_min = 0.0
        n_earth_max = 20.0
        n_earth = parameter.Uniform(n_earth_min, n_earth_max)
        priors[key_ + "_min"] = n_earth_min
        priors[key_ + "_max"] = n_earth_max
    else:
        print('nearth constant')
        n_earth = parameter.Constant(4)

    deter_sw = solar_wind(n_earth=n_earth)
    mean_sw = deterministic_signals.Deterministic(deter_sw)#, name='n_earth')
    sw = mean_sw

    #vary gp_sw for pulsars with constrained gp_sw parameters
    if psr.name[1:5] in ['0437', '0711', '0900', '1024', '1643', '1713', '1730', '1744', '1909', '2145'] and not n_earth_only:
        key_ = psr.name + '_gp_sw_log10_A'
        print('getting prior for {}'.format(key_))
        log10_A_min = -10
        log10_A_max = -3
        gamma_min = -4
        gamma_max = 4
        log10_A_sw = parameter.Uniform(log10_A_min, log10_A_max)
        gamma_sw = parameter.Uniform(gamma_min, gamma_max)
        print(f"""{psr.name}_gp_sw_noise prior:
        log10_A in [{log10_A_min}, {log10_A_max}]
        gamma in [{gamma_min}, {gamma_max}]
        """)
        key_ = psr.name + '_gp_sw_log10_A'
       	priors[key_ + "_min"] = log10_A_min
       	priors[key_ + "_max"] = log10_A_max
       	key_ = psr.name + '_gp_sw_gamma'
       	priors[key_ + "_min"] = gamma_min
       	priors[key_ + "_max"] = gamma_max        

        Tspan = psr.toas.max() - psr.toas.min()
        max_cadence_sw = 60
        sw_components = int(Tspan / (max_cadence_sw*86400))
        sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
        sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)
        sw += gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

    return sw

sw = get_informed_nearth_priors(psr, priors)


"""
Define band noise model
"""
def low_frequencies(freqs):
    """Selection for obs frequencies <=960MHz"""
    return delete_empty_keys(dict(zip(['low'], [freqs <= 960])))
low_freq = selections.Selection(low_frequencies)

max_cadence = 60  # days
band_components = int(Tspan / (max_cadence*86400))
bn = gp_signals.FourierBasisGP(rn_model, components=band_components,
                               selection=low_freq, name='band_noise')


"""
Define system noise model
"""
max_cadence_gn = 30 #days
log10_A_gn_prior = parameter.Uniform(-20, -11)
gamma_gn_prior = parameter.Uniform(0, 7)
gn_model = gp_priors.powerlaw(log10_A = log10_A_gn_prior, gamma = gamma_gn_prior)

def delete_empty_keys(d):
    delkeys = []
    for key in d.keys():
        if np.sum(d[key]) == 0:  # all False
            delkeys.append(key)
    for key in delkeys:
        del d[key]
    return d

psr_groupnoiselist_dict_dr2 = {'J0437-4715': ['CASPSR_40CM'],#,-->taken out after fixing number of harmonics for group noise 'PDFB_20CM'],
                               #'J0613-0200': ['PDFB_40CM'],
                               #'J1017-7156': ['PDFB_20CM', 'PDFB_40CM']
                               #'J1600-3053': ['CASPSR_40CM', 'PDFB_20CM', 'PDFB_10CM'],
                               #'J1643-1224': ['PDFB_10CM', 'PDFB_40CM']#,
                               'J1713+0747': ['WBCORR_10CM'], #PDFB1_early_20CM, -->taken out after fixing number of harmonics for group noise'CPSR2_20CM' 
                               'J1909-3744': ['CPSR2_50CM'] #-->taken out after fixing number of harmonics for group noise#'CASPSR_40CM', 'PDFB1_1433', 'PDFB1_early_20CM'
}

psr_groupnoiselist_dict_uwl = {'J0437-4715': ['UWL_PDFB4_20CM', 'UWL_sbA', 'UWL_sbG'],
                               #'J0613-0200': ['UWL_sbG'],
                               'J1017-7156': ['UWL_sbA', 'UWL_sbD'], #UNCOMMENT ME
                               'J1022+1001': ['UWL_sbE', 'UWL_sbH'],
                               #'J1600-3053': ['UWL_sbA'],
                               #'J1643-1224': ['UWL_sbA', 'UWL_sbD', 'UWL_sbE', 'UWL_sbF', 'UWL_sbH'],
                               'J1713+0747': ['UWL_sbA', 'UWL_sbE', 'UWL_sbF']
                               #'J1744-1134': ['UWL_PDFB4_10CM',  'UWL_sbE'],
                               #'J1909-3744': ['UWL_PDFB4_10CM', 'UWL_PDFB4_20CM', 'UWL_sbG', 'UWL_sbH'],
                               #'J2241-5236': ['UWL_PDFB4_20CM', 'UWL_sbD']
}
class sel_by_group_factory:

    def __init__(self, flagvals = None):
        if isinstance(flagvals, list) or isinstance(flagvals, np.ndarray):
            self.flagvals = np.unique(flagvals)
        elif flagvals == None:
            self.flagvals = flagvals
        else:
            raise ValueError(f'flagvals should be None or array-like, not {type(self.flagvals)}')
        
    def _sel_by_group(self, flags):
        if self.flagvals is None:
            #return all flags
            return delete_empty_keys({val: [val in g for g in flags["group"]] for val in np.unique(flags["group"])})
        else:
            return delete_empty_keys({val: np.array([val in g for g in flags["group"]]) for val in self.flagvals})

psr_groupnoiselist_dict_all = {}
for psr, item in psr_groupnoiselist_dict_dr2.items():
    if psr in psr_groupnoiselist_dict_uwl:
        psr_groupnoiselist_dict_all[psr] = psr_groupnoiselist_dict_uwl[psr] + item
    else:
        psr_groupnoiselist_dict_all[psr] = item

for psr, item in psr_groupnoiselist_dict_uwl.items():
    if psr in psr_groupnoiselist_dict_dr2:
        continue
    psr_groupnoiselist_dict_all[psr] = item

psr_groupnoise_dict_dict = {'dr2': psr_groupnoiselist_dict_dr2, 'uwl': psr_groupnoiselist_dict_uwl, 'all': psr_groupnoiselist_dict_all}
psr_groupnoiselist_dict = {psr: None for psr in psrnames}
by_group_dict = {key: selections.Selection(sel_by_group_factory(item)._sel_by_group) for key, item in psr_groupnoiselist_dict.items()}
gn = gp_signals.FourierBasisGP(gn_model, components=None,#, fmax=1/(max_cadence_gn*86400),
                            selection = by_group_dict[p.name],
                            name = 'group_noise')


"""
Achromatic JUMP
"""
def get_achrom_jump(toas, jump_search = False):
    if jump_search == True:
        MJD_JUMP_boundaries = np.arange(49990,60000,182.625)
        start_ind = np.squeeze(np.argwhere(toas.min()/86400.0 > MJD_JUMP_boundaries)[-1])
        end_ind = np.squeeze(np.argwhere(toas.max()/86400.0 < MJD_JUMP_boundaries)[0])
        print(start_ind, end_ind)
        log10_Amp_JUMP = parameter.Uniform(-10,-6)
        t0_JUMP = parameter.Uniform(toas.min()/86400.0, toas.max()/86400.0)
        signpar_JUMP = parameter.Uniform(-1, 1)
        
        wf = step_achrom_jump(log10_Amp=log10_Amp_JUMP, sign_param=signpar_JUMP, t0=t0_JUMP)
        jump_wfs = [wf]
        jump_labels = ['ALL']
        for _ind in range(start_ind, end_ind):
            _tmin = MJD_JUMP_boundaries[_ind]
            _tmax = MJD_JUMP_boundaries[_ind + 1]
            log10_Amp_JUMP_ = parameter.Uniform(-10,-6)
            t0_JUMP_ = parameter.Uniform(_tmin - 30.0, _tmax + 30.0) #enable some overlap between JUMP search blocks
            signpar_JUMP_ = parameter.Uniform(-1, 1)
            jump_wfs.append(step_achrom_jump(log10_Amp=log10_Amp_JUMP_, sign_param=signpar_JUMP_, t0=t0_JUMP_))
            jump_labels.append('{}'.format(_ind))
            achrom_JUMP = deterministic_signals.Deterministic(jump_wfs[0], name=f'MJD_JUMP_{jump_labels[0]}')
        for i, jump_wf_ in enumerate(jump_wfs[1:]):
            print(i+1, jump_labels[i+1])
            achrom_JUMP += deterministic_signals.Deterministic(jump_wf_, name=f'MJD_JUMP_{jump_labels[i+1]}')
    else:
        if dir == 'uwl':
            log10_Amp_JUMP = parameter.Uniform(-10,-6)
            t0_JUMP_1 = parameter.Uniform(58256, 59200)
            t0_JUMP_2 = parameter.Uniform(59200, 59645)
            signpar_JUMP = parameter.Uniform(-1, 1)
            wf = step_achrom_jump(log10_Amp=log10_Amp_JUMP, sign_param=signpar_JUMP, t0=t0_JUMP_1)
            achrom_JUMP = deterministic_signals.Deterministic(wf, name='MJD_JUMP_1')
            wf = step_achrom_jump(log10_Amp=log10_Amp_JUMP, sign_param=signpar_JUMP, t0=t0_JUMP_2)
            achrom_JUMP += deterministic_signals.Deterministic(wf, name='MJD_JUMP_2')
        elif dir == 'dr2':
            log10_Amp_JUMP = parameter.Uniform(-10,-6)
            t0_JUMP_1 = parameter.Uniform(53040, 59144)
            signpar_JUMP = parameter.Uniform(-1, 1)
            wf = step_achrom_jump(log10_Amp=log10_Amp_JUMP, sign_param=signpar_JUMP, t0=t0_JUMP_1)
            achrom_JUMP = deterministic_signals.Deterministic(wf, name='MJD_JUMP_1')
        else:
            log10_Amp_JUMP = parameter.Uniform(-10,-6)
            t0_JUMP_1 = parameter.Uniform(58256, 59200)
            t0_JUMP_2 = parameter.Uniform(59200, 59645)
            t0_JUMP_3 = parameter.Uniform(53040, 58256)
            signpar_JUMP = parameter.Uniform(-1, 1)
            wf = step_achrom_jump(log10_Amp=log10_Amp_JUMP, sign_param=signpar_JUMP, t0=t0_JUMP_1)
            achrom_JUMP = deterministic_signals.Deterministic(wf, name='MJD_JUMP_1')
            wf = step_achrom_jump(log10_Amp=log10_Amp_JUMP, sign_param=signpar_JUMP, t0=t0_JUMP_2)
            achrom_JUMP += deterministic_signals.Deterministic(wf, name='MJD_JUMP_2')
            wf = step_achrom_jump(log10_Amp=log10_Amp_JUMP, sign_param=signpar_JUMP, t0=t0_JUMP_3)
            achrom_JUMP += deterministic_signals.Deterministic(wf, name='MJD_JUMP_3')
    return achrom_JUMP
jump_search = False
#achrom_JUMP = get_achrom_jump(psr.toas, jump_search = jump_search)


"""
High frequency achromatic noise
"""
no_selection = selections.Selection(selections.no_selection)
max_cadence = 30  # days
hf_components = int(Tspan / (max_cadence*86400))
print("Using {} hf achromatic noise components".format(hf_components))
hf = gp_signals.FourierBasisGP(rn_model, components=hf_components,
                               selection=no_selection, name='hf_noise')


