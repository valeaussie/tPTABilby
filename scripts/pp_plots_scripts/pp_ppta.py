#!/usr/bin/env python
# coding: utf-8

import bilby
import tbilby
from bilby.core.prior import  Uniform
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import corner
import multiprocessing
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm
from enterprise_extensions.blocks import common_red_noise_block
import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing
import enterprise.constants as const
import json

from enterprise_warp import bilby_warp
from noise_definition import *

import gc

### define noise model
noise_list = noise_models(tm, ef)
noise_list.add_noise_model(eq, 'eq')
noise_list.add_noise_model(ec, 'ec')
noise_list.add_noise_model(rn, 'rn')
noise_list.add_noise_model(dm, 'dm')
noise_list.add_noise_model(chrom, 'chrom')
#noise_list.add_noise_model(sw, 'sw')




ranking = []
for i in range(0, 99):
    print(f"Running sim {i}")
    data = {
    'datadir' : f"/fred/oz005/users/vdimarco/tBilby/sims/100_sims_7/test_sim_{i}",
    'ephem' : 'DE421',
    'psrname' : "J0437-4715",
    'outdir' : f"/fred/oz005/users/vdimarco/tBilby/100_outs_7/test_sim_{i}-model_full_J0437-4715_5000_livepoints",
    'label' : f"sims_test_sim_{i}-model_full_J0437-4715_5000_livepoints"
    }
    filename = data['outdir'] + '/' + data['label'] + '_result.json'
    if not os.path.exists(filename):
        print(f"'{filename}' does not exist. Continuing...")
        continue
    
    with open("/fred/oz005/users/vdimarco/tBilby/config.json", "w") as file:
        json.dump(data, file)
    
    parfile = datadir + '/' + psrname + '.par'
    timfile = datadir + '/' + psrname + '.tim'
    psr = Pulsar(parfile, timfile, ephem=data['ephem'])
    p = psr
    s = noise_list.generate_signal()
    model_holder = noise_list.generate_model(p)

    # define priors
    pta = signal_base.PTA(s(p))
    priors = bilby_warp.get_bilby_prior_dict(pta)

    priors_t = bilby.core.prior.dict.ConditionalPriorDict(priors)
    for key in noise_list.noise_model_dict.keys():
        priors_t[key] = tbilby.core.prior.DiscreteUniform(0,1, key)

    print('printing noise models')
    noise_list.print_noise_models()

    print('printing priors')
    print(priors_t.keys())

    model_holder={}
    parameters = dict.fromkeys(priors_t.keys())

    analysis_dict = {}
    for val in noise_list.noise_model_dict.values():
        analysis_dict[val] = (1,)

    analysis_class = analysis(outdir+ '/'+ label + '_result.json', noise_list, analysis_dict)
    # result = bilby.result.read_in_result(filename = outdir+ '/'+ label + '_result.json')

    params_injected = {}
    with open(datadir + '/' + 'parameters.txt', "r") as f:
        for line in f:
            key, value = line.strip().split(": ", 1)
            params_injected[key] = value  # Store in dictionary
    dic_trans = {'efac': 'efac', 'equad': 'equad', 'ecorr': 'ecorr', 
                'red_amp': 'red_noise', 'red_gamma': 'red_noise_gamma', 'dm_amp': 'dm_gp', 
                'dm_gamma': 'dm_gp_gamma', 'ch_amp': 'chrom_gp', 'ch_gamma': 'chrom_gp_gamma', 
                'sw_amp': 'gp_sw', 'sw_gamma': 'sw_gamma'}
    param_array = []
    for key in params_injected.keys():
        param_array.append(dic_trans[key])
    params_injected = param_array
    print(params_injected)
    key = noise_list.get_model_key(list(params_injected))
    print('printing analysis class')
    print(analysis_class.pp_analysis(key))
    ranking.append(analysis_class.pp_analysis(key))
plt.figure()
plt.hist(ranking)
plt.savefig('ranking.png')
plt.close()