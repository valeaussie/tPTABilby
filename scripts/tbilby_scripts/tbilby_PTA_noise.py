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

from enterprise_warp import bilby_warp

from utils import *
#from noise_definition import *
#from noise_definition_2 import *
from noise_definition_Meerkat import *
#from singlePsrNoise import *

### define noise model
noise_list = noise_models(tm, ef)
noise_list.add_noise_model(eq, 'eq')
noise_list.add_noise_model(ec, 'ec')
noise_list.add_noise_model(rn, 'rn')
noise_list.add_noise_model(dm, 'dm')
noise_list.add_noise_model(chrom, 'chrom')
#noise_list.add_noise_model(sw, 'sw')
#noise_list.add_noise_model(gn, 'gn')
#noise_list.add_noise_model(bn, 'bn')
#noise_list.add_noise_model(achrom_JUMP, 'achrom_JUMP')
#noise_list.add_noise_model(hf, 'hf')
noise_list.add_noise_model(gw, 'gw')


parfile = datadir + '/' + psrname + '.par'
timfile = datadir + '/' + psrname + '.tim'
psr = Pulsar(parfile, timfile, ephem=ephem)
p = psr
s = noise_list.generate_signal()

# define priors
pta = signal_base.PTA(s(p))
priors = bilby_warp.get_bilby_prior_dict(pta)

priors_t = bilby.core.prior.dict.ConditionalPriorDict(priors)
for key in noise_list.noise_model_dict.keys():
    priors_t[key] = tbilby.core.prior.DiscreteUniform(0,1, key)
#Fix priors for testing
# priors_t["n0"] = DeltaFunction(peak=1, name="n0")
# priors_t["n1"] = DeltaFunction(peak=1, name="n1")

# If you want to fix some models, do it here
# 0 = absent, 1 = present
priors_t["n0"]=1
priors_t["n1"]=1
# priors_t["n2"]=1
# priors_t["n3"]=0
# priors_t["n4"]=1
priors_t["n5"]=1

print('printing noise models')
noise_list.print_noise_models()

print('printing priors')
print(priors_t.keys())

model_holder={}
parameters = dict.fromkeys(priors_t.keys())

# bilby model
class PTALikelihood(bilby.Likelihood):
    def __init__(self, data, parameters):
        super().__init__(parameters=parameters)
        print(parameters)
        self.data = data
        self.model_holder = noise_list.generate_model(data)
        print(self.model_holder.keys())

    def get_relevant_params(self, key):
        q = {}
        for params in self.model_holder[key].params:
            q[params.name] = self.parameters[params.name]
        return q
 
    def log_likelihood(self):
        arr = []
        for key in noise_list.noise_model_dict.keys():
            arr.append(self.parameters[key])
        non_zero_indices_list = [index for index, value in enumerate(arr) if value != 0]
        key = '-'.join(map(str,np.array(arr).astype(int)))
        q = self.get_relevant_params(key)
        return self.model_holder[key].get_lnlikelihood(q)

likelihood = PTALikelihood(parameters=parameters, data=p)
print(noise_list.model_holder.keys())

# if debug:
#     temp_draw = priors_t.sample(1)
#     print(temp_draw)
#     ppp
#     likelihood.parameters.update(tmp_dict)
#     llog_prob.append(likelihood.log_likelihood())


# set sample to False to only run analysis on existing results
sample = False
if sample :
    # run sampler
    label = label
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors_t,
        outdir=outdir,
        sampler="dynesty",
        nlive=5000,
        resume=True,
        #clean=True,
        label=label,
        max_act=None,
    )

analysis_dict = {}
for val in noise_list.noise_model_dict.values():
    analysis_dict[val] = (1,)

analysis_class = analysis(outdir+ '/'+ label + '_result.json', noise_list, analysis_dict)
analysis_class.margin_over_model()
#analysis_class = analysis("/fred/oz002/vdimarco/tbiby_PTA_noise/3freq_EF_TN_DM_model_full_J0437-4715_5000_livepoints/out_sim_3freq_EF_TN_DM_model_full_J0437-4715_5000_livepoints/sim_3freq_EF_TN_DM_model_full_J0437-4715_5000_livepoints_result.json", noise_list)
analysis_class.run_analysis()
analysis.pp_analysis
