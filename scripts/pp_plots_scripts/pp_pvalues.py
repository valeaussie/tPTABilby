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

i = int(sys.argv[1])

## define noise model
noise_list = noise_models(tm, ef)
noise_list.add_noise_model(eq, 'eq')
noise_list.add_noise_model(ec, 'ec')
noise_list.add_noise_model(rn, 'rn')
noise_list.add_noise_model(dm, 'dm')
noise_list.add_noise_model(chrom, 'chrom')

ephem = 'DE421'

parfile = datadir + '/' + psrname + '.par'
timfile = datadir + '/' + psrname + '.tim'
psr = Pulsar(parfile, timfile, ephem=ephem)
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

result = bilby.result.read_in_result(filename = f"/fred/oz005/users/vdimarco/tBilby/300_sims_tight_uniforms_2/test_sim_{i}-model_full_J0437-4715_5000_livepoints/sims_test_sim_{i}-model_full_J0437-4715_5000_livepoints_result.json")

models = result.posterior.groupby(list(noise_list.model_def.keys()))

model_freq = {}
for model in models:
    n = len(model[1])
    key = noise_list.generate_key(model[1].iloc[0].to_dict())
    param = noise_list.parameter_mapper(key)
    samples = model[1][param].values
    if len(param) > len(samples):
        print('param length greater than samples length')
        continue
    model_freq[binary_to_decimal(key)] = n

# Calculate the sum of all values
total = sum(model_freq.values())

# Create the normalized dictionary
p_values = {k: v / total for k, v in model_freq.items()}

# Find the number of the correct model
datadir = f"/fred/oz005/users/vdimarco/tBilby/sims/300_sims_tight_uniforms_2/test_sim_{i}"
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

key = noise_list.get_model_key(list(params_injected))
injected_model_int = binary_to_decimal(key)

outdir = f"/fred/oz005/users/vdimarco/tBilby/300_sims_tight_uniforms_2/p_values"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Save to JSON file
p_values["injected"] = injected_model_int
with open(f'{outdir}/p_values_model_{i}.json', 'w') as f:
    json.dump(p_values, f, indent=4)
