#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:53:00 2019

@author: dreardon,azic

Runs basic white, red, and DM noise model for all pulsars in datadir
Usage:
python makeCorner.py <psr> <dataset> "<par1,par2,par3...>"
"""

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner


def make_noise_files(psrname, chain, pars, outdir='noisefiles/'):
    x = {}
    for ct, par in enumerate(pars):
        x[par] = np.median(chain[:, ct])

    os.system('mkdir -p {}'.format(outdir))
    with open(outdir + '/{}_noise.json'.format(psrname), 'w') as fout:
        json.dump(x, fout, sort_keys=True, indent=4, separators=(',', ': '))


def get_par_indices(pars, cpars):
    all_indices = []
    
    for cp in cpars:
        indices = [cp in p for p in pars]
        all_indices.append(indices)
    indices = np.any(all_indices, axis = 0)#, dtype = bool)
    return indices

def get_covm(psrname, chain, pars, cpars):

    chain = chain[:, :-4]
    if cpars is not None:

        indices = get_par_indices(pars, cpars)
        
        corner_pars = [p.replace('{}_'.format(psrname), '') for p in pars[indices]]
        corner_file_label = '_' + '_'.join([c.replace('{}_'.format(psrname), '') for c in cpars])
    else:
        #plot all pars
        indices = np.array([True for p in pars])
        corner_pars = [p.replace('{}_'.format(psrname), '') for p in pars[indices]] 
        corner_file_label = ''
        print('No corner pars specified, plotting all parameters')
    
    chain_covm = chain[:, indices]
    cov = np.cov(chain_covm)

    plt.imshow(cov, interpolation = 'nearest')
    plt.savefig(f'./{psrname}_cov_tmp.png', dpi = 300)
    plt.close()

def plot_corner(psrname, chain, pars, cpars, outdir = 'noisefiles/'):
    print(psrname)
    print(pars)
    print(cpars)
    print(chain.shape)
    print(outdir)
    
    chain = chain[:, :-4]
    if cpars is not None:

        indices = get_par_indices(pars, cpars)
        
        corner_pars = [p.replace('{}_'.format(psrname), '') for p in pars[indices]]
        corner_file_label = '_' + '_'.join([c.replace('{}_'.format(psrname), '') for c in cpars])
    else:
        #plot all pars
        indices = np.array([True for p in pars])
        corner_pars = [p.replace('{}_'.format(psrname), '') for p in pars]
        corner_file_label = ''
        print('No corner pars specified, plotting all parameters')



    chain_corner = chain[:, indices]
    fig = corner.corner(chain_corner, bins = 30, labels = corner_pars, quantiles=(0.16, 0.5, 0.84), show_titles = True)
    for ax in fig.axes:
        xlab = ax.get_xlabel()
        ylab = ax.get_ylabel()
        ti = ax.get_title()
        ax.set_title(ti, fontsize = 9)
        ax.set_xlabel(xlab, fontsize = 9)
        ax.set_ylabel(ylab, fontsize = 9)
    os.system('mkdir -p {}'.format(outdir))
    figsavename = outdir + '/{}_corner'.format(psrname) + corner_file_label + '.png'
    print(figsavename)
    plt.savefig(figsavename, dpi = 300, bbox_inches = 'tight')

# psrname = str(sys.argv[1])
# dir = sys.argv[2]
# cpars = None
# if len(sys.argv) > 3:
#     if ',' in sys.argv[3]:
#         cpars = [p.strip() for p in sys.argv[3].split(',')]
#     else:
#         cpars = [p.strip() for p in sys.argv[3].split(' ')]
#     #cpars = [ c for c in cpars]

psrname = "J1713+0747"
dir = "/fred/oz005/users/vdimarco/tBilby/results/chains/singlePsrNoise/J1713+0747_tm+ef+ec+eq+chrom+gw_1"
#dir = '/fred/oz005/users/vdimarco/tBilby/results/chains/singlePsrNoise/J1713+0747_tm+ef+ec+eq+rn+dm+chrom+gw_1'
cpars = None

#datadir = os.path.abspath("./data/" + str(dir) + '/')
datadir = dir
max_nchain = 1  # number of chains to read

first_chain = True
for i in range(1, max_nchain+1):
    outdir = datadir + "/chains/singlePsrNoise/" + psrname + '_' + str(i)
    outdir = dir + '/chain_' + str(i)
    chainfile = dir + '/chain_1.txt'
    #chainfile = outdir + '/chain_1.txt'
    if not os.path.exists(chainfile):
        continue
    if os.path.getsize(chainfile) == 0:
        continue
    chain_i = np.loadtxt(chainfile).squeeze()
    #pars = np.loadtxt(outdir + '/pars.txt', dtype=np.unicode_)
    pars = np.loadtxt(dir + '/pars.txt', dtype=np.unicode_)
    print(pars)
    
    pp = model_utils.PostProcessing(chain_i, pars)
    pp.plot_trace()
    plt.savefig(chainfile.replace('.txt', '.pdf'))
    plt.close()

    # Burn larger of 25% or first 25000
    burn = int(max([0.25*chain_i.shape[0], 25000]))
    chain = chain_i[burn:, :]
    if chain.size == 0:
        continue

    if first_chain:
        chain_total = chain

        first_chain = False
    else:
        chain_total = np.concatenate((chain_total, chain)).squeeze()

# Now, save noise files
try:
    #get_covm(psrname, chain_total, pars, cpars)
    plot_corner(psrname, chain_total, pars, cpars, outdir = dir)
    print('Made corner plot for {}'.format(psrname))
except NameError:
    print('No valid chains for {}'.format(psrname))

