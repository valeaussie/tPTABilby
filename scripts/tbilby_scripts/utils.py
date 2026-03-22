import os
import sys
import glob
import json
import numpy as np
import subprocess
import bilby
import tbilby
import corner
import itertools
import matplotlib.pyplot as plt
from pyhelpers.store import load_json\


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
import enterprise.signals.parameter as parameter
from enterprise.signals import gp_priors

import sys
from pathlib import Path


# To imput stuff from the command line and run parallel in sbatch
if len(sys.argv) != 6:
    try:
        # Read in pulsasr name, ephemeris, data and output directory
        print("load info from config.json file")
        input_data = load_json('config.json')
        psrname = input_data['psrname']
        ephem = input_data['ephem']
        datadir = input_data['datadir']
        label = input_data['label']
        outdir = input_data['outdir']
    except:
        print("failing to read from config.json file")
        print("Usage: script.py <psrname> <ephem> <datadir> <label> <outdir>")
        sys.exit(1)
else:
    psrname = sys.argv[1]
    ephem = sys.argv[2]
    datadir = sys.argv[3]
    label = sys.argv[4]
    outdir = sys.argv[5]

print(f"psrname: {psrname}")
print(f"ephem: {ephem}")
print(f"datadir: {datadir}")
print(f"label: {label}")
print(f"outdir: {outdir}")

#########

# create output directory if it does not exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

class noise_models:
    def __init__(self, tm, ef):
        self.noise_model_dict = {}
        self.i = 0
        self.model_holder = {}
        self.tm = tm
        self.ef = ef
        self.model_def = {}
    def is_func_in_list(self, func):
        for val in self.noise_model_dict.values():
            if func is val:
                return True
        return False
    def print_noise_models(self):
        print(self.model_def)
    def add_noise_model(self, model, user_string):
        if not self.is_func_in_list(model) and model is not self.ef:
            self.noise_model_dict['n{}'.format(self.i)] = model
            self.model_def['n{}'.format(self.i)] = user_string
            self.i += 1
        else:
            print("Model already in list or model is ef")
    def generate_signal(self):
        s = self.tm + self.ef
        for key in self.noise_model_dict.keys():
            s += self.noise_model_dict[key]
        return s
    def generate_model(self, data):
        n = len(self.noise_model_dict)
        combinations = list(itertools.product([0, 1], repeat=n))   
        for comb in combinations:
            key = '-'.join(map(str,comb))
            non_zero_indices_list = [index for index, value in enumerate(comb) if value != 0]
            model = self.tm + self.ef
            for model_i in non_zero_indices_list:
                 model += self.noise_model_dict['n'+str(model_i)]

            self.model_holder[key] = signal_base.PTA(model(data))

        return self.model_holder


    def get_param_list(self):
        param_list = []
        for key in self.model_holder.keys():
            for param in list(self.model_holder[key].params):
                if param.name not in param_list:
                    param_list.append(param.name)
        return param_list 
    def get_model_key(self, list_of_params_to_identify):
        for key in self.model_holder.keys():
            param_list = []
            for param in list(self.model_holder[key].params):
                if param.name not in param_list:
                    param_list.append(param.name)
            processed_list = self.match_strings(list_of_params_to_identify, param_list)
            if set(list_of_params_to_identify) == set(processed_list):
                print("printing param_list")
                print(param_list)
                print("printing list_of_params to identify")
                print(list_of_params_to_identify)
                print("printing processed_list")
                print(processed_list)
                return key
        return None
    def match_strings(self, main_list, reference_list):
        matched_list = []
        for string in main_list: 
            if any (string in ref for ref in reference_list):
                matched_list.append(string)
        return matched_list   
    def parameter_mapper(self, key):
        a = [param.name for param in list(self.model_holder[key].params)]
        return a
    def generate_key(self, dictionary):
        key = ''
        for num in range(len(self.model_def)):
            if dictionary ['n{}'.format(num)] == 1:
                key += '-1'
            else:
                key += '-0'
        return key[1:] #remove first '-'

def get_rednoise_priors(psr, noisename, noisedict_3sig_min, noisedict_3sig_max, priors,
                                log10_A_min_inp=-16, log10_A_max_inp = -12, 
                                gamma_min_inp=0, gamma_max_inp=5):
    key_ = psr.name + '_' + noisename + '_log10_A'
    log10_A_min = log10_A_min_inp
    log10_A_max = log10_A_max_inp
    gamma_min = gamma_min_inp
    gamma_max = gamma_max_inp
    log10_A_prior = parameter.Uniform(log10_A_min, log10_A_max)
    gamma_prior = parameter.Uniform(gamma_min, gamma_max)
    print(f"""{psr.name}_{noisename}_noise prior:
    log10_A in [{log10_A_min}, {log10_A_max}]
    gamma in [{gamma_min}, {gamma_max}]
    """)
    # # powerlaw
    rednoise_model = gp_priors.powerlaw(log10_A=log10_A_prior,
                                        gamma=gamma_prior)
    key_ = psr.name + '_' + noisename + '_log10_A'
    priors[key_ + "_min"] = log10_A_min
    priors[key_ + "_max"] = log10_A_max
    key_ = psr.name + '_' + noisename + '_gamma'
    priors[key_ + "_min"] = gamma_min
    priors[key_ + "_max"] = gamma_max
    
    return rednoise_model, log10_A_prior, gamma_prior, priors

#convert binary list to decimal number
def binary_to_decimal(binary_string):
    binary_list = [int(item) for item in binary_string.split('-')]

    decimal_number = 0
    length = len(binary_list)
    
    for i, bit in enumerate(binary_list):
        decimal_number += bit * (2 ** (length - i - 1))
    
    return decimal_number

# class analysis:
#     def __init__(self, result_file_name, model_list_object, model_dict):
#         self.model_dict = model_dict
#         self.result_file_name = result_file_name
#         self.model_list_object = model_list_object
#         self.result = bilby.result.read_in_result(filename = self.result_file_name)

#     def run_analysis(self, save_corner_plots=True):

#         print('printing list')
#         print(list(self.model_list_object.model_def.keys()))

#         # Find which indicator columns exist in the tables
#         ind_keys = list(self.model_list_object.model_def.keys())
#         cols_post = [k for k in ind_keys if k in self.result.posterior.columns]
#         cols_nest = [k for k in ind_keys if k in self.result.nested_samples.columns]

#         # Group (or fall back to a single “all-on” group if indicators are absent)
#         groups = (self.result.posterior.groupby(cols_post)
#                 if cols_post else [(None, self.result.posterior)])

#         groups_nested = (self.result.nested_samples.groupby(cols_nest)
#                         if cols_nest else [(None, self.result.nested_samples)])

#         self.model_freq = {}
#         model_freq_nested = {}

#         plotdir = outdir + '/' + 'plots'
#         os.makedirs(plotdir, exist_ok=True)

#         # -----------------------
#         # MAIN LOOP (with guards)
#         # -----------------------
#         for _, df in groups:
#             n = len(df)

#             # If indicators absent in df, synthesize an "all-on" dict for key generation
#             src_dict = {k: 1 for k in ind_keys}
#             if cols_post:
#                 for k in cols_post:
#                     # cast to int in case values are floats/bool
#                     src_dict[k] = int(df.iloc[0][k])

#             key = self.model_list_object.generate_key(src_dict)
#             param = self.model_list_object.parameter_mapper(key)

#             print('printing param')
#             print(key)
#             print(param)
#             print(self.result.posterior.columns)

#             print("debug info:")
#             print(f"model {key}: n={n}")
#             print(f"param list: {param}")
#             print(f"posterior columns: {list(self.result.posterior.columns)}")
#             # guard 1: no params at all
#             if not param:
#                 print(f'[SKIP] model {key}: no parameters returned by parameter_mapper.')
#                 self.model_freq[binary_to_decimal(key)] = n
#                 continue

#             # guard 2: keep only params present in the dataframe; report missing
#             valid_params = [p for p in param if p in df.columns]
#             missing_params = [p for p in param if p not in df.columns]
#             if missing_params:
#                 print(f'[INFO] model {key}: missing params not found in posterior: {missing_params}')
#             if not valid_params:
#                 print(f'[SKIP] model {key}: after filtering, no valid parameters remain.')
#                 self.model_freq[binary_to_decimal(key)] = n
#                 continue

#             samples = df[valid_params].to_numpy()

#             # guard 3: drop all-NaN columns; report which were dropped
#             nan_mask = ~np.all(np.isnan(samples), axis=0)
#             if not np.all(nan_mask):
#                 dropped = [p for p, keep in zip(valid_params, nan_mask) if not keep]
#                 print(f'[INFO] model {key}: dropping all-NaN params: {dropped}')
#             samples = samples[:, nan_mask]
#             valid_params = [p for p, keep in zip(valid_params, nan_mask) if keep]

#             # guard 4: zero-dimensional after NaN drop
#             if samples.ndim != 2 or samples.shape[1] == 0:
#                 print(f'[SKIP] model {key}: no plottable parameters after NaN filtering.')
#                 self.model_freq[binary_to_decimal(key)] = n
#                 continue

#             # record model frequency
#             self.model_freq[binary_to_decimal(key)] = n

#             # plots
#             if save_corner_plots:
#                 try:
#                     fig = corner.corner(
#                         samples,
#                         labels=valid_params,
#                         bins=50,
#                         quantiles=[0.025, 0.5, 0.975],
#                         show_titles=True,
#                         title_kwargs={"fontsize": 12},
#                     )
#                     plt.savefig(f"{plotdir}/{binary_to_decimal(key)}.png", bbox_inches='tight')
#                     plt.close(fig)
#                 except Exception as e:
#                     print(f'[WARN] corner plot failed for model {key}: {e}')

#         for _, df_n in groups_nested:
#             src_dict_n = {k: 1 for k in ind_keys}
#             if cols_nest:
#                 for k in cols_nest:
#                     src_dict_n[k] = int(df_n.iloc[0][k])

#             key_n = self.model_list_object.generate_key(src_dict_n)
#             model_freq_nested[binary_to_decimal(key_n)] = len(df_n)

#         plt.figure()
#         plt.subplot(211)
#         ax = plt.gca()
#         plt.bar(model_freq_nested.keys(), model_freq_nested.values(), color='r', alpha=0.3)
#         x_min, x_max = ax.get_xlim()

#         plt.subplot(212)
#         ax = plt.gca()
#         plt.bar(self.model_freq.keys(), self.model_freq.values())
#         plt.xlim(x_min, x_max)
#         plt.savefig(f"{plotdir}/model_freq.png", bbox_inches='tight')
#         plt.close()

#     def margin_over_model(self):
#         temp_results, _ = tbilby.core.base.preprocess_results(
#             result_in=self.result,model_dict=self.model_dict,
#             remove_ghost_samples=True,return_samples_of_most_freq_component_function=False)
#         # import file with injected data if it exists and put into dictionary
#         filename = Path(datadir + "/" + "parameters.txt")
#         injected_dict = {}
#         if filename.exists():
#             with filename.open("r") as file:
#                 for line in file:
#                     key, value = line.strip().split(": ")
#                     injected_dict[key] = float(value)  # Convert value to float
#             has_injected_values = True
#         else:
#             print(f"{filename} does not exist. Ignoring...")
#             has_injected_values = False
#         print("dictionary of injected values:")
#         # plot the marginalized posteriors
#         results_dic = {}
#         for p in self.model_list_object.get_param_list():
#             pulsar_name = p.split("_")[0]
#             param_name = p[len(pulsar_name) + 1:]
#             injected_value = injected_dict.get(param_name, "Not injected") if has_injected_values else None
#             a = temp_results.posterior[p]
#             results_dic[p]=a[~np.isnan(a)]
#             plt.figure()
#             plt.hist(results_dic[p], bins=50, histtype='step')
#             plt.title("Marginalised Posterior")
#             plt.xlabel(p)
#             plt.ylabel("Count")
#             # Add injected value to the plot only if parameters.txt exists
#             if has_injected_values:
#                 textstr = f"Injected: {injected_value}"
#                 plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
#                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
#             # Save the plots
#             plotdir = outdir + '/' + 'plots'
#             os.makedirs(plotdir, exist_ok=True)
#             plt.savefig(plotdir + '/' + p + '_marg_post.png', bbox_inches='tight')
#     def pp_analysis(self, injected_model_key):
#         injected_model_int = binary_to_decimal(injected_model_key)
#         print("injected model key")
#         print(injected_model_int)
#         self.run_analysis(save_corner_plots = False)
#         if injected_model_int not in self.model_freq:
#             print("Model not found in the dictionary")
#             return -1
#         else: 
#             sorted_models = sorted(self.model_freq.items(), key=lambda x: x[1], reverse=True)
#             rank_dict = {model: rank+1 for rank, (model, _) in enumerate(sorted_models)}
#             return rank_dict.get(injected_model_int, None)  # Return None if the model is not in the dict

class analysis:
    def __init__(self, result_file_name, model_list_object, model_dict):
        self.model_dict = model_dict
        self.result_file_name = result_file_name
        self.model_list_object = model_list_object
        self.result = bilby.result.read_in_result(filename = self.result_file_name)
    def run_analysis(self, save_corner_plots = True):
        print('printing list')
        print(list(self.model_list_object.model_def.keys()))
        groups = self.result.posterior.groupby(list(self.model_list_object.model_def.keys()))
        #groups_nested = self.result.nested_samples.groupby(list(self.model_list_object.model_def.keys()))
        cols_nest = [k for k in self.model_list_object.model_def.keys() if k in self.result.nested_samples.columns]
        groups_nested = self.result.nested_samples.groupby(cols_nest)


        self.model_freq = {}
        model_freq_nested = {}
        for group in groups:
            n = len(group[1])
            key = self.model_list_object.generate_key(group[1].iloc[0].to_dict())
            param = self.model_list_object.parameter_mapper(key)
            print('printing param')
            print(key)
            print(param)
            print(self.result.posterior.columns)
            #if param not in self.result.posterior.columns:
                #param = param[:12] + param[-11:]
                #print(f"ecorr param name has changed to {param}" )
            samples = group[1][param].values
            if len(param) > len(samples):
                print('param length greater than samples length')
                continue
            self.model_freq[binary_to_decimal(key)] = n
            if save_corner_plots:
                fig = corner.corner(samples, labels=param, bins=50, quantiles=[0.025, 0.5, 0.975],
                        show_titles=True, title_kwargs={"fontsize": 12})
                plotdir = outdir + '/' + 'plots'
                plt.savefig(plotdir + '/' + '{}.png'.format(binary_to_decimal(key)), bbox_inches='tight')
        # for group in groups_nested:
        #     n = len(group[1])
        #     key = self.model_list_object.generate_key(group[1].iloc[0].to_dict())
        #     model_freq_nested[binary_to_decimal(key)] = n

        for group in groups_nested:
            n = len(group[1])
            d = group[1].iloc[0].to_dict()
            for k in self.model_list_object.model_def.keys():
                d.setdefault(k, 1)  # assume fixed indicators (like n0) are always on
            key = self.model_list_object.generate_key(d)
            model_freq_nested[binary_to_decimal(key)] = n

        plt.figure()
        plt.subplot(211)
        ax = plt.gca()
        plt.bar(model_freq_nested.keys(), model_freq_nested.values(), color = 'r', alpha = 0.3)
        x_min, x_max = ax.get_xlim()
        plt.subplot(212)
        ax = plt.gca()
        plt.bar(self.model_freq.keys(), self.model_freq.values())
        plt.xlim(x_min, x_max)
        plotdir = outdir + '/' + 'plots'
        os.makedirs(plotdir, exist_ok=True)
        plt.savefig(plotdir + '/' + 'model_freq.png', bbox_inches='tight')
        plt.close()
    def margin_over_model(self):
        temp_results, _ = tbilby.core.base.preprocess_results(result_in=self.result,model_dict=self.model_dict,remove_ghost_samples=True,return_samples_of_most_freq_component_function=False)
        # import file with injected data if it exists and put into dictionary
        filename = Path(datadir + "/" + "parameters.txt")
        injected_dict = {}
        if filename.exists():
            with filename.open("r") as file:
                for line in file:
                    key, value = line.strip().split(": ")
                    injected_dict[key] = float(value)  # Convert value to float
            has_injected_values = True
        else:
            print(f"{filename} does not exist. Ignoring...")
            has_injected_values = False
        print("dictionary of injected values:")
        # plot the marginalized posteriors
        results_dic = {}
        for p in self.model_list_object.get_param_list():
            pulsar_name = p.split("_")[0]
            param_name = p[len(pulsar_name) + 1:]
            injected_value = injected_dict.get(param_name, "Not injected") if has_injected_values else None
            a = temp_results.posterior[p]
            results_dic[p]=a[~np.isnan(a)]
            plt.figure()
            plt.hist(results_dic[p], bins=50, histtype='step')
            plt.title("Marginalised Posterior")
            plt.xlabel(p)
            plt.ylabel("Count")
            # Add injected value to the plot only if parameters.txt exists
            if has_injected_values:
                textstr = f"Injected: {injected_value}"
                plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            # Save the plots
            plotdir = outdir + '/' + 'plots'
            os.makedirs(plotdir, exist_ok=True)
            plt.savefig(plotdir + '/' + p + '_marg_post.png', bbox_inches='tight')
    def pp_analysis(self, injected_model_key):
        injected_model_int = binary_to_decimal(injected_model_key)
        print("injected model key")
        print(injected_model_int)
        self.run_analysis(save_corner_plots = False)
        if injected_model_int not in self.model_freq:
            print("Model not found in the dictionary")
            return -1
        else: 
            sorted_models = sorted(self.model_freq.items(), key=lambda x: x[1], reverse=True)
            rank_dict = {model: rank+1 for rank, (model, _) in enumerate(sorted_models)}
            return rank_dict.get(injected_model_int, None)  # Return None if the model is not in the dict



