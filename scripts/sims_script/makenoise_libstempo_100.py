import numpy as np
import libstempo as T
import libstempo.toasim as LT
import libstempo.plot as LP
import matplotlib.pyplot as plt
import scipy.constants as sc
import math
import os


datadir = '/fred/oz005/users/vdimarco/tBilby/sims/WN_TN_PTASim/output/real_0/'
def plotting(psr, i, filename, noise_name):
    plt.figure(100)
    plt.subplot(7,1,1)
    plt.plot(psr.toas(), psr.residuals(), 'o', label=noise_name)
    plt.ylim([-5*10**(-6), 5*10**(-6)])
    plt.legend()
    plt.subplot(7,1,i+1)
    plt.plot(psr.toas(), psr.residuals(), 'o', label=noise_name)
    plt.ylim([-5*10**(-6), 5*10**(-6)])
    plt.legend()
    plt.savefig(filename + '.png')

### Units
day = 24 * 3600 # Days in seconds
year = 365.25 * day # Year in seconds

AU = sc.astronomical_unit
c = sc.speed_of_light
pc = sc.parsec
AU_light_sec = AU/c
AU_pc = AU/pc

########################
### Functions definition
########################

### Dispersion mesure noise
def add_dm(psr, A, gamma, idx=-2, components=120, seed=None):
    """Add dispersion measure noise variations with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""    

    if seed is not None:
        np.random.seed(seed)

    t = psr.toas()
    fref = 1400
    v = (psr.freqs / fref)**idx

    minx, maxx = np.min(t), np.max(t)
    x = (t - minx) / (maxx - minx)
    T = (day / year) * (maxx - minx)

    size = 2 * components
    F = np.zeros((psr.nobs, size), "d")
    f = np.zeros(size, "d")

    for i in range(components):
        F[:, 2 * i] = np.cos(2 * math.pi * (i + 1) * x)
        F[:, 2 * i + 1] = np.sin(2 * math.pi * (i + 1) * x)

        f[2 * i] = f[2 * i + 1] = (i + 1) / T

    norm = A**2 * year**2 / (12 * math.pi**2 * T) 
    prior = norm * f ** (-gamma)

    y = np.sqrt(prior) * np.random.randn(size)
    psr.stoas[:] += (1.0 / day) * v * np.dot(F, y)

### Chromatic noise
def add_ch(psr, A, gamma, idx=-4, components=33, seed=None):
    """Add chromatic noise variations with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""    

    if seed is not None:
        np.random.seed(seed)

    t = psr.toas()
    fref = 1400
    v = (psr.freqs / fref)**idx

    minx, maxx = np.min(t), np.max(t)
    x = (t - minx) / (maxx - minx)
    T = (day / year) * (maxx - minx)

    size = 2 * components
    F = np.zeros((psr.nobs, size), "d")
    f = np.zeros(size, "d")

    for i in range(components):
        F[:, 2 * i] = np.cos(2 * math.pi * (i + 1) * x)
        F[:, 2 * i + 1] = np.sin(2 * math.pi * (i + 1) * x)

        f[2 * i] = f[2 * i + 1] = (i + 1) / T

    norm = A**2 * year**2 / (12 * math.pi**2 * T) 
    prior = norm * f ** (-gamma)

    y = np.sqrt(prior) * np.random.randn(size)
    psr.stoas[:] += (1.0 / day) * v * np.dot(F, y)

### Solar wind
def add_sw(psr, A, gamma, components=33, seed=None):
    """Add Solar wind variations with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases. 
    The electron density at 1 AU distance (nearth) is 1.0
    Optionally take a pseudorandom-number-generator seed."""    

    if seed is not None:
        np.random.seed(seed)

    t = psr.toas()

    theta, R_earth = theta_impact(psr.earth_ssb, psr.sun_ssb, psr.psrPos)
    dt_sol_wind = (4.148808e3 /(psr.freqs**2))*dm_solar(1.0, theta, R_earth)

    minx, maxx = np.min(t), np.max(t)
    x = (t - minx) / (maxx - minx)
    T = (day / year) * (maxx - minx)

    size = 2 * components
    F = np.zeros((psr.nobs, size), "d")
    f = np.zeros(size, "d")

    for i in range(components):
        F[:, 2 * i] = np.cos(2 * math.pi * (i + 1) * x)
        F[:, 2 * i + 1] = np.sin(2 * math.pi * (i + 1) * x)

        f[2 * i] = f[2 * i + 1] = (i + 1) / T

    norm = A**2 * year**2 / (12 * math.pi**2 * T)
    prior = norm * f ** (-gamma)

    y = np.sqrt(prior) * np.random.randn(size)
    psr.stoas[:] += (1.0 / day) * dt_sol_wind * np.dot(F, y)

def theta_impact(earthssb,sunssb,pos_t):
    """Computes the solar impact angle"""
    earth = earthssb[:,:3]
    sun = sunssb[:,:3]
    pulsar = pos_t[:,:3]
    earthsun = earth - sun
    R_earth = np.sqrt(np.einsum('ij,ij->i', earthsun, earthsun))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earthsun, pulsar)
    theta = np.arccos(-Re_cos_theta_impact / R_earth)
    return theta, R_earth

def _dm_solar_close(n_earth, r_earth):
    return (n_earth * AU_light_sec * AU_pc / r_earth)

def _dm_solar(n_earth, theta, r_earth):
    return ((np.pi - theta) *
            (n_earth * AU_light_sec * AU_pc
             / (r_earth * np.sin(theta))))

def dm_solar(n_earth, theta, r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proto/electron density at Earth (1/cm^3)
    ::param :theta: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 2007 for more details.
    """
    return np.where(np.pi - theta >= 1e-5,
                    _dm_solar(n_earth, theta, r_earth),
                    _dm_solar_close(n_earth, r_earth))


############
### Run code
############


for i in range(300):
    base_psr = T.tempopulsar(parfile=datadir + 'J0437-4715.par', 
                        timfile=datadir + 'J0437-4715.tim')

    print(f"Simulating pulsar {i}")
    # draw parameters for noise sources

    params = {
        "efac": np.random.uniform(0.5, 1.5),
        "ecorr": np.random.uniform(-6.5, -6),
        "equad": np.random.uniform(-6.5, -6),
        "red_amp": np.random.uniform(-14.2, -13.8),
        "red_gamma": np.random.uniform(1.5, 3.0),
        "dm_amp": np.random.uniform(-13.6, -13.2),
        "dm_gamma": np.random.uniform(3.5, 4.5),
        "ch_amp": np.random.uniform(-14.8, -14.4),
        "ch_gamma": np.random.uniform(3.5, 4.5)
    }

    # params = {
    #     "efac": 1,
    #     "ecorr": -6,
    #     "equad": -6,
    #     "red_amp": -14,
    #     "red_gamma": 3,
    #     "dm_amp": -13,
    #     "dm_gamma": 3,
    #     "ch_amp": -13.5,
    #     "ch_gamma": 3.5
    # }

    # Ensure `efac` is always included
    selected_params = {
    "efac": params["efac"]
    }

    # Select additional random parameters
    noise_sources = ["red", "dm", "ch"]
    num_selected_sources = np.random.randint(0, len(noise_sources) + 1)
    selected_sources = np.random.choice(noise_sources, num_selected_sources, replace=False)
    for src in selected_sources:
        selected_params[f"{src}_amp"] = params[f"{src}_amp"]
        selected_params[f"{src}_gamma"] = params[f"{src}_gamma"]
    optional_single_params = ["ecorr", "equad"]
    num_wn = np.random.randint(0, len(optional_single_params) + 1)
    selected_wn = np.random.choice(optional_single_params, num_wn, replace=False)
    for param in selected_wn:
        selected_params[param] = params[param]

    # # simulate from fake pulsar for 3 radio frequencies
    # t = base_psr.toas()
    # minx, maxx = np.min(t), np.max(t)
    # obstimes = np.arange(minx, maxx, 30) + np.random.randn(len(np.arange(minx, maxx, 30)))
    # obstimes_600 = obstimes
    # obstimes_1400 = obstimes
    # obstimes_3200 = obstimes

    # # Combine TOAs into a single array
    # combined_obstimes = np.concatenate([obstimes_600, obstimes_1400, obstimes_3200])

    # # Corresponding frequencies and errors for each TOA
    # combined_freq = np.concatenate([
    #     np.full(len(obstimes_600), 600),
    #     np.full(len(obstimes_1400), 1400),
    #     np.full(len(obstimes_3200), 3200)
    # ])
    # combined_toaerr = np.concatenate([
    #     np.full(len(obstimes_600), 0.1),
    #     np.full(len(obstimes_1400), 0.1),
    #     np.full(len(obstimes_3200), 0.1)
    # ])

    parfile = datadir + 'J0437-4715.par'
    timfile = datadir + 'J0437-4715.tim'
    #psr = LT.fakepulsar(parfile=parfile, obstimes=combined_obstimes, toaerr=combined_toaerr, freq=combined_freq, observatory="pks", flags="-or pks")
    #psr = LT.fakepulsar(parfile=parfile, obstimes=obstimes, toaerr=0.1, observatory="pks", flags="-or pks")
    psr = T.tempopulsar(parfile=parfile,timfile=timfile)
    LT.make_ideal(psr)

    
    #plotting(psr, 1, "psr_debugging", "no_noise")
    #psr = LT.fakepulsar(parfile=datadir + 'J0437-4715.par', obstimes=(obstimes), toaerr=0.1, freq=1400)

    # Inject parameters always present
    LT.add_efac(psr, selected_params["efac"])
    ## Debugging plotter
    #plotting(psr, 2, "psr_debugging", "EF")
    #LT.add_rednoise(psr, 10**selected_params["red_amp"], selected_params["red_gamma"])
    #LT.add_rednoise(psr,3.311e-14,2.5)

    # Inject params
    if "equad" in selected_params:
        LT.add_equad(psr, 10**selected_params["equad"])
        ## Debugging plotter
        #plotting(psr, 3, "psr_debugging", "EQ")
    if "ecorr" in selected_params:
        LT.add_jitter(psr, 10**selected_params["ecorr"])
        ## Debugging plotter
        #plotting(psr, 4, "psr_debugging", "EC")
    if "dm_amp" in selected_params:
        #add_dm(psr,2e-14,3.3)
        add_dm(psr, 10**selected_params["dm_amp"], selected_params["dm_gamma"])
        ## Debugging plotter
        #plotting(psr, 5, "psr_debugging", "dm")
    if "ch_amp" in selected_params:
        #add_ch(psr,5e-14,3)
        add_ch(psr, 10**selected_params["ch_amp"], selected_params["ch_gamma"])
        ## Debugging plotter
        #plotting(psr, 6, "psr_debugging", "ch")
    if "red_amp" in selected_params:
        LT.add_rednoise(psr, 10**selected_params["red_amp"], selected_params["red_gamma"], components=33)


    outdir = f'/fred/oz005/users/vdimarco/tBilby/sims/300_sims_tight_uniforms_2/test_sim_{i}/'
    os.makedirs(outdir, exist_ok=True)
    # Save params
    with open(outdir + "parameters.txt", "w") as f:
        for key, value in selected_params.items():
            f.write(f"{key}: {value}\n")
    os.makedirs(outdir, exist_ok=True)
    psr.savepar(outdir + 'J0437-4715.par')
    psr.savetim(outdir + 'J0437-4715.tim')

