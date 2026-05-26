#run simulation study on hierachical phenoflex model
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import lax
import arviz as az
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, Predictive
# Building on numpyro AR2 example: https://num.pyro.ai/en/latest/examples/ar2.html
import pandas as pd

#custom functions
from helpers.helpers_hierachical_model import gen_season_list, prepare_seasons_hierarchical, phenoflex_numpyro_hierarchical, get_jday_from_hour, phenoflex_model_with_custom_priors, solve_for_log_normal_parameters, run_inference

n_chain = 2
n_warmup = 2000
n_samples = 2000

numpyro.set_host_device_count(n_chain)


KA_temp_hourly = pd.read_csv("weather_hourly/klein-altendorf_hourly.csv")

#generate dummy data.frame to prepare temperature data to run model
pheno_data_dummy = pd.DataFrame(columns=['Year', 'pheno', 'cultivar_id', 'species_id', 'location_id'])
#years from 1999 to 2010, for two species and two cultivars each
pheno_data_dummy['Year'] = np.arange(1999, 2010).repeat(4)
pheno_data_dummy['cultivar_id'] = ['Elstar', 'Regona', 'Symphony', 'Sentenniel'] * np.arange(1999, 2010).shape[0]
pheno_data_dummy['species_id'] = ['Apple', 'Apple', 'Sweet Cherry', 'Sweet Cherry'] * np.arange(1999, 2010).shape[0]
pheno_data_dummy['location_id'] = ['Klein-Altendorf'] * 4 * np.arange(1999, 2010).shape[0]
pheno_data_dummy['pheno'] = [10] * 4 * np.arange(1999, 2010).shape[0]

#prepare temperature data
years = np.arange(1999,2010)
seasons = gen_season_list(KA_temp_hourly, years=years)

# The 'seasons' variable is a list of DataFrames, but prepare_seasons_hierarchical expects a dictionary.
# We need to transform 'seasons' into a dictionary with (location_id, year) as keys.
season_dict_for_hierarchical = {}
location = 'Klein-Altendorf' # All seasons currently generated are for this location

for i, year in enumerate(years):
    # Assuming all seasons in 'seasons' list correspond to the same location
    # and are in the order of 'years' array.
    season_dict_for_hierarchical[(location, year)] = seasons[i]

temps, times, bloom_indices, cultivar_idx, location_idx, cultivar_to_species, cultivar_names, species_names, location_names = prepare_seasons_hierarchical(season_dict_for_hierarchical, pheno_data_dummy)


# Two species: apple(0), sw_cherry(1)
#data for apple: Delgado et al 2021: 10.1016/j.eja.2021.126374
#data for cherry: Santolaria et al 2026 10.1016/j.agrformet.2026.111138
#chose both times low and high requirement
# Four cultivars: Elstar(0), Regona(1), Symphony(2), Sentenniel(3)

# Fixed species-level means
# yc_species: shape (n_species,)
# zc_species: shape (n_species,)
FIXED_HIERARCHICAL_PARAMS = dict(
    yc_species = jnp.array([75, 55]),    # apple mean, pear mean
    zc_species = jnp.array([270.0, 200.0]),  # apple mean, pear mean

    # Offsets that place each cultivar relative to its species mean
    # yc_cultivar = yc_species[cultivar_to_species] + yc_sigma * yc_offset
    # To fix yc_cultivar directly, work backwards:
    # yc_offset = (yc_cultivar_target - yc_species[cultivar_to_species]) / yc_sigma
    yc_sigma   = 10.0,
    zc_sigma   = 20.0,

    # Target cultivar means:
    #   Elstar:     yc=65,  zc=270   (low chill apple)
    #   Regona:     yc=90,  zc=210   (high chill apple)
    #   Symphony:   yc=37,  zc=220   (low chill cherry)
    #   Sentenniel: yc=66, zc=180   (high chill cherry)
    #
    # yc_offset = (target - species_mean) / yc_sigma
    # Elstar:     (65  - 75) / 10 = -1.0
    # Regona:     (90  - 75) / 10 = +1.5
    # Symphony:   (37  - 55) / 10 = -1.8
    # Sentenniel: (66 -  55) / 10 = +1.0
    yc_offset  = jnp.array([-1.0,  1.0, -1.0,  1.0]),
    zc_offset  = jnp.array([1.0,  -1.0, 1.0,   -1.0]),

    # Fix all the shape parameters too
    E0=4153.5, E1=12888.8, A0=139500.0, A1=2.567e18,
    Tf=4.0, slope=1.6, Tb=4.0, Tu=26.0, Tc=36.0, Delta=4.0,
    k=0.1, s1=0.5,
)

conditioned_model = numpyro.handlers.condition(
    phenoflex_numpyro_hierarchical,
    data=FIXED_HIERARCHICAL_PARAMS,
)

#predict the bloom date
predictive = Predictive(conditioned_model, num_samples=1)
rng_key = jax.random.PRNGKey(0)

preds = predictive(
    rng_key,
    temp=temps,                          # shape (4, N) — one season per cultivar
    times=times,                         # shape (4, N)
    cultivar_idx=cultivar_idx,
    cultivar_to_species=cultivar_to_species,
    bloom_index=None,                    # prediction mode
    Imodel=0,
)

print(preds["bloom_hour_pred"])          # predicted bloom hour per season
print(preds["yc_cultivar"])             # should match your targets above
print(preds["zc_cultivar"])


#now I want to see, if the pattern can be retrieved by the model. I want the predictions as an input and then run mccm
#extract the most likely bloom date and bring
#preds["most_likely_bloom_hour_pred"]

pheno_data_in = pheno_data_dummy.copy()
#get shape of most likely hour
pheno_data_in['most_likely_hour'] = np.concatenate(preds["most_likely_bloom_hour_pred"])


pheno_data_in['pheno'] = pheno_data_in.apply(get_jday_from_hour, 
                                             args=(season_dict_for_hierarchical,),
                                               axis=1)
#display(pheno_data_in)


temps, times, bloom_indices, cultivar_idx, location_idx, cultivar_to_species, cultivar_names, species_names, location_names = prepare_seasons_hierarchical(season_dict_for_hierarchical, pheno_data_in)


#prior for chill requirement
apple_yc_prior = 80
cherry_yc_prior = 50
apples_yc_std = cherry_yc_std = 5
apple_zc_mean = 250 # Placeholder, will use actual calculated values from df_apple and df_cherries_pheno
cherry_zc_mean = 250 # Placeholder, will use actual calculated values from df_apple and df_cherries_pheno
apple_zc_std = 20 # Placeholder
cherry_zc_std = 20 # Placeholder

yc_std_among_cultivars = jnp.array(10.0, dtype=jnp.float32)
zc_std_among_cultivars = jnp.array(10.0, dtype=jnp.float32)

yc_offset = jnp.array(0.0, dtype=jnp.float32)
yc_offset_std = jnp.array(1.0, dtype=jnp.float32)
zc_offset = jnp.array(0.0, dtype=jnp.float32)
zc_offset_std = jnp.array(1.0, dtype=jnp.float32)

apple_yc_prior_mu, apple_yc_prior_sigma2 = solve_for_log_normal_parameters(apple_yc_prior, apples_yc_std**2)
cherry_yc_prior_mu, cherry_yc_prior_sigma2 = solve_for_log_normal_parameters(cherry_yc_prior, cherry_yc_std**2)

# The FIXED_HIERARCHICAL_PARAMS now only contain the fixed shape parameters
FIXED_PARAMS_FOR_CONDITIONING = dict(
    # Fix all the shape parameters to single values
    E0=4153.5, E1=12888.8, A0=139500.0, A1=2.567e18,
    Tf=4.0, slope=1.6, Tb=4.0, Tu=26.0, Tc=36.0, Delta=4.0,
    k=0.1, s1=0.5,
)

# Define custom priors for the hierarchical parameters that will be sampled
# from these distributions instead of the default ones in the model.
# The species order is ['Apple', 'Sweet Cherry'] as inferred from species_names.
custom_priors = {
    # yc_species: forcing requirement, two species (Apple, Sweet Cherry)
    "yc_species": dist.LogNormal(jnp.array([apple_yc_prior_mu, cherry_yc_prior_mu], dtype=jnp.float32), jnp.array([jnp.sqrt(apple_yc_prior_sigma2), jnp.sqrt(cherry_yc_prior_sigma2)], dtype=jnp.float32)),

    # zc_species: chilling requirement (CP), two species (Apple, Sweet Cherry)
    # Using the calculated means and stds: apple_cp_mean for Apple, cherry_mean for Sweet Cherry
    #"zc_species": dist.Normal(jnp.array([apple_zc_mean, cherry_zc_mean], dtype=jnp.float32), jnp.array([apple_zc_std, cherry_zc_std], dtype=jnp.float32)),
    #"zc_species": dist.unif Normal(jnp.array([apple_zc_mean, cherry_zc_mean], dtype=jnp.float32), jnp.array([apple_zc_std, cherry_zc_std], dtype=jnp.float32)),
    'zc_species' : dist.Uniform(150.0, 350.0),

    # yc_sigma, zc_sigma: variability around species mean, HalfNormal priors
    "yc_sigma": dist.HalfNormal(yc_std_among_cultivars),
    "zc_sigma": dist.HalfNormal(zc_std_among_cultivars),

    # yc_offset, zc_offset: cultivar-specific deviations, Normal priors
    "yc_offset": dist.Normal(yc_offset, yc_offset_std),
    "zc_offset": dist.Normal(zc_offset, zc_offset_std),
}

dat_phenoflex = {
    'temp': temps,
    'times': times,
    'bloom_index': bloom_indices,
    'cultivar_idx': cultivar_idx,
    'cultivar_to_species': cultivar_to_species,
    'custom_priors': custom_priors,  # Pass the custom priors to the model
    'Imodel': 0,    # Using the same Imodel as in the predictive run
    'n_species':   int(np.array(cultivar_to_species).max()) + 1,
    'n_cultivars': int(np.array(cultivar_idx).max()) + 1,
    'return_traces': False
}

# Condition the wrapper model with fixed parameters
conditioned_model = numpyro.handlers.condition(
    phenoflex_model_with_custom_priors,
    data=FIXED_PARAMS_FOR_CONDITIONING,
)

#print(conditioned_model)

#settings for the optimization
# Define args for run_inference
args_phenoflex = {}
args_phenoflex['num_warmup'] = n_warmup
args_phenoflex['num_samples'] = n_samples
args_phenoflex['num_chains'] = n_chain


# Run the inference
mcmc_phenoflex, mcmc_samples_phenoflex, az_mcmc_phenoflex = run_inference(conditioned_model, args_phenoflex, rng_key, dat_phenoflex)

from datetime import datetime
# Get current date and hour for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# --- Saving az_mcmc_phenoflex (ArviZ InferenceData object) ---
# This is the recommended way to save full inference results.
# It saves all chains, samples, and diagnostics in a single file.
az_mcmc_phenoflex.to_netcdf(f"calibrated_models/hierarchical_model_mcmc_inference_data_{timestamp}.nc")
print("Saved ArviZ InferenceData to hierarchical_model_mcmc_inference_data.nc")

# --- Saving mcmc_samples_phenoflex (dictionary of JAX arrays) ---
# This saves the raw samples for each parameter into a .npz file.
# First, convert JAX arrays to NumPy arrays if they aren't already.
# NumPyro's MCMC.get_samples() typically returns JAX arrays.
samples_to_save = {k: np.asarray(v) for k, v in mcmc_samples_phenoflex.items()}
np.savez(f"calibrated_models/hierarchical_model_mcmc_raw_samples_{timestamp}.npz", **samples_to_save)
print("Saved raw MCMC samples to mcmc_raw_samples.npz")