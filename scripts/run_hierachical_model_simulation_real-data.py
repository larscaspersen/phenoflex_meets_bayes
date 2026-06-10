#run hierachical model on real data

#run simulation study on hierachical phenoflex model
import argparse
import os
import sys
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
import json
import pandas as pd

parser = argparse.ArgumentParser(description="Run hierarchical model on real data.")
parser.add_argument(
    "--run_cluster",
    action="store_true",
    help="Use cluster-specific absolute paths for helpers, input data, and outputs.",
)
args = parser.parse_args()

run_cluster = args.run_cluster

if run_cluster:
    HELPERS_FILE = "/home/lcaspers_hpc/code/pheno_bayes/helpers/helpers_hierachical_model.py"
    INPUT_BASE_DIR = "/home/lcaspers_hpc/data/calibration_hierach_model/in"
    OUTPUT_DIR = "/home/lcaspers_hpc/data/calibration_hierach_model/out/calibrated_models"

    helpers_dir = os.path.dirname(HELPERS_FILE)
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)

    from helpers_hierachical_model import gen_season_list, prepare_seasons_hierarchical, phenoflex_numpyro_hierarchical, get_jday_from_hour, phenoflex_model_with_custom_priors, solve_for_log_normal_parameters, run_inference, DEFAULT_PRIORS
    from helpers_hierachical_model import phenoflex_model_with_custom_priors_slim, phenoflex_numpyro_hierarchical_slim
else:
    INPUT_BASE_DIR = "."
    OUTPUT_DIR = "calibrated_models"

    # custom functions (local workspace layout)
    from helpers.helpers_hierachical_model import gen_season_list, prepare_seasons_hierarchical, phenoflex_numpyro_hierarchical, get_jday_from_hour, phenoflex_model_with_custom_priors, solve_for_log_normal_parameters, run_inference, DEFAULT_PRIORS
    from helpers.helpers_hierachical_model import phenoflex_model_with_custom_priors_slim, phenoflex_numpyro_hierarchical_slim

PHENOLOGY_DIR = os.path.join(INPUT_BASE_DIR, "phenology")
WEATHER_DIR = os.path.join(INPUT_BASE_DIR, "weather_hourly")
PRIORS_DIR = os.path.join(INPUT_BASE_DIR, "priors")

os.makedirs(OUTPUT_DIR, exist_ok=True)


n_chain = 1
n_warmup = 100
n_samples = 200
use_slim_model = True
truncate_days_after_bloom=45

numpyro.set_host_device_count(n_chain)


# Free-text notes saved to the protocol JSON — describe what you changed or observed
NOTES = """all adamedor data, longer"""

# Map location names (as they appear in phenology data) to weather CSV filenames
LOCATION_WEATHER_FILES = {
    'Cieza':          os.path.join(WEATHER_DIR, 'cieza_hourly.csv'),
    'Klein-Altendorf':os.path.join(WEATHER_DIR, 'klein-altendorf_hourly.csv'),
    'Meknes':         os.path.join(WEATHER_DIR, 'meknes_hourly.csv'),
    'Santomera':      os.path.join(WEATHER_DIR, 'santomera_hourly.csv'),
    'Sfax':           os.path.join(WEATHER_DIR, 'sfax_hourly.csv'),
    'Zaragoza':       os.path.join(WEATHER_DIR, 'zaragoza_hourly.csv'),
}

# Read all weather files
weather_data = {loc: pd.read_csv(fpath) for loc, fpath in LOCATION_WEATHER_FILES.items()}

#read adamedor data
adamedor = pd.read_csv(os.path.join(PHENOLOGY_DIR, "adamedor_sub.csv"))
adamedor = adamedor.rename(columns={'year': 'Year', 'cultivar': 'cultivar_id', 'species': 'species_id', 'location': 'location_id', 'pheno': 'pheno'})

# Make cultivar names unique across species by appending a species abbreviation
SPECIES_ABBREV = {
    'Almond':        'AL',
    'Apple':         'AP',
    'Apricot':       'AR',
    'European Plum': 'EP',
    'Japanese Plum': 'JP',
    'Pear':          'PE',
    'Pistachio':     'PI',
    'Sweet Cherry':  'SC',
}
adamedor['cultivar_id'] = adamedor['cultivar_id'] + '_' + adamedor['species_id'].map(SPECIES_ABBREV)

cka_pheno = adamedor.copy()

# Drop Zaragoza 2022 observations (no matching weather data)
cka_pheno = cka_pheno[~((cka_pheno['location_id'] == 'Zaragoza') & (cka_pheno['Year'] == 2022))]

# Build season_dict_for_hierarchical: keys are (location, year), covering all locations
season_dict_for_hierarchical = {}
for loc, temps_df in weather_data.items():
    loc_years = cka_pheno.loc[cka_pheno['location_id'] == loc, 'Year'].unique()
    if len(loc_years) == 0:
        continue
    loc_seasons = gen_season_list(temps_df, years=loc_years)
    for year, season in zip(loc_years, loc_seasons):
        season_dict_for_hierarchical[(loc, year)] = season

temps, times, bloom_indices, cultivar_idx, location_idx, cultivar_to_species, cultivar_names, species_names, location_names = prepare_seasons_hierarchical(season_dict=season_dict_for_hierarchical, bloom_df=cka_pheno, truncate_days_after_bloom=truncate_days_after_bloom)

#load prior species
prior_species = pd.read_csv(os.path.join(PRIORS_DIR, "prior_species_stats_hierarchical.csv"))
prior_cultivar = pd.read_csv(os.path.join(PRIORS_DIR, "prior_allcultivars_hierarchical.csv"))

#print(prior_species.head())
#print(prior_cultivar.head())

#print(species_names)
#print(cultivar_names)
#filter prior species and prior cultivar depending on cka_pheno
#make sure the order is the same
prior_species_filtered = prior_species[prior_species['Species'].isin(species_names)]
prior_species_filtered = (
    prior_species_filtered
    .set_index('Species')
    .reindex(species_names)          # NaN rows for species not in prior
    .fillna(prior_species_filtered.set_index('Species').mean())  # fill with across-species mean
    .reset_index()
)

#filter for cultivars
prior_cultivar_filtered = prior_cultivar[prior_cultivar['Cultivar'].isin(cultivar_names)]
#if cultivar not covered by prior, add row with zero for CP_standardized and GDH_standardized, and add it to the prior_cultivar_filtered dataframe
for cultivar in cultivar_names:
    if cultivar not in prior_cultivar_filtered['Cultivar'].values:
        new_row = pd.DataFrame({'Cultivar': [cultivar], 'CP_standardized': [0.0], 'GDH_standardized': [0.0]})
        prior_cultivar_filtered = pd.concat([prior_cultivar_filtered, new_row], ignore_index=True)

prior_cultivar_filtered = prior_cultivar_filtered.set_index('Cultivar').loc[cultivar_names].reset_index()

#calculate mean offset per cultivar, for CP_standardized and GDH_standardized, in case there are duplications
# fill NaN with 0.0 = no offset, use species mean as prior (e.g. when GDH was not reported in the literature)
prior_cultivar_filtered['CP_offset'] = prior_cultivar_filtered.groupby('Cultivar')['CP_standardized'].transform('mean').fillna(0.0)
prior_cultivar_filtered['GDH_offset'] = prior_cultivar_filtered.groupby('Cultivar')['GDH_standardized'].transform('mean').fillna(0.0)
# Deduplicate to exactly one row per cultivar, in cultivar_names order
prior_cultivar_filtered = prior_cultivar_filtered.drop_duplicates(subset='Cultivar').set_index('Cultivar').loc[cultivar_names].reset_index()

#transform species yc prior to log-normal parameters
yc_mu, yc_sigma2 = solve_for_log_normal_parameters(prior_species_filtered['CP_mean'], prior_species_filtered['CP_std']**2)
#print(yc_mu)
#print(yc_sigma2)

#
zc_upper = 400
zc_lower = 100 
zc_std_among_cultivars = jnp.array(10.0, dtype=jnp.float32)
zc_offset_std = jnp.array(1.0, dtype=jnp.float32)
yc_offset_std = jnp.array(1.0, dtype=jnp.float32)

"""
#previously used priors, before calculating them from the data
#prior for chill requirement
apple_yc_prior = 65
cherry_yc_prior = 50
apples_yc_std = cherry_yc_std = 10
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
"""

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
    # solve_for_log_normal_parameters returns sigma^2 (log-space variance); LogNormal needs sigma (std dev)
    "yc_species": dist.LogNormal(jnp.array(yc_mu.values, dtype=jnp.float32), jnp.array(np.sqrt(yc_sigma2.values), dtype=jnp.float32)),

    # zc_species: chilling requirement (GDH), two species — model default, set explicitly for protocol logging
    'zc_species' : dist.Uniform(zc_lower, zc_upper),

    # cultivar-level offsets — zc_offset informed by literature GDH data, yc_offset uses model default
    "zc_offset": dist.Normal(jnp.array(prior_cultivar_filtered['GDH_offset'].values, dtype=jnp.float32), zc_offset_std),
    "yc_offset": dist.Normal(jnp.array(prior_cultivar_filtered['CP_offset'].values, dtype=jnp.float32), yc_offset_std),
    #"yc_offset": dist.Normal(0.0, 1.0),

    # among-cultivar spread — model defaults, set explicitly for protocol logging
    "yc_sigma": dist.HalfNormal(10.0),
    "zc_sigma": dist.HalfNormal(20.0),
}

#print(custom_priors)

dat_phenoflex = {
    'temp': temps,
    'times': times,
    'bloom_index': bloom_indices,
    'cultivar_idx': cultivar_idx,
    'cultivar_to_species': cultivar_to_species,
    'custom_priors': custom_priors,  # Pass the custom priors to the model
    'Imodel': 0,    # Using the same Imodel as in the predictive run
    'n_species':   int(np.array(cultivar_to_species).max()) + 1,
    'n_cultivars': int(np.array(cultivar_idx).max()) + 1
}

# Condition the wrapper model with fixed parameters

if use_slim_model:
    conditioned_model = numpyro.handlers.condition(
        phenoflex_model_with_custom_priors_slim,
        data=FIXED_PARAMS_FOR_CONDITIONING,
    )
else:
    #in case of normal model, set traces to no
    dat_phenoflex['return_traces'] = False

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
rng_key, _ = random.split(random.PRNGKey(1))

# Run the inference
mcmc_phenoflex, mcmc_samples_phenoflex, az_mcmc_phenoflex = run_inference(conditioned_model, args_phenoflex, rng_key, dat_phenoflex)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# ── File names (shared across all saved artefacts) ────────────────────────────
pheno_file           = os.path.join(PHENOLOGY_DIR, "adamedor_sub.csv")
inference_data_fname = f"hierarchical_model_mcmc_inference_data_{timestamp}.nc"
raw_samples_fname    = f"hierarchical_model_mcmc_raw_samples_{timestamp}.npz"
training_data_fname  = f"hierarchical_model_training_data_{timestamp}.csv"
protocol_fname       = f"hierarchical_model_protocol_{timestamp}.json"

# --- ArviZ InferenceData ---
az_mcmc_phenoflex.to_netcdf(os.path.join(OUTPUT_DIR, inference_data_fname))
print(f"Saved ArviZ InferenceData to {inference_data_fname}")

# --- Raw MCMC samples (.npz) ---
samples_to_save = {k: np.asarray(v) for k, v in mcmc_samples_phenoflex.items()}
np.savez(os.path.join(OUTPUT_DIR, raw_samples_fname), **samples_to_save)
print(f"Saved raw MCMC samples to {raw_samples_fname}")

# --- Training phenology data ---
cka_pheno[["species_id", "cultivar_id", "Year", "pheno", "location_id"]].to_csv(
    os.path.join(OUTPUT_DIR, training_data_fname), index=False
)
print(f"Saved training data to {training_data_fname}")

# --- Simulation protocol ---
def _prior_to_dict(d):
    """Serialize a numpyro distribution to a JSON-safe dict."""
    params = {}
    for attr in ("loc", "scale", "concentration1", "concentration0", "rate", "low", "high"):
        if hasattr(d, attr):
            val = getattr(d, attr)
            params[attr] = val.tolist() if hasattr(val, "tolist") else val
    return {"distribution": type(d).__name__, **params}

protocol = {
    "timestamp": timestamp,
    "notes": NOTES,
    "model_files": {
        "inference_data": inference_data_fname,
        "raw_samples":    raw_samples_fname,
        "training_data":  training_data_fname,
    },
    "model_config": {
        "use_slim_model":            use_slim_model,
        "n_chains":                  n_chain,
        "n_warmup":                  n_warmup,
        "n_samples":                 n_samples,
        "truncate_days_after_bloom": truncate_days_after_bloom,
    },
    "training_data": {
        #"phenology_file": pheno_file,
        "n_seasons":      int(len(cka_pheno)),
        "species":        species_names,
        "cultivars":      cultivar_names,
    },
    "priors": {
        "fixed_params":    FIXED_PARAMS_FOR_CONDITIONING,
        "default_priors":  {k: _prior_to_dict(v) for k, v in DEFAULT_PRIORS.items()},
        "custom_priors":   {k: _prior_to_dict(v) for k, v in custom_priors.items()},
    },
}
with open(os.path.join(OUTPUT_DIR, protocol_fname), "w") as f:
    json.dump(protocol, f, indent=2)
print(f"Saved simulation protocol to {protocol_fname}")
