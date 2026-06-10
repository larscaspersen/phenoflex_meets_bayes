import pandas as pd
import numpy as np
import numpyro
from jax import random
import os
import time
import arviz as az
from datetime import datetime
from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.phenoflex_numpyro_hazard import phenoflex_numpyro_hazard
from helpers.helpers_phenoflex_pp import (
    prepare_seasons_hazard,
    gen_season_list,
    plot_bloom_pmf,
)

run_forward_prediction = False
run_mcmc = True  # Set to False to skip MCMC and just run the predictive model
show_plots = False
timestamp_model = None  # Example timestamp for loading specific MCMC results
intermed_par = True
n_chain = 1
n_warmup = 5
n_samples = 10

numpyro.set_host_device_count(n_chain)


# read klein-altendorf temperature
cka_temp = pd.read_csv("weather_hourly/klein-altendorf_hourly.csv")

# read phenology data from cka
cka_bloom = pd.read_csv("KA_bloom.csv")
cka_bloom = cka_bloom.dropna()

FIXED_PARAMS = dict(
    # yc=40.0,
    # zc=190.0,
    # s1=0.5,
    #E0=4153.5,
    #E1=12888.8,
    #A0=139500.0,
    #A1=2.567e18,
    #Tf=4.0,
    #slope=1.6,
    #Tb=4.0,
    #Tu=26.0,
    #Tc=36.0,
    Delta=4.0,
)

# ── Example ───────────────────────────────────────────────────────────────────────
if run_forward_prediction:
    years = np.arange(1986, 1991)
    seasons = gen_season_list(cka_temp, years=years)
    temps, times, bloom_index = prepare_seasons_hazard(seasons, cka_bloom, years)

    print(f"Temps shape : {temps.shape}")
    print(f"Times shape : {times.shape}")
    print(f"Bloom Index  : {bloom_index}")


    conditioned_model = numpyro.handlers.condition(
        phenoflex_numpyro_hazard, data=FIXED_PARAMS
    )
    predictive = Predictive(conditioned_model, num_samples=1000)
    rng_key, _ = random.split(random.PRNGKey(1))

    preds = predictive(
        rng_key,
        temp=temps,
        times=times,
        # bloom_index=bloom_index,
        Imodel=0,
        return_traces=True,
    )

    print(preds.keys())

    # Get predicted bloom PMFs (Probability Mass Functions)
    # Shape: (num_samples, num_seasons, num_timesteps - 1)
    bloom_pmf_pred = preds["bloom_pmf_pred"]

    # Plot the results with DOY on x-axis
    fig = plot_bloom_pmf(
        bloom_pmf_pred,
        times,
        years,
        bloom_index=bloom_index,
        season_list=seasons,
        x_axis="doy",
        xlim=(60, 150),
    )
    plt.show()


# ── Run MCMC Inference ────────────────────────────────────────────────────────────
#'''
if run_mcmc:

    years = np.arange(1986, 2000)
    seasons = gen_season_list(cka_temp, years=years)
    temps, times, bloom_index = prepare_seasons_hazard(seasons, cka_bloom, years)

    def run_inference(model, args, rng_key, dat):
        start = time.time()
        sampler = numpyro.infer.NUTS(model)
        mcmc = numpyro.infer.MCMC(
            sampler,
            num_warmup=args["num_warmup"],
            num_samples=args["num_samples"],
            num_chains=args["num_chains"],
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        )
        mcmc.run(rng_key, **dat)
        mcmc.print_summary()
        az_mcmc = az.from_numpyro(mcmc)
        print("\nMCMC elapsed time:", time.time() - start)
        return mcmc, mcmc.get_samples(), az_mcmc


    # Prepare the 'dat' dictionary for phenoflex_numpyro
    # Reusing the data prepared in previous cells (e.g., from plSeY2Y-C2NC)
    dat_phenoflex = {
        "temp": temps,
        "times": times,
        "bloom_index": bloom_index,
        "Imodel": 0,  # Using the same Imodel as in the predictive run
        "use_intermediate_params": intermed_par,  # Include intermediate parameters if the flag is set
        "return_traces": False
    }

    # Define args for run_inference
    args_phenoflex = {}
    args_phenoflex["num_warmup"] = n_warmup
    args_phenoflex["num_samples"] = n_samples
    args_phenoflex["num_chains"] = n_chain

    # Set the model to phenoflex_numpyro
    model_phenoflex = phenoflex_numpyro_hazard

    # Condition the model with fixed parameters as done previously (e.g., in qmpen104LyfF)
    conditioned_model_phenoflex = numpyro.handlers.condition(
        model_phenoflex, data=FIXED_PARAMS
    )
    rng_key, _ = random.split(random.PRNGKey(1))
    # Run the inference
    mcmc_phenoflex, mcmc_samples_phenoflex, az_mcmc_phenoflex = run_inference(
        conditioned_model_phenoflex, args_phenoflex, rng_key, dat_phenoflex
    )

    # Get current date and hour for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # --- Saving az_mcmc_phenoflex (ArviZ InferenceData object) ---
    # This is the recommended way to save full inference results.
    # It saves all chains, samples, and diagnostics in a single file.
    #az_mcmc_phenoflex.to_netcdf(f"calibrated_models/mcmc_inference_data_{timestamp}.nc")
    print("Saved ArviZ InferenceData to mcmc_inference_data.nc")

    # --- Saving mcmc_samples_phenoflex (dictionary of JAX arrays) ---
    # This saves the raw samples for each parameter into a .npz file.
    # First, convert JAX arrays to NumPy arrays if they aren't already.
    # NumPyro's MCMC.get_samples() typically returns JAX arrays.
    samples_to_save = {k: np.asarray(v) for k, v in mcmc_samples_phenoflex.items()}
    #np.savez(f"calibrated_models/mcmc_raw_samples_{timestamp}.npz", **samples_to_save)
    print("Saved raw MCMC samples to mcmc_raw_samples.npz")

if show_plots:
    #read mcmc results 
    # Load the ArviZ InferenceData object depending on the timestamp

    #incase the timesstamp of the model to read was not specified, take the most recent one from the calibrated_models folder
    if timestamp_model is None:
        model_files = [f for f in os.listdir("calibrated_models") if f.startswith("mcmc_inference_data_") and f.endswith(".nc")]
        if not model_files:
            raise FileNotFoundError("No MCMC inference data files found in calibrated_models folder.")
        # Extract timestamps and find the most recent one
        timestamps = [f[len("mcmc_inference_data_"):-len(".nc")] for f in model_files]
        most_recent_timestamp = max(timestamps)
        timestamp_model = most_recent_timestamp
        print(f"No timestamp specified. Using most recent MCMC inference data: {timestamp_model}")


    loaded_az_mcmc = az.from_netcdf(f"calibrated_models/mcmc_inference_data_{timestamp}.nc")
    print("\nLoaded ArviZ InferenceData:")
    print(loaded_az_mcmc)

    # Load the raw MCMC samples (Numpy array format)
    loaded_samples = np.load(f"calibrated_models/mcmc_raw_samples_{timestamp}.npz")
    print("\nLoaded raw MCMC samples keys:")
    for key in loaded_samples.keys():
        print(f" - {key}")


    #vizualize the bloom PMF predictions from the loaded MCMC samples
    flat_yc = loaded_samples['yc'].flatten()
    flat_zc = loaded_samples['zc'].flatten()
    flat_s1 = loaded_samples['s1'].flatten()
    flat_k = loaded_samples['k'].flatten()


    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # Increased figsize for better visibility

    # First plot (yc)
    axes[0, 0].hist(flat_yc, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Chill requirement (yc)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of yc')

    # Second plot (zc)
    axes[0, 1].hist(flat_zc, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Heat requirement (zc)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of zc')

    # Third plot (s1)
    axes[1, 0].hist(flat_s1, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Slope of transition function (s1)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of s1')

    # Fourth plot (k)
    axes[1, 1].hist(flat_k, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Rate parameter (k)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of k')

    plt.tight_layout()
    plt.show()


    # Plot the posterior
    plot_data = {
        'yc': loaded_samples['yc'],
        'zc': loaded_samples['zc'],
        's1': loaded_samples['s1'],
        'k': loaded_samples['k'],

        
    }
    g = sns.PairGrid(pd.DataFrame.from_dict(plot_data))
    g.map_lower(plt.scatter,s=0.1)
    g.map_diag(sns.histplot, lw=3, legend=False)
    # Hide upper plots of grid
    def hide_current_axis(*args, **kwds):
        plt.gca().set_visible(False)
    g.map_upper(hide_current_axis)
    g.tight_layout()
    plt.show()


    # --- How to load them back (for demonstration) ---
    # To load ArviZ InferenceData:
    # loaded_az_mcmc = az.from_netcdf("mcmc_inference_data.nc")
    # print("Loaded ArviZ InferenceData keys:", loaded_az_mcmc.posterior.data_vars.keys())

    # To load raw MCMC samples:
    # loaded_samples = np.load("mcmc_raw_samples.npz")
    # print("Loaded raw samples keys:", loaded_samples.keys())
    # # You can access individual parameters like: loaded_samples['yc']
    #'''
