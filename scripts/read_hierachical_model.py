# After sampling, check these in ArviZ
import arviz as az
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from helpers.helpers_hierachical_model import (
    gen_season_list, prepare_seasons_hierarchical,
    phenoflex_numpyro_hierarchical,
    solve_for_log_normal_parameters, DEFAULT_PRIORS,
)
import json
timestamp_model = "20260605_1626"  # Set to None to automatically find the most recent model

#calibrated_models\hierarchical_model_mcmc_inference_data_20260601_1629.nc

if timestamp_model is None:
    model_files = [f for f in os.listdir("calibrated_models") if f.startswith("hierarchical_model_mcmc_inference_data_") and f.endswith(".nc")]
    if not model_files:
        raise FileNotFoundError("No MCMC inference data files found in calibrated_models folder.")
    # Extract timestamps and find the most recent one
    timestamps = [f[len("hierarchical_model_mcmc_inference_data_"):-len(".nc")] for f in model_files]
    most_recent_timestamp = max(timestamps)
    timestamp_model = most_recent_timestamp
    print(f"No timestamp specified. Using most recent MCMC inference data: {timestamp_model}")


loaded_az_mcmc = az.from_netcdf(f"calibrated_models/hierarchical_model_mcmc_inference_data_{timestamp_model}.nc")
print("\nLoaded ArviZ InferenceData:")
print(loaded_az_mcmc)

# Load the raw MCMC samples (Numpy array format)
loaded_samples = np.load(f"calibrated_models/hierarchical_model_mcmc_raw_samples_{timestamp_model}.npz")
print("\nLoaded raw MCMC samples keys:")

#data = az.from_numpyro(mcmc)   # or az.from_pymc(trace)

# Divergences — the smoking gun for funnel geometry
n_diverging = int(loaded_az_mcmc.sample_stats["diverging"].values.sum())
print(f"Total divergences: {n_diverging}")

# R-hat > 1.01 → chains not mixing → possible multimodality or funnel
print(az.summary(loaded_az_mcmc)[["r_hat", "ess_bulk", "ess_tail"]])

# Pair plots reveal correlations and funnels
# Restrict to key hierarchical scale parameters to avoid 1024-subplot overflow
pair_param_vars = ["yc_sigma", "zc_sigma", "yc_species", "zc_species"]

# Flatten chains x draws and split species-dimensioned arrays into per-species columns
pair_data = {}
for var in ["yc_sigma", "zc_sigma"]:
    pair_data[var] = loaded_az_mcmc.posterior[var].values.reshape(-1)
for var in ["yc_species", "zc_species"]:
    arr = loaded_az_mcmc.posterior[var].values.reshape(
        -1, loaded_az_mcmc.posterior[var].values.shape[-1])  # (n_post, n_species)
    for s_i in range(arr.shape[1]):
        pair_data[f"{var}[{s_i}]"] = arr[:, s_i]

col_names = list(pair_data.keys())
n_cols = len(col_names)
fig_pair, axes_pair = plt.subplots(n_cols, n_cols,
                                    figsize=(2.5 * n_cols, 2.5 * n_cols))
for i, yi in enumerate(col_names):
    for j, xj in enumerate(col_names):
        ax = axes_pair[i, j]
        if i == j:
            ax.hist(pair_data[yi], bins=30, color="steelblue", alpha=0.7)
        else:
            ax.scatter(pair_data[xj], pair_data[yi], s=2, alpha=0.3, color="steelblue")
        if i == n_cols - 1:
            ax.set_xlabel(xj, fontsize=7)
        if j == 0:
            ax.set_ylabel(yi, fontsize=7)
        ax.tick_params(labelsize=6)
fig_pair.suptitle("Posterior pair plot: hierarchical model parameters")
plt.tight_layout()
plt.show()

# Density plots for yc_offset and zc_offset (all cultivars in one plot each)
for param in ["yc_offset", "zc_offset"]:
    samples = loaded_az_mcmc.posterior[param].values  # shape: (chains, draws, cultivars)
    samples_flat = samples.reshape(-1, samples.shape[-1])  # (chains*draws, cultivars)
    n_cultivars = samples_flat.shape[1]

    ncols = 4
    nrows = int(np.ceil(n_cultivars / ncols))
    x_min, x_max = samples_flat.min(), samples_flat.max()
    bins = np.linspace(x_min, x_max, 31)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), sharey=True)
    axes = axes.flatten()

    for i in range(n_cultivars):
        axes[i].hist(samples_flat[:, i], bins=bins, density=True, color="steelblue", alpha=0.7)
        axes[i].set_title(f"cultivar {i}")
        axes[i].set_xlabel(param)
    for j in range(n_cultivars, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(param)
    plt.tight_layout()
    plt.show()

# ─── Phenology predictions: predicted vs observed bloom DOY ─────────────────

# 1. Load protocol to get fixed params and model config
protocol_file = f"calibrated_models/hierarchical_model_protocol_{timestamp_model}.json"
if os.path.exists(protocol_file):
    with open(protocol_file) as f:
        protocol = json.load(f)
    fixed_params = {k: float(v) for k, v in protocol["priors"]["fixed_params"].items()}
    truncate_days_after_bloom = protocol["model_config"]["truncate_days_after_bloom"]
else:
    print("No protocol file found; using default fixed params.")
    fixed_params = dict(E0=4153.5, E1=12888.8, A0=139500.0, A1=2.567e18,
                        Tf=4.0, slope=1.6, Tb=4.0, Tu=26.0, Tc=36.0, Delta=4.0, k=0.1, s1=0.5)
    truncate_days_after_bloom = None

# 2. Reload the same training data used for calibration
KA_temp_hourly = pd.read_csv("weather_hourly/klein-altendorf_hourly.csv")
adamedordf = pd.read_csv("phenology/adamedor_sub.csv")
adamedordf = adamedordf.rename(columns={'year': 'Year', 'cultivar': 'cultivar_id',
                                        'species': 'species_id', 'location': 'location_id'})
adamedordf_apple  = adamedordf[(adamedordf['species_id'] == 'Apple') &
                               (adamedordf['location_id'] == 'Klein-Altendorf') &
                               (adamedordf['Year'] > 1958)]
adamedordf_cherry = adamedordf[(adamedordf['species_id'] == 'Sweet Cherry') &
                               (adamedordf['location_id'] == 'Klein-Altendorf') &
                               (adamedordf['Year'] > 1958)]
cka_pheno = pd.concat([adamedordf_apple, adamedordf_cherry], ignore_index=True)

unique_years = cka_pheno['Year'].unique()
seasons      = gen_season_list(KA_temp_hourly, years=unique_years)
location     = 'Klein-Altendorf'
season_dict  = {(location, year): seasons[i] for i, year in enumerate(unique_years)}

# Use truncate=None for prediction so the full season is visible to the model
temps, times, bloom_indices, cultivar_idx, location_idx, cultivar_to_species, \
    cultivar_names, species_names, _ = prepare_seasons_hierarchical(
        season_dict=season_dict, bloom_df=cka_pheno, truncate_days_after_bloom=None)

n_species   = int(np.array(cultivar_to_species).max()) + 1
n_cultivars = int(np.array(cultivar_idx).max()) + 1

# 3. Build posterior_samples dict from loaded ArviZ InferenceData
posterior = loaded_az_mcmc.posterior
posterior_samples = {
    var: jnp.array(posterior[var].values.reshape(-1, *posterior[var].values.shape[2:]))
    for var in posterior.data_vars
}

# 4. Run Predictive with bloom_index=None → triggers prediction branch
conditioned_pred_model = numpyro.handlers.condition(
    phenoflex_numpyro_hierarchical,
    data={k: jnp.array(v, dtype=jnp.float32) for k, v in fixed_params.items()},
)

pred_kwargs = dict(
    temp=temps, times=times, bloom_index=None,
    cultivar_idx=cultivar_idx, cultivar_to_species=cultivar_to_species,
    priors=DEFAULT_PRIORS,
    n_species=n_species, n_cultivars=n_cultivars,
)

predictive  = Predictive(conditioned_pred_model, posterior_samples=posterior_samples)
pred        = predictive(random.PRNGKey(42), **pred_kwargs)
# bloom_hour_pred shape: (n_posterior_samples, n_seasons)

# 5. Convert predicted hour index → DOY
# bloom_indices[i] is the hour index that corresponds to observed DOY cka_pheno['pheno'][i]
# So: predicted_DOY = observed_DOY + (pred_hour - obs_hour) / 24
observed_doy     = cka_pheno['pheno'].values.astype(float)
bloom_hour_pred  = np.array(pred["bloom_hour_pred"])           # (n_post, n_seasons)
bloom_idx_np     = np.array(bloom_indices, dtype=float)          # (n_seasons,)
predicted_doy    = observed_doy[None, :] + (bloom_hour_pred - bloom_idx_np[None, :]) / 24.0

pred_mean = predicted_doy.mean(axis=0)
pred_q05  = np.percentile(predicted_doy, 5,  axis=0)
pred_q95  = np.percentile(predicted_doy, 95, axis=0)

# 6. Plot predicted vs observed — one subplot per species
species_of_season = np.array([
    int(cultivar_to_species[int(c)]) for c in cultivar_idx
])
n_sp = len(species_names)

lims = [min(observed_doy.min(), pred_mean.min()) - 5,
        max(observed_doy.max(), pred_mean.max()) + 5]

fig, axes = plt.subplots(1, n_sp, figsize=(6 * n_sp, 5), sharey=True)
if n_sp == 1:
    axes = [axes]

for sp_i, (sname, ax) in enumerate(zip(species_names, axes)):
    sp_mask = species_of_season == sp_i
    for c_i, cname in enumerate(cultivar_names):
        mask = (np.array(cultivar_idx) == c_i) & sp_mask
        if not mask.any():
            continue
        ax.errorbar(
            observed_doy[mask], pred_mean[mask],
            yerr=[np.maximum(pred_mean[mask] - pred_q05[mask], 0),
                  np.maximum(pred_q95[mask] - pred_mean[mask], 0)],
            fmt='^', alpha=0.6, label=cname,
        )
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed bloom DOY")
    ax.set_title(sname)
    ax.legend(fontsize=7, ncol=2)

axes[0].set_ylabel("Predicted bloom DOY (posterior mean \u00b1 90% CI)")
plt.suptitle("Predicted vs Observed Bloom Dates")
plt.tight_layout()
plt.show()

# 7. Accuracy metrics per cultivar and per species
errors = pred_mean - observed_doy   # signed: positive = predicted too late

rows_cultivar = []
for c_i, cname in enumerate(cultivar_names):
    mask = np.array(cultivar_idx) == c_i
    if not mask.any():
        continue
    e = errors[mask]
    sp_name = species_names[int(cultivar_to_species[c_i])]
    rows_cultivar.append({
        "species":  sp_name,
        "cultivar": cname,
        "n":        int(mask.sum()),
        "ME":       round(float(e.mean()), 2),
        "MAE":      round(float(np.abs(e).mean()), 2),
        "RMSE":     round(float(np.sqrt((e**2).mean())), 2),
    })

metrics_cultivar = pd.DataFrame(rows_cultivar)
print("\n\u2500\u2500 Accuracy metrics per cultivar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
print(metrics_cultivar.to_string(index=False))

rows_species = []
for sp_i, sname in enumerate(species_names):
    mask = species_of_season == sp_i
    if not mask.any():
        continue
    e = errors[mask]
    rows_species.append({
        "species": sname,
        "n":       int(mask.sum()),
        "ME":      round(float(e.mean()), 2),
        "MAE":     round(float(np.abs(e).mean()), 2),
        "RMSE":    round(float(np.sqrt((e**2).mean())), 2),
    })

metrics_species = pd.DataFrame(rows_species)
print("\n\u2500\u2500 Accuracy metrics per species \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
print(metrics_species.to_string(index=False))