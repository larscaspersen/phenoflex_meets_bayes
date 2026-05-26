import os
import time
import numpy as np
from numpy import log
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import lax
import arviz as az
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
# Building on numpyro AR2 example: https://num.pyro.ai/en/latest/examples/ar2.html


def _p1z_jax(T, Tu, Tb, Tc):
    in_lower = (T >= Tb) & (T <= Tu)
    in_upper = (T > Tu) & (T <= Tc)
    val_lower = 0.5 * (1 + jnp.cos(jnp.pi + jnp.pi * (T - Tb) / (Tu - Tb)))
    val_upper = 1 + jnp.cos(jnp.pi / 2 + jnp.pi / 2 * (T - Tu) / (Tc - Tu))
    return jnp.where(in_lower, val_lower, jnp.where(in_upper, val_upper, 0.0))

def _p2z_jax(T, Tu, Delta):
    return jnp.exp(-((T - Tu) / (2 * Delta)) ** 2)

def _pfcn_jax(T, Tf, slope):
   # Guard against T=0 (occurs at season start when y_prev=0)
    T_safe = jnp.where(T == 0.0, 1.0, T)  # safe value for division; masked out below
    x = slope * Tf * (T_safe - Tf) / T_safe
    sr = jnp.exp(jnp.clip(x, -20, 17))
    result = jnp.where(x >= 17, 1.0, jnp.where(x <= -20, 0.0, sr / (1 + sr)))
    return jnp.where(T == 0.0, 0.0, result)  # when y=0, no forcing

    #x = slope * Tf * (T - Tf) / T
    #sr = jnp.exp(jnp.clip(x, -20, 17))
    #return jnp.where(x >= 17, 1.0, jnp.where(x <= -20, 0.0, sr / (1 + sr)))

# ── Hazard function ─────────────────────────────────────────────────
#used to compare predicted and observed bloom
#processes accumulated heat and returns a predicted bloom date
#based on event analysis
def discrete_hazard_loglik(z_trace, bloom_step, zc, k):
    """
    Log-likelihood of observing bloom at bloom_step under a
    discrete-time survival model.

    Parameters
    ----------
    z_trace   : jnp array (N,) — cumulative heat units at each time step
    bloom_step: int — observed index at which bloom occurred
    zc        : scalar — heat requirement threshold (sampled parameter)
    k         : scalar — logistic sharpness (sampled parameter)

    Returns
    -------
    Scalar log-likelihood
    """
    # Hazard at each step: probability of blooming NOW given survival
    hazards = sigmoid(k * (z_trace - zc))          # shape (N,)

    # Clip hazards for numerical stability when taking logarithms
    epsilon = 1e-7
    hazards = jnp.clip(hazards, epsilon, 1.0 - epsilon)

    # Calculate log_survival using boolean masking
    # Create a mask for indices up to (but not including) bloom_step
    # jnp.arange(hazards.shape[0]) creates [0, 1, ..., N-1]
    # (jnp.arange(hazards.shape[0]) < bloom_step) creates a boolean mask [True, True, ..., False, False]
    mask = jnp.arange(hazards.shape[0]) < bloom_step

    # Apply the mask: only sum jnp.log1p(-hazards) where mask is True
    # For elements where mask is False, jnp.where returns 0.0, which doesn't affect the sum
    log_survival = jnp.sum(jnp.where(mask, jnp.log1p(-hazards), 0.0))

    # Log hazard at the bloom step
    log_hazard_at_bloom = jnp.log(hazards[bloom_step])

    return log_survival + log_hazard_at_bloom

# make forward prediction (when there is no observation)
def expected_bloom_time(z_trace, hours, zc, k):
    hazards  = sigmoid(k * (z_trace - zc))
    survival = jnp.cumprod(1 - hazards)
    pmf      = jnp.concatenate([
        jnp.array([hazards[0]]),
        survival[:-1] * hazards[1:]
    ])
    pmf = pmf / pmf.sum() # Ensure PMF sums to 1
    expected_hour = jnp.dot(pmf, hours)
    most_likely_hour = hours[jnp.argmax(pmf)] # Mode of the PMF
    return expected_hour, pmf, most_likely_hour

def phenoflex_numpyro_hierarchical(
    temp,
    times,
    cultivar_idx,        # int array (num_seasons,) — which cultivar each season belongs to
    cultivar_to_species, # int array (num_cultivars,) — maps cultivar → species
    bloom_index=None,
    Imodel=0,
    deg_celsius=True,
    return_traces=False,
    priors=None,
    n_species=None,
    n_cultivars=None,
):
    # ── Default priors ─────────────────────────────────────────────────────────
    # Override any of these by passing priors={"yc_species": dist.Normal(...), ...}
    default_priors = {
        "E0":         dist.Normal(4153.5,  200.0),
        "E1":         dist.Normal(12888.8, 500.0),
        "A0":         dist.HalfNormal(139500),
        "A1":         dist.HalfNormal(2.567e18),
        "Tf":         dist.Normal(4.0,  1.0),
        "slope":      dist.HalfNormal(1.6),
        "Tb":         dist.Normal(4.0,  2.0),
        "Tu":         dist.Normal(26.0, 3.0),
        "Tc":         dist.Normal(36.0, 3.0),
        "Delta":      dist.HalfNormal(5.0),
        "k":          dist.LogNormal(0.0, 1.0),
        "s1":         dist.Beta(2.0, 2.0),
        "yc_species": dist.LogNormal(jnp.log(65.0), 0.3),
        "zc_species": dist.Normal(220.0, 30.0),
        "yc_sigma":   dist.HalfNormal(10.0),
        "zc_sigma":   dist.HalfNormal(20.0),
        "yc_offset":  dist.Normal(0.0, 1.0),
        "zc_offset":  dist.Normal(0.0, 1.0),
    }

    # User-supplied priors overwrite defaults — anything not supplied keeps its default
    p = {**default_priors, **(priors or {})}

    # ── Sampling ───────────────────────────────────────────────────────────────
    if n_species is None:
        n_species   = int(np.array(cultivar_to_species).max()) + 1
    if n_cultivars is None:
        n_cultivars = int(np.array(cultivar_idx).max()) + 1

    E0    = numpyro.sample("E0",    p["E0"])
    E1    = numpyro.sample("E1",    p["E1"])
    A0    = numpyro.sample("A0",    p["A0"])
    A1    = numpyro.sample("A1",    p["A1"])
    Tf    = numpyro.sample("Tf",    p["Tf"])
    slope = numpyro.sample("slope", p["slope"])
    Tb    = numpyro.sample("Tb",    p["Tb"])
    Tu    = numpyro.sample("Tu",    p["Tu"])
    Tc    = numpyro.sample("Tc",    p["Tc"])
    Delta = numpyro.sample("Delta", p["Delta"])
    k     = numpyro.sample("k",     p["k"])
    s1    = numpyro.sample("s1",    p["s1"])

    with numpyro.plate("species", n_species):
        yc_species = numpyro.sample("yc_species", p["yc_species"])
        zc_species = numpyro.sample("zc_species", p["zc_species"])

    yc_sigma = numpyro.sample("yc_sigma", p["yc_sigma"])
    zc_sigma = numpyro.sample("zc_sigma", p["zc_sigma"])

    with numpyro.plate("cultivars", n_cultivars):
        yc_offset = numpyro.sample("yc_offset", p["yc_offset"])
        zc_offset = numpyro.sample("zc_offset", p["zc_offset"])

    # Actual cultivar values: species mean + scaled offset
    yc_cultivar = yc_species[cultivar_to_species] + yc_sigma * yc_offset  # (n_cultivars,)
    zc_cultivar = zc_species[cultivar_to_species] + zc_sigma * zc_offset  # (n_cultivars,)

    # Register for posterior inspection
    numpyro.deterministic("yc_cultivar", yc_cultivar)
    numpyro.deterministic("zc_cultivar", zc_cultivar)

    # ── Index into per-season cultivar params ─────────────────────────────────
    # Each season gets its cultivar's yc and zc
    yc = yc_cultivar[cultivar_idx]   # (num_seasons,)
    zc = zc_cultivar[cultivar_idx]   # (num_seasons,)

    # Register per-season yc and zc for posterior inspection
    numpyro.deterministic("yc", yc)
    numpyro.deterministic("zc", zc)

    # ── Everything below is unchanged ─────────────────────────────────────────
    if temp.ndim == 1:
        temp  = temp[None, :]
        times = times[None, :]
        if bloom_index is not None:
            bloom_index = jnp.atleast_1d(bloom_index)

    num_seasons = temp.shape[0]
    offset = 273.0 if deg_celsius else 0.0
    _Tf = Tf + offset
    _Tu = Tu + offset
    _Tc = Tc + offset
    _Tb = Tb + offset

    dt = times[:, 1:] - times[:, :-1]

    def transition(carry, inputs):
        x_prev, y_prev, z_prev = carry
        ti_raw, dt_i = inputs
        ti   = ti_raw + offset
        xs_i = A0 / A1 * jnp.exp(-(E0 - E1) / ti)
        k1   = A1 * jnp.exp(-E1 / ti)
        x_new = xs_i - (xs_i - x_prev) * jnp.exp(-k1 * dt_i)
        y_new = y_prev
        heat_rate = _p1z_jax(ti, _Tu, _Tb, _Tc) if Imodel == 0 else _p2z_jax(ti, _Tu, Delta)

        # yc and zc are now (num_seasons,) — broadcast correctly via y_prev
        z_new = z_prev + heat_rate * _pfcn_jax(y_prev, yc, s1) * dt_i

        delta   = _pfcn_jax(ti, _Tf, slope) * x_new
        convert = jnp.where(x_new >= 1.0, 1.0, 0.0)
        y_new   = y_new + convert * delta
        x_new   = x_new - convert * delta
        return (x_new, y_new, z_new), (x_new, y_new, z_new)

    init   = (jnp.zeros(num_seasons), jnp.zeros(num_seasons), jnp.zeros(num_seasons))
    inputs = (temp[:, :-1].T, dt.T)
    (_, _, _), (x_trace, y_trace, z_trace) = lax.scan(transition, init, inputs)
    x_trace = x_trace.T
    y_trace = y_trace.T
    z_trace = z_trace.T

    if return_traces:
        numpyro.deterministic("x_trace", x_trace)
        numpyro.deterministic("y_trace", y_trace)
        numpyro.deterministic("z_trace", z_trace)

    if bloom_index is not None:
        log_liks = jax.vmap(
            lambda z, t, threshold, sharpness: discrete_hazard_loglik(z, t, threshold, sharpness),
            in_axes=(0, 0, 0, None) # z_trace, bloom_index, zc are batched, k is constant
        )(z_trace, bloom_index, zc, k)
        numpyro.factor("bloom_obs", jnp.sum(log_liks))
    else:
        expected_hours, pmfs, most_likely_hours = jax.vmap(
            lambda z, h, threshold, sharpness: expected_bloom_time(z, h, threshold, sharpness),
            in_axes=(0, 0, 0, None) # z_trace, times[:, 1:], zc are batched, k is constant
        )(z_trace, times[:, 1:], zc, k)
        numpyro.deterministic("bloom_hour_pred",           expected_hours)
        numpyro.deterministic("bloom_pmf_pred",            pmfs)
        numpyro.deterministic("most_likely_bloom_hour_pred", most_likely_hours)

def prepare_seasons_hierarchical(season_dict, bloom_df):
    """
    Prepare hierarchical model inputs from multi-cultivar, multi-location bloom data.

    Parameters
    ----------
    season_dict : dict mapping (location_id, year) → season DataFrame
                  e.g. {("Karlsruhe", 1999): df, ("Münster", 1999): df, ...}
                  Each DataFrame has columns: Temp, JDay, Year
    bloom_df    : DataFrame with columns:
                    Year        — calendar year of bloom
                    pheno       — observed bloom DOY
                    cultivar_id — cultivar name  (e.g. "Gala")
                    species_id  — species name   (e.g. "apple")
                    location_id — location name  (e.g. "Karlsruhe")

    Returns
    -------
    temps               : (S, T)  float32 — padded hourly temperatures
    times               : (S, T)  float32 — hourly time index
    bloom_indices       : (S,)    int32   — hourly index of observed bloom
    cultivar_idx        : (S,)    int32   — cultivar index per season
    location_idx        : (S,)    int32   — location index per season
    cultivar_to_species : (C,)    int32   — maps cultivar index → species index
    cultivar_names      : list[str]
    species_names       : list[str]
    location_names      : list[str]
    """

    # ── Validate required columns ─────────────────────────────────────────────
    required_cols = {"Year", "pheno", "cultivar_id", "species_id", "location_id"}
    missing_cols  = required_cols - set(bloom_df.columns)
    if missing_cols:
        raise ValueError(f"bloom_df is missing columns: {missing_cols}")

    if bloom_df["pheno"].isna().any():
        bad = bloom_df[bloom_df["pheno"].isna()][["Year", "cultivar_id", "location_id"]]
        raise ValueError(f"Missing pheno values:\n{bad}")

    # ── Build integer encodings ───────────────────────────────────────────────
    cultivar_names = sorted(bloom_df["cultivar_id"].unique().tolist())
    species_names  = sorted(bloom_df["species_id"].unique().tolist())
    location_names = sorted(bloom_df["location_id"].unique().tolist())

    cultivar_to_int  = {n: i for i, n in enumerate(cultivar_names)}
    species_to_int   = {n: i for i, n in enumerate(species_names)}
    location_to_int  = {n: i for i, n in enumerate(location_names)}

    # Derive cultivar → species mapping
    cultivar_species_lookup = (
        bloom_df[["cultivar_id", "species_id"]]
        .drop_duplicates()
        .set_index("cultivar_id")["species_id"]
    )
    if cultivar_species_lookup.index.duplicated().any():
        raise ValueError(
            "Some cultivars map to more than one species. "
            "Each cultivar_id must belong to exactly one species_id."
        )

    cultivar_to_species = jnp.array(
        [species_to_int[cultivar_species_lookup[c]] for c in cultivar_names],
        dtype=jnp.int32,
    )

    # ── Validate that all bloom_df rows have a matching season in season_dict ─
    missing_seasons = [
        (row.location_id, row.Year)
        for row in bloom_df.itertuples()
        if (row.location_id, row.Year) not in season_dict
    ]
    if missing_seasons:
        raise ValueError(
            f"The following (location, year) combinations are in bloom_df "
            f"but missing from season_dict:\n{missing_seasons}"
        )

    # ── Build one entry per row in bloom_df ───────────────────────────────────
    max_len = max(len(df) for df in season_dict.values())

    padded_temps  = []
    padded_times  = []
    bloom_indices = []
    cultivar_idx  = []
    location_idx  = []

    for row in bloom_df.itertuples():
        season_df = season_dict[(row.location_id, row.Year)]

        # Pad temperature and time
        temp  = jnp.asarray(season_df.Temp.values, dtype=jnp.float32)
        t_idx = jnp.arange(len(temp), dtype=jnp.float32)
        pad   = max_len - len(temp)

        padded_temps.append(
            jnp.pad(temp, (0, pad), mode="edge") if pad else temp
        )
        padded_times.append(
            jnp.concatenate([t_idx, t_idx[-1] + jnp.arange(1, pad + 1, dtype=jnp.float32)])
            if pad else t_idx
        )

        # Locate bloom hour
        match_mask = (season_df["JDay"] == row.pheno) & (season_df["Year"] == row.Year)
        matching   = jnp.where(match_mask.values)[0]

        if len(matching) == 0:
            raise ValueError(
                f"No hourly record found for bloom DOY {row.pheno} "
                f"in season {row.Year} at {row.location_id}."
            )
        if len(matching) < 12:
            raise ValueError(
                f"Fewer than 12 hourly records on bloom day {row.pheno} "
                f"in season {row.Year} at {row.location_id}. Found {len(matching)}."
            )

        idx = int(jnp.clip(matching[11], 0, len(season_df) - 1))
        bloom_indices.append(idx)

        cultivar_idx.append(cultivar_to_int[row.cultivar_id])
        location_idx.append(location_to_int[row.location_id])

    return (
        jnp.stack(padded_temps),                        # (S, T)
        jnp.stack(padded_times),                        # (S, T)
        jnp.asarray(bloom_indices, dtype=jnp.int32),    # (S,)
        jnp.asarray(cultivar_idx,  dtype=jnp.int32),    # (S,)
        jnp.asarray(location_idx,  dtype=jnp.int32),    # (S,)
        cultivar_to_species,                            # (C,)
        cultivar_names,
        species_names,
        location_names,
    )

def gen_season_list(temps, mrange=(8, 6), years=None):
    """Python equivalent of chillR::genSeasonList."""
    assert len(mrange) == 2 and mrange[0] > mrange[1]
    assert years is not None and temps is not None
    start_month, end_month = mrange

    return [
        temps.loc[
            ((temps["Month"].between(start_month, 12)) & (temps["Year"] == y - 1)) |
            ((temps["Month"].between(1, end_month))    & (temps["Year"] == y)),
            ["Temp", "JDay", "Year"]
        ].copy()
        for y in years
    ]

#get jday corresponding to the hour
def get_jday_from_hour(row,season_dict_for_hierarchical):
    year = int(row['Year'])
    location = row['location_id']
    most_likely_hour = int(np.round(row['most_likely_hour']))

    # Get the corresponding season DataFrame
    current_season_df = season_dict_for_hierarchical[(location, year)]

    # Ensure hour index is within bounds
    max_actual_idx = len(current_season_df) - 1
    hour_idx_clipped = min(most_likely_hour, max_actual_idx)

    return current_season_df.iloc[hour_idx_clipped]['JDay']

def solve_for_log_normal_parameters(mean, variance):
    sigma2 = log(variance/mean**2 + 1)
    mu = log(mean) - sigma2/2
    return (mu, sigma2)

# Define a wrapper function that passes custom_priors to the model
def phenoflex_model_with_custom_priors(temp, times, cultivar_idx, cultivar_to_species,
                                       custom_priors, # ← new argument for custom priors
                                     bloom_index=None, Imodel=0, deg_celsius=True, return_traces=False,
                                       n_species=None, n_cultivars=None):
    return phenoflex_numpyro_hierarchical(
        temp=temp,
        times=times,
        cultivar_idx=cultivar_idx,
        cultivar_to_species=cultivar_to_species,
        bloom_index=bloom_index,
        Imodel=Imodel,
        deg_celsius=deg_celsius,
        return_traces=return_traces,
        priors=custom_priors, # Pass custom_priors here
        n_species=n_species,       # ← pass through
        n_cultivars=n_cultivars,   # ← pass through
    )

def run_inference(model, args, rng_key, dat):
    start = time.time()
    sampler = numpyro.infer.NUTS(model,
                                  init_strategy=numpyro.infer.init_to_median(num_samples=500))
    mcmc = numpyro.infer.MCMC(
        sampler,
        num_warmup=args['num_warmup'],
        num_samples=args['num_samples'],
        num_chains=args['num_chains'],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, **dat)
    mcmc.print_summary()
    az_mcmc = az.from_numpyro(mcmc)
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc, mcmc.get_samples(), az_mcmc