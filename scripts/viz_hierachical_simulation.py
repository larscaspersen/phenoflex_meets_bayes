"""
compare_prior_posterior.py

Compares:
  1. Fixed (true) simulation parameters vs posterior estimates
  2. Prior distributions vs posterior distributions

Reads the most recent (or a specified) MCMC inference data file from
the calibrated_models folder.

Usage:
    python compare_prior_posterior.py
    python compare_prior_posterior.py --timestamp 20240101_120000
    python compare_prior_posterior.py --models_dir /path/to/calibrated_models
"""

import os
import sys
import argparse
import warnings
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Compare priors, posteriors and true parameters.")
parser.add_argument("--timestamp", default=None, help="MCMC run timestamp to load (default: most recent).")
parser.add_argument(
    "--models_dir",
    default=None,
    help="Path to calibrated_models directory (default: ../calibrated_models relative to this script).",
)
parser.add_argument("--output_dir", default=None, help="Where to save figures (default: figures/ in main directory).")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATED_MODELS_DIR = args.models_dir or os.path.join(SCRIPT_DIR, "..", "calibrated_models")
OUTPUT_DIR = args.output_dir or os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Locate the most recent (or requested) MCMC file
# ---------------------------------------------------------------------------

timestamp = args.timestamp
if timestamp is None:
    model_files = [
        f
        for f in os.listdir(CALIBRATED_MODELS_DIR)
        if f.startswith("hierarchical_model_mcmc_") and f.endswith(".nc")
    ]
    if not model_files:
        raise FileNotFoundError("No MCMC inference data files found in calibrated_models folder.")
    prefix = "hierarchical_model_mcmc_inference_data_"
    timestamps = [f[len(prefix) : -len(".nc")] for f in model_files if f.startswith(prefix)]
    timestamp = max(timestamps)
    print(f"Using most recent MCMC inference data: {timestamp}")

idata = az.from_netcdf(
    os.path.join(CALIBRATED_MODELS_DIR, f"hierarchical_model_mcmc_inference_data_{timestamp}.nc")
)
print("Loaded ArviZ InferenceData successfully.")

# ---------------------------------------------------------------------------
# Reconstruct temperature / season input data
# ---------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)
from helpers.helpers_hierachical_model import (
    gen_season_list, prepare_seasons_hierarchical, phenoflex_model_with_custom_priors,
)

_weather_file = os.path.join(SCRIPT_DIR, "..", "weather_hourly", "klein-altendorf_hourly.csv")
_ka_temp      = pd.read_csv(_weather_file)
_years        = np.arange(1999, 2010)
_seasons_list = gen_season_list(_ka_temp, years=_years)
_location     = "Klein-Altendorf"
season_dict   = {(_location, yr): _seasons_list[i] for i, yr in enumerate(_years)}

_pheno_dummy = pd.DataFrame({
    "Year":        np.arange(1999, 2010).repeat(4),
    "cultivar_id": ["Elstar", "Regona", "Symphony", "Sentenniel"] * len(_years),
    "species_id":  ["Apple", "Apple", "Sweet Cherry", "Sweet Cherry"] * len(_years),
    "location_id": [_location] * 4 * len(_years),
    "pheno":       [10] * 4 * len(_years),
})
_pheno_dummy["Year"] = _pheno_dummy["Year"].astype(int)

temps, times, _, cultivar_idx, _, cultivar_to_species, cultivar_names_data, _, _ = \
    prepare_seasons_hierarchical(season_dict, _pheno_dummy)
print("Season data reconstructed.")

# ---------------------------------------------------------------------------
# True (fixed) parameter values used for data generation
# ---------------------------------------------------------------------------

CULTIVAR_NAMES = ["Elstar", "Regona", "Symphony", "Sentenniel"]
SPECIES_NAMES  = ["Apple", "Cherry"]

# Species-level true values
TRUE_SPECIES = {
    "yc_species": np.array([75.0, 55.0]),   # apple, cherry
    "zc_species": np.array([270.0, 200.0]),  # apple, cherry
    "yc_sigma":   10.0,
    "zc_sigma":   20.0,
}

# Cultivar-level offsets (raw offsets, i.e. standardised deviations)
TRUE_CULTIVAR_OFFSETS = {
    "yc_offset": np.array([-1.0,  1.0, -1.0,  1.0]),
    "zc_offset": np.array([ 1.0, -1.0,  1.0, -1.0]),
}

# Derived true cultivar values  (yc_cultivar = yc_species[s] + yc_sigma * yc_offset)
CULTIVAR_TO_SPECIES = [0, 0, 1, 1]  # Elstar & Regona → Apple; Symphony & Sentenniel → Cherry
TRUE_CULTIVAR = {
    "yc_cultivar": np.array([
        TRUE_SPECIES["yc_species"][s] + TRUE_SPECIES["yc_sigma"] * TRUE_CULTIVAR_OFFSETS["yc_offset"][i]
        for i, s in enumerate(CULTIVAR_TO_SPECIES)
    ]),
    "zc_cultivar": np.array([
        TRUE_SPECIES["zc_species"][s] + TRUE_SPECIES["zc_sigma"] * TRUE_CULTIVAR_OFFSETS["zc_offset"][i]
        for i, s in enumerate(CULTIVAR_TO_SPECIES)
    ]),
}

# ---------------------------------------------------------------------------
# Prior samplers
# ---------------------------------------------------------------------------
# Mirrors the custom_priors dict in the original script so we can draw
# prior predictive samples analytically / via MC.

RNG = np.random.default_rng(42)
N_PRIOR = 20_000  # samples drawn to represent each prior

def lognormal_samples(mu, sigma, n=N_PRIOR):
    return RNG.lognormal(mu, sigma, size=n)

def normal_samples(mu, sigma, n=N_PRIOR):
    return RNG.normal(mu, sigma, size=n)

def halfnormal_samples(sigma, n=N_PRIOR):
    return np.abs(RNG.normal(0, sigma, size=n))

def uniform_samples(low, high, n=N_PRIOR):
    return RNG.uniform(low, high, size=n)

def solve_for_log_normal_parameters(mean, variance):
    sigma2 = np.log(1 + variance / mean**2)
    mu     = np.log(mean) - sigma2 / 2
    return mu, sigma2

# yc_species priors
apple_yc_mu,  apple_yc_s2  = solve_for_log_normal_parameters(80, 5**2)
cherry_yc_mu, cherry_yc_s2 = solve_for_log_normal_parameters(50, 5**2)

PRIOR_SAMPLES = {
    "yc_species[Apple]":        lognormal_samples(apple_yc_mu,  np.sqrt(apple_yc_s2)),
    "yc_species[Cherry]":       lognormal_samples(cherry_yc_mu, np.sqrt(cherry_yc_s2)),
    "zc_species[Apple]":        uniform_samples(150, 350),
    "zc_species[Cherry]":       uniform_samples(150, 350),
    "yc_sigma":                 halfnormal_samples(10.0),
    "zc_sigma":                 halfnormal_samples(10.0),
    "yc_offset[Elstar]":        normal_samples(0, 1),
    "yc_offset[Regona]":        normal_samples(0, 1),
    "yc_offset[Symphony]":      normal_samples(0, 1),
    "yc_offset[Sentenniel]":    normal_samples(0, 1),
    "zc_offset[Elstar]":        normal_samples(0, 1),
    "zc_offset[Regona]":        normal_samples(0, 1),
    "zc_offset[Symphony]":      normal_samples(0, 1),
    "zc_offset[Sentenniel]":    normal_samples(0, 1),
}

# ---------------------------------------------------------------------------
# Posterior extractor helpers
# ---------------------------------------------------------------------------

def get_posterior_flat(idata, var_name):
    """Return flattened 1-D posterior samples for a scalar variable."""
    da = idata.posterior[var_name]
    return da.values.flatten()

def get_posterior_flat_idx(idata, var_name, idx):
    """Return flattened samples for a 1-D array variable at position idx."""
    da = idata.posterior[var_name]
    return da.values[:, :, idx].flatten()

# ---------------------------------------------------------------------------
# Build a unified list of panels to plot
# ---------------------------------------------------------------------------
# Each entry: (panel_label, prior_samples, posterior_samples, true_value_or_None)

def build_panels(idata):
    panels = []

    # --- Species-level parameters ---
    for sp_idx, sp_name in enumerate(SPECIES_NAMES):
        # yc_species
        prior = PRIOR_SAMPLES[f"yc_species[{sp_name}]"]
        post  = get_posterior_flat_idx(idata, "yc_species", sp_idx)
        true  = TRUE_SPECIES["yc_species"][sp_idx]
        panels.append((f"yc_species [{sp_name}]", prior, post, true))

        # zc_species
        prior = PRIOR_SAMPLES[f"zc_species[{sp_name}]"]
        post  = get_posterior_flat_idx(idata, "zc_species", sp_idx)
        true  = TRUE_SPECIES["zc_species"][sp_idx]
        panels.append((f"zc_species [{sp_name}]", prior, post, true))

    # --- Variance parameters ---
    for param in ("yc_sigma", "zc_sigma"):
        prior = PRIOR_SAMPLES[param]
        post  = get_posterior_flat(idata, param)
        true  = TRUE_SPECIES[param]
        panels.append((param, prior, post, true))

    # --- Cultivar offsets ---
    for cv_idx, cv_name in enumerate(CULTIVAR_NAMES):
        for prefix in ("yc", "zc"):
            label  = f"{prefix}_offset [{cv_name}]"
            prior  = PRIOR_SAMPLES[f"{prefix}_offset[{cv_name}]"]
            post   = get_posterior_flat_idx(idata, f"{prefix}_offset", cv_idx)
            true   = TRUE_CULTIVAR_OFFSETS[f"{prefix}_offset"][cv_idx]
            panels.append((label, prior, post, true))

    return panels


# ---------------------------------------------------------------------------
# Figure 1: Prior vs Posterior (all panels)
# ---------------------------------------------------------------------------

def plot_prior_posterior(panels, save_path):
    n_panels = len(panels)
    ncols    = 4
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes      = axes.flatten()

    for ax, (label, prior, post, true_val) in zip(axes, panels):
        # Determine shared x range
        combined  = np.concatenate([prior, post])
        lo, hi    = np.percentile(combined, 0.5), np.percentile(combined, 99.5)
        x_grid    = np.linspace(lo, hi, 500)

        # KDE for prior
        kde_prior = stats.gaussian_kde(prior)
        ax.fill_between(x_grid, kde_prior(x_grid), alpha=0.35, color="#4C9ED9", label="Prior")
        ax.plot(x_grid, kde_prior(x_grid), color="#4C9ED9", lw=1.2)

        # KDE for posterior
        kde_post  = stats.gaussian_kde(post)
        ax.fill_between(x_grid, kde_post(x_grid), alpha=0.45, color="#E07B39", label="Posterior")
        ax.plot(x_grid, kde_post(x_grid), color="#E07B39", lw=1.5)

        # True value line
        if true_val is not None:
            ax.axvline(true_val, color="#2ca02c", lw=2, ls="--", label=f"True = {true_val:.2f}")

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlim(lo, hi)

    # Hide unused axes
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle("Prior vs Posterior (dashed = true simulation value)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: True vs Posterior — summary plot (mean ± 94% HDI)
# ---------------------------------------------------------------------------

def plot_true_vs_posterior(panels, save_path):
    labels     = [p[0] for p in panels]
    posteriors = [p[2] for p in panels]
    true_vals  = [p[3] for p in panels]

    post_means = np.array([np.mean(s) for s in posteriors])
    hdis       = np.array([az.hdi(s, prob=0.94) for s in posteriors])
    lo_err     = post_means - hdis[:, 0]
    hi_err     = hdis[:, 1]  - post_means
    true_arr   = np.array([v if v is not None else np.nan for v in true_vals])

    n          = len(labels)
    y_pos      = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45)))
    ax.errorbar(
        post_means, y_pos,
        xerr=[lo_err, hi_err],
        fmt="o", color="#E07B39", ecolor="#E07B39",
        elinewidth=1.5, capsize=3, ms=5, label="Posterior mean ± 94% HDI",
    )
    ax.scatter(true_arr, y_pos, marker="|", color="#2ca02c", s=120, linewidths=2.5, zorder=5, label="True value")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Parameter value", fontsize=10)
    ax.set_title("True vs Posterior Estimates (mean ± 94 % HDI)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Derived cultivar values  (yc_cultivar, zc_cultivar)
# ---------------------------------------------------------------------------


def plot_cultivar_derived(idata, save_path):
    """
    Compute posterior predictive cultivar values from sampled offsets + species means,
    compare against true cultivar values.
    """
    yc_sp   = idata.posterior["yc_species"].values   # (chains, draws, 2)
    zc_sp   = idata.posterior["zc_species"].values
    yc_sig  = idata.posterior["yc_sigma"].values      # (chains, draws)
    zc_sig  = idata.posterior["zc_sigma"].values
    yc_off  = idata.posterior["yc_offset"].values     # (chains, draws, 4)
    zc_off  = idata.posterior["zc_offset"].values

    # Flatten chain/draw dims
    def flat(arr): return arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1)

    yc_sp_f  = flat(yc_sp)   # (S, 2)
    zc_sp_f  = flat(zc_sp)
    yc_sig_f = flat(yc_sig)  # (S,)
    zc_sig_f = flat(zc_sig)
    yc_off_f = flat(yc_off)  # (S, 4)
    zc_off_f = flat(zc_off)

    # Derived cultivar posteriors
    yc_cult_post = np.array([
        yc_sp_f[:, CULTIVAR_TO_SPECIES[i]] + yc_sig_f * yc_off_f[:, i]
        for i in range(4)
    ])  # (4, S)
    zc_cult_post = np.array([
        zc_sp_f[:, CULTIVAR_TO_SPECIES[i]] + zc_sig_f * zc_off_f[:, i]
        for i in range(4)
    ])  # (4, S)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for cv_idx, cv_name in enumerate(CULTIVAR_NAMES):
        for row, (label, post_cv, true_cv) in enumerate([
            ("yc_cultivar", yc_cult_post[cv_idx], TRUE_CULTIVAR["yc_cultivar"][cv_idx]),
            ("zc_cultivar", zc_cult_post[cv_idx], TRUE_CULTIVAR["zc_cultivar"][cv_idx]),
        ]):
            ax   = axes[row, cv_idx]
            x    = np.linspace(post_cv.min(), post_cv.max(), 400)
            kde  = stats.gaussian_kde(post_cv)
            ax.fill_between(x, kde(x), alpha=0.45, color="#E07B39")
            ax.plot(x, kde(x), color="#E07B39", lw=1.5, label="Posterior")
            ax.axvline(true_cv, color="#2ca02c", lw=2, ls="--", label=f"True = {true_cv:.1f}")
            ax.set_title(f"{cv_name}\n{label}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Value", fontsize=7)
            ax.set_ylabel("Density", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)

    fig.suptitle("Derived Cultivar Values: Posterior vs True", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Posterior predictive bloom DOY vs true simulation values
# ---------------------------------------------------------------------------

# Fixed shape parameters that were conditioned during inference
_FIXED_SHAPE_PARAMS = dict(
    E0=4153.5, E1=12888.8, A0=139500.0, A1=2.567e18,
    Tf=4.0, slope=1.6, Tb=4.0, Tu=26.0, Tc=36.0, Delta=4.0,
    k=0.1, s1=0.5,
)

# Fixed true parameters used to generate the simulation data
_FIXED_TRUE_PARAMS = dict(
    **_FIXED_SHAPE_PARAMS,
    yc_species=jnp.array([75.0, 55.0]),
    zc_species=jnp.array([270.0, 200.0]),
    yc_sigma=10.0, zc_sigma=20.0,
    yc_offset=jnp.array([-1.0,  1.0, -1.0,  1.0]),
    zc_offset=jnp.array([ 1.0, -1.0,  1.0, -1.0]),
)

# Dummy priors needed for the phenoflex_model_with_custom_priors signature
# (won't be sampled when all params are conditioned)
_dummy_priors = {
    "yc_species": dist.LogNormal(4.0, 0.3),
    "zc_species": dist.Uniform(150.0, 350.0),
    "yc_sigma":   dist.HalfNormal(10.0),
    "zc_sigma":   dist.HalfNormal(20.0),
    "yc_offset":  dist.Normal(0.0, 1.0),
    "zc_offset":  dist.Normal(0.0, 1.0),
}

_model_kwargs = dict(
    temp=temps, times=times,
    cultivar_idx=cultivar_idx,
    cultivar_to_species=cultivar_to_species,
    bloom_index=None,
    Imodel=0,
    n_species=2, n_cultivars=4,
    custom_priors=_dummy_priors,
)


def _hour_to_jday(hour_idx_float, season_df):
    """Convert a fractional hour index to the corresponding Julian day."""
    idx = min(int(np.round(float(hour_idx_float))), len(season_df) - 1)
    return season_df.iloc[idx]["JDay"]


def compute_true_bloom_jdays():
    """Run the model with true parameters (num_samples=1) to get reference bloom dates."""
    true_model = numpyro.handlers.condition(phenoflex_model_with_custom_priors, data=_FIXED_TRUE_PARAMS)
    preds = Predictive(true_model, num_samples=1)(jax.random.PRNGKey(0), **_model_kwargs)
    # most_likely_bloom_hour_pred: (1, num_seasons)
    hours = np.array(preds["most_likely_bloom_hour_pred"][0])  # (num_seasons,)
    num_seasons = hours.shape[0]
    jdays = []
    for s_idx in range(num_seasons):
        yr    = int(_pheno_dummy.iloc[s_idx]["Year"])
        s_df  = season_dict[(_location, yr)]
        jdays.append(_hour_to_jday(hours[s_idx], s_df))
    return np.array(jdays)  # (num_seasons,)


def run_posterior_predictive(idata, n_samples=300):
    """
    Sample bloom predictions from the posterior.
    Subsample posterior draws to keep memory manageable.
    """
    param_keys = ("yc_species", "zc_species", "yc_sigma", "zc_sigma", "yc_offset", "zc_offset")
    post = {
        k: np.array(idata.posterior[k].values).reshape(
            -1, *np.array(idata.posterior[k].values).shape[2:]
        )
        for k in param_keys
    }
    total = next(iter(post.values())).shape[0]
    idx   = np.random.default_rng(42).choice(total, size=min(n_samples, total), replace=False)
    post_sub = {k: jnp.array(v[idx]) for k, v in post.items()}

    model = numpyro.handlers.condition(phenoflex_model_with_custom_priors, data=_FIXED_SHAPE_PARAMS)
    print(f"Running posterior predictive with {len(idx)} samples …")
    return Predictive(model, posterior_samples=post_sub)(jax.random.PRNGKey(1), **_model_kwargs)


def plot_posterior_bloom_predictions(post_preds, true_jdays, save_path):
    """
    One subplot per cultivar showing:
      - Box plot of posterior predictive bloom DOY per year
      - True simulation bloom DOY marked as a green diamond
    """
    # most_likely_bloom_hour_pred: (n_post_samples, num_seasons)
    most_likely_post = np.array(post_preds["most_likely_bloom_hour_pred"])
    n_post, num_seasons = most_likely_post.shape

    # Build long-form records
    records = []
    for s_idx in range(num_seasons):
        cv_name  = cultivar_names_data[cultivar_idx[s_idx]]
        yr       = int(_pheno_dummy.iloc[s_idx]["Year"])
        s_df     = season_dict[(_location, yr)]
        true_doy = true_jdays[s_idx]
        for samp in range(n_post):
            records.append({
                "Year":    yr,
                "Cultivar": cv_name,
                "Pred_DOY": _hour_to_jday(most_likely_post[samp, s_idx], s_df),
                "True_DOY": true_doy,
            })

    df = pd.DataFrame(records)
    years_sorted = sorted(df["Year"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, cv_name in zip(axes, CULTIVAR_NAMES):
        sub      = df[df["Cultivar"] == cv_name]
        true_doy = sub.drop_duplicates("Year").set_index("Year")["True_DOY"]
        data_per_year = [sub[sub["Year"] == yr]["Pred_DOY"].values for yr in years_sorted]

        ax.boxplot(
            data_per_year,
            positions=years_sorted,
            widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor="#E07B39", alpha=0.6),
            medianprops=dict(color="#c0392b", lw=2),
            whiskerprops=dict(color="#E07B39"),
            capprops=dict(color="#E07B39"),
            flierprops=dict(marker=".", color="#E07B39", alpha=0.3, ms=3),
        )
        ax.scatter(
            years_sorted,
            [true_doy[yr] for yr in years_sorted],
            color="#2ca02c", zorder=5, s=60, marker="D", label="True DOY",
        )
        ax.set_title(cv_name, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Bloom DOY")
        ax.set_xticks(years_sorted)
        ax.set_xticklabels(years_sorted, rotation=45, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Posterior Predictive Bloom DOY vs True Simulation Values\n"
        "(boxes = posterior uncertainty, diamonds = true value)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    panels = build_panels(idata)

    fig1_path = os.path.join(OUTPUT_DIR, f"prior_posterior_comparison_{timestamp}.png")
    fig2_path = os.path.join(OUTPUT_DIR, f"true_vs_posterior_summary_{timestamp}.png")
    fig3_path = os.path.join(OUTPUT_DIR, f"derived_cultivar_comparison_{timestamp}.png")
    fig4_path = os.path.join(OUTPUT_DIR, f"posterior_bloom_predictions_{timestamp}.png")

    plot_prior_posterior(panels, fig1_path)
    plot_true_vs_posterior(panels, fig2_path)
    plot_cultivar_derived(idata, fig3_path)

    true_jdays  = compute_true_bloom_jdays()
    post_preds  = run_posterior_predictive(idata, n_samples=300)
    plot_posterior_bloom_predictions(post_preds, true_jdays, fig4_path)

    print("\nAll figures saved. Done.")