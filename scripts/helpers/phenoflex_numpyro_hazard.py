import numpyro
import numpyro.distributions as dist
import jax
from jax.nn import sigmoid
import jax.numpy as jnp  # Ensure jnp is imported
from numpyro.contrib.control_flow import scan
from helpers.phenoflex_numpyro import _p1z_jax, _p2z_jax, _pfcn_jax, convert_intermediate_params


# ── Hazard function ─────────────────────────────────────────────────
# used to compare predicted and observed bloom
# processes accumulated heat and returns a predicted bloom date
# based on event analysis
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
    hazards = sigmoid(k * (z_trace - zc))  # shape (N,)
    # Clamp to avoid log(0) at boundaries
    hazards = jnp.clip(hazards, 1e-6, 1 - 1e-6)

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
    hazards = sigmoid(k * (z_trace - zc))
    survival = jnp.cumprod(1 - hazards)
    pmf = jnp.concatenate([jnp.array([hazards[0]]), survival[:-1] * hazards[1:]])
    pmf = pmf / pmf.sum()  # Ensure PMF sums to 1
    expected_hour = jnp.dot(pmf, hours)
    most_likely_hour = hours[jnp.argmax(pmf)]  # Mode of the PMF
    return expected_hour, pmf, most_likely_hour


# phenofelx model with hazard function to calculate bloom
def phenoflex_numpyro_hazard(
    temp,
    times,
    bloom_index=None,
    Imodel=0,
    deg_celsius=True,
    return_traces=False,  # New flag for conditional return
    use_intermediate_params=False,  # If True, sample theta_star/theta_c/tau/pie_c and convert
):
    """
    Numpyro probabilistic wrapper around PhenoFlex.

    The forward pass accumulates heat (z) over the season. Hazard function calculates
    likelihood of bloom at the day of observation, given the accumulated heat and
    the estimated heat requirement.

    Parameters
    ----------
    temp          : jnp array of hourly temperatures, length N or (num_seasons, N)
    times         : jnp array of hours since season start, length N or (num_seasons, N)
                    e.g. [0, 1, 2, ..., N-1] for consecutive hourly records
    bloom_index : time-step when bloom was recorded.
                    Pass None for prior predictive / generative mode.
    Imodel        : 0 = GDH triangular bell, 1 = Gaussian
    deg_celsius   : True if temperatures arrive in Celsius
    return_traces : If True, x_trace, y_trace, and z_trace will be returned as deterministic outputs.
    use_intermediate_params : If True, sample the intermediate chill-submodel parameters
                    (theta_star, theta_c, tau, pie_c) instead of E0/E1/A0/A1 directly.
                    Conversion to E0/E1/A0/A1 follows Fishman et al. (1987) /
                    Egea et al. (2021).  The converted values are registered as
                    numpyro deterministics so they appear in the posterior.
    """

    # ── Priors ────────────────────────────────────────────────────────────────
    yc = numpyro.sample("yc", dist.Normal(65.0, 10.0))
    zc = numpyro.sample("zc", dist.Normal(220.0, 30.0))
    k = numpyro.sample("k", dist.LogNormal(0.0, 1.0))
    s1 = numpyro.sample("s1", dist.Beta(2.0, 2.0))

    if use_intermediate_params:
        # Sample intermediate parameters — temperatures in Kelvin, constrained positive
        # Using TransformedDistribution via constraints to keep theta > 0
        theta_star = numpyro.sample("theta_star", dist.Uniform(279.0, 281.0))   # K, Egea 2021
        theta_c    = numpyro.sample("theta_c",    dist.Uniform(286.0, 287.0))   # K, Egea 2021
        tau        = numpyro.sample("tau",        dist.Uniform(16.0,  48.0))    # h, Egea 2021
        pie_c      = numpyro.sample("pie_c",      dist.Uniform(24.0,  28.0))    # h, Egea 2021
        # Convert to standard PhenoFlex parameters — pure JAX, works inside traced model
        E0, E1, A0, A1 = convert_intermediate_params(
            theta_star, theta_c, tau, pie_c
        )
        E0 = numpyro.deterministic("E0", E0)
        E1 = numpyro.deterministic("E1", E1)
        A0 = numpyro.deterministic("A0", A0)
        A1 = numpyro.deterministic("A1", A1)
    else:
        E0 = numpyro.sample("E0", dist.Normal(4153.5, 200.0))
        E1 = numpyro.sample("E1", dist.Normal(12888.8, 500.0))
        A0 = numpyro.sample(
            "A0", dist.HalfNormal(139500)
        )  # Changed to HalfNormal as values are positive
        A1 = numpyro.sample("A1", dist.HalfNormal(2.567e18))  # Changed to HalfNormal

    Tf = numpyro.sample("Tf", dist.Normal(4.0, 1.0))
    slope = numpyro.sample("slope", dist.HalfNormal(1.6))
    Tb = numpyro.sample("Tb", dist.Normal(4.0, 2.0))
    Tu = numpyro.sample("Tu", dist.Normal(26.0, 3.0))
    Tc = numpyro.sample("Tc", dist.Normal(36.0, 3.0))
    Delta = numpyro.sample("Delta", dist.HalfNormal(5.0))

    # Ensure temp and times are at least 1D arrays, and if they are single season, make them 2D
    if temp.ndim == 1:
        temp = temp[None, :]
        times = times[None, :]
        if bloom_index is not None:  # only make obs 1D if it's not None
            bloom_index = jnp.atleast_1d(bloom_index)

    num_seasons = temp.shape[0]

    # Convert threshold temperatures to Kelvin
    offset = 273.0 if deg_celsius else 0.0
    _Tf = Tf + offset
    _Tu = Tu + offset
    _Tc = Tc + offset
    _Tb = Tb + offset

    # ── Forward pass via scan ─────────────────────────────────────────────────
    # dt will be (num_seasons, N-1)
    dt = times[:, 1:] - times[:, :-1]

    def transition(carry, inputs):
        x_prev, y_prev, z_prev = carry
        ti_raw, dt_i = inputs  # ti_raw and dt_i are now (num_seasons,)

        ti = ti_raw + offset

        # Ensure A0, A1, E0, E1, etc. are broadcastable if they are scalars from priors
        xs_i = A0 / A1 * jnp.exp(-(E0 - E1) / ti)
        k1 = A1 * jnp.exp(-E1 / ti)

        x_new = xs_i - (xs_i - x_prev) * jnp.exp(-k1 * dt_i)
        y_new = y_prev

        heat_rate = (
            _p1z_jax(ti, _Tu, _Tb, _Tc) if Imodel == 0 else _p2z_jax(ti, _Tu, Delta)
        )
        z_new = z_prev + heat_rate * _pfcn_jax(y_prev, yc, s1) * dt_i

        # Labile -> stable chill conversion
        delta = _pfcn_jax(ti, _Tf, slope) * x_new
        convert = jnp.where(x_new >= 1.0, 1.0, 0.0)
        y_new = y_new + convert * delta
        x_new = x_new - convert * delta

        return (x_new, y_new, z_new), (
            x_new,
            y_new,
            z_new,
        )  # Return all states to be traced

    # Initial state for each season is (num_seasons,)
    init = (jnp.zeros(num_seasons), jnp.zeros(num_seasons), jnp.zeros(num_seasons))
    # Inputs for scan need to be (num_time_steps, num_seasons) if we want to iterate over time
    # So we transpose temp[:, :-1] and dt
    inputs = (temp[:, :-1].T, dt.T)

    (_, _, _), (x_trace, y_trace, z_trace) = scan(transition, init, inputs)
    # x_trace, y_trace, z_trace will be (num_time_steps, num_seasons)
    # We want them back as (num_seasons, num_time_steps)
    x_trace = x_trace.T
    y_trace = y_trace.T
    z_trace = z_trace.T

    if return_traces:
        numpyro.deterministic("x_trace", x_trace)
        numpyro.deterministic("y_trace", y_trace)
        numpyro.deterministic("z_trace", z_trace)

    # ── Likelihood (training) or prediction ───────────────────────────────────
    if bloom_index is not None:
        log_liks = jax.vmap(lambda z, t: discrete_hazard_loglik(z, t, zc, k))(
            z_trace, bloom_index
        )
        numpyro.factor("bloom_obs", jnp.sum(log_liks))
    else:
        # Prediction mode: return expected bloom time per season
        # `expected_bloom_time` now returns a tuple: (expected_hour, pmf, most_likely_hour)
        expected_hours, pmfs, most_likely_hours = jax.vmap(
            lambda z, h: expected_bloom_time(z, h, zc, k)
        )(z_trace, times[:, 1:])

        numpyro.deterministic("bloom_hour_pred", expected_hours)
        numpyro.deterministic(
            "bloom_pmf_pred", pmfs
        )  # This will be (num_seasons, num_timesteps-1) and represents the likelihood distribution
        numpyro.deterministic("most_likely_bloom_hour_pred", most_likely_hours)
