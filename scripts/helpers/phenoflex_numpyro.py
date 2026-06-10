import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
import numpy as np


def _p1z_jax(T, Tu, Tb, Tc):
    in_lower = (T >= Tb) & (T <= Tu)
    in_upper = (T > Tu) & (T <= Tc)
    val_lower = 0.5 * (1 + jnp.cos(jnp.pi + jnp.pi * (T - Tb) / (Tu - Tb)))
    val_upper = 1 + jnp.cos(jnp.pi / 2 + jnp.pi / 2 * (T - Tu) / (Tc - Tu))
    return jnp.where(in_lower, val_lower, jnp.where(in_upper, val_upper, 0.0))


def _p2z_jax(T, Tu, Delta):
    return jnp.exp(-(((T - Tu) / (2 * Delta)) ** 2))


def _pfcn_jax(T, Tf, slope):
    # Guard against T=0 (occurs at season start when y_prev=0)
    T_safe = jnp.where(T == 0.0, 1.0, T)  # safe value for division; masked out below
    x = slope * Tf * (T_safe - Tf) / T_safe
    sr = jnp.exp(jnp.clip(x, -20, 17))
    result = jnp.where(x >= 17, 1.0, jnp.where(x <= -20, 0.0, sr / (1 + sr)))
    return jnp.where(T == 0.0, 0.0, result)  # when y=0, no forcing

    # x = slope * Tf * (T - Tf) / T
    # sr = jnp.exp(jnp.clip(x, -20, 17))
    # return jnp.where(x >= 17, 1.0, jnp.where(x <= -20, 0.0, sr / (1 + sr)))


# ── Intermediate-parameter conversion ───────────────────────────────────────
def convert_intermediate_params(theta_star, theta_c, tau, pie_c):
    """
    Convert intermediate chill-submodel parameters to standard PhenoFlex
    parameters (E0, E1, A0, A1).

    Fully JAX-native: uses Newton's method via jax.lax.fori_loop so it works
    inside a traced/JIT-compiled numpyro model.

    Follows Fishman et al. (1987) and Egea et al. (2021), equations 5-8.

    Parameters
    ----------
    theta_star : JAX scalar — reference temperature 1 (K), typically ~279-281
    theta_c    : JAX scalar — reference temperature 2 (K), typically ~286-287
    tau        : JAX scalar — time constant at theta_star (h), typically 16-48
    pie_c      : JAX scalar — equilibrium chill at theta_c, typically 24-28

    Returns
    -------
    E0, E1, A0, A1 : JAX scalars
    """
    def nle_and_jac(E):
        E0, E1 = E[0], E[1]
        # Clip to avoid overflow in exp
        e0s = jnp.exp(jnp.clip(E0 / theta_star, -500.0, 500.0))
        e1s = jnp.exp(jnp.clip(E1 / theta_star, -500.0, 500.0))
        e0c = jnp.exp(jnp.clip(E0 / theta_c,    -500.0, 500.0))
        e1c = jnp.exp(jnp.clip(E1 / theta_c,    -500.0, 500.0))
        # NLE system (chillR / Egea 2021)
        f = jnp.array([
            2.0 * e0s - pie_c * (e1s + 1.0),
            2.0 * e0c - pie_c * (e1c + 1.0),
        ])
        # Analytical Jacobian
        J = jnp.array([
            [ 2.0 / theta_star * e0s,  -pie_c / theta_star * e1s],
            [ 2.0 / theta_c   * e0c,  -pie_c / theta_c   * e1c],
        ])
        return f, J

    def newton_step(i, E):
        f, J = nle_and_jac(E)
        delta = jnp.linalg.solve(J, f)
        return E - delta

    E_init = jnp.array([500.0, 15000.0])
    E_sol  = jax.lax.fori_loop(0, 50, newton_step, E_init)

    E0, E1 = E_sol[0], E_sol[1]

    # Analytical A1 and A0 (Egea 2021 Eq. 36-37)
    q  = 1.0 / theta_star - 1.0 / theta_c
    A1 = -jnp.exp(E1 / theta_star) / tau * jnp.log(1.0 - jnp.exp((E0 - E1) * q))
    A0 = A1 * jnp.exp((E0 - E1) / theta_c)

    return E0, E1, A0, A1


# ── Soft bloom-date estimator ─────────────────────────────────────────────────


def soft_bloom_hour(z_trace, hours, zc, sharpness=1.0):
    """
    Differentiable approximation of the hour at which z first crosses zc.

    A hard argmax is not differentiable, so we use a softmax over (z - zc).
    As sharpness -> inf this converges to the true crossing point.
    Values of 0.5–2.0 work well in practice.

    Parameters
    ----------
    z_trace   : jnp array (N-1,) — accumulated heat at each time step
    hours     : jnp array (N-1,) — hours[i] is the time at which z_trace[i] was computed
    zc        : heat requirement threshold (scalar, can be a sampled parameter)
    sharpness : controls how peaked the soft-argmax is

    Returns
    -------
    Scalar: differentiable estimate of the bloom hour
    """
    # logits  = sharpness * (z_trace - zc)
    # this favpours to wait forever, until the difference is maxed.
    # this is ALWAYS at the end of the time series

    # MODIFIED: Use negative squared difference for logits
    logits = -sharpness * (z_trace - zc) ** 2

    # new approach: negative bi
    # Numerically stable softmax
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(logits)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.sum(
        weights * hours, axis=-1
    )  # Use sum instead of dot for batched operation


# ── Numpyro model ─────────────────────────────────────────────────────────────


def phenoflex_numpyro(
    temp,
    times,
    bloom_doy_obs=None,
    start_doy=0.0,
    Imodel=0,
    deg_celsius=True,
    sharpness=1.0,
    leap_years=None,
    return_traces=False,  # New flag for conditional return
):
    """
    Numpyro probabilistic wrapper around PhenoFlex.

    The forward pass accumulates heat (z) over the season. The predicted bloom
    date is the hour at which z crosses zc, converted to day-of-year via
    start_doy. Because argmax is not differentiable we use soft_bloom_hour()
    as a smooth approximation.

    Parameters
    ----------
    temp          : jnp array of hourly temperatures, length N or (num_seasons, N)
    times         : jnp array of hours since season start, length N or (num_seasons, N)
                    e.g. [0, 1, 2, ..., N-1] for consecutive hourly records
    bloom_doy_obs : observed bloom date as day-of-year (scalar or 1-D array
                    if multiple years are handled outside this function).
                    Pass None for prior predictive / generative mode.
    start_doy     : day-of-year corresponding to times[0].
                    E.g. if the season starts on Nov 1 = DOY 305, pass 305.
    Imodel        : 0 = GDH triangular bell, 1 = Gaussian
    deg_celsius   : True if temperatures arrive in Celsius
    sharpness     : steepness of the soft bloom-date estimator.
                    Increase if the posterior is diffuse; decrease if gradients vanish.
    leap_years    : either none or vector of boolean, indicating if it is a leap year
    return_traces : If True, x_trace, y_trace, and z_trace will be returned as deterministic outputs.
    """

    # ── Priors ────────────────────────────────────────────────────────────────
    yc = numpyro.sample("yc", dist.Normal(65.0, 10.0))
    zc = numpyro.sample("zc", dist.Normal(220.0, 30.0))
    s1 = numpyro.sample("s1", dist.Beta(2.0, 2.0))
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
    sigma = numpyro.sample(
        "sigma", dist.HalfNormal(3.0)
    )  # days (shared observation noise)

    # Ensure temp and times are at least 1D arrays, and if they are single season, make them 2D
    if temp.ndim == 1:
        temp = temp[None, :]
        times = times[None, :]
        if bloom_doy_obs is not None:  # only make obs 1D if it's not None
            bloom_doy_obs = bloom_doy_obs[None]
        start_doy = start_doy[None]
        # If leap_years is a single boolean, convert it to a 1-element array for consistency
        if leap_years is not None and jnp.ndim(jnp.asarray(leap_years)) == 0:
            leap_years = jnp.asarray([leap_years])  # Convert to 1-element array

    num_seasons = temp.shape[0]

    # knowing leap years etc is only important to return prediction before Dec-31 (are coded negative numbers)
    # Determine days_in_year_per_season (vector of 365.0 or 366.0 for each season)
    if leap_years is None:
        days_in_year_per_season = jnp.full(num_seasons, 365.0)
    else:
        # Ensure leap_years is a JAX array for jnp.ndim to work correctly
        _leap_years_arr = jnp.asarray(leap_years)
        if jnp.ndim(_leap_years_arr) == 0:  # Single boolean value
            days_in_year_scalar = jnp.where(_leap_years_arr, 366.0, 365.0)
            days_in_year_per_season = jnp.full(num_seasons, days_in_year_scalar)
        else:  # Array of boolean values
            if _leap_years_arr.shape[0] != num_seasons:
                raise ValueError(
                    "Length of 'leap_years' must match 'num_seasons' or be a single boolean."
                )
            days_in_year_per_season = jnp.where(_leap_years_arr, 366.0, 365.0)

    # Convert threshold temperatures to Kelvin
    offset = 273.0 if deg_celsius else 0.0
    _Tf = Tf + offset
    _Tu = Tu + offset
    _Tc = Tc + offset
    _Tb = Tb + offset

    # The plate is not needed here as scan already handles the batching over seasons.
    # The 'plate' will be implicitly handled by NumPyro's vmap when `bloom_doy_obs` is provided as a batch.
    # with numpyro.plate("seasons", num_seasons):
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

    # ── Predicted bloom date ──────────────────────────────────────────────────
    # times[:, 1:] are the hours at which each z value was recorded, (num_seasons, N-1)
    # z_trace is (num_seasons, N-1)
    # zc is scalar, but will broadcast correctly
    bloom_hour = soft_bloom_hour(z_trace, times[:, 1:], zc, sharpness=sharpness)

    # Redefine bloom_doy_pred to match user's request:
    # Dec 31 of (season start year) = 0
    # Jan 1 of (season start year + 1) = 1
    # raw_bloom_doy_pred is the Julian day count since Jan 1st of the season start year, as a float.
    # Subtracting days_in_year_per_season shifts the reference point so Dec 31 of the previous year is 0.

    # -0.5 so that julian day is centered at midday and not midnight = 1.0 = Jan-01 midday
    # -days in the year per season so that there is no break between Dec-31 and Jan-01
    # (otherwise optimizer can get stuck early on in the season)

    raw_bloom_doy_pred = start_doy - 0.5 - days_in_year_per_season + (bloom_hour / 24.0)

    bloom_doy_pred = numpyro.deterministic("bloom_doy_pred", raw_bloom_doy_pred)

    # ── Observation noise ─────────────────────────────────────────────────────
    # Bloom phenology observations carry real uncertainty from observer error,
    # the definition of first-flower, spatial variability, etc.
    # A few days of SD is typical for field phenology data.
    # sigma is defined outside the plate, so it's shared.

    # ── Likelihood ────────────────────────────────────────────────────────────
    if bloom_doy_obs is not None:
        # Inference mode: condition on observed bloom date.
        # bloom_doy_obs (from KA_bloom) are standard Julian DOY (Jan 1=1).
        # On the new scale, Jan 1 is also 1, so no transformation needed for obs.
        numpyro.sample(
            "bloom_doy", dist.Normal(bloom_doy_pred, sigma), obs=bloom_doy_obs
        )
