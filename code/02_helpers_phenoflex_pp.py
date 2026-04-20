# ── Helper functions

#gdh model
def P1z(T: float, Tu: float, Tb: float, Tc: float) -> float:
    """GDH heat-accumulation rate (triangular bell, model 0)."""
    if Tb <= T <= Tu:
        return 0.5 * (1 + np.cos(np.pi + np.pi * (T - Tb) / (Tu - Tb)))
    elif Tu < T <= Tc:
        return 1 + np.cos(np.pi / 2 + np.pi / 2 * (T - Tu) / (Tc - Tu))
    return 0.0

 #alternative heat accumulation model
def P2z(T: float, Tu: float, Delta: float) -> float:
    """Gaussian heat-accumulation rate (model 1)."""
    return np.exp(-((T - Tu) / (2 * Delta)) ** 2)

#transition function (either PDBF to DBF or Py for effective heat accumulation)
def PFcn(T: float, Tf: float, slope: float) -> float:
    """Sigmoid that controls labile→stable chill conversion and heat sensitivity."""
    x = slope * Tf * (T - Tf) / T
    if x >= 17:
        return 1.0
    if x <= -20:
        return 0.0
    sr = np.exp(x)
    return sr / (1 + sr)


def phenoflex(
    temp,
    times,
    yc: float = 40.0,
    zc: float = 190.0,
    s1: float = 0.5,
    E0: float = 4153.5,
    E1: float = 12888.8,
    A0: float = 139500,
    A1: float = 2567000000000000000,
    Tf: float = 4.0,
    slope: float = 1.6,
    Tb: float = 4.0,
    Tu: float = 26.0,
    Tc: float = 36.0,
    Delta: float = 4.0,
    Imodel: int = 0,
    stopatzc: bool = True,
    deg_celsius: bool = True,
    basic_output: bool = True,
) -> dict:
    """
    Python translation of the PhenoFlex C++ / Rcpp model.

    Parameters
    ----------
    temp       : array-like of hourly temperatures
    times      : array-like of corresponding time stamps (hours)
    yc         : chill requirement (stable chill units)
    zc         : heat requirement (GDH or GDD units)
    s1         : slope of PFcn sigmoid for heat sensitivity
    E0, E1     : activation energies for chill pool dynamics
    A0, A1     : pre-exponential factors for chill pool dynamics
    Tf         : base temperature for labile-to-stable chill conversion (°C)
    slope      : steepness of the labile-to-stable sigmoid
    Tb, Tu, Tc : base, optimum, ceiling temperatures for heat model 0 (°C)
    Delta      : half-width for Gaussian heat model 1 (°C)
    Imodel     : 0 = GDH triangular bell, 1 = Gaussian
    stopatzc   : stop simulation once zc is reached
    deg_celsius: True if temperatures are in °C (will be converted to K internally)
    basic_output: True → return only bloomindex; False → return full state arrays

    Returns
    -------
    dict with 'bloomindex' (and optionally 'x', 'y', 'z', 'xs')
    """
    temp = np.asarray(temp, dtype=float)
    times = np.asarray(times, dtype=float)
    N = len(temp)

    x = np.zeros(N)   # labile chill pool
    y = np.zeros(N)   # stable chill pool
    z = np.zeros(N)   # accumulated heat
    xs = np.zeros(N)  # equilibrium labile chill

    # Convert threshold temperatures to Kelvin if needed
    _Tf = Tf + 273.0 if deg_celsius else Tf
    _Tu = Tu + 273.0 if deg_celsius else Tu
    _Tc = Tc + 273.0 if deg_celsius else Tc
    _Tb = Tb + 273.0 if deg_celsius else Tb

    bloomindex = 0

    for i in range(N - 1):
        ti = temp[i] + 273.0 if deg_celsius else temp[i]
        dt = times[i + 1] - times[i]

        # Equilibrium labile chill and rate constant
        xs[i] = A0 / A1 * np.exp(-(E0 - E1) / ti)
        k1 = A1 * np.exp(-E1 / ti)

        # Update labile chill pool (exponential relaxation toward equilibrium)
        x[i + 1] = xs[i] - (xs[i] - x[i]) * np.exp(-k1 * dt)

        # Carry stable chill forward (only modified below if x >= 1)
        y[i + 1] = y[i]

        # Accumulate heat
        if Imodel == 0:
            z[i + 1] = z[i] + P1z(ti, _Tu, _Tb, _Tc) * PFcn(y[i], yc, s1) * dt
        else:
            z[i + 1] = z[i] + P2z(ti, _Tu, Delta) * PFcn(y[i], yc, s1) * dt

        # Convert labile to stable chill when pool is saturated
        if x[i + 1] >= 1.0:
            delta = PFcn(ti, _Tf, slope) * x[i + 1]
            y[i + 1] += delta
            x[i + 1] -= delta

        # Check heat requirement
        if z[i + 1] >= zc:
            bloomindex = i + 2  # +2 for Fortran/R 1-based index convention
            if stopatzc:
                break

    if basic_output:
        return {"bloomindex": bloomindex}
    return {"x": x, "y": y, "z": z, "xs": xs, "bloomindex": bloomindex}
  
#taken from ChatGPT
def gen_season_list(temps, mrange=(8, 6), years=None):
    """
    Python equivalent of chillR::genSeasonList

    Parameters:
        temps (pd.DataFrame): Must contain columns ['Year', 'Month', 'Temp', 'JDay']
        mrange (tuple): (start_month, end_month), e.g. (8, 6)
        years (list or iterable): Years to generate seasons for

    Returns:
        list of pd.DataFrame
    """

    assert len(mrange) == 2, "mrange must have length 2"
    assert years is not None, "years must be provided"
    assert temps is not None, "temps must be provided"
    assert mrange[0] > mrange[1], "start month must be greater than end month"

    start_month, end_month = mrange
    season_list = []

    for y in years:
        # Previous year portion (start_month → Dec)
        prev_mask = (
            (temps["Month"].between(start_month, 12)) &
            (temps["Year"] == y - 1)
        )

        # Current year portion (Jan → end_month)
        curr_mask = (
            (temps["Month"].between(1, end_month)) &
            (temps["Year"] == y)
        )

        season_df = temps.loc[prev_mask | curr_mask, ["Temp", "JDay", "Year"]].copy()

        season_list.append(season_df)

    return season_list
  
  
# ── JAX helpers (vectorised, no Python loops) ─────────────────────────────────

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
    #logits  = sharpness * (z_trace - zc)
    #this favpours to wait forever, until the difference is maxed.
    #this is ALWAYS at the end of the time series

    # MODIFIED: Use negative squared difference for logits
    logits  = -sharpness * (z_trace - zc)**2

    #new approach: negative bi
    # Numerically stable softmax
    logits  = logits - jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(logits)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.sum(weights * hours, axis=-1) # Use sum instead of dot for batched operation

# ── Numpyro model ─────────────────────────────────────────────────────────────

def phenoflex_numpyro(
    temp,
    times,
    bloom_doy_obs=None,
    start_doy=0.0,
    Imodel=0,
    deg_celsius=True,
    sharpness=1.0,
    leap_years = None,
    return_traces=False, # New flag for conditional return
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
    yc    = numpyro.sample("yc",    dist.Normal(65.0,  10.0))
    zc    = numpyro.sample("zc",    dist.Normal(220.0, 30.0))
    s1    = numpyro.sample("s1",    dist.Beta(2.0, 2.0))
    E0    = numpyro.sample("E0",    dist.Normal(4153.5, 200.0))
    E1    = numpyro.sample("E1",    dist.Normal(12888.8, 500.0))
    A0    = numpyro.sample("A0",    dist.HalfNormal(139500)) # Changed to HalfNormal as values are positive
    A1    = numpyro.sample("A1",    dist.HalfNormal(2.567e18)) # Changed to HalfNormal
    Tf    = numpyro.sample("Tf",    dist.Normal(4.0,   1.0))
    slope = numpyro.sample("slope", dist.HalfNormal(1.6))
    Tb    = numpyro.sample("Tb",    dist.Normal(4.0,   2.0))
    Tu    = numpyro.sample("Tu",    dist.Normal(26.0,  3.0))
    Tc    = numpyro.sample("Tc",    dist.Normal(36.0,  3.0))
    Delta = numpyro.sample("Delta", dist.HalfNormal(5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(3.0))  # days (shared observation noise)

    # Ensure temp and times are at least 1D arrays, and if they are single season, make them 2D
    if temp.ndim == 1:
        temp = temp[None, :]
        times = times[None, :]
        if bloom_doy_obs is not None: # only make obs 1D if it's not None
            bloom_doy_obs = bloom_doy_obs[None]
        start_doy = start_doy[None]
        # If leap_years is a single boolean, convert it to a 1-element array for consistency
        if leap_years is not None and jnp.ndim(jnp.asarray(leap_years)) == 0:
            leap_years = jnp.asarray([leap_years]) # Convert to 1-element array

    num_seasons = temp.shape[0]

    #knowing leap years etc is only important to return prediction before Dec-31 (are coded negative numbers)
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
                raise ValueError("Length of 'leap_years' must match 'num_seasons' or be a single boolean.")
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
        ti_raw, dt_i = inputs # ti_raw and dt_i are now (num_seasons,)

        ti = ti_raw + offset

        # Ensure A0, A1, E0, E1, etc. are broadcastable if they are scalars from priors
        xs_i = A0 / A1 * jnp.exp(-(E0 - E1) / ti)
        k1   = A1 * jnp.exp(-E1 / ti)

        x_new = xs_i - (xs_i - x_prev) * jnp.exp(-k1 * dt_i)
        y_new = y_prev

        heat_rate = _p1z_jax(ti, _Tu, _Tb, _Tc) if Imodel == 0 else _p2z_jax(ti, _Tu, Delta)
        z_new = z_prev + heat_rate * _pfcn_jax(y_prev, yc, s1) * dt_i

        # Labile -> stable chill conversion
        delta   = _pfcn_jax(ti, _Tf, slope) * x_new
        convert = jnp.where(x_new >= 1.0, 1.0, 0.0)
        y_new   = y_new + convert * delta
        x_new   = x_new - convert * delta

        return (x_new, y_new, z_new), (x_new, y_new, z_new) # Return all states to be traced

    # Initial state for each season is (num_seasons,)
    init   = (jnp.zeros(num_seasons), jnp.zeros(num_seasons), jnp.zeros(num_seasons))
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
    bloom_hour     = soft_bloom_hour(z_trace, times[:, 1:], zc, sharpness=sharpness)

    # Redefine bloom_doy_pred to match user's request:
    # Dec 31 of (season start year) = 0
    # Jan 1 of (season start year + 1) = 1
    # raw_bloom_doy_pred is the Julian day count since Jan 1st of the season start year, as a float.
    # Subtracting days_in_year_per_season shifts the reference point so Dec 31 of the previous year is 0.

    #-0.5 so that julian day is centered at midday and not midnight = 1.0 = Jan-01 midday
    #-days in the year per season so that there is no break between Dec-31 and Jan-01
    #(otherwise optimizer can get stuck early on in the season)

    raw_bloom_doy_pred = start_doy - 0.5 - days_in_year_per_season  + (bloom_hour / 24.0)

    bloom_doy_pred = numpyro.deterministic(
        "bloom_doy_pred",
        raw_bloom_doy_pred
    )

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
        numpyro.sample("bloom_doy", dist.Normal(bloom_doy_pred, sigma), obs=bloom_doy_obs)
        

# Helpers ───────────────────────────────────────────────────────────────────

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


def check_leap_year(years):
    y = np.asarray(years)
    return (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))


def prepare_seasons(season_list, bloom_df, years):
    """
    Pad seasons to a common length and return stacked JAX arrays
    ready to pass into phenoflex_numpyro.

    Returns
    -------
    temps      : (S, T) float32
    times      : (S, T) float32
    start_doys : (S,)   float32
    bloom_doys : (S,)   float32
    leap_years : (S,)   bool
    """
    # Ensure season_list and years have consistent lengths
    if len(season_list) != len(years):
        raise ValueError(
            f"The `season_list` must contain one entry for each year in the `years` argument. "
            f"Expected {len(years)} seasons, but got {len(season_list)}."
        )

    # Check for empty season dataframes
    for i, season_df in enumerate(season_list):
        if season_df.empty:
            raise ValueError(
                f"Season data for year {years[i]} is empty. This indicates a lack of temperature data for that season."
            )

    max_len = max(len(s) for s in season_list)
    if max_len == 0:
        raise ValueError("All seasons in `season_list` are empty. Cannot prepare data.")

    padded_temps, padded_times, start_doys = [], [], []

    for season_df in season_list:
        temp  = jnp.asarray(season_df.Temp.values, dtype=jnp.float32)
        times = jnp.arange(len(temp),              dtype=jnp.float32)
        pad   = max_len - len(temp)

        padded_temps.append(
            jnp.pad(temp,  (0, pad), mode="edge") if pad else temp
        )
        padded_times.append(
            jnp.concatenate([times, times[-1] + jnp.arange(1, pad + 1, dtype=jnp.float32)])
            if pad else times
        )

        first = season_df.iloc[0]
        hour_offset = (first["Hour"] / 24.0) if "Hour" in season_df.columns else 0.0
        start_doys.append(float(first["JDay"]) - hour_offset)

    # Filter bloom_df for the requested years and drop NaNs
    # Create a DataFrame containing only the 'pheno' column and 'Year' as index for requested years
    filtered_bloom_pheno = bloom_df[bloom_df["Year"].isin(years)][["Year", "pheno"]].set_index("Year")

    # Reindex to the exact `years` requested, filling missing years with NaN
    bloom_doys_series = filtered_bloom_pheno.reindex(years)["pheno"]

    # Identify years with missing or NaN bloom data
    missing_bloom_data_years = bloom_doys_series[bloom_doys_series.isna()].index.tolist()

    if missing_bloom_data_years:
        raise ValueError(
            f"No valid bloom observation found for years: {sorted(missing_bloom_data_years)}. "
            f"These years must be present in `bloom_df` and have non-NaN 'pheno' values."
        )

    bloom_doys = jnp.asarray(bloom_doys_series.values, dtype=jnp.float32)

    # Final check for length consistency after dropping NaNs, although the above check should cover it
    if len(bloom_doys) != len(years):
        # This case should ideally not be hit if the prior check is robust, but for safety
        raise RuntimeError("Internal error: Mismatch in bloom_doys length after validation.")

    return (
        jnp.stack(padded_temps),
        jnp.stack(padded_times),
        jnp.asarray(start_doys,  dtype=jnp.float32),
        bloom_doys,
        check_leap_year([y - 1 for y in years]),  # season starts in prior year
    )
