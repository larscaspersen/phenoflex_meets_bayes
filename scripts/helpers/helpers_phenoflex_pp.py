import numpy as np
import jax.numpy as jnp

# Helpers ───────────────────────────────────────────────────────────────────


def gen_season_list(temps, mrange=(8, 6), years=None):
    """Python equivalent of chillR::genSeasonList."""
    assert len(mrange) == 2 and mrange[0] > mrange[1]
    assert years is not None and temps is not None
    start_month, end_month = mrange

    return [
        temps.loc[
            ((temps["Month"].between(start_month, 12)) & (temps["Year"] == y - 1))
            | ((temps["Month"].between(1, end_month)) & (temps["Year"] == y)),
            ["Temp", "JDay", "Year"],
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
        temp = jnp.asarray(season_df.Temp.values, dtype=jnp.float32)
        times = jnp.arange(len(temp), dtype=jnp.float32)
        pad = max_len - len(temp)

        padded_temps.append(jnp.pad(temp, (0, pad), mode="edge") if pad else temp)
        padded_times.append(
            jnp.concatenate(
                [times, times[-1] + jnp.arange(1, pad + 1, dtype=jnp.float32)]
            )
            if pad
            else times
        )

        first = season_df.iloc[0]
        hour_offset = (first["Hour"] / 24.0) if "Hour" in season_df.columns else 0.0
        start_doys.append(float(first["JDay"]) - hour_offset)

    # Filter bloom_df for the requested years and drop NaNs
    # Create a DataFrame containing only the 'pheno' column and 'Year' as index for requested years
    filtered_bloom_pheno = bloom_df[bloom_df["Year"].isin(years)][
        ["Year", "pheno"]
    ].set_index("Year")

    # Reindex to the exact `years` requested, filling missing years with NaN
    bloom_doys_series = filtered_bloom_pheno.reindex(years)["pheno"]

    # Identify years with missing or NaN bloom data
    missing_bloom_data_years = bloom_doys_series[
        bloom_doys_series.isna()
    ].index.tolist()

    if missing_bloom_data_years:
        raise ValueError(
            f"No valid bloom observation found for years: {sorted(missing_bloom_data_years)}. "
            f"These years must be present in `bloom_df` and have non-NaN 'pheno' values."
        )

    bloom_doys = jnp.asarray(bloom_doys_series.values, dtype=jnp.float32)

    # Final check for length consistency after dropping NaNs, although the above check should cover it
    if len(bloom_doys) != len(years):
        # This case should ideally not be hit if the prior check is robust, but for safety
        raise RuntimeError(
            "Internal error: Mismatch in bloom_doys length after validation."
        )

    return (
        jnp.stack(padded_temps),
        jnp.stack(padded_times),
        jnp.asarray(start_doys, dtype=jnp.float32),
        bloom_doys,
        check_leap_year([y - 1 for y in years]),  # season starts in prior year
    )


def prepare_seasons_hazard(season_list, bloom_df, years):
    """
    Pad seasons to a common length and return stacked JAX arrays
    ready to pass into phenoflex_numpyro_hazard.

    Returns
    -------
    temps        : (S, T) float32
    times        : (S, T) float32
    bloom_indices: (S,)   int
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

    padded_temps, padded_times = [], []

    for season_df in season_list:
        temp = jnp.asarray(season_df.Temp.values, dtype=jnp.float32)
        times_local = jnp.arange(
            len(temp), dtype=jnp.float32
        )  # Use local variable to avoid confusion
        pad = max_len - len(temp)

        padded_temps.append(jnp.pad(temp, (0, pad), mode="edge") if pad else temp)
        padded_times.append(
            jnp.concatenate(
                [
                    times_local,
                    times_local[-1] + jnp.arange(1, pad + 1, dtype=jnp.float32),
                ]
            )
            if pad
            else times_local
        )

    # Filter bloom_df for the requested years and drop NaNs
    filtered_bloom_pheno = bloom_df[bloom_df["Year"].isin(years)][
        ["Year", "pheno"]
    ].set_index("Year")
    bloom_doys_series = filtered_bloom_pheno.reindex(years)["pheno"]

    missing_bloom_data_years = bloom_doys_series[
        bloom_doys_series.isna()
    ].index.tolist()
    if missing_bloom_data_years:
        raise ValueError(
            f"No valid bloom observation found for years: {sorted(missing_bloom_data_years)}. "
            f"These years must be present in `bloom_df` and have non-NaN 'pheno' values."
        )

    observed_bloom_doys = jnp.asarray(bloom_doys_series.values, dtype=jnp.float32)

    bloom_indices = []
    for i_season, season_df in enumerate(season_list):
        observed_bloom_doy = observed_bloom_doys[i_season]
        season_bloom_year = years[i_season]  # The calendar year of the observed bloom

        # Find all indices where JDay matches observed_bloom_doy AND Year matches season_bloom_year
        # .values is used to convert pandas Series to numpy array for jnp.where
        match_mask = (season_df["JDay"] == observed_bloom_doy) & (
            season_df["Year"] == season_bloom_year
        )
        all_matching_hourly_indices = jnp.where(match_mask.values)[0]

        if len(all_matching_hourly_indices) == 0:
            raise ValueError(
                f"No matching JDay found for observed bloom DOY {observed_bloom_doy} in season {season_bloom_year}. "
                f"This might indicate an issue with the observed bloom data or temperature data range."
            )
        # Assume bloom recorded at noon (12th hour, which is index 11 for 0-indexed hours)
        if len(all_matching_hourly_indices) < 12:
            raise ValueError(
                f"Less than 12 hourly records found for bloom day {observed_bloom_doy} in season {season_bloom_year}. "
                f"Cannot determine noon (index 11) bloom hour reliably. Found {len(all_matching_hourly_indices)} records."
            )

        # Take the index corresponding to the 12th hour (index 11) of the bloom day
        bloom_index_for_season = all_matching_hourly_indices[11].astype(int)

        # Ensure the index is within the bounds of the actual (unpadded) season data length
        actual_season_len = len(
            season_list[i_season]
        )  # Length of original season_df before padding
        bloom_index_for_season = jnp.clip(
            bloom_index_for_season, 0, actual_season_len - 1
        )

        bloom_indices.append(bloom_index_for_season)

    return (
        jnp.stack(padded_temps),
        jnp.stack(padded_times),
        jnp.asarray(bloom_indices, dtype=jnp.int32),
    )


def _hours_to_fractional_doy(hourly_indices, season_df):
    """
    Convert hourly indices to fractional DOY with sub-daily precision.

    Assumes hourly data with 24 entries per day. Noon (hour 12) of each day
    corresponds to DOY.0, so earlier hours subtract fractional values and
    later hours add them.

    Parameters
    ----------
    hourly_indices : array-like
        Array of hourly indices (0, 1, 2, ...)
    season_df : DataFrame
        Season dataframe with 'JDay' column

    Returns
    -------
    fractional_doys : array
        DOY values with sub-daily precision
    """
    jdays_array = jnp.asarray(season_df["JDay"].values, dtype=jnp.float32)

    fractional_doys = []
    for h in hourly_indices:
        h_int = int(h)
        # Ensure index is within bounds
        h_int = min(h_int, len(jdays_array) - 1)

        # Get base DOY for this hour's date
        base_doy = jdays_array[h_int]

        # Calculate hour within the day (0-23)
        hour_within_day = h_int % 24

        # Calculate fractional part: noon (12) maps to +0.0, midnight (0) maps to -0.5
        fractional_part = (hour_within_day - 12) / 24.0

        fractional_doys.append(base_doy + fractional_part)

    return jnp.asarray(fractional_doys, dtype=jnp.float32)


def plot_bloom_pmf(
    bloom_pmf_pred,
    times,
    years,
    bloom_index=None,
    season_list=None,
    xlim=None,
    x_axis="doy",
):
    """
    Plot predicted bloom probability distributions with credible intervals.

    Parameters
    ----------
    bloom_pmf_pred : jnp array (num_samples, num_seasons, num_timesteps - 1)
        Predicted bloom probability mass functions from predictive sampling
    times : jnp array (num_seasons, num_timesteps)
        Time steps for each season (hourly indices)
    years : array-like
        Array of years for season labels
    bloom_index : jnp array (num_seasons,), optional
        Observed bloom indices for each season
    season_list : list of DataFrame, optional
        List of season DataFrames containing 'JDay' column for DOY conversion.
        If provided with x_axis='doy', will convert hourly indices to DOY values
        with sub-daily precision (noon = DOY.0, midnight = DOY±0.5).
    xlim : tuple, optional
        X-axis limits (min, max)
    x_axis : str, default 'doy'
        Type of x-axis: 'hours' for hourly indices, 'doy' for day-of-year

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    num_seasons = bloom_pmf_pred.shape[1]

    fig, axes = plt.subplots(num_seasons, 1, figsize=(15, 4 * num_seasons), sharex=True)
    if num_seasons == 1:
        axes = [axes]

    for i_season in range(num_seasons):
        ax = axes[i_season]

        # Extract PMFs for the current season across all samples
        pmfs_for_season = bloom_pmf_pred[:, i_season, :]

        # Calculate mean PMF and 90% credible interval
        mean_pmf = jnp.mean(pmfs_for_season, axis=0)
        lower_pmf = jnp.percentile(pmfs_for_season, 5, axis=0)
        upper_pmf = jnp.percentile(pmfs_for_season, 95, axis=0)

        hours_for_plotting = times[i_season, 1:]

        # Convert to DOY if requested and season_list is provided
        if x_axis == "doy" and season_list is not None:
            season_df = season_list[i_season]
            x_axis_data = _hours_to_fractional_doy(hours_for_plotting, season_df)
            x_label = "Day of Year"
        else:
            x_axis_data = hours_for_plotting
            x_label = "Hours since season start"

        # Plot the mean PMF and credible interval
        ax.plot(x_axis_data, mean_pmf, label="Mean PMF of Bloom", color="blue")
        ax.fill_between(
            x_axis_data,
            lower_pmf,
            upper_pmf,
            color="blue",
            alpha=0.2,
            label="90% CI of PMF",
        )

        # Plot observed bloom index
        if bloom_index is not None:
            observed_idx = bloom_index[i_season]
            # Convert observed index to DOY if needed
            if x_axis == "doy" and season_list is not None:
                season_df = season_list[i_season]
                observed_x = float(
                    _hours_to_fractional_doy(jnp.array([observed_idx]), season_df)[0]
                )
            else:
                observed_x = observed_idx
            ax.axvline(
                x=observed_x,
                color="red",
                linestyle="--",
                label=f"Observed Bloom (Index: {observed_idx})",
            )

        # Find the most likely bloom hour from the mean PMF
        most_likely_idx = jnp.argmax(mean_pmf)
        if x_axis == "doy" and season_list is not None:
            season_df = season_list[i_season]
            most_likely_hour_idx = hours_for_plotting[most_likely_idx]
            most_likely_x = float(
                _hours_to_fractional_doy(jnp.array([most_likely_hour_idx]), season_df)[
                    0
                ]
            )
            label_text = f"Most Likely Bloom (DOY: {most_likely_x:.2f})"
        else:
            most_likely_x = float(hours_for_plotting[most_likely_idx])
            label_text = f"Most Likely Bloom (Hour: {most_likely_x:.0f})"

        ax.axvline(x=most_likely_x, color="green", linestyle=":", label=label_text)

        ax.set_title(f"Season {years[i_season]} - Bloom Probability Distribution")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Probability")
        ax.legend()

        # Set weekly intervals on x-axis if using DOY
        if x_axis == "doy":
            ax.xaxis.set_major_locator(MultipleLocator(7))
            ax.xaxis.set_minor_locator(MultipleLocator(1))

        ax.grid(True, which="major", alpha=0.7)
        ax.grid(True, which="minor", alpha=0.2)

        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    return fig
