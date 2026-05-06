# ── Helper functions
import numpy as np

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