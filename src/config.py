# ===================== #
# CONFIGURE ENVIRONMENT #
# ===================== #

from __future__ import annotations # Annotations are not evaluated immediately when the file is imported.
from colossus.cosmology import cosmology as colossus_cosmology
colossus_cosmology.setCosmology("planck18")
from colossus.halo import concentration as colossus_concentration
from functools import lru_cache
import math # to be optimized
import numpy as np
import scipy
from scipy import interpolate, optimize
from typing import Tuple
import re
import warnings

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
AU        = 1.495978707e11        # astronomical unit [m] (reference: https://en.wikipedia.org/wiki/Astronomical_unit)
c         = 2.99792458e8          # speed of light [m·s⁻¹] (reference: https://en.wikipedia.org/wiki/Speed_of_light)
day       = 24 * 3600             # day [s]
e         = 1.602176634e-19       # elementary charge [C]
epsilon_0 = 8.854187817e-12       # vacuum permittivity [F·m⁻¹]
G         = 6.6743015e-11         # gravitational constant [m³·kg⁻¹·s⁻²]
G_Arepo   = 4.300931494278067e-3  # gravitational constant [pc·(km/s)²·M☉⁻¹]
G_kpc     = 4.300931494278067e-6  # gravitational constant [kpc·(km/s)²·M☉⁻¹]
G_astro   = 0.004498517029175462  # gravitational constant [pc³·M☉⁻¹·Myr⁻²]
h         = 6.62607015e-34        # Planck constant [J·s]
k_B       = 1.380649e-23          # Boltzmann constant [J·K⁻¹]
m_e       = 9.109383713928e-31    # electron mass [kg] (reference: https://en.wikipedia.org/wiki/Electron_mass)
m_e_c2    = 0.5109989506916       # electron mass [MeV] (reference: https://en.wikipedia.org/wiki/Electron_mass)
m_p       = 1.6726219259552e-27   # proton mass [kg] (reference: https://en.wikipedia.org/wiki/Proton)
m_u       = 1.6605390689252e-27   # unified atomic mass unit [kg] (reference: https://en.wikipedia.org/wiki/Dalton_(unit))
M_sun     = 1.988416e30           # solar mass [kg] (reference: https://en.wikipedia.org/wiki/Solar_mass)
N_A       = 6.02214076e23         # Avogadro constant [mol⁻¹] (reference: https://en.wikipedia.org/wiki/Avogadro_constant)
pc        = 3.0856775814913673e16 # parsec [m] (reference: https://en.wikipedia.org/wiki/Parsec)
PI        = np.pi                 # π
sigma_T   = 6.652458705162e-29    # Thomson cross section [m^2] (reference: https://en.wikipedia.org/wiki/Thomson_scattering)
yr        = 365.25 * 24 * 3600    # Julian year [s]

kpc       = 1.0e3 * pc            # kiloparsec [m]
Mpc       = 1.0e6 * pc            # megaparsec [m]
Myr       = 1.0e6 * yr            # megayear [s]
Gyr       = 1.0e9 * yr            # gigayear [s]

# DESI 2024 + CMB
H0            = 67.97 # Hubble constant [(km/s)/Mpc]
Omega_Lambda0 = 0.693 # present-day dark-energy density parameter
Omega_m0      = 0.307 # present-day matter density parameter
ReducedH0     = H0 / 100.0 # reduced Hubble constant h; H_0 = 100 h (km/s)/Mpc
t_universe    = 13.780 # age of the universe [Gyr]

# WMAP + eCMB + BAO + H0
#H0            = 69.7 # Hubble constant [(km/s)/Mpc]
#Omega_Lambda0 = 0.7181 # present-day dark-energy density parameter
#Omega_m0      = 0.2819 # present-day matter density parameter
#ReducedH0     = H0 / 100.0 # reduced Hubble constant h; H_0 = 100 h (km/s)/Mpc
#t_universe    = 13.75 # age of the universe [Gyr]

SqrtOmega_Lambda0OverOmega_m0 = math.sqrt(Omega_Lambda0 / Omega_m0)
t_Lambda_Gyr  = 2.0 / (3.0 * H0 * math.sqrt(Omega_Lambda0)) * Mpc / 1.0e3 / Gyr

MIN_RAD_PC = 1.0 # inner aperture/bin edge
NSC_RAD_PC = 6.0 # public stellar NSC aperture
Eddington_varepsilon = 0.1 # Eddington radiative efficiency
Eddington_time_Gyr = Eddington_varepsilon * sigma_T * c / (4.0 * PI * G * m_p * (1.0 - Eddington_varepsilon)) / Gyr # Eddington time [Gyr]
M_BH_warning = 1.0e12 # BH mass threshold for warnings about excessive Eddington growth [Msun]
STD_DPI = 512

# ================== #
# HELPER FUNCTION(S) #
# ================== #

# checking utilities for input parameters

def check_finite(val, name="val"):
    val = float(val)
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite, but got {val}!")
    return val
    
def check_finite_non_negative(val, name="val"):
    val = float(val)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, but got {val}!")
    return val

def check_finite_positive(val, name="val"):
    val = float(val)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and positive, but got {val}!")
    return val

def parse_exact_int64(value, name="integer"):
    """Parse one exact signed 64-bit integer from an integer or decimal token."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an exact decimal integer, not a boolean: {value!r}")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
    elif isinstance(value, str):
        token = value.strip()
        if re.fullmatch(r"[+-]?[0-9]+", token) is None:
            raise ValueError(f"{name} must be an exact decimal integer token; got {value!r}")
        integer = int(token, 10)
    else:
        raise ValueError(f"{name} must be an exact integer value or decimal token; got {value!r}")
    limits = np.iinfo(np.int64)
    if integer < int(limits.min) or integer > int(limits.max):
        raise ValueError(
            f"{name}={integer} is outside the signed int64 range "
            f"[{int(limits.min)}, {int(limits.max)}]."
        )
    return np.int64(integer)

def parse_fixed_tree_id(value, name="fixed-tree identifier", *, allow_minus_one=False):
    """Parse a fixed-tree ID and enforce its schema-specific lower bound."""

    identifier = parse_exact_int64(value, name=name)
    minimum = -1 if allow_minus_one else 0
    if identifier < minimum:
        bound = "-1" if allow_minus_one else "0"
        raise ValueError(f"{name} must be greater than or equal to {bound}; got {identifier}.")
    return identifier

def fixed_tree_mpb_branch_id(log_mh, branch_id): # to be checked
    """Return the project MPB branch ID from fixed-tree rows.

    This intentionally follows the Gao+2024 formation scripts and the
    previous ``src/main.py`` behaviour: choose the branch ID at the first row
    with maximum log10(M_h/Msun).
    """

    log_mh_arr = np.asarray(log_mh, dtype=float)
    branch_raw = np.asarray(branch_id)
    if log_mh_arr.ndim != 1 or branch_raw.ndim != 1:
        raise ValueError("Fixed-tree MPB inputs must be one-dimensional arrays.")
    if len(log_mh_arr) == 0:
        raise ValueError("Cannot identify the MPB branch from empty fixed-tree arrays.")
    if len(log_mh_arr) != len(branch_raw):
        raise ValueError("Fixed-tree log_mh and branch_id arrays must have matching lengths.")
    if np.any(~np.isfinite(log_mh_arr)):
        raise ValueError("Fixed-tree log_mh values must be finite.")

    branch_values = []
    for value in branch_raw:
        branch = int(parse_exact_int64(value, name="fixed-tree branch ID"))
        if branch < 0:
            raise ValueError(f"Fixed-tree branch ID must be non-negative; got {branch}")
        branch_values.append(branch)

    return int(branch_values[int(np.argmax(log_mh_arr))])

# Eddington-limited BH growth utilities

def grow_eddington_mass_msun(M_BH: float, dt_Gyr: float, f_Eddington: float):
    """
    Grow BH mass by simplified Eddington-limited accretion.

    M_BH (t + dt) = M_BH (t) * exp(f_Eddington * dt / t_Eddington)
    """
    M_BH        = check_finite_non_negative(M_BH, name="BH mass M_BH before Eddington accretion")
    dt_Gyr      = check_finite_non_negative(dt_Gyr, name="Eddington accretion timestep dt_Gyr")
    f_Eddington = check_finite_non_negative(f_Eddington, name="Eddington ratio f_Eddington")

    if M_BH > 0.0 and dt_Gyr > 0.0 and f_Eddington > 0.0:
        M_BH *= math.exp(f_Eddington * dt_Gyr / Eddington_time_Gyr)
        check_finite_non_negative(M_BH, name="BH mass M_BH after Eddington accretion")

    if M_BH > M_BH_warning:
        warnings.warn(f"BH mass M_BH after Eddington accretion exceeds {M_BH_warning:.0e} Msun.", RuntimeWarning)

    return M_BH

# linear interpolation on a uniformly spaced grid

def lininterp_uniform(xq, x_grid, y_grid, dx_inv=None, *, allow_extrapolate=False): # to be checked
    """
    Linear interpolation on a uniformly spaced grid.

    Parameters
    ----------
    xq : float
        Query coordinate.
    x_grid : array-like
        Uniformly spaced grid coordinates.
    y_grid : array-like
        Function values tabulated on x_grid.
    dx_inv : float, optional
        Inverse grid spacing, 1 / dx. If None, it is computed from x_grid.
    allow_extrapolate : bool
        If False, raise an error outside the grid.
        If True, linearly extrapolate using the edge interval.

    Returns
    -------
    yq : float
        Interpolated value at xq.
    """

    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be 1D arrays.")

    if len(x_grid) != len(y_grid):
        raise ValueError("x_grid and y_grid must have the same length.")

    if len(x_grid) < 2:
        raise ValueError("Need at least two grid points for interpolation.")

    if dx_inv is None:
        dx = x_grid[1] - x_grid[0]
        if dx == 0.0:
            raise ValueError("x_grid spacing cannot be zero.")
        dx_inv = 1.0 / dx

    u = (xq - x_grid[0]) * dx_inv

    if not allow_extrapolate:
        if u < 0.0 or u > len(x_grid) - 1:
            raise ValueError(f"xq={xq} is outside the interpolation grid.")

    # Exact upper boundary: return the final value directly.
    if u == len(x_grid) - 1:
        return float(y_grid[-1])

    i = math.floor(u)

    if allow_extrapolate:
        i = max(0, min(i, len(x_grid) - 2))
    else:
        i = max(0, min(i, len(x_grid) - 2))

    f = u - i

    return float(y_grid[i] + f * (y_grid[i + 1] - y_grid[i]))

# cosmology

def E(z: float) -> float:
    """
    dimensionless Hubble parameter E(z) = H(z) / H0 for flat ΛCDM without radiation
    """
    check_finite_non_negative(z, name="Redshift z")

    return np.sqrt(Omega_m0 * (1.0 + z)**3 + Omega_Lambda0)

def H(z: float) -> float:
    """
    Hubble parameter H(z) in (km/s)/Mpc for flat ΛCDM without radiation
    """
    return H0 * E(z)

def Omega_m(z: float) -> float:
    """matter density parameter Ω_m(z) for flat ΛCDM without radiation"""
    check_finite_non_negative(z, name="Redshift z")

    zPlus1Cubed = (1.0 + z) ** 3
    return Omega_m0 * zPlus1Cubed / (1.0 - Omega_m0 + Omega_m0 * zPlus1Cubed)

def Redshift2CosmicAge(z: float, time_unit: str = "Gyr") -> float:
    """
    For flat ΛCDM without radiation, the age at redshift z has the analytic form:
        t(z) = 2 / (3 H0 sqrt(Omega_L)) * asinh(sqrt(Omega_L / Omega_M) / (1 + z)^(3/2))
    """

    check_finite_non_negative(z, name="Redshift z")
    if time_unit == "Gyr":
        t_Lambda = t_Lambda_Gyr
    elif time_unit == "Myr":
        t_Lambda = t_Lambda_Gyr * 1.0e3
    elif time_unit == "yr":
        t_Lambda = t_Lambda_Gyr * 1.0e9
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")

    return t_Lambda * math.asinh(SqrtOmega_Lambda0OverOmega_m0 / ((1.0 + z) ** 1.5))

def Rv(Mhalo: float, z: float) -> float:
    """
    Virial radius in kpc for halo mass Mhalo in Msun for flat ΛCDM without radiation

    Uses the Bryan & Norman virial overdensity relative to the critical density.
    """
    check_finite_positive(Mhalo, name="Halo mass in M☉ Mhalo")
    check_finite_non_negative(z, name="Redshift z")

    # critical density
    H_kpc = H(z=z) * 1.0e-3 # [(km/s)/Mpc] --> [(km/s)/kpc]
    rho_crit = 3.0 * (H_kpc ** 2) / (8.0 * PI * G_kpc) # [M☉/kpc³]

    # Bryan & Norman virial overdensity relative to critical density
    x = Omega_m(z=z) - 1.0
    Delta_c = 18.0 * (PI ** 2) + 82.0 * x - 39.0 * (x ** 2)

    # virial radius in kpc
    return np.cbrt(3.0 * Mhalo / (4.0 * PI * Delta_c * rho_crit))

def CosmicAge2Redshift(t: float, time_unit: str = "Gyr") -> float:
    """cosmic age to redshift conversion for flat ΛCDM without radiation"""
    check_finite_positive(t, name="Cosmic age t")
    if time_unit == "Gyr":
        t = t
    elif time_unit == "Myr":
        t = t / 1.0e3
    elif time_unit == "yr":
        t = t / 1.0e9
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")

    z = (SqrtOmega_Lambda0OverOmega_m0 / math.sinh(t / t_Lambda_Gyr)) ** (2.0 / 3.0) - 1.0
    return check_finite_non_negative(z, name="Redshift z")

def v_v(Mhalo: float, z: float) -> float:
    """virial velocity in km/s for halo mass Mhalo in M☉ at redshift z for flat ΛCDM without radiation"""
    return np.sqrt(G_kpc * Mhalo / Rv(Mhalo=Mhalo, z=z))

@lru_cache(maxsize=4096)
def halo_concn_IshiyamaP2021(Mhalo: float, z: float) -> float:
    """Ishiyama+2021 median NFW-fit virial concentration for all haloes."""
    check_finite_positive(Mhalo, name="Halo virial mass in M☉ Mhalo")
    check_finite_non_negative(z, name="Redshift z")

    c_vir = colossus_concentration.concentration(
        Mhalo * ReducedH0, # [M☉] --> [M☉/h]
        "vir",
        z,
        model="ishiyama21",
        c_type="fit",
        halo_sample="all")

    return check_finite_positive(c_vir, name="Ishiyama+2021 virial concentration c_vir")

# Behroozi+2013 stellar mass-halo mass(SMHM) relation

def f_x_SMHM(x: float, z: float) -> float:
    check_finite(x, name="lg(Mhalo/M1) x")
    check_finite_non_negative(z, name="Redshift z")

    a     = 1.0 / (1.0 + z)
    nu    = math.exp(- 4.0 * a * a)
    alpha = - 1.412 + 0.731 * (a - 1.0) * nu
    delta = 3.508 + (2.608 * (a - 1.0) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu
    exp   = 10.0 ** (-x)
    coef  = 0.0 if exp > 700.0 else 1.0 / (1.0 + math.exp(exp))
    return - math.log10(10.0 ** (alpha * x) + 1.0) + delta * (math.log10(1.0 + math.exp(x))) ** gamma * coef

def Mstar_SMHM(Mhalo: float, z: float, scatter: bool = False) -> float:
    check_finite_positive(Mhalo, name="Halo mass in M☉ Mhalo")
    check_finite_non_negative(z, name="Redshift z")

    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    epsilon = 10.0 ** (- 1.777 - 0.006 * (a - 1.0) * nu - 0.119 * (a - 1.0))
    M1 = 10.0 ** (11.514 - (1.793 * (a - 1.0) + 0.251 * z) * nu)
    lg_Mstar = math.log10(epsilon * M1) + f_x_SMHM(math.log10(Mhalo / M1), z) - f_x_SMHM(0.0, z)
    if scatter:
        xi = np.random.normal(0.0, 0.218 + 0.023 * z / (1.0 + z))
        lg_Mstar += xi
    return check_finite_positive(10 ** lg_Mstar, name="Stellar mass in M☉ Mstar")

# Schechter star cluster initial mass function

def upperIncompleteGamma0(x):
    """
    upper incomplete gamma function Gamma(0, x) = exponential integral E_1(x) for x > 0
    """
    check_finite_positive(x, name="x for upper incomplete gamma function Gamma(0, x)")

    return scipy.special.exp1(x)

def upperIncompleteGammaMinus1(x):
    """
    upper incomplete gamma function Gamma(-1, x) = exp(-x) / x - Gamma(0, x) for x > 0
    """
    check_finite_positive(x, name="x for upper incomplete gamma function Gamma(-1, x)")

    return np.exp(-x) / x - scipy.special.exp1(x)

def makeLogMgcToLogMmaxInterpolator(Mc: float, Mmin: float = 3.0e4, dlog_mmax: float = 0.02): # to be checked
    """
    Build an interpolator from log10(total GC mass) to log10(Mmax)
    for a Schechter cluster initial mass function with alpha = -2.

    The CIMF is

        dN/dM = A M^-2 exp(-M / Mc)

    The normalization A is set by requiring one expected cluster above Mmax:

        1 = int_{Mmax}^{inf} dN/dM dM

    Then the total GC mass formed in the event is

        M_GC = int_{Mmin}^{Mmax} M dN/dM dM

    For alpha = -2, this gives

        M_GC =
            Mc * [Gamma(0, Mmin/Mc) - Gamma(0, Mmax/Mc)]
               / Gamma(-1, Mmax/Mc)

    Parameters
    ----------
    mc : float
        Schechter cutoff mass Mc in Msun.

    mmin : float
        Minimum cluster mass in Msun.

    dlog_mmax : float
        Grid spacing in log10(Mmax).

    Returns
    -------
    callable
        Callable with usage:

            log_mmax = interp(log_mgc)

        where log_mgc = log10(total GC mass formed in one event).
    """
    check_finite_positive(Mc, name="Schechter cutoff mass Mc")
    check_finite_positive(Mmin, name="Minimum cluster mass Mmin")
    check_finite_positive(dlog_mmax, name="log Mmax grid spacing dlog_mmax")
    if Mmin >= 1.0e6:
        raise ValueError(f"Minimum cluster mass Mmin must be less than 1e6 Msun, but got Mmin = {Mmin}!")

    log_mmin = np.log10(Mmin)

    # Mmax must be larger than Mmin, so start one grid step above Mmin.
    log_mmax_grid = np.arange(log_mmin + dlog_mmax, 8.6, dlog_mmax, dtype=float)

    mmax_grid = 10.0 ** log_mmax_grid

    # Dimensionless mass ratios x = M / Mc.
    x_min = Mmin / Mc
    x_max_grid = mmax_grid / Mc

    # Gamma(0, Mmin / Mc), scalar.
    gamma0_min = upperIncompleteGamma0(x_min)

    # Use the updated gamma functions on each grid point.
    gamma0_max_grid = np.array([
        upperIncompleteGamma0(x) for x in x_max_grid
    ])

    gamma_minus1_max_grid = np.array([
        upperIncompleteGammaMinus1(x) for x in x_max_grid
    ])

    # Total GC mass corresponding to each Mmax.
    mgc_grid = Mc * (gamma0_min - gamma0_max_grid) / gamma_minus1_max_grid

    if np.any(~np.isfinite(mgc_grid)) or np.any(mgc_grid <= 0.0):
        raise RuntimeError("Generated invalid M_GC values while building the interpolator.")

    log_mgc_grid = np.log10(mgc_grid)

    if not np.all(np.diff(log_mgc_grid) > 0.0):
        raise RuntimeError(
            "log10(M_GC) is not strictly increasing with log10(Mmax). "
            "Cannot build a safe inverse interpolator."
        )

    tabulated_inverse = interpolate.interp1d(
        log_mgc_grid,
        log_mmax_grid,
        bounds_error=True,
        assume_sorted=True,
    )

    def log_mgc_to_log_mmax(log_mgc):
        log_mgc_value = check_finite(log_mgc, name="log10 total GC mass")
        if log_mgc_value >= float(log_mgc_grid[0]):
            return float(tabulated_inverse(log_mgc_value))

        target_mgc = 10.0 ** log_mgc_value
        if not np.isfinite(target_mgc) or target_mgc <= 0.0:
            raise ValueError(f"Total GC mass is not representable as a positive finite value: {log_mgc_value}")

        def residual(log_mmax):
            mmax = 10.0 ** float(log_mmax)
            x_max = mmax / Mc
            mgc = Mc * (gamma0_min - upperIncompleteGamma0(x_max)) / upperIncompleteGammaMinus1(x_max)
            return float(mgc - target_mgc)

        lower = float(log_mmin)
        upper = float(log_mmax_grid[0])
        log_mmax = optimize.brentq(residual, lower, upper, xtol=1.0e-14, rtol=4.0 * np.finfo(float).eps)
        return float(max(log_mmax, lower))

    return log_mgc_to_log_mmax

def upper_gamma2_log_mass(log_m: float, Mc: float) -> float: # to be checked
    """Return Gamma(-1, M/Mc) for a base-10 log mass."""

    check_finite(log_m, name="log10 cluster mass")
    check_finite_positive(Mc, name="Schechter cutoff mass Mc")
    return float(upperIncompleteGammaMinus1((10.0 ** float(log_m)) / float(Mc)))
    
# Sersic profile utilities

def Sersic_coefs(N_S: float) -> Tuple[float, float]:
    check_finite_positive(N_S, name="Sersic index N_S")
    p = 1.0 - 0.6097 / N_S + 0.05563 / (N_S * N_S)
    b = 2.0 * N_S - 1.0 / 3.0 + 0.009876 / N_S
    return p, b

# Gao-only effective-radius helpers used by the formation and evolution stages.

"""
def resolve_birth_re_kpc(halomass_msun: float, redshift: float, jsp: float) -> float:
    j_kpc_kms = float(jsp) * ReducedH0
    Rv_kpc = Rv(Mhalo=halomass_msun, z=redshift)
    hz_km_s_kpc = H(float(redshift)) * 1.0e-3
    Re = j_kpc_kms / (20.0 * hz_km_s_kpc * Rv_kpc)
    return check_finite_positive(Re, name="Gao+2024 birth effective radius in kpc")
"""

def calcRe(Mhalo_1e9Msun: float, t_Gyr: float, j: float) -> float:
    """compute the effective radius of the galactic disc in kpc"""
    check_finite_positive(Mhalo_1e9Msun, name="Halo mass in 1e9 M☉ Mhalo_1e9Msun")
    check_finite_positive(t_Gyr, name="Cosmic age in Gyr t_Gyr")
    check_finite_positive(j, name="Specific angular momentum in pc(km/s) j")

    Mhalo = float(Mhalo_1e9Msun) * 1.0e9
    Rv_kpc = Rv(Mhalo=Mhalo, z=CosmicAge2Redshift(t_Gyr, time_unit="Gyr"))
    lambdaB = j / math.sqrt(2.0 * G_Arepo * Mhalo * Rv_kpc * 1.0e3)
    return check_finite_positive(lambdaB * Rv_kpc / math.sqrt(2.0), name="Effective radius of the galactic disc in kpc Re")

"""
Function-only scalar IMBH seeding estimator for GC formation outputs.

Metallicity inputs are the literal ratio Z/Zsun, not [Fe/H].  The Rantala+2026
fit uses the cluster surface density and metallicity, while the Vergara+2026
fits use only the birth stellar cluster mass.
"""

IMBH_FIT_RANTALA = "Rantala+2026"
IMBH_FIT_VERGARA_CONSERVATIVE = "Vergara+2026conservative"
IMBH_FIT_VERGARA_OPTIMISTIC = "Vergara+2026optimistic"
IMBH_FIT_CHOICES = (
    IMBH_FIT_RANTALA,
    IMBH_FIT_VERGARA_CONSERVATIVE,
    IMBH_FIT_VERGARA_OPTIMISTIC,
)
DEFAULT_IMBH_FIT = IMBH_FIT_RANTALA

def validate_imbh_fit(fit: str) -> str:
    """Validate and return an IMBH formation-mass fit name."""

    if fit not in IMBH_FIT_CHOICES:
        choices = ", ".join(IMBH_FIT_CHOICES)
        raise ValueError(f"Unknown IMBH fit {fit!r}; choose one of: {choices}.")
    return fit

def initMRRofSCs(Mcl: float, f_h: float = 0.125) -> float:
    """Eq.7: initial mass-radius relation of star clusters,
    returning the star cluster 3D half-mass radius in pc."""

    check_finite_positive(Mcl, name="Star cluster mass Mcl in M☉")
    r_h = f_h * 2.365 / 1.3 * ((Mcl / 1.0e4) ** 0.18)
    return check_finite_positive(r_h, name="Star cluster 3D half-mass radius r_h in pc")

def calcSigma_h(Mcl: float, r_h: float) -> float:
    """Projected half-mass surface density in M☉/pc².

    The input radius is the 3D half-mass radius.  For a Plummer profile,
    Sigma_h = M / (2 pi (2 ** (2/3) - 1) r_h^2).
    """
    check_finite_positive(Mcl, name="Star cluster mass Mcl in M☉")
    check_finite_positive(r_h, name="Star cluster 3D half-mass radius r_h in pc")

    Sigma_h = Mcl / (2.0 * PI * (np.cbrt(4.0) - 1) * (r_h ** 2))
    return check_finite_positive(Sigma_h, name="Projected half-mass surface density Sigma_h in M☉/pc²")

def calcMimbhEq9(Sigma_h: float, Z: float) -> float:
    """Eq.9: IMBH mass fit within the calibrated surface-density range."""
    Sigma_h = check_finite_positive(Sigma_h, name="Projected 2D half-mass surface density Sigma_h in M☉/pc²")
    lgZ     = math.log10(check_finite_positive(Z, name="Metallicity Z in Z☉"))

    if Z < 0.126:
        A, B, C, lgSigma_crit = - 1790.07 * lgZ - 7392.65, 162.46 * lgZ + 829.42, 4734.11 * lgZ + 16556.82, 0.0386 * lgZ + 4.53
    elif Z < 0.398:
        A, B, C, lgSigma_crit = 9707.84 * lgZ + 2627.71, - 1015.72 * lgZ - 211.38, - 23585.20 * lgZ - 7814.04, 0.91 * lgZ + 5.22
    else:
        A, B, C, lgSigma_crit = 1147.68 * lgZ - 721.18, - 166.92 * lgZ + 126.15, - 2002.25 * lgZ + 471.59, 0.91 * lgZ + 5.22
    lgSigma_h = math.log10(Sigma_h)
    Mimbh = 0.0 if Sigma_h < 10.0**lgSigma_crit else A * lgSigma_h + B * lgSigma_h**2 + C
    Mimbh = check_finite(Mimbh, name="Eq.9 IMBH mass Mimbh in M☉")
    Mimbh = Mimbh if Mimbh >= 100.0 else 0.0
    return Mimbh

def calcMimbhEq10(Sigma_h: float, Z: float) -> float:
    """Eq.10: high-surface-density extrapolation."""
    Sigma_h = check_finite_positive(Sigma_h, name="Projected 2D half-mass surface density Sigma_h in M☉/pc²")
    lgZ     = math.log10(check_finite_positive(Z, name="Metallicity Z in Z☉"))

    if Z < 0.079:
        D, E = - 37.37 * lgZ + 1452.33, 81.66 * lgZ - 6892.57
    elif Z < 0.316:
        D, E = - 922.15 * lgZ + 466.71, 4242.58 * lgZ - 2280.02
    else:
        D, E = - 611.27 * lgZ + 628.40, 2620.25 * lgZ - 3137.93

    Mimbh = check_finite(D * math.log10(Sigma_h) + E, name="Eq.10 IMBH mass Mimbh in M☉")
    Mimbh = Mimbh if Mimbh >= 100.0 else 0.0
    return Mimbh

def imbh_mass_from_sigma_metallicity(sigma_h_msun_pc2: float, z_ratio: float) -> float:
    """Estimate IMBH mass from Sigma_h and metallicity Z/Zsun."""
    Sigma_h = check_finite_positive(sigma_h_msun_pc2, name="Projected 2D half-mass surface density Sigma_h in M☉/pc²")
    Z       = check_finite_positive(z_ratio, name="Metallicity Z in Z☉")
    #if Z < 0.01 or Z > 1.0:
    #    warnings.warn(f"Rantala+2026 IMBH fit evaluated outside 0.01 <= Z/Z☉ <= 1.0: Z/Z☉ = {Z:.6g}.",
    #        RuntimeWarning, stacklevel=2)

    Mimbh = calcMimbhEq10(Sigma_h, Z) if math.log10(Sigma_h) > 5.2 else calcMimbhEq9(Sigma_h, Z)
    return Mimbh if Mimbh >= 100.0 else 0.0

def imbh_mass_from_cluster_mass(Mcl: float, fit: str) -> float:
    """Estimate a Vergara+2026 IMBH mass from birth stellar cluster mass.

    Parameters
    ----------
    Mcl : float
        Birth stellar cluster mass in M☉.
    fit : str
        One of the two Vergara+2026 fit names.

    Returns
    -------
    float
        Formation-time IMBH mass in M☉.  The power law is evaluated directly
        for every finite positive input mass, without the Rantala 100 M☉ cut.
    """
    Mcl = check_finite_positive(Mcl, name="Birth stellar cluster mass Mcl in M☉")
    fit = validate_imbh_fit(fit)
    if fit == IMBH_FIT_VERGARA_CONSERVATIVE:
        intercept, slope = -2.0, 0.88
    elif fit == IMBH_FIT_VERGARA_OPTIMISTIC:
        intercept, slope = -0.76, 0.76
    else:
        raise ValueError(f"Mass-only IMBH helper does not support fit {fit!r}.")

    log_Mimbh = intercept + slope * math.log10(Mcl)
    Mimbh = 10.0**log_Mimbh
    return check_finite_positive(Mimbh, name="Vergara IMBH mass Mimbh in M☉")

def estimate_for_gc(Mcl: float, Z: float, fit: str = DEFAULT_IMBH_FIT) -> dict:
    """Full GC-level IMBH estimate.

    Parameters
    ----------
    Mcl : float
        GC mass in Msun.
    Z : float
        Metallicity ratio Z/Zsun.
    fit : str, optional
        Formation-time IMBH mass prescription.  The default is Rantala+2026.

    Returns
    -------
    dict
        Same output keys as the original class-based implementation.
    """
    Mcl = check_finite_positive(Mcl, name="Star cluster mass Mcl in M☉")
    Z   = check_finite_positive(Z, name="Metallicity Z in Z☉")
    fit = validate_imbh_fit(fit)

    r_h     = initMRRofSCs(Mcl=Mcl, f_h=0.125)
    Sigma_h = calcSigma_h(Mcl=Mcl, r_h=r_h)
    if fit == IMBH_FIT_RANTALA:
        Mimbh = imbh_mass_from_sigma_metallicity(sigma_h_msun_pc2=Sigma_h, z_ratio=Z)
    else:
        Mimbh = imbh_mass_from_cluster_mass(Mcl=Mcl, fit=fit)

    return {
        "r_h_pc": r_h,
        "sigma_h_msun_pc2": Sigma_h,
        "Z": Z,
        "imbh_mass_msun": Mimbh,
    }
