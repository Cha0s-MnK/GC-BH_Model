#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Self-contained Kong & Li 2026 suite b plot suite for the High-z SMBH Seeds project."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
import warnings

# CONFIGURE ENVIRONMENT
THREAD_CAP_DEFAULT = str(min(64, max(1, os.cpu_count() or 1)))
for env_name in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ.setdefault(env_name, THREAD_CAP_DEFAULT)

import matplotlib as mpl
mpl.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman",
                     #"font.size": 10,
                     "mathtext.default": "regular",
                     "xtick.direction": "in",
                     "ytick.direction": "in",
                     #"xtick.top": True,
                     #"ytick.right": True,
                     #"axes.grid": False,
                     "text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}"})
import numpy as np
import pandas as pd
from scipy import special


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    Sersic_coefs,
    Rv,
    ReducedH0,
    CosmicAge2Redshift,
    calcRe,
    G_Arepo,
    Mstar_SMHM,
    Redshift2CosmicAge,
    STD_DPI,
    check_finite,
    check_finite_non_negative,
    parse_fixed_tree_id,
    parse_exact_int64,
)
from evo import STAT_SUNK_GC, read_haloevo_mpb  # noqa: E402
from run import (  # noqa: E402
    VALID_EVOLUTION_STATUS,
    _interpolate_mpb_logmh_at_redshift,
    _mpb_branch_id,
    _read_full_tree_numeric,
)

# input params
DATA_ROOT = PROJECT_ROOT / "data"
FIGURE_01_FILENAME = "Fig.01_RotationCurve.pdf"
FIGURE_02_FILENAME = "Fig.02_BHmasses.pdf"
FIGURE_03_FILENAME = "Fig.03_UVmag.pdf"
FIGURE_04_FILENAME = "Fig.04_distr.pdf"
FIGURE_05_FILENAME = "Fig.05_assembly.pdf"
SCORE_FILENAME = "Fig.01_Fig.05_candidate_scores.csv"
FIG01_SCORE_COLUMNS = (
    "index",
    "halo_id_z0",
    "redshift",
    "nsc_mass_msun",
    "log10_nsc_mass",
    "central_bh_mass_msun",
    "log10_central_bh_mass",
    "formed_mass_6pc_msun",
    "weighted_age_gyr",
    "weighted_feh",
    "m1500_per_msun",
    "M_UV",
    "uv_term",
    "keplerian_term",
    "keplerian_chi2_weighted",
    "keplerian_n_points",
    "score_keplerian_uv",
    "n_gc_contributors",
    "n_uv_valid_contributors",
    "n_uv_age_nearest_grid",
    "n_uv_feh_nearest_grid",
    "n_uv_any_nearest_grid",
    "min_raw_age_gyr",
    "max_raw_age_gyr",
    "min_raw_feh",
    "max_raw_feh",
    "min_eval_age_gyr",
    "max_eval_age_gyr",
    "min_eval_feh",
    "max_eval_feh",
    "uv_table_path",
    "uv_mode",
    "missing_reason",
)
FIG01_TARGET_REDSHIFT = 7.04
FIG01_SERSIC_INDEX = 2.2
FIG01_REDSHIFT_ATOL = 0.1
FIG01_MATCH_RADIUS_RANGE_PC = (10.0, 160.0)
FIG01_SCATTER_PERCENTILES = (16.0, 84.0)
FIG01_VELOCITY_SIN_I = np.sin(np.radians(52.0)) # or 1.0 = sin(90)
FIG01_RADIUS_MAX_PC = 160.0
FIG01_HALO_MASS_WINDOW_DEX = 0.1
QSO1_REDSHIFT = 7.04
UV_AGE_REDSHIFT = 7.0
QSO1_MUV_AB = - 16.98
QSO1_MUV_TOL_MAG = 0.5
QSO1_MOKA3D_LOGMBH = 7.7
QSO1_MOKA3D_LOGMBH_ERR = 0.3
QSO1_DIRECT_LOWER_LIMIT_LOGM = 6.94
QSO1_NSC_APERTURE_PC = 6.0
QSO1_SCORE_WEIGHT_MUV = 1.0
QSO1_SCORE_WEIGHT_KEPLERIAN = 1.0
UV_APERTURES_PC = np.arange(1.0, 8.0, 1.0)
QSO1_VELOCITY_GROUP_WEIGHTS = (0.20, 0.20, 0.20, 0.20, 0.20)
FIG02_XLIM_LOGM = (5.3, 8.4)
UV_MODE_LABEL = "FSPS-MIST/Chabrier pure-stellar M1500(age,[Fe/H]) table"
UV_MIN_TABLE_AGE_GYR = 1.0e-4
UV_MIN_TABLE_FEH = -2.50
UV_MAX_TABLE_FEH = 0.50
UV_CALIBRATION_COLUMNS = ("age_gyr", "log10_age_gyr", "feh", "z_ratio", "M1500_AB_per_Msun")
UV_CALIBRATION_PATH = DATA_ROOT / "UV" / "fsps_mist_chabrier_m1500_grid.csv"
FIG04_DISTR_BIN_WIDTH_DEX = 0.5
FIG05_TARGET_REDSHIFT = 7.0
FIG05_REDSHIFT_ROW_ATOL = 1.0e-10
FIG05_MAX_PANELS = 9
FIG05_SATELLITE_CMAP = "jet"
FIG05_SCORE_RTOL = 1.0e-10
FIG05_SCORE_ATOL = 1.0e-10
TNG_CATALOGUE_ROOT = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark")
TNG_TARGET_MANIFEST_FILENAME = "target_manifest_dark.csv"
TNG_TARGET_METADATA_FILENAME = "targets_z0_dark.json"
TNG_TREE_LOOKUP_FILENAME = "halo_tree_lookup.csv"
TNG_FIXED_TREE_DIRNAME = "fixed_trees_large_spin_dark"
TNG_ORIGINAL_LOOKUP_FILENAME = "id_lookup_original.csv"
TNG_SHIFTED_LOOKUP_FILENAME = "id_lookup_large_dark.csv"
TNG_SUITE_KEYS = ("tng50_1_dark", "tng100_1_dark")
TNG100_HALO_ID_OFFSET = 1_000_000
JUODZBALIS2026_FIG3_BH_MASS_ROWS = [
    {
        "method": r"Furtak+24 virial H$\beta$",
        "group": "virial",
        "log10_mass": 7.60,
        "err_low": 0.25,
        "err_high": 0.25,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": r"Ji+25 virial H$\beta$",
        "group": "virial",
        "log10_mass": 7.59,
        "err_low": 0.20,
        "err_high": 0.20,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": r"D'Eugenio/Maiolino+25 virial H$\alpha$",
        "group": "virial",
        "log10_mass": 7.30,
        "err_low": 0.30,
        "err_high": 0.30,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": "Scattering-cocoon scenario",
        "group": "scattering",
        "log10_mass": 5.90,
        "err_low": 0.10,
        "err_high": 0.10,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": r"Bolometric luminosity, $L/L_{\rm Edd}=1$",
        "group": "bolometric",
        "log10_mass": 5.90,
        "err_low": 0.10,
        "err_high": 0.10,
        "is_lower_limit": False,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": "This work, spectroastrometry lower limit",
        "group": "direct",
        "log10_mass": QSO1_DIRECT_LOWER_LIMIT_LOGM,
        "err_low": 0.0,
        "err_high": 0.0,
        "is_lower_limit": True,
        "show_moka_band": False,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
    {
        "method": "This work, MOKA3D direct measurement",
        "group": "direct",
        "log10_mass": QSO1_MOKA3D_LOGMBH,
        "err_low": QSO1_MOKA3D_LOGMBH_ERR,
        "err_high": QSO1_MOKA3D_LOGMBH_ERR,
        "is_lower_limit": False,
        "show_moka_band": True,
        "source_note": "Approximate vector extraction from Juodzbalis+2026 source Fig. 3.",
    },
]


# HELPER FUNCTION(S)
def _stellar_mass_from_halo_mass_at_redshift(halo_mass, redshift):
    mh = np.asarray(halo_mass, dtype=float)
    z = np.asarray(redshift, dtype=float)
    mh_b, z_b = np.broadcast_arrays(mh, z)
    out = np.full(mh_b.shape, np.nan, dtype=float)
    for i, (mass, z_val) in enumerate(zip(mh_b.ravel(), z_b.ravel())):
        if np.isfinite(mass) and mass > 0.0 and np.isfinite(z_val) and z_val >= 0.0:
            out.ravel()[i] = Mstar_SMHM(float(mass), float(z_val), scatter=False)
    if np.isscalar(halo_mass) and np.isscalar(redshift):
        return float(out.reshape(-1)[0])
    return out
def _regular_log_bin_edges(values, step_dex):
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0, float(step_dex)], dtype=float)
    lo = float(step_dex) * math.floor(float(vals.min()) / float(step_dex))
    hi = float(step_dex) * math.ceil(float(vals.max()) / float(step_dex))
    if hi <= lo:
        hi = lo + float(step_dex)
    return np.arange(lo, hi + 0.5 * float(step_dex), float(step_dex), dtype=float)
def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
def _read_comment_columns(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                text = line[1:].strip()
                if text:
                    return text.split()
    raise ValueError(f"Cannot find header columns in {path}")
def _read_headered_whitespace_table(path):
    columns = _read_comment_columns(path)
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, dtype=str, engine="python")
    raw = raw.iloc[:, : len(columns)].copy()
    raw.columns = columns[: raw.shape[1]]
    return raw


def _integer_values(values, name, non_negative=False):
    """Parse exact signed int64 values without a floating-point round trip."""

    raw = np.asarray(values, dtype=object).reshape(-1)
    parsed = []
    for index, value in enumerate(raw):
        integer = parse_exact_int64(value, name=f"{name}[{index}]")
        if non_negative and integer < 0:
            raise ValueError(f"{name}[{index}] must be non-negative; got {integer}.")
        parsed.append(integer)
    return np.asarray(parsed, dtype=np.int64)


def _coerce_physical_columns(table, integer_columns):
    """Convert non-identifier columns in a parsed table to numeric values."""

    out = table.copy()
    for column in out.columns:
        if column not in integer_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
def _final_gcs_path(out_dir):
    path = Path(out_dir).resolve() / "finalGCs.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing final-GC catalogue: {path}")
    return path
def _halo_summary_by_z_path(out_dir):
    path = Path(out_dir).resolve() / "haloSummaryByZ.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing redshift-resolved halo summary: {path}")
    return path


def _deposit_path(out_dir):
    path = Path(out_dir).resolve() / "depos.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing deposit profile: {path}")
    return path


def _rename_existing_columns(table, mapping):
    return table.rename(columns={old: new for old, new in mapping.items() if old in table.columns and new not in table.columns})
def _add_aliases(table, aliases):
    out = table.copy()
    for alias, source in aliases.items():
        if source in out.columns and alias not in out.columns:
            out[alias] = out[source]
    return out
def load_run_metadata(out_dir):
    path = Path(out_dir).resolve() / "run_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
def load_halo_summary_by_z(out_dir):
    mapping = {
        "hid_z0": "halo_id_z0",
        "z_out": "redshift",
        "logMh_z_msun": "log10_halo_mass_at_redshift",
        "M_NSC": "nsc_mass_msun",
        "M_SMBH_init": "central_bh_mass_init_msun",
        "M_SMBH_final": "central_bh_mass_final_msun",
        "z_depos_sampled": "deposit_sample_redshift",
        "lookback_depos_sampled_gyr": "deposit_sample_lookback_gyr",
        "depos_time_match_delta_gyr": "deposit_sample_time_delta_gyr",
    }
    table = _rename_existing_columns(pd.read_csv(_halo_summary_by_z_path(out_dir), dtype=str), mapping)
    required = ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift", "central_bh_mass_final_msun"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing required columns after normalisation: {missing}")
    table["halo_id_z0"] = _integer_values(table["halo_id_z0"], "haloSummaryByZ halo IDs", non_negative=True)
    table = _coerce_physical_columns(table, {"halo_id_z0"})
    if table[["halo_id_z0", "redshift", "central_bh_mass_final_msun"]].isna().any().any():
        raise ValueError("haloSummaryByZ contains non-finite halo IDs, redshifts, or central BH masses.")
    bh = table["central_bh_mass_final_msun"].to_numpy(dtype=float)
    if np.any(bh < 0.0):
        raise ValueError("haloSummaryByZ contains negative central BH masses.")
    available = table["halo_mass_available"].to_numpy(dtype=float) == 1.0
    logmh = table["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    table["halo_mass_at_redshift_msun"] = np.nan
    valid = available & np.isfinite(logmh)
    table.loc[valid, "halo_mass_at_redshift_msun"] = np.power(10.0, logmh[valid])
    mstar = _stellar_mass_from_halo_mass_at_redshift(table["halo_mass_at_redshift_msun"].to_numpy(dtype=float), table["redshift"].to_numpy(dtype=float))
    table["mstar_z_smhm_msun"] = mstar
    table["logMstar_z_smhm_msun"] = np.where(np.isfinite(mstar) & (mstar > 0.0), np.log10(mstar), np.nan)
    return _add_aliases(
        table,
        {
            "hid_z0": "halo_id_z0",
            "z_out": "redshift",
            "logMh_z_msun": "log10_halo_mass_at_redshift",
            "M_NSC": "nsc_mass_msun",
            "M_SMBH_init": "central_bh_mass_init_msun",
            "M_SMBH_final": "central_bh_mass_final_msun",
            "z_depos_sampled": "deposit_sample_redshift",
            "lookback_depos_sampled_gyr": "deposit_sample_lookback_gyr",
            "depos_time_match_delta_gyr": "deposit_sample_time_delta_gyr",
        },
    )
def _build_fig04_halo_distribution(summary_by_z, best):
    required = ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift"]
    missing = [name for name in required if name not in summary_by_z.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing Fig. 04 distribution columns: {missing}")

    table = summary_by_z.loc[:, required].copy()
    table["halo_id_z0"] = _integer_values(table["halo_id_z0"], "Fig. 04 haloSummaryByZ halo IDs", non_negative=True)
    for column in required:
        if column != "halo_id_z0":
            table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id = table["halo_id_z0"].to_numpy(dtype=np.int64)
    redshift = table["redshift"].to_numpy(dtype=float)
    available_raw = table["halo_mass_available"].to_numpy(dtype=float)
    log10_halo_mass = table["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("Fig. 04 haloSummaryByZ contains non-finite or negative redshifts.")
    if np.any(~np.isfinite(available_raw)) or np.any(~np.isin(available_raw, np.asarray([0.0, 1.0]))):
        raise ValueError("Fig. 04 haloSummaryByZ halo_mass_available must contain only finite 0/1 values.")

    available = available_raw == 1.0
    key_table = pd.DataFrame({"halo_id_z0": halo_id, "redshift": redshift})
    if key_table.duplicated().any():
        duplicate_keys = key_table.loc[key_table.duplicated(keep=False)].drop_duplicates().to_dict("records")
        raise ValueError(f"Fig. 04 requires one haloSummaryByZ row per (halo_id_z0, redshift); duplicates={duplicate_keys[:10]}.")

    invalid_available_mass = available & ~np.isfinite(log10_halo_mass)
    if np.any(invalid_available_mass):
        bad_keys = key_table.loc[invalid_available_mass].to_dict("records")
        raise ValueError(f"Fig. 04 has unavailable numeric halo masses for rows marked available: {bad_keys[:10]}.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        halo_mass_msun = np.power(10.0, log10_halo_mass)
    valid_mass = available & np.isfinite(log10_halo_mass) & np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0)
    invalid_mass = available & ~valid_mass
    if np.any(invalid_mass):
        bad_keys = key_table.loc[invalid_mass].to_dict("records")
        raise ValueError(f"Fig. 04 has non-positive or non-finite available halo masses: {bad_keys[:10]}.")
    if not np.any(valid_mass):
        raise ValueError("Fig. 04 cannot plot a halo distribution because no valid MPB halo masses remain.")

    log_edges = _regular_log_bin_edges(log10_halo_mass[valid_mass], FIG04_DISTR_BIN_WIDTH_DEX)
    if len(log_edges) < 2 or np.any(~np.isfinite(log_edges)) or np.any(np.diff(log_edges) <= 0.0):
        raise ValueError("Fig. 04 generated invalid common logarithmic halo-mass bin edges.")
    all_redshifts = np.sort(np.unique(redshift))
    distributions = []
    empty_redshifts = []
    for z_value in all_redshifts:
        z_mask = redshift == float(z_value)
        valid_z = z_mask & valid_mass
        if not np.any(valid_z):
            empty_redshifts.append(float(z_value))
            continue
        samples_log = log10_halo_mass[valid_z]
        counts, _ = np.histogram(samples_log, bins=log_edges)
        if int(np.sum(counts)) != int(len(samples_log)):
            raise ValueError(f"Fig. 04 histogram dropped halo masses at z={float(z_value):.6g}.")
        distributions.append(
            {
                "redshift": float(z_value),
                "log10_halo_mass": samples_log.copy(),
                "halo_mass_msun": halo_mass_msun[valid_z].copy(),
                "counts": counts.astype(int, copy=False),
            }
        )
    if len(distributions) == 0:
        raise ValueError("Fig. 04 has no redshift snapshot with valid halo masses.")

    try:
        best_id = int(parse_exact_int64(best["halo_id_z0"], name="Fig. 04 best halo_id_z0"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fig. 04 best-halo selection has an invalid halo_id_z0.") from exc
    best_rows = halo_id == best_id
    if not np.any(best_rows):
        raise ValueError(f"Fig. 04 best halo_id_z0={best_id} is absent from haloSummaryByZ.")
    best_track = []
    best_missing_redshifts = []
    for z_value in all_redshifts:
        z_mask = best_rows & (redshift == float(z_value))
        if not np.any(z_mask):
            best_missing_redshifts.append(float(z_value))
            continue
        index = int(np.flatnonzero(z_mask)[0])
        if valid_mass[index]:
            best_track.append(
                {
                    "redshift": float(z_value),
                    "log10_halo_mass": float(log10_halo_mass[index]),
                    "halo_mass_msun": float(halo_mass_msun[index]),
                }
            )
        else:
            best_missing_redshifts.append(float(z_value))
    if len(best_track) == 0:
        raise ValueError(f"Fig. 04 best halo_id_z0={best_id} has no valid MPB halo mass at any output redshift.")

    return {
        "redshifts": np.asarray([item["redshift"] for item in distributions], dtype=float),
        "all_redshifts": all_redshifts.astype(float),
        "empty_redshifts": np.asarray(empty_redshifts, dtype=float),
        "log_bin_edges": log_edges,
        "distributions": distributions,
        "excluded_unavailable_rows": int(np.count_nonzero(~available)),
        "best_halo_id_z0": best_id,
        "best_halo_track": best_track,
        "best_halo_missing_redshifts": np.asarray(best_missing_redshifts, dtype=float),
    }
def load_final_gc(out_dir):
    mapping = {
        "M_IMBH_final": "imbh_mass_final_msun",
        "halo_id_z0": "halo_id_z0",
        "status": "status",
    }
    table = _rename_existing_columns(_read_headered_whitespace_table(_final_gcs_path(out_dir)), mapping)
    table = _add_aliases(table, {"M_IMBH_final": "imbh_mass_final_msun"})
    missing = [name for name in ["status", "M_IMBH_final"] if name not in table.columns]
    if missing:
        raise ValueError(f"Final-GC table is missing required columns: {missing}")
    if "halo_id_z0" in table.columns:
        table["halo_id_z0"] = _integer_values(table["halo_id_z0"], "finalGCs halo IDs", non_negative=True)
    if "gc_index_halo" in table.columns:
        table["gc_index_halo"] = _integer_values(table["gc_index_halo"], "finalGCs GC indices", non_negative=True)
    status = _integer_values(table["status"], "finalGCs statuses")
    table["status"] = status
    table = _coerce_physical_columns(table, {"halo_id_z0", "gc_index_halo", "status"})
    table["M_IMBH_final"] = pd.to_numeric(table["M_IMBH_final"], errors="coerce")
    if table["M_IMBH_final"].isna().any() or (table["M_IMBH_final"] < 0.0).any():
        raise ValueError("Final-GC table contains non-finite or negative M_IMBH_final values.")
    return table
def _fig05_allcat_path(out_dir):
    paths = sorted(Path(out_dir).resolve().glob("allcat_*.txt"))
    if len(paths) != 1:
        raise ValueError(
            f"Fig. 05 requires exactly one allcat_*.txt catalogue in {Path(out_dir).resolve()}, "
            f"found {len(paths)}: {[path.name for path in paths[:10]]}."
        )
    path = paths[0]
    if not path.is_file():
        raise ValueError(f"Fig. 05 allcat catalogue is not a regular file: {path}")
    return path
def _fig05_read_allcat_header(path):
    header = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            fields = stripped.lstrip("#").strip().split()
            if "hid_z0" in fields and "isMPB" in fields and "subfind_form" in fields:
                header = fields
                break
    if header is None or len(header) != len(set(header)):
        raise ValueError(f"Could not identify a unique allcat header in {path}.")
    required = ("hid_z0", "logMh_form", "zform", "isMPB", "subfind_form")
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"Fig. 05 allcat catalogue {path} is missing required fields: {missing}")
    return header
def _fig05_integer_values(values, name, non_negative=False):
    return _integer_values(values, f"Fig. 05 {name}", non_negative=non_negative)
def _load_fig05_formation_catalogue(out_dir, final_gc):
    """Read only the allcat fields needed for branch membership and auditing."""

    path = _fig05_allcat_path(out_dir)
    header = _fig05_read_allcat_header(path)
    required = ["hid_z0", "logMh_form", "zform", "isMPB", "subfind_form"]
    dtype = {name: str for name in required}
    try:
        table = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=header,
            usecols=required,
            dtype=dtype,
            engine="c",
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not read the required Fig. 05 allcat fields from {path}") from exc
    if table.empty:
        raise ValueError(f"Fig. 05 allcat catalogue is empty: {path}")

    halo_id = _fig05_integer_values(table["hid_z0"].to_numpy(dtype=object), "allcat halo IDs", non_negative=True)
    subfind_form = _fig05_integer_values(table["subfind_form"].to_numpy(dtype=object), "allcat subfind_form", non_negative=True)
    is_mpb = _fig05_integer_values(table["isMPB"].to_numpy(dtype=object), "allcat isMPB flags", non_negative=True)
    if not np.all(np.isin(is_mpb, np.asarray([0, 1], dtype=np.int64))):
        raise ValueError("Fig. 05 allcat isMPB flags must be exactly 0 or 1.")
    logmh_form = table["logMh_form"].to_numpy(dtype=float)
    zform = table["zform"].to_numpy(dtype=float)
    if np.any(~np.isfinite(logmh_form)) or np.any(~np.isfinite(zform)) or np.any(zform < 0.0):
        raise ValueError("Fig. 05 allcat formation log masses and redshifts must be finite, with zform >= 0.")

    final_required = ["halo_id_z0", "gc_index_halo", "status"]
    missing = [name for name in final_required if name not in final_gc.columns]
    if missing:
        raise ValueError(f"finalGCs.dat is missing the Fig. 05 alignment fields: {missing}")
    final_halo_id = _fig05_integer_values(final_gc["halo_id_z0"].to_numpy(dtype=object), "finalGCs halo IDs", non_negative=True)
    final_gc_index = _fig05_integer_values(final_gc["gc_index_halo"].to_numpy(dtype=object), "finalGCs GC indices", non_negative=True)
    final_status = _fig05_integer_values(final_gc["status"].to_numpy(dtype=object), "finalGCs statuses")
    if np.any(final_gc_index < 1):
        raise ValueError("Fig. 05 finalGCs GC indices must be positive and one-based.")
    if len(final_halo_id) != len(halo_id):
        raise ValueError(
            f"Fig. 05 allcat/finalGCs row counts disagree: allcat={len(halo_id)}, finalGCs={len(final_halo_id)}."
        )
    if not np.array_equal(final_halo_id, halo_id):
        mismatch = np.flatnonzero(final_halo_id != halo_id)
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(
            "Fig. 05 allcat/finalGCs parent halo IDs are not aligned "
            f"(first mismatch row={first}, allcat={int(halo_id[first]) if first >= 0 else 'n/a'}, "
            f"finalGCs={int(final_halo_id[first]) if first >= 0 else 'n/a'})."
        )

    expected_gc_index = np.empty(len(halo_id), dtype=np.int64)
    for hid in np.unique(halo_id):
        rows = np.flatnonzero(halo_id == int(hid))
        expected_gc_index[rows] = np.arange(1, len(rows) + 1, dtype=np.int64)
    if not np.array_equal(final_gc_index, expected_gc_index):
        mismatch = np.flatnonzero(final_gc_index != expected_gc_index)
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(
            "Fig. 05 allcat/finalGCs GC indices are not aligned "
            f"(first mismatch row={first}, expected={int(expected_gc_index[first]) if first >= 0 else 'n/a'}, "
            f"finalGCs={int(final_gc_index[first]) if first >= 0 else 'n/a'})."
        )

    return {
        "path": path,
        "halo_id_z0": halo_id,
        "logMh_form": logmh_form,
        "zform": zform,
        "isMPB": is_mpb,
        "subfind_form": subfind_form,
        "status": final_status,
    }
def _fig05_fixed_tree_path(fixed_tree_basename, tree_root=None):
    root = (Path(tree_root) if tree_root is not None else TNG_CATALOGUE_ROOT / TNG_FIXED_TREE_DIRNAME).resolve()
    basename = str(fixed_tree_basename).strip()
    if not basename or Path(basename).is_absolute():
        raise ValueError(f"Fig. 05 fixed-tree basename must be a non-empty relative path: {fixed_tree_basename!r}")
    path = (root / basename).resolve()
    try:
        inside_root = os.path.commonpath([str(root), str(path)]) == str(root)
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"Fig. 05 fixed tree escapes the declared tree directory: {fixed_tree_basename!r}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Fig. 05 fixed tree is missing or not a regular file: {path}")
    return path
def _fig05_read_full_tree_numeric(path):
    """Validate raw fixed-tree rows before using the shared project reader."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first = stripped.split()[0].lower()
            if first.startswith("logmh") or first == "log10_mhalo_msun":
                continue
            parts = stripped.split()
            if len(parts) < 9:
                raise ValueError(f"Fig. 05 raw fixed tree has fewer than nine columns at line {line_no}: {path}")
            try:
                float(parts[0])
                parse_fixed_tree_id(parts[1], name=f"{path} first progenitor ID", allow_minus_one=True)
                parse_fixed_tree_id(parts[2], name=f"{path} subhalo ID")
                parse_fixed_tree_id(parts[3], name=f"{path} branch ID")
                parse_fixed_tree_id(parts[4], name=f"{path} descendant ID", allow_minus_one=True)
                float(parts[5])
                float(parts[6])
                float(parts[7])
                float(parts[8])
            except ValueError as exc:
                raise ValueError(f"Fig. 05 raw fixed tree has a malformed numeric row at line {line_no}: {path}") from exc
    with warnings.catch_warnings():
        # The shared reader recognises the legacy logMh header but the
        # current fixed-tree files use log10_mhalo_msun for the same header.
        warnings.filterwarnings(
            "ignore",
            message="Malformed fixed-tree row .*row=log10_mhalo_msun first_progenitor_id.*",
            category=RuntimeWarning,
        )
        rows = _read_full_tree_numeric(path)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError(f"Fig. 05 raw fixed tree contains no numeric rows: {path}")
    return rows
def _fig05_tree_logmass_and_redshift(tree_rows, context):
    rows = np.asarray(tree_rows, dtype=object)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 6:
        raise ValueError(f"Fig. 05 {context} fixed-tree rows are empty or malformed: shape={rows.shape}")
    logmh = np.asarray(rows[:, 0], dtype=float)
    redshift = np.asarray(rows[:, 5], dtype=float)
    if np.any(~np.isfinite(logmh)) or np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError(f"Fig. 05 {context} fixed-tree log masses and redshifts must be finite, with z >= 0.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mass = np.power(10.0, logmh)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError(f"Fig. 05 {context} fixed-tree halo masses must be finite and positive.")
    return logmh, redshift, mass
def _fig05_main_track(path, mpb_rows=None):
    if mpb_rows is None:
        mpb_rows = read_haloevo_mpb(path)
    rows = np.asarray(mpb_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 5:
        raise ValueError(f"Fig. 05 raw MPB is empty or malformed: {path}")
    logmh = np.asarray(rows[:, 0], dtype=float)
    redshift = np.asarray(rows[:, 1], dtype=float)
    if np.any(~np.isfinite(logmh)) or np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError(f"Fig. 05 raw MPB has non-finite log mass/redshift values: {path}")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mass = np.power(10.0, logmh)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError(f"Fig. 05 raw MPB masses must be finite and positive: {path}")
    if float(np.max(redshift)) < FIG05_TARGET_REDSHIFT - FIG05_REDSHIFT_ROW_ATOL or float(np.min(redshift)) > FIG05_TARGET_REDSHIFT + FIG05_REDSHIFT_ROW_ATOL:
        raise ValueError(f"Fig. 05 raw MPB does not bracket z={FIG05_TARGET_REDSHIFT:g}: {path}")

    exact = np.flatnonzero(np.abs(redshift - FIG05_TARGET_REDSHIFT) <= FIG05_REDSHIFT_ROW_ATOL)
    if len(exact) > 1:
        raise ValueError(f"Fig. 05 raw MPB contains duplicate z={FIG05_TARGET_REDSHIFT:g} rows: {path}")
    if len(exact) == 1:
        endpoint_logmh = float(logmh[int(exact[0])])
    else:
        endpoint_logmh, available = _interpolate_mpb_logmh_at_redshift(rows, FIG05_TARGET_REDSHIFT)
        if int(available) != 1 or not np.isfinite(endpoint_logmh):
            raise ValueError(f"Fig. 05 raw MPB cannot be interpolated to z={FIG05_TARGET_REDSHIFT:g}: {path}")

    visible = redshift >= FIG05_TARGET_REDSHIFT
    if len(exact) == 1:
        visible[int(exact[0])] = False
    if not np.any(visible) and len(exact) == 0:
        raise ValueError(f"Fig. 05 raw MPB has no raw rows above z={FIG05_TARGET_REDSHIFT:g}: {path}")
    visible_logmh = logmh[visible]
    visible_redshift = redshift[visible]
    visible_time = np.asarray([Redshift2CosmicAge(float(z), time_unit="Gyr") for z in visible_redshift], dtype=float)
    order = np.argsort(visible_time, kind="mergesort")
    visible_logmh = visible_logmh[order]
    visible_redshift = visible_redshift[order]
    visible_time = visible_time[order]
    visible_x = float(Redshift2CosmicAge(0.0, time_unit="Gyr")) - visible_time
    endpoint_time = float(Redshift2CosmicAge(FIG05_TARGET_REDSHIFT, time_unit="Gyr"))
    endpoint_x = float(Redshift2CosmicAge(0.0, time_unit="Gyr")) - endpoint_time
    track_logmh = np.concatenate([visible_logmh, np.asarray([endpoint_logmh], dtype=float)])
    track_redshift = np.concatenate([visible_redshift, np.asarray([FIG05_TARGET_REDSHIFT], dtype=float)])
    track_x = np.concatenate([visible_x, np.asarray([endpoint_x], dtype=float)])
    track_mass = np.power(10.0, track_logmh)
    if np.any(~np.isfinite(track_mass)) or np.any(track_mass <= 0.0) or np.any(track_redshift < FIG05_TARGET_REDSHIFT):
        raise ValueError(f"Fig. 05 raw MPB produced an invalid z >= 7 track: {path}")
    return {
        "x_gyr": track_x,
        "redshift": track_redshift,
        "log10_halo_mass": track_logmh,
        "halo_mass_msun": track_mass,
        "endpoint_log10_halo_mass": float(endpoint_logmh),
    }
def _fig05_map_formation_rows_to_branches(logmh_form, zform, subfind_form, tree_rows):
    logmh_form = np.asarray(logmh_form, dtype=float)
    zform = np.asarray(zform, dtype=float)
    subfind_form = _fig05_integer_values(subfind_form, "formation subfind IDs", non_negative=True)
    if len(logmh_form) != len(zform) or len(logmh_form) != len(subfind_form):
        raise ValueError("Fig. 05 formation fields have inconsistent lengths.")
    if np.any(~np.isfinite(logmh_form)) or np.any(~np.isfinite(zform)) or np.any(zform < 0.0):
        raise ValueError("Fig. 05 formation fields must be finite, with zform >= 0.")
    rows = np.asarray(tree_rows, dtype=object)
    candidates_by_subfind = {}
    for row in rows:
        branch = int(row[3])
        subfind = int(row[2])
        if branch < 0:
            raise ValueError(f"Fig. 05 fixed-tree branch IDs must be non-negative; got {branch}")
        candidates_by_subfind.setdefault(subfind, []).append((branch, float(row[5]), float(row[0])))
    branch_ids = np.empty(len(logmh_form), dtype=np.int64)
    for index, (logmh, z_value, subfind) in enumerate(zip(logmh_form, zform, subfind_form)):
        candidates = candidates_by_subfind.get(int(subfind), ())
        if not candidates:
            raise ValueError(
                f"Fig. 05 cannot map formation row {index}: subfind_form={int(subfind)} is absent from the raw tree."
            )
        scored = [
            (abs(float(z_tree) - float(z_value)) + abs(float(logmh_tree) - float(logmh)), int(branch), float(z_tree), float(logmh_tree))
            for branch, z_tree, logmh_tree in candidates
        ]
        scored.sort(key=lambda item: item)
        best_score, best_branch, _, _ = scored[0]
        if best_score > 1.0e-3:
            raise ValueError(
                f"Fig. 05 cannot robustly map formation row {index} to a raw-tree branch; "
                f"nearest score={best_score:.6g}, branch={best_branch}."
            )
        branch_ids[index] = int(best_branch)
    return branch_ids
def _fig05_satellite_tracks(tree_rows, mpb_branch, branch_ids, formation):
    branch_ids = np.asarray(branch_ids, dtype=np.int64)
    statuses = np.asarray(formation["status"], dtype=np.int64)
    zform = np.asarray(formation["zform"], dtype=float)
    valid_status = np.isin(statuses, np.asarray(sorted(VALID_EVOLUTION_STATUS), dtype=np.int64))
    high_z_gc = valid_status & np.isfinite(zform) & (zform >= FIG05_TARGET_REDSHIFT)
    tree_branch = np.asarray(tree_rows[:, 3], dtype=np.int64)
    tree_logmh, tree_redshift, tree_mass = _fig05_tree_logmass_and_redshift(tree_rows, "satellite")
    tracks = []
    for branch in sorted(set(int(value) for value in tree_branch if int(value) != int(mpb_branch))):
        branch_gc_count = int(np.count_nonzero(high_z_gc & (branch_ids == int(branch))))
        visible = (tree_branch == int(branch)) & (tree_redshift >= FIG05_TARGET_REDSHIFT)
        if not np.any(visible):
            # Omit a branch with no visible raw-tree row from both the plot and
            # the reported satellite count.
            continue
        visible_logmh = tree_logmh[visible]
        visible_redshift = tree_redshift[visible]
        visible_mass = tree_mass[visible]
        visible_time = np.asarray([Redshift2CosmicAge(float(z), time_unit="Gyr") for z in visible_redshift], dtype=float)
        order = np.argsort(visible_time, kind="mergesort")
        visible_logmh = visible_logmh[order]
        visible_redshift = visible_redshift[order]
        visible_mass = visible_mass[order]
        visible_time = visible_time[order]
        x_gyr = float(Redshift2CosmicAge(0.0, time_unit="Gyr")) - visible_time
        maximum_logmh = float(np.max(visible_logmh))
        maximum_indices = np.flatnonzero(visible_logmh == maximum_logmh)
        marker_index = int(maximum_indices[-1])
        tracks.append(
            {
                "branch_id": int(branch),
                "n_gc_high_z": branch_gc_count,
                "x_gyr": x_gyr,
                "redshift": visible_redshift,
                "log10_halo_mass": visible_logmh,
                "halo_mass_msun": visible_mass,
                "marker_x_gyr": float(x_gyr[marker_index]),
                "marker_log10_halo_mass": maximum_logmh,
                "marker_halo_mass_msun": float(visible_mass[marker_index]),
            }
        )
    return tracks
def _fig05_load_history(candidate, formation):
    tree_path = _fig05_fixed_tree_path(candidate["fixed_tree_basename"])
    tree_rows = _fig05_read_full_tree_numeric(tree_path)
    _fig05_tree_logmass_and_redshift(tree_rows, "full")
    mpb_branch = _mpb_branch_id(tree_rows)
    mpb_rows = read_haloevo_mpb(tree_path)
    main = _fig05_main_track(tree_path, mpb_rows=mpb_rows)
    row_indices = np.flatnonzero(formation["halo_id_z0"] == int(candidate["halo_id_z0"]))
    if len(row_indices):
        branch_ids = _fig05_map_formation_rows_to_branches(
            formation["logMh_form"][row_indices],
            formation["zform"][row_indices],
            formation["subfind_form"][row_indices],
            tree_rows,
        )
        allcat_is_mpb = formation["isMPB"][row_indices].astype(bool)
        expected_is_mpb = branch_ids == int(mpb_branch)
        if not np.array_equal(allcat_is_mpb, expected_is_mpb):
            mismatch = np.flatnonzero(allcat_is_mpb != expected_is_mpb)
            first = int(mismatch[0]) if len(mismatch) else -1
            raise ValueError(
                f"Fig. 05 allcat isMPB disagrees with raw-tree branch membership for halo {int(candidate['halo_id_z0'])}; "
                f"first local mismatch={first}."
            )
        satellite_tracks = _fig05_satellite_tracks(
            tree_rows,
            mpb_branch,
            branch_ids,
            {
                "status": formation["status"][row_indices],
                "zform": formation["zform"][row_indices],
            },
        )
    else:
        satellite_tracks = _fig05_satellite_tracks(
            tree_rows,
            mpb_branch,
            np.asarray([], dtype=np.int64),
            {
                "status": np.asarray([], dtype=np.int64),
                "zform": np.asarray([], dtype=float),
            },
        )
    history = dict(candidate)
    history.update(
        {
            "tree_path": tree_path,
            "mpb_branch_id": int(mpb_branch),
            "main": main,
            "satellites": satellite_tracks,
            "n_satellites": len(satellite_tracks),
        }
    )
    return history
def _fig05_lookup_table(tng_volume_context):
    lookup = tng_volume_context.get("lookup") if isinstance(tng_volume_context, dict) else None
    if not isinstance(lookup, pd.DataFrame):
        raise ValueError("Fig. 05 requires the validated TNG lookup in tng_volume_context.")
    required = ["halo_id_z0", "simulation_key", "fixed_tree_basename"]
    missing = [name for name in required if name not in lookup.columns]
    if missing:
        raise ValueError(f"Fig. 05 TNG lookup is missing required provenance fields: {missing}")
    out = lookup.loc[:, [name for name in lookup.columns if name in required + ["simulation"]]].copy()
    out["halo_id_z0"] = _fig05_integer_values(out["halo_id_z0"].to_numpy(dtype=object), "TNG lookup halo IDs", non_negative=True)
    out["simulation_key"] = out["simulation_key"].fillna("").astype(str).str.strip()
    out["fixed_tree_basename"] = out["fixed_tree_basename"].fillna("").astype(str).str.strip()
    if "simulation" in out.columns:
        out["simulation"] = out["simulation"].fillna("").astype(str).str.strip()
    if out["simulation_key"].eq("").any() or out["fixed_tree_basename"].eq("").any():
        raise ValueError("Fig. 05 TNG lookup contains blank suite or fixed-tree provenance values.")
    if not set(out["simulation_key"]).issubset(set(TNG_SUITE_KEYS)):
        raise ValueError(f"Fig. 05 TNG lookup contains unsupported suites: {sorted(set(out['simulation_key']) - set(TNG_SUITE_KEYS))}")
    if out["halo_id_z0"].duplicated().any():
        raise ValueError("Fig. 05 TNG lookup contains duplicate model-facing halo IDs.")
    return out
def _fig05_exact_catalogue_rows(summary_by_z, best_id):
    column_aliases = {
        "halo_id_z0": "halo_id_z0",
        "redshift": "redshift" if "redshift" in summary_by_z.columns else "z_out",
        "halo_mass_available": "halo_mass_available",
        "log10_halo_mass_at_redshift": "log10_halo_mass_at_redshift" if "log10_halo_mass_at_redshift" in summary_by_z.columns else "logMh_z_msun",
    }
    missing = [name for name, source in column_aliases.items() if source not in summary_by_z.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing Fig. 05 catalogue fields: {missing}")
    table = summary_by_z.loc[:, list(column_aliases.values())].copy()
    table.columns = list(column_aliases.keys())
    table["halo_id_z0"] = _fig05_integer_values(table["halo_id_z0"].to_numpy(dtype=object), "haloSummaryByZ halo IDs", non_negative=True)
    for column in table.columns:
        if column != "halo_id_z0":
            table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id = table["halo_id_z0"].to_numpy(dtype=np.int64)
    redshift = table["redshift"].to_numpy(dtype=float)
    available = table["halo_mass_available"].to_numpy(dtype=float)
    logmh = table["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("haloSummaryByZ contains invalid redshifts for Fig. 05.")
    exact = np.abs(redshift - FIG05_TARGET_REDSHIFT) <= FIG05_REDSHIFT_ROW_ATOL
    if not np.any(exact):
        raise ValueError(f"haloSummaryByZ contains no exact z={FIG05_TARGET_REDSHIFT:g} catalogue row.")
    exact_table = pd.DataFrame(
        {
            "halo_id_z0": halo_id[exact],
            "catalogue_redshift": redshift[exact],
            "halo_mass_available": available[exact],
            "catalogue_log10_halo_mass": logmh[exact],
        }
    )
    if exact_table["halo_id_z0"].duplicated().any():
        duplicate_ids = exact_table.loc[exact_table["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].drop_duplicates().tolist()
        raise ValueError(f"haloSummaryByZ contains duplicate exact z={FIG05_TARGET_REDSHIFT:g} rows: {duplicate_ids[:10]}")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exact_mass = np.power(10.0, exact_table["catalogue_log10_halo_mass"].to_numpy(dtype=float))
    exact_table["catalogue_halo_mass_msun"] = exact_mass
    best_rows = exact_table.loc[exact_table["halo_id_z0"] == int(best_id)]
    if len(best_rows) != 1:
        raise ValueError(f"Fig. 05 Fig. 01 best halo {int(best_id)} lacks one unique exact z=7 catalogue row.")
    best_row = best_rows.iloc[0]
    if float(best_row["halo_mass_available"]) != 1.0 or not np.isfinite(float(best_row["catalogue_log10_halo_mass"])) or not np.isfinite(float(best_row["catalogue_halo_mass_msun"])) or float(best_row["catalogue_halo_mass_msun"]) <= 0.0:
        raise ValueError(f"Fig. 05 Fig. 01 best halo {int(best_id)} has an invalid exact z=7 catalogue mass gate row.")
    exact_table["catalogue_mass_valid"] = (
        (exact_table["halo_mass_available"] == 1.0)
        & np.isfinite(exact_table["catalogue_log10_halo_mass"])
        & np.isfinite(exact_table["catalogue_halo_mass_msun"])
        & (exact_table["catalogue_halo_mass_msun"] > 0.0)
    )
    return exact_table, best_row
def _fig05_candidate_row(row, suite_label=None):
    simulation_key = str(row["simulation_key"])
    if suite_label is None:
        suite_label = {"tng50_1_dark": "TNG50", "tng100_1_dark": "TNG100"}.get(simulation_key, simulation_key)
    candidate = row.to_dict()
    candidate["halo_id_z0"] = int(row["halo_id_z0"])
    candidate["catalogue_halo_mass_msun"] = float(row["catalogue_halo_mass_msun"])
    candidate["catalogue_log10_halo_mass"] = float(row["catalogue_log10_halo_mass"])
    candidate["suite_label"] = str(suite_label)
    return candidate
def _fig05_unavailable_score_fields():
    return {
        "score_keplerian": np.nan,
        "score_uv": np.nan,
        "score_keplerian_uv": np.nan,
        "score_keplerian_available": False,
        "score_uv_available": False,
        "score_keplerian_uv_available": False,
    }
def _fig05_score_lookup(score_table, fig01_best):
    """Join validated Fig. 01 scores to Fig. 05 histories by integer halo ID."""

    if not isinstance(score_table, pd.DataFrame):
        raise ValueError("Fig. 05 score propagation requires the Fig. 01 score table as a pandas DataFrame.")
    redshift_column = "redshift" if "redshift" in score_table.columns else "z_out" if "z_out" in score_table.columns else None
    required = ["halo_id_z0", "keplerian_term", "uv_term", "score_keplerian_uv"]
    if redshift_column is None:
        required.append("redshift")
    else:
        required.append(redshift_column)
    missing = [name for name in required if name not in score_table.columns]
    if missing:
        raise ValueError(f"Fig. 05 score table is missing required columns: {missing}")

    table = score_table.loc[:, ["halo_id_z0", redshift_column, "keplerian_term", "uv_term", "score_keplerian_uv"]].copy()
    table["halo_id_z0"] = _fig05_integer_values(table["halo_id_z0"].to_numpy(dtype=object), "Fig. 05 score halo IDs", non_negative=True)
    for column in table.columns:
        if column != "halo_id_z0":
            table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id = table["halo_id_z0"].to_numpy(dtype=np.int64)
    redshift = table[redshift_column].to_numpy(dtype=float)
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("Fig. 05 score rows contain non-finite or negative redshifts.")
    keplerian = table["keplerian_term"].to_numpy(dtype=float)
    uv = table["uv_term"].to_numpy(dtype=float)
    combined = table["score_keplerian_uv"].to_numpy(dtype=float)
    if np.any(np.isfinite(keplerian) & (keplerian < 0.0)):
        raise ValueError("Fig. 05 score table contains a finite negative keplerian_term.")
    if np.any(np.isfinite(combined) & (combined < 0.0)):
        raise ValueError("Fig. 05 score table contains a finite negative score_keplerian_uv.")

    target = np.abs(redshift - FIG05_TARGET_REDSHIFT) <= FIG05_REDSHIFT_ROW_ATOL
    target_table = table.loc[target].copy()
    if len(target_table) == 0:
        raise ValueError(f"Fig. 05 score table contains no row at target z={FIG05_TARGET_REDSHIFT:g}.")
    if target_table["halo_id_z0"].duplicated().any():
        duplicate_ids = target_table.loc[target_table["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].drop_duplicates().tolist()
        raise ValueError(f"Fig. 05 score table contains duplicate target-redshift rows for halo IDs: {duplicate_ids[:10]}")

    target_keplerian = target_table["keplerian_term"].to_numpy(dtype=float)
    target_uv = target_table["uv_term"].to_numpy(dtype=float)
    target_combined = target_table["score_keplerian_uv"].to_numpy(dtype=float)
    finite_combined_missing_component = np.isfinite(target_combined) & (~np.isfinite(target_keplerian) | ~np.isfinite(target_uv))
    if np.any(finite_combined_missing_component):
        bad_ids = target_table.loc[finite_combined_missing_component, "halo_id_z0"].to_numpy(dtype=np.int64).tolist()
        raise ValueError(f"Fig. 05 finite combined scores require finite keplerian and UV components: halo IDs={bad_ids[:10]}")
    finite_all = np.isfinite(target_keplerian) & np.isfinite(target_uv) & np.isfinite(target_combined)
    if np.any(finite_all):
        expected_combined = np.sqrt(
            QSO1_SCORE_WEIGHT_KEPLERIAN * target_keplerian[finite_all] ** 2
            + QSO1_SCORE_WEIGHT_MUV * target_uv[finite_all] ** 2
        )
        if not np.allclose(target_combined[finite_all], expected_combined, rtol=FIG05_SCORE_RTOL, atol=FIG05_SCORE_ATOL):
            bad_ids = target_table.loc[finite_all, "halo_id_z0"].to_numpy(dtype=np.int64)
            mismatch = ~np.isclose(target_combined[finite_all], expected_combined, rtol=FIG05_SCORE_RTOL, atol=FIG05_SCORE_ATOL)
            raise ValueError(f"Fig. 05 score table violates the weighted score formula for halo IDs={bad_ids[mismatch][:10].tolist()}")

    try:
        best_id = int(parse_exact_int64(fig01_best["halo_id_z0"], name="Fig. 05 Fig. 01 best halo ID"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fig. 05 cannot validate the Fig. 01 best-halo score without an exact halo_id_z0.") from exc
    best_rows = target_table.loc[target_table["halo_id_z0"] == best_id]
    if len(best_rows) != 1:
        raise ValueError(f"Fig. 05 Fig. 01 best halo {best_id} lacks one unique target-redshift score row.")
    best_row = best_rows.iloc[0]
    for column in ("keplerian_term", "uv_term", "score_keplerian_uv"):
        try:
            best_value = float(fig01_best[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Fig. 05 Fig. 01 best-halo score is missing {column}.") from exc
        joined_value = float(best_row[column])
        if not np.isfinite(best_value) or not np.isfinite(joined_value) or not np.isclose(joined_value, best_value, rtol=FIG05_SCORE_RTOL, atol=FIG05_SCORE_ATOL):
            raise ValueError(
                f"Fig. 05 best-halo {column} mismatch for halo_id_z0={best_id}: "
                f"joined={joined_value!r}, Fig. 01={best_value!r}."
            )

    by_halo_id = {}
    for row in target_table.itertuples(index=False):
        row_keplerian = float(row.keplerian_term)
        row_uv = float(row.uv_term)
        row_combined = float(row.score_keplerian_uv)
        keplerian_available = bool(np.isfinite(row_keplerian))
        uv_available = bool(np.isfinite(row_uv))
        combined_available = bool(np.isfinite(row_combined) and keplerian_available and uv_available)
        by_halo_id[int(row.halo_id_z0)] = {
            "score_keplerian": row_keplerian if keplerian_available else np.nan,
            "score_uv": row_uv if uv_available else np.nan,
            "score_keplerian_uv": row_combined if combined_available else np.nan,
            "score_keplerian_available": keplerian_available,
            "score_uv_available": uv_available,
            "score_keplerian_uv_available": combined_available,
        }
    return {"by_halo_id": by_halo_id, "target_rows": target_table, "target_redshift": FIG05_TARGET_REDSHIFT}
def _fig05_score_text(value, available):
    return f"{float(value):.2f}" if bool(available) and np.isfinite(float(value)) else "n/a"
def _fig05_score_annotation(history):
    keplerian = _fig05_score_text(history["score_keplerian"], history["score_keplerian_available"])
    uv = _fig05_score_text(history["score_uv"], history["score_uv_available"])
    combined_available = (
        history["score_keplerian_available"]
        and history["score_uv_available"]
        and history["score_keplerian_uv_available"]
    )
    combined = _fig05_score_text(history["score_keplerian_uv"], combined_available)
    keplerian_text = rf"$S_{{\rm K}}={keplerian}$" if keplerian != "n/a" else r"$S_{\rm K}$=n/a"
    uv_text = rf"$S_{{\rm UV}}={uv}$" if uv != "n/a" else r"$S_{\rm UV}$=n/a"
    combined_text = rf"$S_{{\rm K+UV}}={combined}$" if combined != "n/a" else r"$S_{\rm K+UV}$=n/a"
    return f"{keplerian_text}, {uv_text}\n{combined_text}"


def _fig05_score_diagnostic(history):
    keplerian = _fig05_score_text(history["score_keplerian"], history["score_keplerian_available"])
    uv = _fig05_score_text(history["score_uv"], history["score_uv_available"])
    combined_available = (
        history["score_keplerian_available"]
        and history["score_uv_available"]
        and history["score_keplerian_uv_available"]
    )
    combined = _fig05_score_text(history["score_keplerian_uv"], combined_available)
    return f"S_K={keplerian}, S_UV={uv}, S_K+UV={combined}"


def select_fig05_assembly_histories(out_dir, summary_by_z, final_gc, metadata, tng_volume_context, score_table, fig01_best):
    """Select Fig. 05 panels and build their raw-tree assembly histories."""

    if not isinstance(metadata, dict):
        raise ValueError("Fig. 05 requires the validated run metadata mapping.")
    try:
        best_id = int(parse_exact_int64(fig01_best["halo_id_z0"], name="Fig. 05 Fig. 01 best halo ID"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fig. 05 cannot read the Fig. 01 best halo ID.") from exc
    score_lookup = _fig05_score_lookup(score_table, fig01_best)
    scores_by_halo_id = score_lookup["by_halo_id"]
    formation = _load_fig05_formation_catalogue(out_dir, final_gc)
    exact_table, best_row = _fig05_exact_catalogue_rows(summary_by_z, best_id)
    lookup = _fig05_lookup_table(tng_volume_context)
    missing_provenance = sorted(set(exact_table["halo_id_z0"].tolist()) - set(lookup["halo_id_z0"].tolist()))
    if missing_provenance:
        raise ValueError(f"Fig. 05 exact z=7 catalogue rows lack strict TNG provenance: {missing_provenance[:10]}")
    candidate_table = exact_table.merge(lookup, on="halo_id_z0", how="left", validate="one_to_one")
    best_lookup = candidate_table.loc[candidate_table["halo_id_z0"] == best_id]
    if len(best_lookup) != 1:
        raise ValueError(f"Fig. 05 Fig. 01 best halo {best_id} lacks one unique strict TNG provenance row.")
    best_candidate = _fig05_candidate_row(best_lookup.iloc[0])
    # The best halo is a fatal gate: the first panel cannot be replaced by a
    # nearby halo if its raw MPB is incomplete at z=7.
    best_history = _fig05_load_history(best_candidate, formation)
    best_history.update(scores_by_halo_id.get(best_id, _fig05_unavailable_score_fields()))

    best_mass = float(best_row["catalogue_halo_mass_msun"])
    comparisons = candidate_table.loc[
        candidate_table["catalogue_mass_valid"] & (candidate_table["halo_id_z0"] != best_id)
    ].copy()
    comparisons["mass_delta_msun"] = np.abs(comparisons["catalogue_halo_mass_msun"].to_numpy(dtype=float) - best_mass)
    comparisons = comparisons.sort_values(
        ["mass_delta_msun", "halo_id_z0", "fixed_tree_basename"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    histories = [best_history]
    rejected = []
    for _, row in comparisons.iterrows():
        if len(histories) >= FIG05_MAX_PANELS:
            break
        candidate = _fig05_candidate_row(row)
        try:
            history = _fig05_load_history(candidate, formation)
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
            rejected.append({"halo_id_z0": int(candidate["halo_id_z0"]), "reason": str(exc)})
            continue
        history.update(scores_by_halo_id.get(int(candidate["halo_id_z0"]), _fig05_unavailable_score_fields()))
        histories.append(history)
    return {
        "histories": histories,
        "rejected": rejected,
        "formation_catalogue_path": formation["path"],
        "best_halo_id_z0": best_id,
        "target_redshift": FIG05_TARGET_REDSHIFT,
    }
def _read_tng_lookup(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing TNG lookup: {path}")
    lookup = pd.read_csv(path, dtype=str)
    required = [
        "file_index",
        "simulation",
        "simulation_key",
        "halo_id_z0",
        "subhalo_id_z0",
        "label",
        "raw_tree_basename",
        "fixed_tree_basename",
    ]
    missing = [name for name in required if name not in lookup.columns]
    if missing:
        raise ValueError(f"{path} is missing TNG lookup columns: {missing}")
    for column in ("file_index", "halo_id_z0", "subhalo_id_z0"):
        lookup[column] = _integer_values(lookup[column], f"{path} {column}", non_negative=True)
    for column in ("simulation", "simulation_key", "label", "raw_tree_basename", "fixed_tree_basename"):
        lookup[column] = lookup[column].fillna("").astype(str).str.strip()
        if lookup[column].eq("").any():
            raise ValueError(f"{path} contains empty {column} values.")
    if not set(lookup["simulation_key"]).issubset(set(TNG_SUITE_KEYS)):
        raise ValueError(f"{path} contains an unsupported TNG suite.")
    if lookup.duplicated(["simulation_key", "fixed_tree_basename"]).any():
        raise ValueError(f"{path} contains duplicate suite/fixed-tree provenance keys.")
    return lookup
def _load_tng_catalogue_provenance():
    manifest_path = TNG_CATALOGUE_ROOT / TNG_TARGET_MANIFEST_FILENAME
    metadata_path = TNG_CATALOGUE_ROOT / TNG_TARGET_METADATA_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing TNG target manifest: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing TNG target metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    simulations = metadata.get("simulations")
    full_box = metadata.get("full_box_selection")
    if not isinstance(simulations, dict) or not isinstance(full_box, dict):
        raise ValueError("TNG metadata must contain simulations and full_box_selection.")
    if full_box.get("geometry") != "native_full_simulation_box" or bool(full_box.get("coordinate_filter_applied")) or bool(full_box.get("periodic_wrapping")):
        raise ValueError("TNG metadata does not describe unfiltered native full-box selection.")

    volume_by_suite = {}
    for simulation_key in TNG_SUITE_KEYS:
        spec = simulations.get(simulation_key)
        if not isinstance(spec, dict):
            raise ValueError(f"TNG metadata is missing {simulation_key}.")
        h = float(spec.get("h", np.nan))
        box_size_ckpc_h = float(spec.get("box_size_ckpc_h", np.nan))
        if not np.isfinite(h) or h <= 0.0 or not np.isfinite(box_size_ckpc_h) or box_size_ckpc_h <= 0.0:
            raise ValueError(f"TNG metadata has invalid h or box size for {simulation_key}.")
        side_native = box_size_ckpc_h / 1000.0
        side_physical = side_native / h
        if not np.isclose(float(full_box["side_native_cmpc_h"][simulation_key]), side_native, rtol=0.0, atol=1.0e-10) or not np.isclose(float(full_box["side_physical_cmpc"][simulation_key]), side_physical, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"TNG full-box side metadata is inconsistent for {simulation_key}.")
        if not np.isclose(float(full_box["volume_physical_cmpc3"][simulation_key]), side_physical**3, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"TNG full-box physical volume metadata is inconsistent for {simulation_key}.")
        volume_by_suite[simulation_key] = float(full_box["volume_physical_cmpc3"][simulation_key])

    rules = {
        "tng50_1_dark": "full_box_Group_M_Mean200_gt_1e10_msun_and_le_1e13_msun",
        "tng100_1_dark": "full_box_Group_M_Mean200_gt_1e13_msun",
    }
    try:
        manifest = pd.read_csv(manifest_path, dtype=str)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read TNG target manifest: {manifest_path}") from error
    required_manifest = ["simulation", "simulation_key", "halo_id_z0", "subhalo_id_z0", "selection_rule", "raw_tree_basename", "fixed_tree_basename"]
    missing = [name for name in required_manifest if name not in manifest.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing TNG manifest columns: {missing}")
    manifest["simulation_key"] = manifest["simulation_key"].fillna("").astype(str).str.strip()
    manifest["selection_rule"] = manifest["selection_rule"].fillna("").astype(str).str.strip()
    manifest["fixed_tree_basename"] = manifest["fixed_tree_basename"].fillna("").astype(str).str.strip()
    manifest["raw_tree_basename"] = manifest["raw_tree_basename"].fillna("").astype(str).str.strip()
    manifest["halo_id_z0"] = _integer_values(manifest["halo_id_z0"], f"{manifest_path} halo_id_z0", non_negative=True)
    manifest["subhalo_id_z0"] = _integer_values(manifest["subhalo_id_z0"], f"{manifest_path} subhalo_id_z0", non_negative=True)
    if not set(manifest["simulation_key"]).issubset(set(TNG_SUITE_KEYS)):
        raise ValueError(f"{manifest_path} contains an unsupported TNG suite.")
    if manifest.duplicated("fixed_tree_basename").any():
        raise ValueError(f"{manifest_path} contains duplicate fixed-tree basenames.")
    for simulation_key, selection_rule in rules.items():
        rows = manifest.loc[manifest["simulation_key"].eq(simulation_key)]
        if len(rows) and set(rows["selection_rule"]) != {selection_rule}:
            raise ValueError(f"{manifest_path} contains an unexpected selection rule for {simulation_key}.")
    metadata_counts = metadata.get("counts", {}).get("selected_by_simulation", {})
    manifest_counts = {key: int((manifest["simulation_key"] == key).sum()) for key in TNG_SUITE_KEYS}
    for key in TNG_SUITE_KEYS:
        if int(metadata_counts.get(key, -1)) != manifest_counts[key]:
            raise ValueError(f"TNG metadata and manifest counts disagree for {key}.")

    fixed_tree_dir = TNG_CATALOGUE_ROOT / TNG_FIXED_TREE_DIRNAME
    original = _read_tng_lookup(fixed_tree_dir / TNG_ORIGINAL_LOOKUP_FILENAME)
    shifted = _read_tng_lookup(fixed_tree_dir / TNG_SHIFTED_LOOKUP_FILENAME)
    expected = manifest.set_index(["simulation_key", "fixed_tree_basename"], drop=False)
    original = original.set_index(["simulation_key", "fixed_tree_basename"], drop=False)
    shifted = shifted.set_index(["simulation_key", "fixed_tree_basename"], drop=False)
    if len(original) != len(manifest) or len(shifted) != len(manifest) or set(original.index) != set(expected.index) or set(shifted.index) != set(expected.index):
        raise ValueError("TNG lookup files do not contain exactly one row per current manifest target.")
    final_ids = []
    for key in expected.index:
        manifest_row = expected.loc[key]
        original_row = original.loc[key]
        shifted_row = shifted.loc[key]
        if int(original_row["halo_id_z0"]) != int(manifest_row["halo_id_z0"]):
            raise ValueError(f"Original lookup halo ID disagrees with manifest for {key}.")
        required_shifted = int(manifest_row["halo_id_z0"]) + (TNG100_HALO_ID_OFFSET if key[0] == "tng100_1_dark" else 0)
        if int(shifted_row["halo_id_z0"]) != required_shifted:
            raise ValueError(f"Shifted lookup halo ID disagrees with the required offset for {key}.")
        final_ids.append(required_shifted)
    if len(final_ids) != len(set(final_ids)):
        raise ValueError("The model-facing TNG lookup contains duplicate final halo IDs.")
    return {
        "manifest": manifest.reset_index(drop=True),
        "metadata": metadata,
        "lookup_original": original.reset_index(drop=True),
        "lookup_final": shifted.reset_index(drop=True),
        "volume_tng50_cmpc3": volume_by_suite["tng50_1_dark"],
        "volume_tng100_cmpc3": volume_by_suite["tng100_1_dark"],
        "tng100_weight": float(volume_by_suite["tng50_1_dark"] / volume_by_suite["tng100_1_dark"]),
        "manifest_counts": manifest_counts,
    }
def _load_tng_tree_lookup(out_dir, catalogue):
    path = Path(out_dir).resolve() / TNG_TREE_LOOKUP_FILENAME
    lookup = pd.read_csv(path, dtype=str) if path.exists() else None
    if lookup is None:
        raise FileNotFoundError(f"Missing required TNG halo-tree lookup: {path}")
    if "halo_id_z0" not in lookup.columns and "hid_z0" in lookup.columns:
        lookup = lookup.rename(columns={"hid_z0": "halo_id_z0"})
    required = ["halo_id_z0", "simulation_key", "fixed_tree_basename"]
    missing = [name for name in required if name not in lookup.columns]
    if missing:
        raise ValueError(f"{path} is missing TNG halo-tree lookup columns: {missing}")
    lookup["halo_id_z0"] = _integer_values(lookup["halo_id_z0"], f"{path} halo IDs", non_negative=True)
    lookup["simulation_key"] = lookup["simulation_key"].fillna("").astype(str).str.strip()
    lookup["fixed_tree_basename"] = lookup["fixed_tree_basename"].fillna("").astype(str).str.strip()
    if lookup["simulation_key"].eq("").any() or lookup["fixed_tree_basename"].eq("").any() or lookup.duplicated(["simulation_key", "fixed_tree_basename"]).any():
        raise ValueError(f"{path} contains invalid or duplicate suite/fixed-tree provenance.")
    catalogue_lookup = catalogue["lookup_final"][["simulation_key", "fixed_tree_basename", "halo_id_z0"]].rename(columns={"halo_id_z0": "catalogue_halo_id_z0"})
    joined = lookup.merge(catalogue_lookup, on=["simulation_key", "fixed_tree_basename"], how="left", validate="one_to_one")
    if joined["catalogue_halo_id_z0"].isna().any():
        missing_names = joined.loc[joined["catalogue_halo_id_z0"].isna(), "fixed_tree_basename"].tolist()
        raise ValueError(f"{path} contains fixed-tree names absent from the shifted catalogue lookup: {missing_names[:10]}")
    mismatch = joined["halo_id_z0"].to_numpy(dtype=np.int64) != joined["catalogue_halo_id_z0"].to_numpy(dtype=np.int64)
    if np.any(mismatch):
        raise ValueError("Model halo_tree_lookup.csv does not use the shifted catalogue halo-ID namespace.")
    return joined
def attach_tng_volume_weights(out_dir, summary_by_z, final_gc):
    catalogue = _load_tng_catalogue_provenance()
    lookup = _load_tng_tree_lookup(out_dir, catalogue)
    summary_ids_int = _integer_values(summary_by_z["halo_id_z0"], "haloSummaryByZ IDs", non_negative=True)
    lookup_id_set = set(lookup["halo_id_z0"].tolist())
    missing_summary_ids = sorted(set(summary_ids_int.tolist()) - lookup_id_set)
    if missing_summary_ids:
        raise ValueError(f"TNG halo-tree lookup is missing haloSummaryByZ IDs: {missing_summary_ids[:10]}")

    weight_by_halo = dict(zip(lookup["halo_id_z0"].tolist(), np.where(lookup["simulation_key"].eq("tng100_1_dark"), catalogue["tng100_weight"], 1.0)))
    weighted_summary = summary_by_z.copy()
    weighted_summary["volume_weight_tng50"] = [float(weight_by_halo[int(halo_id)]) for halo_id in summary_ids_int]

    if "halo_id_z0" not in final_gc.columns:
        raise ValueError("finalGCs.dat must contain halo_id_z0 for strict TNG Satellite volume weighting.")
    final_ids_int = _integer_values(final_gc["halo_id_z0"], "finalGCs parent halo IDs", non_negative=True)
    missing_final_ids = sorted(set(final_ids_int.tolist()) - lookup_id_set)
    if missing_final_ids:
        raise ValueError(f"TNG halo-tree lookup is missing finalGCs.dat parent IDs: {missing_final_ids[:10]}")
    weighted_final_gc = final_gc.copy()
    weighted_final_gc["volume_weight_tng50"] = [float(weight_by_halo[int(halo_id)]) for halo_id in final_ids_int]

    output_counts = {key: int(value) for key, value in lookup["simulation_key"].value_counts().to_dict().items()}
    return weighted_summary, weighted_final_gc, {
        "lookup": lookup,
        "manifest_counts": catalogue["manifest_counts"],
        "output_counts": output_counts,
        "volume_tng50_cmpc3": float(catalogue["volume_tng50_cmpc3"]),
        "volume_tng100_cmpc3": float(catalogue["volume_tng100_cmpc3"]),
        "tng100_weight": float(catalogue["tng100_weight"]),
    }
def _set_tng_volume_context(context):
    """Use the current full-box metadata for all physical-density figures."""

    global BHMF_REFERENCE_SIDE_CMPC, BHMF_REFERENCE_VOLUME_CMPC3
    volume_tng50 = float(context["volume_tng50_cmpc3"])
    if not np.isfinite(volume_tng50) or volume_tng50 <= 0.0:
        raise ValueError("TNG50 full-box volume must be finite and positive.")
    BHMF_REFERENCE_VOLUME_CMPC3 = volume_tng50
    BHMF_REFERENCE_SIDE_CMPC = float(volume_tng50 ** (1.0 / 3.0))
def _read_csv_required(path, required, numeric=()):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required cached data file: {path}")
    table = pd.read_csv(path)
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in numeric:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if numeric and table[list(numeric)].isna().any().any():
        raise ValueError(f"{path} contains non-finite numeric values in required columns.")
    return table
def load_uv_calibration(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing FSPS UV calibration table: {path}")
    table = pd.read_csv(path, comment="#")
    missing = [name for name in UV_CALIBRATION_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required FSPS UV columns: {missing}")
    table = table.loc[:, list(UV_CALIBRATION_COLUMNS)].copy()
    for col in UV_CALIBRATION_COLUMNS:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if table[list(UV_CALIBRATION_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} contains non-finite FSPS UV calibration values.")
    if np.any(table["age_gyr"].to_numpy(dtype=float) <= 0.0) or np.any(table["z_ratio"].to_numpy(dtype=float) <= 0.0):
        raise ValueError(f"{path} contains non-positive age_gyr or z_ratio values.")

    log_age_unique = np.sort(table["log10_age_gyr"].unique().astype(float))
    age_unique = np.sort(table["age_gyr"].unique().astype(float))
    feh_unique = np.sort(table["feh"].unique().astype(float))
    if len(log_age_unique) != len(age_unique):
        raise ValueError(f"{path} has inconsistent age_gyr and log10_age_gyr axes.")
    if len(table) != len(log_age_unique) * len(feh_unique):
        raise ValueError(f"{path} does not form a complete rectangular age-[Fe/H] grid.")
    if table.duplicated(subset=["log10_age_gyr", "feh"]).any():
        raise ValueError(f"{path} contains duplicate age-[Fe/H] grid nodes.")

    age_by_log = table.sort_values("log10_age_gyr").drop_duplicates("log10_age_gyr")
    age_axis = age_by_log["age_gyr"].to_numpy(dtype=float)
    log_age_axis = age_by_log["log10_age_gyr"].to_numpy(dtype=float)
    if not np.all(np.diff(log_age_axis) > 0.0) or not np.all(np.diff(age_axis) > 0.0):
        raise ValueError(f"{path} age axis is not strictly increasing.")
    if not np.allclose(np.log10(age_axis), log_age_axis, rtol=0.0, atol=2.0e-10):
        raise ValueError(f"{path} has inconsistent log10_age_gyr values.")

    sorted_table = table.sort_values(["log10_age_gyr", "feh"])
    expected_log_age = np.repeat(log_age_axis, len(feh_unique))
    expected_feh = np.tile(feh_unique, len(log_age_axis))
    if not np.allclose(sorted_table["log10_age_gyr"].to_numpy(dtype=float), expected_log_age, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{path} age rows are not rectangular after sorting.")
    if not np.allclose(sorted_table["feh"].to_numpy(dtype=float), expected_feh, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{path} [Fe/H] rows are not rectangular after sorting.")
    if not np.allclose(sorted_table["z_ratio"].to_numpy(dtype=float), np.power(10.0, expected_feh), rtol=2.0e-10, atol=1.0e-12):
        raise ValueError(f"{path} z_ratio values are inconsistent with feh = log10(Z/Zsun).")

    m1500_grid = sorted_table["M1500_AB_per_Msun"].to_numpy(dtype=float).reshape(len(log_age_axis), len(feh_unique))
    if np.any(~np.isfinite(m1500_grid)):
        raise ValueError(f"{path} contains non-finite M1500_AB_per_Msun values.")
    if not np.isclose(float(feh_unique[0]), UV_MIN_TABLE_FEH, rtol=0.0, atol=1.0e-8) or not np.isclose(float(feh_unique[-1]), UV_MAX_TABLE_FEH, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"{path} metallicity bounds must be {UV_MIN_TABLE_FEH:.2f} <= feh <= {UV_MAX_TABLE_FEH:.2f}.")
    if float(age_axis[0]) > UV_MIN_TABLE_AGE_GYR * (1.0 + 1.0e-8):
        raise ValueError(f"{path} minimum age {float(age_axis[0]):.6g} Gyr exceeds the planned minimum {UV_MIN_TABLE_AGE_GYR:.6g} Gyr.")

    return {
        "path": path,
        "age_gyr": age_axis,
        "log10_age_gyr": log_age_axis,
        "feh": feh_unique,
        "m1500_grid": m1500_grid,
        "age_min_gyr": float(age_axis[0]),
        "age_max_gyr": float(age_axis[-1]),
        "feh_min": float(feh_unique[0]),
        "feh_max": float(feh_unique[-1]),
        "uv_mode": UV_MODE_LABEL,
    }
def load_juodzbalis2026_fig2():
    point_path = DATA_ROOT / "Juodzbalis+2026Fig2" / "juodzbalis2026_fig2_points.csv"
    curve_path = DATA_ROOT / "Juodzbalis+2026Fig2" / "juodzbalis2026_fig2_curves.csv"
    point_cols = ["component", "r_pc", "r_err_low_pc", "r_err_high_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s", "source_kind", "source_note"]
    curve_cols = ["curve", "r_pc", "v_km_s", "log10_mass_reference", "chi2_reduced", "source_kind", "source_note"]
    points = _read_csv_required(point_path, point_cols, numeric=["r_pc", "r_err_low_pc", "r_err_high_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"])
    curves = _read_csv_required(curve_path, curve_cols, numeric=["r_pc", "v_km_s", "log10_mass_reference", "chi2_reduced"])
    if sorted(set(["resolved_kinematics", "spectroastrometry", "spectroastrometry_fine"]) - set(points["component"])):
        raise ValueError("Juodzbalis Fig. 2 point table is missing one or more expected components.")
    if sorted(set(["point_mass_keplerian", "mw_nsc"]) - set(curves["curve"])):
        raise ValueError("Juodzbalis Fig. 2 curve table is missing one or more expected curves.")
    if np.any(curves["r_pc"].to_numpy(dtype=float) == 0.0):
        raise ValueError("Juodzbalis Fig. 2 curve table must not contain r_pc == 0 rows.")
    err_cols = ["r_err_low_pc", "r_err_high_pc", "v_err_low_km_s", "v_err_high_km_s"]
    if np.any(points[err_cols].dropna().to_numpy(dtype=float) < 0.0):
        raise ValueError("Juodzbalis Fig. 2 point table contains negative error values.")
    return points, curves
def load_juodzbalis2026_fig3_bh_masses():
    table = pd.DataFrame(JUODZBALIS2026_FIG3_BH_MASS_ROWS)
    required = ["method", "group", "log10_mass", "err_low", "err_high", "is_lower_limit", "show_moka_band", "source_note"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"Juodzbalis+2026 Fig. 3 table is missing columns: {missing}")
    table["log10_mass"] = [check_finite(value, name="Juodzbalis Fig. 3 log10 mass") for value in table["log10_mass"]]
    for name in ["err_low", "err_high"]:
        table[name] = [check_finite_non_negative(value, name=f"Juodzbalis Fig. 3 {name}") for value in table[name]]
    table["is_lower_limit"] = table["is_lower_limit"].map(_as_bool)
    table["show_moka_band"] = table["show_moka_band"].map(_as_bool)
    if len(table) != 7:
        raise ValueError("Juodzbalis+2026 Fig. 3 table should contain exactly seven reference rows.")
    return table
def _lookback_to_z0_gyr(redshift):
    return Redshift2CosmicAge(0.0) - Redshift2CosmicAge(float(redshift))
def interpolate_formed_mass_inside_aperture(deposit_profile, halo_id, aperture_pc=QSO1_NSC_APERTURE_PC):
    if "cumulative_formed_mass_msun" not in deposit_profile:
        return np.nan
    halo_ids = _integer_values(deposit_profile["halo_ids"], "deposit halo IDs", non_negative=True)
    matches = np.flatnonzero(halo_ids == np.int64(parse_exact_int64(halo_id, name="deposit lookup halo ID")))
    if len(matches) != 1:
        return np.nan
    index = int(matches[0])
    rout_pc = np.asarray(deposit_profile["r_outer_kpc"][index], dtype=float) * 1.0e3
    cumulative = np.asarray(deposit_profile["cumulative_formed_mass_msun"][index], dtype=float)
    if (
        len(rout_pc) == 0
        or len(rout_pc) != len(cumulative)
        or np.any(~np.isfinite(rout_pc))
        or np.any(~np.isfinite(cumulative))
        or np.any(cumulative < 0.0)
        or np.any(np.diff(rout_pc) <= 0.0)
        or float(aperture_pc) < rout_pc[0]
        or float(aperture_pc) > rout_pc[-1]
    ):
        return np.nan
    return float(np.interp(float(aperture_pc), rout_pc, cumulative))
def select_qso1_gc_contributors(final_gc, halo_id, lookback_qso1_gyr):
    required = ["halo_id_z0", "status", "M_GC_final", "m_init_msun", "lookback_time_final_gyr", "lookback_time_init_gyr", "feh"]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        return final_gc.iloc[0:0].copy(), f"final-GC catalogue missing {missing}"
    halo_ids = _integer_values(final_gc["halo_id_z0"], "final-GC parent halo IDs", non_negative=True)
    rows = final_gc.loc[halo_ids == np.int64(parse_exact_int64(halo_id, name="QSO1 final-GC halo ID"))].copy()
    if len(rows) == 0:
        return rows, "no final-GC rows for halo"
    status = _integer_values(rows["status"], "selected final-GC statuses")
    lookback_final = pd.to_numeric(rows["lookback_time_final_gyr"], errors="coerce").to_numpy(dtype=float)
    lookback_init = pd.to_numeric(rows["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    mask = (
        np.isfinite(status)
        & (status == STAT_SUNK_GC)
        & np.isfinite(lookback_final)
        & np.isfinite(lookback_init)
        & (lookback_final >= float(lookback_qso1_gyr))
        & (lookback_init >= float(lookback_qso1_gyr))
    )
    contributors = rows.loc[mask].copy()
    if len(contributors) == 0:
        return contributors, "no clean STAT_SUNK_GC contributors before QSO1"
    return contributors, ""
def _blank_uv_result(formed_mass_msun, contributors, uv_calibration, missing_reason=""):
    formed_mass = float(formed_mass_msun) if np.isfinite(formed_mass_msun) else np.nan
    return {
        "formed_mass_msun": formed_mass,
        "weighted_age_gyr": np.nan,
        "weighted_feh": np.nan,
        "m1500_per_msun": np.nan,
        "M_UV": np.nan,
        "n_contributors": int(len(contributors)),
        "n_uv_valid_contributors": 0,
        "n_uv_age_nearest_grid": 0,
        "n_uv_feh_nearest_grid": 0,
        "n_uv_any_nearest_grid": 0,
        "min_raw_age_gyr": np.nan,
        "max_raw_age_gyr": np.nan,
        "min_raw_feh": np.nan,
        "max_raw_feh": np.nan,
        "min_eval_age_gyr": np.nan,
        "max_eval_age_gyr": np.nan,
        "min_eval_feh": np.nan,
        "max_eval_feh": np.nan,
        "uv_table_path": str(uv_calibration["path"]),
        "uv_mode": str(uv_calibration["uv_mode"]),
        "missing_reason": missing_reason,
    }
def _interpolate_uv_m1500(uv_calibration, age_gyr, feh):
    age_raw = np.asarray(age_gyr, dtype=float)
    feh_raw = np.asarray(feh, dtype=float)
    if age_raw.shape != feh_raw.shape:
        raise ValueError("UV interpolation age and [Fe/H] arrays must have the same shape.")

    log_age_grid = np.asarray(uv_calibration["log10_age_gyr"], dtype=float)
    age_grid = np.asarray(uv_calibration["age_gyr"], dtype=float)
    feh_grid = np.asarray(uv_calibration["feh"], dtype=float)
    m1500_grid = np.asarray(uv_calibration["m1500_grid"], dtype=float)
    if len(log_age_grid) < 2 or len(feh_grid) < 2:
        raise ValueError("FSPS UV calibration must have at least two age and [Fe/H] grid points.")

    m1500 = np.full(age_raw.shape, np.nan, dtype=float)
    eval_age = np.full(age_raw.shape, np.nan, dtype=float)
    eval_feh = np.full(age_raw.shape, np.nan, dtype=float)
    valid = np.isfinite(age_raw) & (age_raw > 0.0) & np.isfinite(feh_raw)
    age_used_nearest = valid & ((age_raw < age_grid[0]) | (age_raw > age_grid[-1]))
    feh_used_nearest = valid & ((feh_raw < feh_grid[0]) | (feh_raw > feh_grid[-1]))
    if np.any(feh_used_nearest):
        warnings.warn(
            f"FSPS UV metallicity outside native MIST grid clipped to {feh_grid[0]:.2f} <= feh <= {feh_grid[-1]:.2f}.",
            RuntimeWarning,
            stacklevel=2,
        )

    eval_age[valid] = np.clip(age_raw[valid], age_grid[0], age_grid[-1])
    eval_feh[valid] = np.clip(feh_raw[valid], feh_grid[0], feh_grid[-1])
    log_age_eval = np.full(age_raw.shape, np.nan, dtype=float)
    log_age_eval[valid] = np.log10(eval_age[valid])

    flat_valid = np.flatnonzero(valid.ravel())
    for flat_index in flat_valid:
        idx = np.unravel_index(int(flat_index), age_raw.shape)
        x = float(log_age_eval[idx])
        y = float(eval_feh[idx])
        i = int(np.clip(np.searchsorted(log_age_grid, x, side="right") - 1, 0, len(log_age_grid) - 2))
        j = int(np.clip(np.searchsorted(feh_grid, y, side="right") - 1, 0, len(feh_grid) - 2))
        x0, x1 = float(log_age_grid[i]), float(log_age_grid[i + 1])
        y0, y1 = float(feh_grid[j]), float(feh_grid[j + 1])
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        f00 = float(m1500_grid[i, j])
        f10 = float(m1500_grid[i + 1, j])
        f01 = float(m1500_grid[i, j + 1])
        f11 = float(m1500_grid[i + 1, j + 1])
        m1500[idx] = (1.0 - tx) * (1.0 - ty) * f00 + tx * (1.0 - ty) * f10 + (1.0 - tx) * ty * f01 + tx * ty * f11

    return {
        "M1500_AB_per_Msun": m1500,
        "age_eval_gyr": eval_age,
        "feh_eval": eval_feh,
        "valid": valid,
        "age_used_nearest_grid": age_used_nearest,
        "feh_used_nearest_grid": feh_used_nearest,
        "any_used_nearest_grid": age_used_nearest | feh_used_nearest,
    }
def estimate_old_nsc_uv_mag(formed_mass_msun, contributors, lookback_eval_gyr, uv_calibration):
    result = _blank_uv_result(formed_mass_msun, contributors, uv_calibration)
    if not (np.isfinite(formed_mass_msun) and float(formed_mass_msun) > 0.0):
        result["missing_reason"] = "formed aperture mass unavailable"
        return result
    if len(contributors) == 0:
        result["missing_reason"] = "no usable GC-origin age weights"
        return result

    lookback_init = pd.to_numeric(contributors["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    age_gyr = lookback_init - float(lookback_eval_gyr)
    weights = pd.to_numeric(contributors["M_GC_final"], errors="coerce").to_numpy(dtype=float)
    fallback = pd.to_numeric(contributors["m_init_msun"], errors="coerce").to_numpy(dtype=float)
    use_fallback = ~np.isfinite(weights) | (weights <= 0.0)
    weights = weights.copy()
    weights[use_fallback] = fallback[use_fallback]
    feh = pd.to_numeric(contributors["feh"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(age_gyr) & (age_gyr > 0.0) & np.isfinite(feh) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        result["missing_reason"] = "no finite positive GC-origin UV age, [Fe/H], and weight tuples"
        return result

    age_sel = age_gyr[valid]
    feh_sel = feh[valid]
    weight_sel = weights[valid]
    interp = _interpolate_uv_m1500(uv_calibration, age_sel, feh_sel)
    m1500 = np.asarray(interp["M1500_AB_per_Msun"], dtype=float)
    interp_valid = np.isfinite(m1500)
    if not np.any(interp_valid):
        result["missing_reason"] = "non-finite FSPS UV interpolation"
        return result

    age_sel = age_sel[interp_valid]
    feh_sel = feh_sel[interp_valid]
    weight_sel = weight_sel[interp_valid]
    m1500 = m1500[interp_valid]
    eval_age = np.asarray(interp["age_eval_gyr"], dtype=float)[interp_valid]
    eval_feh = np.asarray(interp["feh_eval"], dtype=float)[interp_valid]
    age_nearest = np.asarray(interp["age_used_nearest_grid"], dtype=bool)[interp_valid]
    feh_nearest = np.asarray(interp["feh_used_nearest_grid"], dtype=bool)[interp_valid]
    any_nearest = np.asarray(interp["any_used_nearest_grid"], dtype=bool)[interp_valid]

    luminosity_per_msun = np.power(10.0, -0.4 * m1500)
    weighted_luminosity = float(np.average(luminosity_per_msun, weights=weight_sel))
    if not (np.isfinite(weighted_luminosity) and weighted_luminosity > 0.0):
        result["missing_reason"] = "non-finite composite stellar UV luminosity"
        return result

    m1500_per_msun = float(-2.5 * np.log10(weighted_luminosity))
    m_uv = float(m1500_per_msun - 2.5 * np.log10(float(formed_mass_msun)))

    result.update(
        {
            "weighted_age_gyr": float(np.average(age_sel, weights=weight_sel)),
            "weighted_feh": float(np.average(feh_sel, weights=weight_sel)),
            "m1500_per_msun": m1500_per_msun,
            "M_UV": m_uv,
            "n_uv_valid_contributors": int(len(age_sel)),
            "n_uv_age_nearest_grid": int(np.sum(age_nearest)),
            "n_uv_feh_nearest_grid": int(np.sum(feh_nearest)),
            "n_uv_any_nearest_grid": int(np.sum(any_nearest)),
            "min_raw_age_gyr": float(np.min(age_sel)),
            "max_raw_age_gyr": float(np.max(age_sel)),
            "min_raw_feh": float(np.min(feh_sel)),
            "max_raw_feh": float(np.max(feh_sel)),
            "min_eval_age_gyr": float(np.min(eval_age)),
            "max_eval_age_gyr": float(np.max(eval_age)),
            "min_eval_feh": float(np.min(eval_feh)),
            "max_eval_feh": float(np.max(eval_feh)),
            "missing_reason": "",
        }
    )
    return result
def estimate_uv_magnitude_apertures(deposit_profile, final_gc, halo_id, uv_calibration):
    selection_lookback_gyr = _lookback_to_z0_gyr(QSO1_REDSHIFT)
    age_lookback_gyr = _lookback_to_z0_gyr(UV_AGE_REDSHIFT)
    contributors, contributor_reason = select_qso1_gc_contributors(final_gc, halo_id, selection_lookback_gyr)
    if contributor_reason:
        raise ValueError(f"Fig. 03 selected halo {int(halo_id)} has no usable 7.04-selected UV contributors: {contributor_reason}.")

    rows = []
    for aperture_pc in UV_APERTURES_PC:
        formed_mass = interpolate_formed_mass_inside_aperture(deposit_profile, halo_id, float(aperture_pc))
        uv = estimate_old_nsc_uv_mag(formed_mass, contributors, age_lookback_gyr, uv_calibration)
        if not (np.isfinite(formed_mass) and float(formed_mass) > 0.0):
            raise ValueError(f"Fig. 03 selected halo {int(halo_id)} has no positive initially formed mass at R_UV={float(aperture_pc):.1f} pc.")
        if uv["missing_reason"] or not np.isfinite(float(uv["M_UV"])):
            reason = uv["missing_reason"] or "non-finite UV magnitude"
            raise ValueError(f"Fig. 03 selected halo {int(halo_id)} cannot evaluate R_UV={float(aperture_pc):.1f} pc: {reason}.")
        row = {"aperture_pc": float(aperture_pc)}
        row.update({key: value for key, value in uv.items() if key != "formed_mass_msun"})
        row["formed_mass_msun"] = float(formed_mass)
        rows.append(row)

    aperture_table = pd.DataFrame(rows)
    expected_apertures = np.arange(1.0, 8.0, 1.0)
    actual_apertures = aperture_table["aperture_pc"].to_numpy(dtype=float)
    if len(aperture_table) != len(expected_apertures) or not np.array_equal(actual_apertures, expected_apertures):
        raise ValueError(f"Fig. 03 selected halo {int(halo_id)} aperture series must contain exactly 1--7 pc, got {actual_apertures.tolist()}.")
    formed_mass_values = aperture_table["formed_mass_msun"].to_numpy(dtype=float)
    uv_values = aperture_table["M_UV"].to_numpy(dtype=float)
    if np.any(~np.isfinite(formed_mass_values)) or np.any(~np.isfinite(uv_values)):
        raise ValueError(f"Fig. 03 selected halo {int(halo_id)} aperture series contains non-finite values.")
    mass_tolerance = 1.0e-12 * max(1.0, float(np.max(formed_mass_values)))
    if np.any(np.diff(formed_mass_values) < -mass_tolerance):
        raise ValueError(f"Fig. 03 selected halo {int(halo_id)} initially formed mass is not cumulative over 1--7 pc.")
    if np.any(np.diff(uv_values) > 1.0e-10):
        raise ValueError(f"Fig. 03 selected halo {int(halo_id)} UV magnitude becomes fainter at larger aperture.")
    return aperture_table
def _fig01_weighted_velocity_points(points):
    required = ["component", "r_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"]
    missing = [name for name in required if name not in points.columns]
    if missing:
        raise ValueError(f"Juodzbalis Fig. 2 point table is missing velocity-score columns: {missing}")
    weights = np.asarray(QSO1_VELOCITY_GROUP_WEIGHTS, dtype=float)

    table = points.copy()
    for col in ["r_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    numeric = table[["r_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("Juodzbalis Fig. 2 point table contains non-finite velocity-score values.")
    if np.any(table["r_pc"].to_numpy(dtype=float) == 0.0):
        raise ValueError("Juodzbalis Fig. 2 velocity-score points must not have r_pc == 0.")
    if np.any(table[["v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float) <= 0.0):
        raise ValueError("Juodzbalis Fig. 2 velocity-score points must have positive velocity uncertainties.")
    table["abs_r_pc"] = np.abs(table["r_pc"].to_numpy(dtype=float))

    def _component_rows(component, expected_count):
        rows = table.loc[table["component"].eq(component)].copy()
        if len(rows) != expected_count:
            raise ValueError(f"Juodzbalis Fig. 2 velocity-score component {component!r} has {len(rows)} rows, expected {expected_count}.")
        return rows.sort_values("abs_r_pc").reset_index(drop=True)

    spectro = _component_rows("spectroastrometry", 2)
    spectro_fine = _component_rows("spectroastrometry_fine", 4)
    resolved = _component_rows("resolved_kinematics", 4)
    groups = [
        ("spectroastrometry", spectro, float(weights[0])),
        ("spectroastrometry_inner", spectro_fine.iloc[:2].copy(), float(weights[1])),
        ("spectroastrometry_outer", spectro_fine.iloc[2:].copy(), float(weights[2])),
        ("resolved_inner", resolved.iloc[:2].copy(), float(weights[3])),
        ("resolved_outer", resolved.iloc[2:].copy(), float(weights[4])),
    ]

    weighted = []
    for name, rows, group_weight in groups:
        rows = rows.sort_values("r_pc").copy()
        rows["velocity_group"] = name
        rows["velocity_group_weight"] = group_weight
        rows["point_weight"] = group_weight / float(len(rows))
        weighted.append(rows)
    out = pd.concat(weighted, ignore_index=True)
    if not np.isclose(float(out["point_weight"].sum()), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Juodzbalis Fig. 2 velocity-score point weights do not sum to unity.")
    return out
def _fig01_observed_velocity_score_by_halo(points, deposit_profile, z_rows, background_by_halo=None):
    observed = _fig01_weighted_velocity_points(points)
    halo_ids, radius_pc, _stellar_cumulative, velocity_profiles = _fig01_total_velocity_profiles(
        deposit_profile,
        z_rows,
        background_by_halo=background_by_halo,
    )
    obs_abs_r = np.abs(observed["r_pc"].to_numpy(dtype=float))
    obs_sign = np.sign(observed["r_pc"].to_numpy(dtype=float))
    obs_velocity = observed["v_km_s"].to_numpy(dtype=float)
    point_weight = observed["point_weight"].to_numpy(dtype=float)
    err_low = observed["v_err_low_km_s"].to_numpy(dtype=float)
    err_high = observed["v_err_high_km_s"].to_numpy(dtype=float)

    if np.any(obs_abs_r < radius_pc[0]) or np.any(obs_abs_r > radius_pc[-1]):
        reason = f"observed velocity radii {float(np.min(obs_abs_r)):.6g}-{float(np.max(obs_abs_r)):.6g} pc exceed model velocity grid {float(radius_pc[0]):.6g}-{float(radius_pc[-1]):.6g} pc"
        return {
            int(hid): {
                "keplerian_term": np.nan,
                "keplerian_chi2_weighted": np.nan,
                "keplerian_n_points": 0,
                "missing_reason": reason,
            }
            for hid in halo_ids
        }

    scores = {}
    for hid, velocity in zip(halo_ids, velocity_profiles):
        model_velocity = np.interp(obs_abs_r, radius_pc, velocity)
        signed_model_velocity = obs_sign * model_velocity
        residual = signed_model_velocity - obs_velocity
        sigma = np.where(residual >= 0.0, err_high, err_low)
        bad_sigma = ~np.isfinite(sigma) | (sigma <= 0.0)
        if np.any(bad_sigma):
            for idx in np.flatnonzero(bad_sigma):
                alternatives = np.asarray([err_low[idx], err_high[idx]], dtype=float)
                alternatives = alternatives[np.isfinite(alternatives) & (alternatives > 0.0)]
                if len(alternatives) > 0:
                    sigma[idx] = float(np.mean(alternatives))
        valid = np.isfinite(model_velocity) & np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0.0)
        if not np.all(valid):
            scores[int(hid)] = {
                "keplerian_term": np.nan,
                "keplerian_chi2_weighted": np.nan,
                "keplerian_n_points": int(np.sum(valid)),
                "missing_reason": "non-finite observed-velocity residual or uncertainty",
            }
            continue
        normalised_residual = residual / sigma
        chi2_weighted = float(np.sum(point_weight * normalised_residual**2))
        scores[int(hid)] = {
            "keplerian_term": float(np.sqrt(chi2_weighted)),
            "keplerian_chi2_weighted": chi2_weighted,
            "keplerian_n_points": int(len(observed)),
            "missing_reason": "",
        }
    return scores
def _candidate_no_score_error(out_dir, score_table):
    n_candidates = int(len(score_table))
    def _numeric_column(name):
        if name not in score_table:
            return np.asarray([], dtype=float)
        return pd.to_numeric(score_table[name], errors="coerce").to_numpy(dtype=float)

    finite_score = int(np.isfinite(_numeric_column("score_keplerian_uv")).sum())
    finite_keplerian = int(np.isfinite(_numeric_column("keplerian_term")).sum())
    finite_uv = int(np.isfinite(_numeric_column("M_UV")).sum())
    formed_mass = _numeric_column("formed_mass_6pc_msun")
    finite_formed = int((np.isfinite(formed_mass) & (formed_mass > 0.0)).sum())
    usable_age = int(np.isfinite(_numeric_column("weighted_age_gyr")).sum())
    reasons = []
    if "missing_reason" in score_table:
        for row in score_table[["halo_id_z0", "missing_reason"]].itertuples(index=False):
            reason = str(row.missing_reason).strip()
            if reason:
                reasons.append(f"halo {int(row.halo_id_z0)}: {reason}")
            if len(reasons) >= 5:
                break
    reason_text = "; ".join(reasons) if reasons else "no candidate-specific missing-producer reason recorded"
    return (
        "No finite Keplerian+UV score is available for Fig. 01/Fig. 05 selection. "
        f"out_dir={Path(out_dir).resolve()}, "
        f"redshift-selected candidates={n_candidates}, finite Keplerian terms={finite_keplerian}, "
        f"finite UV values={finite_uv}, finite Keplerian+UV scores={finite_score}, "
        f"finite formed 6 pc mass={finite_formed}, usable GC-origin age weights={usable_age}, "
        f"first missing producers: {reason_text}."
    )
def score_fig01_candidate_haloes(out_dir, points, z_rows, deposit_profile, final_gc, uv_calibration, background_by_halo=None):
    if len(z_rows) == 0:
        return pd.DataFrame(columns=FIG01_SCORE_COLUMNS), None
    lookback_qso1 = _lookback_to_z0_gyr(QSO1_REDSHIFT)
    deposit_halo_ids = _integer_values(deposit_profile["halo_ids"], "Fig. 01 candidate deposit halo IDs", non_negative=True)
    velocity_scores = _fig01_observed_velocity_score_by_halo(
        points,
        deposit_profile,
        z_rows,
        background_by_halo=background_by_halo,
    )
    rows = []
    for row in z_rows.sort_values("halo_id_z0").itertuples(index=False):
        hid = int(getattr(row, "halo_id_z0"))
        redshift = float(getattr(row, "z_out", getattr(row, "redshift", np.nan)))
        nsc_mass = float(getattr(row, "M_NSC", getattr(row, "nsc_mass_msun", np.nan)))
        central_bh = float(getattr(row, "M_SMBH_final", getattr(row, "central_bh_mass_final_msun", np.nan)))
        log_nsc = float(np.log10(nsc_mass)) if np.isfinite(nsc_mass) and nsc_mass > 0.0 else np.nan
        log_bh = float(np.log10(central_bh)) if np.isfinite(central_bh) and central_bh > 0.0 else np.nan
        formed_mass = interpolate_formed_mass_inside_aperture(deposit_profile, hid, QSO1_NSC_APERTURE_PC)
        contributors, contributor_reason = select_qso1_gc_contributors(final_gc, hid, lookback_qso1)
        uv = estimate_old_nsc_uv_mag(formed_mass, contributors, lookback_qso1, uv_calibration)

        missing = []
        if not (np.isfinite(formed_mass) and formed_mass > 0.0):
            missing.append("formed 6 pc mass unavailable")
        if contributor_reason:
            missing.append(contributor_reason)
        if uv["missing_reason"]:
            missing.append(uv["missing_reason"])
        m_uv = float(uv["M_UV"])
        uv_term = (m_uv - QSO1_MUV_AB) / QSO1_MUV_TOL_MAG if np.isfinite(m_uv) else np.nan
        velocity_score = velocity_scores.get(
            hid,
            {
                "keplerian_term": np.nan,
                "keplerian_chi2_weighted": np.nan,
                "keplerian_n_points": 0,
                "missing_reason": "observed-velocity score unavailable",
            },
        )
        if velocity_score["missing_reason"]:
            missing.append(str(velocity_score["missing_reason"]))
        keplerian_term = float(velocity_score["keplerian_term"])
        keplerian_chi2_weighted = float(velocity_score["keplerian_chi2_weighted"])
        keplerian_n_points = int(velocity_score["keplerian_n_points"])
        if np.isfinite(keplerian_term) and np.isfinite(uv_term):
            score_keplerian_uv = float(np.sqrt(QSO1_SCORE_WEIGHT_KEPLERIAN * keplerian_term**2 + QSO1_SCORE_WEIGHT_MUV * uv_term**2))
        else:
            score_keplerian_uv = np.nan

        if hid in set(deposit_halo_ids.tolist()):
            velocity_index = int(np.flatnonzero(deposit_halo_ids == hid)[0])
        else:
            velocity_index = -1
            missing.append("deposit-profile halo index unavailable")

        rows.append(
            {
                "index": velocity_index,
                "halo_id_z0": hid,
                "redshift": redshift,
                "nsc_mass_msun": nsc_mass,
                "log10_nsc_mass": log_nsc,
                "central_bh_mass_msun": central_bh,
                "log10_central_bh_mass": log_bh,
                "formed_mass_6pc_msun": float(uv["formed_mass_msun"]) if np.isfinite(uv["formed_mass_msun"]) else np.nan,
                "weighted_age_gyr": float(uv["weighted_age_gyr"]),
                "weighted_feh": float(uv["weighted_feh"]) if np.isfinite(uv["weighted_feh"]) else np.nan,
                "m1500_per_msun": float(uv["m1500_per_msun"]) if np.isfinite(uv["m1500_per_msun"]) else np.nan,
                "M_UV": m_uv,
                "uv_term": float(uv_term) if np.isfinite(uv_term) else np.nan,
                "keplerian_term": keplerian_term,
                "keplerian_chi2_weighted": keplerian_chi2_weighted,
                "keplerian_n_points": keplerian_n_points,
                "score_keplerian_uv": score_keplerian_uv,
                "n_gc_contributors": int(uv["n_contributors"]),
                "n_uv_valid_contributors": int(uv["n_uv_valid_contributors"]),
                "n_uv_age_nearest_grid": int(uv["n_uv_age_nearest_grid"]),
                "n_uv_feh_nearest_grid": int(uv["n_uv_feh_nearest_grid"]),
                "n_uv_any_nearest_grid": int(uv["n_uv_any_nearest_grid"]),
                "min_raw_age_gyr": float(uv["min_raw_age_gyr"]) if np.isfinite(uv["min_raw_age_gyr"]) else np.nan,
                "max_raw_age_gyr": float(uv["max_raw_age_gyr"]) if np.isfinite(uv["max_raw_age_gyr"]) else np.nan,
                "min_raw_feh": float(uv["min_raw_feh"]) if np.isfinite(uv["min_raw_feh"]) else np.nan,
                "max_raw_feh": float(uv["max_raw_feh"]) if np.isfinite(uv["max_raw_feh"]) else np.nan,
                "min_eval_age_gyr": float(uv["min_eval_age_gyr"]) if np.isfinite(uv["min_eval_age_gyr"]) else np.nan,
                "max_eval_age_gyr": float(uv["max_eval_age_gyr"]) if np.isfinite(uv["max_eval_age_gyr"]) else np.nan,
                "min_eval_feh": float(uv["min_eval_feh"]) if np.isfinite(uv["min_eval_feh"]) else np.nan,
                "max_eval_feh": float(uv["max_eval_feh"]) if np.isfinite(uv["max_eval_feh"]) else np.nan,
                "uv_table_path": str(uv["uv_table_path"]),
                "uv_mode": str(uv["uv_mode"]),
                "missing_reason": "; ".join(dict.fromkeys(reason for reason in missing if reason)),
            }
        )

    score_table = pd.DataFrame(rows, columns=FIG01_SCORE_COLUMNS)
    finite = np.isfinite(score_table["score_keplerian_uv"].to_numpy(dtype=float))
    if not np.any(finite):
        raise ValueError(_candidate_no_score_error(out_dir, score_table))
    best = score_table.loc[finite].sort_values(["score_keplerian_uv", "halo_id_z0"], ascending=[True, True]).iloc[0].to_dict()
    return score_table, best
def load_deposit_profile_for_redshift_summary(deposit_path, summary_rows, final_redshift):
    try:
        table = _read_headered_whitespace_table(deposit_path)
    except pd.errors.EmptyDataError:
        table = pd.DataFrame()
    required = ["halo_id_z0", "lookback_time_gyr", "bin_index", "r_inner_kpc", "r_outer_kpc", "m_star_no_evo_msun", "m_star_with_evo_msun"]
    missing = [name for name in required if name not in table.columns]
    if missing and len(table) > 0:
        raise ValueError(f"{deposit_path} is missing required deposit columns: {missing}")
    if len(table) > 0:
        table["halo_id_z0"] = _integer_values(table["halo_id_z0"], f"{deposit_path} halo IDs", non_negative=True)
        table["bin_index"] = _integer_values(table["bin_index"], f"{deposit_path} bin indices", non_negative=True)
        for col in required:
            if col not in {"halo_id_z0", "bin_index"}:
                table[col] = pd.to_numeric(table[col], errors="coerce")
        if table[required].isna().any().any():
            raise ValueError(f"{deposit_path} contains non-finite values in required deposit columns.")

    summary = summary_rows.copy()
    if "redshift" not in summary.columns and "z_out" in summary.columns:
        summary["redshift"] = summary["z_out"]
    summary["halo_id_z0"] = _integer_values(summary["halo_id_z0"], "selected halo-summary IDs", non_negative=True)
    for col in summary.columns:
        if col != "halo_id_z0":
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    if summary["halo_id_z0"].duplicated().any():
        dupes = summary.loc[summary["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].drop_duplicates().tolist()
        raise ValueError(f"Selected halo summary has duplicated halo_id_z0 values: {dupes[:10]}")

    final_age_gyr = Redshift2CosmicAge(float(final_redshift))
    grouped = {} if len(table) == 0 else {int(hid): group for hid, group in table.groupby("halo_id_z0", sort=True)}
    halo_ids, r_outer, cumulative, cumulative_formed = [], [], [], []
    missing_halo_ids = []
    for row in summary.sort_values("halo_id_z0").itertuples(index=False):
        hid = int(getattr(row, "halo_id_z0"))
        if hid not in grouped:
            missing_halo_ids.append(hid)
            continue
        group = grouped[hid]
        unique_lookbacks = np.unique(np.sort(group["lookback_time_gyr"].to_numpy(dtype=float)))
        target_lookback = getattr(row, "lookback_depos_sampled_gyr", np.nan)
        if not np.isfinite(target_lookback):
            target_lookback = getattr(row, "deposit_sample_lookback_gyr", np.nan)
        if np.isfinite(target_lookback):
            block_lookback = float(unique_lookbacks[np.argmin(np.abs(unique_lookbacks - float(target_lookback)))])
            if abs(block_lookback - float(target_lookback)) > 1.0e-6:
                raise ValueError(f"Deposit profile for halo_id_z0={hid} does not contain the requested lookback.")
        else:
            target_redshift = getattr(row, "z_depos_sampled", np.nan)
            if not np.isfinite(target_redshift):
                target_redshift = getattr(row, "deposit_sample_redshift", np.nan)
            if not np.isfinite(target_redshift):
                target_redshift = getattr(row, "z_out", getattr(row, "redshift"))
            block_redshifts = np.array([CosmicAge2Redshift(final_age_gyr - float(lb)) for lb in unique_lookbacks], dtype=float)
            block_lookback = float(unique_lookbacks[np.argmin(np.abs(block_redshifts - float(target_redshift)))])

        block = group[np.isclose(group["lookback_time_gyr"].to_numpy(dtype=float), block_lookback, rtol=0.0, atol=1.0e-8)]
        ordered = block.sort_values("bin_index")
        bin_index = ordered["bin_index"].to_numpy(dtype=np.int64)
        if len(bin_index) == 0 or not np.array_equal(bin_index, np.arange(1, len(bin_index) + 1, dtype=np.int64)):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has non-contiguous bin_index values.")
        rin = ordered["r_inner_kpc"].to_numpy(dtype=float)
        rout = ordered["r_outer_kpc"].to_numpy(dtype=float)
        shell_formed = ordered["m_star_no_evo_msun"].to_numpy(dtype=float)
        shell = ordered["m_star_with_evo_msun"].to_numpy(dtype=float)
        if not np.isclose(rin[0], 0.0, rtol=0.0, atol=1.0e-10) or not np.isclose(rout[0], 1.0e-3, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has an unexpected innermost radial bin.")
        if np.any(~np.isfinite(rin)) or np.any(~np.isfinite(rout)) or np.any(~np.isfinite(shell)) or np.any(~np.isfinite(shell_formed)) or np.any(shell < 0.0) or np.any(shell_formed < 0.0):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} contains invalid radial or mass values.")
        if np.any(np.diff(rout) <= 0.0) or np.any(rout <= rin):
            raise ValueError(f"Deposit profile for halo_id_z0={hid} has invalid radial bin edges.")
        halo_ids.append(hid)
        r_outer.append(rout)
        cumulative.append(np.cumsum(shell))
        cumulative_formed.append(np.cumsum(shell_formed))
    return {
        "halo_ids": np.asarray(halo_ids, dtype=np.int64),
        "r_outer_kpc": r_outer,
        "cumulative_mass_msun": cumulative,
        "cumulative_formed_mass_msun": cumulative_formed,
        "missing_halo_ids": np.asarray(missing_halo_ids, dtype=np.int64),
    }
def _fig01_bh_column(summary):
    for name in ["M_SMBH_final", "central_bh_mass_final_msun", "M_BH"]:
        if name in summary.columns:
            return name
    raise ValueError("haloSummaryByZ is missing a central-BH mass column required for Fig. 01.")
def _select_fig01_z_rows(summary_by_z):
    redshift = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(redshift) & (np.abs(redshift - FIG01_TARGET_REDSHIFT) < FIG01_REDSHIFT_ATOL)
    if not np.any(mask):
        raise ValueError(f"No haloSummaryByZ rows satisfy |z - {FIG01_TARGET_REDSHIFT:.2f}| < {FIG01_REDSHIFT_ATOL:.2f}.")
    rows = summary_by_z.loc[mask].copy()
    rows["halo_id_z0"] = _integer_values(rows["halo_id_z0"], "Fig. 01 selected halo IDs", non_negative=True)
    for col in rows.columns:
        if col != "halo_id_z0":
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.sort_values("halo_id_z0").reset_index(drop=True)
    if rows["halo_id_z0"].duplicated().any():
        dupes = rows.loc[rows["halo_id_z0"].duplicated(keep=False), "halo_id_z0"].drop_duplicates().tolist()
        raise ValueError(f"Fig. 01 z selection produced duplicate halo rows: {dupes[:10]}")
    bh = rows[_fig01_bh_column(rows)].to_numpy(dtype=float)
    if np.any(~np.isfinite(bh)) or np.any(bh < 0.0):
        raise ValueError("Fig. 01 selected rows contain invalid central BH masses.")
    return rows
def _fig01_error_arrays(rows):
    xerr = rows[["r_err_low_pc", "r_err_high_pc"]].to_numpy(dtype=float).T
    yerr = rows[["v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float).T
    return np.where(np.isfinite(xerr), xerr, 0.0), np.where(np.isfinite(yerr), yerr, 0.0)
def _plot_fig01_observed_curve(ax, curves, curve_name, label, **kwargs):
    used_label = False
    for sign in [-1.0, 1.0]:
        rows = curves.loc[curves["curve"].eq(curve_name) & (np.sign(curves["r_pc"].to_numpy(dtype=float)) == sign)]
        if len(rows) == 0:
            continue
        ordered = rows.sort_values("r_pc")
        ax.plot(ordered["r_pc"].to_numpy(dtype=float), ordered["v_km_s"].to_numpy(dtype=float), label=label if not used_label else None, **kwargs)
        used_label = True
def _fig01_signed_profile(radius_pc, velocity_km_s):
    radius = np.asarray(radius_pc, dtype=float)
    velocity = np.asarray(velocity_km_s, dtype=float)
    return np.concatenate([-radius[::-1], radius]), np.concatenate([-velocity[::-1], velocity])


def _fig01_subset_deposit_profile(deposit_profile, keep):
    keep = np.asarray(keep, dtype=bool)
    profile_halo_ids = _integer_values(deposit_profile["halo_ids"], "Fig. 01 deposit halo IDs", non_negative=True)
    if len(keep) != len(profile_halo_ids):
        raise ValueError("Fig. 01 deposit-profile subset mask has the wrong length.")
    out = dict(deposit_profile)
    out["halo_ids"] = profile_halo_ids[keep]
    for name in ("r_outer_kpc", "cumulative_mass_msun", "cumulative_formed_mass_msun"):
        if name not in deposit_profile:
            continue
        values = deposit_profile[name]
        if len(values) != len(keep):
            raise ValueError(f"Fig. 01 deposit-profile field {name} has the wrong length.")
        out[name] = [value for value, keep_value in zip(values, keep) if keep_value]
    return out


def _fig01_interpolate_mpb_spin(mpb_rows, z_out):
    target_redshift = check_finite_non_negative(float(z_out), name="Fig. 01 background redshift")
    rows = np.asarray(mpb_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 5:
        raise ValueError("Fig. 01 MPB rows are empty or missing spin columns 2--4.")
    redshift = rows[:, 1]
    spin = rows[:, 2:5]
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("Fig. 01 MPB redshifts must be finite and non-negative.")
    if np.any(~np.isfinite(spin)):
        raise ValueError("Fig. 01 MPB spin components must be finite.")
    cosmic_time = np.asarray(
        [Redshift2CosmicAge(float(value), time_unit="Gyr") for value in redshift],
        dtype=float,
    )
    order = np.argsort(cosmic_time, kind="mergesort")
    cosmic_time = cosmic_time[order]
    spin = spin[order]
    keep_last_duplicate = np.r_[cosmic_time[1:] != cosmic_time[:-1], True]
    unique_time = cosmic_time[keep_last_duplicate]
    unique_spin = spin[keep_last_duplicate]
    target_time = float(Redshift2CosmicAge(target_redshift, time_unit="Gyr"))
    if target_time < unique_time[0] - 1.0e-10 or target_time > unique_time[-1] + 1.0e-10:
        raise ValueError(f"Fig. 01 MPB spin history does not cover z={target_redshift:.6g}.")
    interpolated_spin = np.asarray(
        [np.interp(target_time, unique_time, unique_spin[:, axis]) for axis in range(3)],
        dtype=float,
    )
    if np.any(~np.isfinite(interpolated_spin)):
        raise ValueError("Fig. 01 interpolated MPB spin is non-finite.")
    with np.errstate(over="ignore", invalid="ignore"):
        j_pc_kms = float(np.linalg.norm(interpolated_spin) * 1.0e3 / ReducedH0)
    if not np.isfinite(j_pc_kms) or j_pc_kms <= 0.0:
        raise ValueError(f"Fig. 01 interpolated MPB specific angular momentum is non-positive: {j_pc_kms!r}.")
    return j_pc_kms


def _fig01_background_parameters_for_halo(halo_id_z0, lookup):
    halo_id = int(parse_exact_int64(halo_id_z0, name="Fig. 01 background halo ID"))
    lookup_ids = _integer_values(lookup["halo_id_z0"], "Fig. 01 TNG lookup halo IDs", non_negative=True)
    lookup_rows = lookup.loc[lookup_ids == np.int64(halo_id)]
    if len(lookup_rows) != 1:
        raise ValueError(f"Fig. 01 TNG lookup has {len(lookup_rows)} rows for halo_id_z0={halo_id}, expected one.")
    fixed_tree_basename = str(lookup_rows.iloc[0]["fixed_tree_basename"]).strip()
    tree_path = _fig05_fixed_tree_path(fixed_tree_basename)
    mpb_rows = np.asarray(read_haloevo_mpb(tree_path), dtype=float)
    if mpb_rows.ndim != 2 or mpb_rows.shape[0] == 0 or mpb_rows.shape[1] < 5:
        raise ValueError(f"Fig. 01 fixed tree has no usable MPB rows: {tree_path}")
    log_halo_mass = mpb_rows[:, 0]
    redshift = mpb_rows[:, 1]
    if np.any(~np.isfinite(log_halo_mass)) or np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError(f"Fig. 01 MPB mass/redshift values are invalid: {tree_path}")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        raw_halo_mass = np.power(10.0, log_halo_mass)
    if np.any(~np.isfinite(raw_halo_mass)) or np.any(raw_halo_mass <= 0.0):
        raise ValueError(f"Fig. 01 MPB halo masses must be finite and positive: {tree_path}")
    log_halo_mass_bg, available = _interpolate_mpb_logmh_at_redshift(
        mpb_rows,
        FIG01_TARGET_REDSHIFT,
    )
    if int(available) != 1 or not np.isfinite(log_halo_mass_bg):
        raise ValueError(f"Fig. 01 MPB cannot be interpolated to z={FIG01_TARGET_REDSHIFT:.2f}: {tree_path}")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        halo_mass = float(10.0 ** float(log_halo_mass_bg))
    if not np.isfinite(halo_mass) or halo_mass <= 0.0:
        raise ValueError(f"Fig. 01 interpolated halo mass is invalid: {halo_mass!r}")

    j_pc_kms = _fig01_interpolate_mpb_spin(mpb_rows, FIG01_TARGET_REDSHIFT)
    background_time_gyr = float(Redshift2CosmicAge(FIG01_TARGET_REDSHIFT, time_unit="Gyr"))
    effective_radius_kpc = float(
        calcRe(
            Mhalo_1e9Msun=halo_mass / 1.0e9,
            t_Gyr=background_time_gyr,
            j=j_pc_kms,
        )
    )
    sersic_index = float(FIG01_SERSIC_INDEX)
    sersic_p, sersic_b = Sersic_coefs(sersic_index)
    stellar_mass = float(Mstar_SMHM(halo_mass, FIG01_TARGET_REDSHIFT, scatter=False))
    concentration = float(
        10.0 ** 0.971
        / ((halo_mass / 1.0e9) * ReducedH0 / 1.0e3) ** 0.094
    )
    virial_radius_kpc = float(Rv(Mhalo=halo_mass, z=FIG01_TARGET_REDSHIFT))
    scale_radius_kpc = virial_radius_kpc / concentration
    scalar_values = [
        effective_radius_kpc,
        stellar_mass,
        concentration,
        virial_radius_kpc,
        scale_radius_kpc,
        sersic_p,
        sersic_b,
    ]
    if np.any(~np.isfinite(scalar_values)) or np.any(np.asarray(scalar_values[:5]) <= 0.0):
        raise ValueError(f"Fig. 01 background parameters are invalid: {tree_path}")
    return {
        "halo_id_z0": halo_id,
        "tree_path": str(tree_path),
        "halo_mass_msun": halo_mass,
        "mstar_msun": stellar_mass,
        "r_e_kpc": effective_radius_kpc,
        "r_v_kpc": virial_radius_kpc,
        "r_s_kpc": scale_radius_kpc,
        "concentration": concentration,
        "sersic_index": sersic_index,
        "sersic_p": float(sersic_p),
        "sersic_b": float(sersic_b),
        "j_pc_kms": j_pc_kms,
    }


def _fig01_build_background_parameters(z_rows, lookup):
    if not isinstance(lookup, pd.DataFrame):
        raise ValueError("Fig. 01 background calculation requires the validated TNG lookup table.")
    required = ["halo_id_z0", "simulation_key", "fixed_tree_basename"]
    missing = [name for name in required if name not in lookup.columns]
    if missing:
        raise ValueError(f"Fig. 01 TNG lookup is missing background provenance columns: {missing}")
    lookup_ids = _integer_values(lookup["halo_id_z0"], "Fig. 01 TNG lookup halo IDs", non_negative=True)
    if len(lookup_ids) != len(np.unique(lookup_ids)):
        raise ValueError("Fig. 01 TNG lookup contains duplicate halo IDs for background profiles.")
    halo_id_values = _integer_values(z_rows["halo_id_z0"], "Fig. 01 selected halo IDs", non_negative=True)
    if len(np.unique(halo_id_values)) != len(halo_id_values):
        raise ValueError("Fig. 01 selected rows contain duplicate halo IDs for background profiles.")

    valid = {}
    excluded = {}
    for halo_id in halo_id_values:
        try:
            valid[int(halo_id)] = _fig01_background_parameters_for_halo(int(halo_id), lookup)
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, OverflowError, IndexError) as exc:
            excluded[int(halo_id)] = f"Fig. 01 background unavailable: {exc}"
    return valid, excluded


def _fig01_enclosed_background_masses(radius_pc, parameters, sersic_index=FIG01_SERSIC_INDEX):
    radius = np.asarray(radius_pc, dtype=float)
    if radius.ndim != 1 or np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("Fig. 01 background enclosed-mass radii must be finite, positive, and one-dimensional.")
    sersic_index = float(sersic_index)
    sersic_p, sersic_b = Sersic_coefs(sersic_index)
    halo_mass = float(parameters["halo_mass_msun"])
    stellar_mass = float(parameters["mstar_msun"])
    effective_radius_pc = float(parameters["r_e_kpc"]) * 1.0e3
    scale_radius_pc = float(parameters["r_s_kpc"]) * 1.0e3
    concentration = float(parameters["concentration"])
    scalar_values = [halo_mass, stellar_mass, effective_radius_pc, scale_radius_pc, concentration]
    if np.any(~np.isfinite(scalar_values)) or np.any(np.asarray(scalar_values) <= 0.0):
        raise ValueError("Fig. 01 background enclosed-mass parameters are invalid.")
    nfw_denominator = np.log1p(concentration) - concentration / (1.0 + concentration)
    if not np.isfinite(nfw_denominator) or nfw_denominator <= 0.0:
        raise ValueError("Fig. 01 NFW concentration has an invalid normalisation.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        radius_over_scale = radius / scale_radius_pc
        nfw_mass = halo_mass * (
            np.log1p(radius_over_scale) - radius_over_scale / (1.0 + radius_over_scale)
        ) / nfw_denominator
        sersic_shape = sersic_index * (3.0 - sersic_p)
        sersic_argument = sersic_b * (radius / effective_radius_pc) ** (1.0 / sersic_index)
        sersic_mass = stellar_mass * special.gammainc(sersic_shape, sersic_argument)
    if np.any(~np.isfinite(nfw_mass)) or np.any(nfw_mass < 0.0):
        raise ValueError("Fig. 01 NFW enclosed mass is non-finite or negative.")
    if np.any(~np.isfinite(sersic_mass)) or np.any(sersic_mass < 0.0):
        raise ValueError("Fig. 01 Sérsic enclosed mass is non-finite or negative.")
    return np.asarray(sersic_mass, dtype=float), np.asarray(nfw_mass, dtype=float)


def _fig01_velocity_profile_data(deposit_profile, z_rows, background_by_halo, best_halo_id=None):
    profile_halo_ids = _integer_values(deposit_profile["halo_ids"], "Fig. 01 profile halo IDs", non_negative=True)
    if best_halo_id is not None:
        mass_column = next(
            (name for name in ("log10_halo_mass_at_redshift", "logMh_z_msun") if name in z_rows.columns),
            None,
        )
        if mass_column is None:
            raise ValueError("Fig. 01 mass filter requires a redshift-resolved halo-mass column.")
        halo_id_values = _integer_values(z_rows["halo_id_z0"], "Fig. 01 velocity-profile halo IDs", non_negative=True)
        log_halo_mass = pd.to_numeric(z_rows[mass_column], errors="coerce").to_numpy(dtype=float)
        best_matches = np.flatnonzero(halo_id_values == np.int64(parse_exact_int64(best_halo_id, name="Fig. 01 best halo ID")))
        if len(best_matches) != 1:
            raise ValueError(f"Fig. 01 best halo_id_z0={int(best_halo_id)} is not unique in z_rows.")
        best_index = int(best_matches[0])
        available = np.ones(len(z_rows), dtype=bool)
        if "halo_mass_available" in z_rows.columns:
            available = pd.to_numeric(z_rows["halo_mass_available"], errors="coerce").to_numpy(dtype=float) == 1.0
        if not available[best_index] or not np.isfinite(log_halo_mass[best_index]):
            raise ValueError(f"Fig. 01 best halo_id_z0={int(best_halo_id)} has no valid redshift-resolved halo mass.")
        keep = available & np.isfinite(log_halo_mass) & (np.abs(log_halo_mass - log_halo_mass[best_index]) <= FIG01_HALO_MASS_WINDOW_DEX)
        z_rows = z_rows.loc[keep].copy()
        profile_keep = np.isin(profile_halo_ids, _integer_values(z_rows["halo_id_z0"], "Fig. 01 filtered halo IDs", non_negative=True))
        if not np.any(profile_keep):
            raise ValueError("Fig. 01 halo-mass filter removed every deposit profile.")
        deposit_profile = _fig01_subset_deposit_profile(deposit_profile, profile_keep)

    bh_column = _fig01_bh_column(z_rows)
    bh_by_halo = {int(row.halo_id_z0): float(getattr(row, bh_column)) for row in z_rows[["halo_id_z0", bh_column]].itertuples(index=False)}
    summary_halos = set(bh_by_halo)
    profile_halos = set(int(value) for value in _integer_values(deposit_profile["halo_ids"], "Fig. 01 available profile halo IDs", non_negative=True))
    if summary_halos != profile_halos:
        raise ValueError("Fig. 01 deposit-profile halo IDs do not match selected redshift rows.")
    first_radii_pc = [float(rout[0]) * 1.0e3 for rout in deposit_profile["r_outer_kpc"]]
    last_radii_pc = [float(rout[-1]) * 1.0e3 for rout in deposit_profile["r_outer_kpc"]]
    r_min = max(1.0, max(first_radii_pc))
    r_max = min(FIG01_RADIUS_MAX_PC, min(last_radii_pc))
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max < FIG01_MATCH_RADIUS_RANGE_PC[1]:
        raise ValueError(f"Fig. 01 deposit radial coverage is insufficient: {r_min:.6g}-{r_max:.6g} pc.")
    radius_pc = np.unique(np.concatenate([np.geomspace(r_min, r_max, 256), np.asarray(FIG01_MATCH_RADIUS_RANGE_PC, dtype=float)]))
    if not isinstance(background_by_halo, dict):
        raise ValueError("Fig. 01 velocity profiles require the shared background-parameter mapping.")
    missing_background = sorted(summary_halos - set(int(value) for value in background_by_halo))
    if missing_background:
        raise ValueError(f"Fig. 01 background parameters are missing halo IDs: {missing_background[:10]}")
    deposited_cumulative = []
    nfw_cumulative = []
    sersic_cumulative = []
    non_bh_stellar_cumulative = []
    velocity_profiles = []
    component_velocity_profiles = {
        "deposited_stars": [],
        "sersic_stars": [],
        "nfw": [],
        "central_bh": [],
    }
    for hid, rout_kpc, cumulative in zip(deposit_profile["halo_ids"], deposit_profile["r_outer_kpc"], deposit_profile["cumulative_mass_msun"]):
        rout_pc = np.asarray(rout_kpc, dtype=float) * 1.0e3
        cum_mass = np.asarray(cumulative, dtype=float)
        if radius_pc[0] < rout_pc[0] or radius_pc[-1] > rout_pc[-1]:
            raise ValueError(f"Fig. 01 common radius grid exceeds deposit coverage for halo_id_z0={int(hid)}.")
        if len(rout_pc) != len(cum_mass) or np.any(~np.isfinite(rout_pc)) or np.any(~np.isfinite(cum_mass)) or np.any(cum_mass < 0.0):
            raise ValueError(f"Fig. 01 deposited profile is invalid for halo_id_z0={int(hid)}.")
        stellar = np.interp(radius_pc, rout_pc, cum_mass)
        sersic_mass, nfw_mass = _fig01_enclosed_background_masses(radius_pc, background_by_halo[int(hid)])
        #non_bh_stellar = stellar + nfw_mass + sersic_mass
        non_bh_stellar = stellar + sersic_mass
        central_bh = float(bh_by_halo[int(hid)])
        enclosed_mass = central_bh + non_bh_stellar
        if not np.isfinite(central_bh) or central_bh < 0.0 or np.any(~np.isfinite(non_bh_stellar)) or np.any(non_bh_stellar < 0.0) or np.any(~np.isfinite(enclosed_mass)) or np.any(enclosed_mass < 0.0):
            raise ValueError(f"Fig. 01 enclosed mass is invalid for halo_id_z0={int(hid)}.")
        velocity = FIG01_VELOCITY_SIN_I * np.sqrt(G_Arepo * enclosed_mass / radius_pc)
        component_masses = {
            "deposited_stars": stellar,
            "sersic_stars": sersic_mass,
            "nfw": nfw_mass,
            "central_bh": np.full(radius_pc.shape, central_bh, dtype=float),
        }
        component_velocities = {
            name: FIG01_VELOCITY_SIN_I * np.sqrt(G_Arepo * mass / radius_pc)
            for name, mass in component_masses.items()
        }
        if np.any(~np.isfinite(velocity)) or any(np.any(~np.isfinite(value)) for value in component_velocities.values()):
            raise ValueError(f"Fig. 01 velocity profile is non-finite for halo_id_z0={int(hid)}.")
        deposited_cumulative.append(stellar)
        nfw_cumulative.append(nfw_mass)
        sersic_cumulative.append(sersic_mass)
        non_bh_stellar_cumulative.append(non_bh_stellar)
        velocity_profiles.append(velocity)
        for name, value in component_velocities.items():
            component_velocity_profiles[name].append(value)
    return {
        "halo_ids": _integer_values(deposit_profile["halo_ids"], "Fig. 01 velocity-profile halo IDs", non_negative=True),
        "radius_pc": radius_pc,
        "deposited_cumulative": np.asarray(deposited_cumulative, dtype=float),
        "nfw_cumulative": np.asarray(nfw_cumulative, dtype=float),
        "sersic_cumulative": np.asarray(sersic_cumulative, dtype=float),
        "non_bh_stellar_cumulative": np.asarray(non_bh_stellar_cumulative, dtype=float),
        "velocity_profiles": np.asarray(velocity_profiles, dtype=float),
        "component_velocity_profiles": {
            name: np.asarray(values, dtype=float)
            for name, values in component_velocity_profiles.items()
        },
    }


def _fig01_total_velocity_profiles(deposit_profile, z_rows, best_halo_id=None, background_by_halo=None):
    if background_by_halo is None:
        raise ValueError("Fig. 01 total velocity profiles require the shared background-parameter mapping.")
    profile_data = _fig01_velocity_profile_data(
        deposit_profile,
        z_rows,
        background_by_halo,
        best_halo_id=best_halo_id,
    )
    return (
        profile_data["halo_ids"],
        profile_data["radius_pc"],
        profile_data["non_bh_stellar_cumulative"],
        profile_data["velocity_profiles"],
    )


def plot_fig01_rotation_curve(points, curves, z_rows, deposit_profile, best, background_by_halo=None):
    if background_by_halo is None:
        raise ValueError("Fig. 01 rotation-curve plotting requires the shared background-parameter mapping.")
    profile_data = _fig01_velocity_profile_data(
        deposit_profile,
        z_rows,
        background_by_halo,
        best_halo_id=int(best["halo_id_z0"]),
    )
    halo_ids = profile_data["halo_ids"]
    radius_pc = profile_data["radius_pc"]
    velocity_profiles = profile_data["velocity_profiles"]
    matches = np.flatnonzero(halo_ids == int(best["halo_id_z0"]))
    if len(matches) != 1:
        raise ValueError(f"Fig. 01 Keplerian+UV best halo {int(best['halo_id_z0'])} is not present in the velocity-profile grid.")
    best_index = int(matches[0])
    median_velocity = np.median(velocity_profiles, axis=0)
    mean_velocity = np.mean(velocity_profiles, axis=0)
    low_velocity, high_velocity = np.percentile(velocity_profiles, FIG01_SCATTER_PERCENTILES, axis=0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(5.4, 4.4))
    _plot_fig01_observed_curve(ax, curves, "point_mass_keplerian", r"Keplerian point mass $\log M_\bullet = 6.75$", color="black", linewidth=1.7, zorder=4)
    _plot_fig01_observed_curve(ax, curves, "mw_nsc", "MW-like NSC model", color="0.45", linestyle="dashdot", linewidth=1.5, zorder=3)
    for component, colour, marker, size, label in [
        ("resolved_kinematics", "tab:blue", "o", 5.0, "Resolved kinematics"),
        ("spectroastrometry", "magenta", "X", 6.0, "Spectroastrometry"),
        ("spectroastrometry_fine", "orchid", "P", 5.5, "Spectroastrometry, fine split"),
    ]:
        rows = points.loc[points["component"].eq(component)]
        if len(rows) == 0:
            continue
        xerr, yerr = _fig01_error_arrays(rows)
        ax.errorbar(rows["r_pc"].to_numpy(dtype=float), rows["v_km_s"].to_numpy(dtype=float), xerr=xerr, yerr=yerr, fmt=marker, ms=size, color=colour, ecolor=colour, elinewidth=1.0, markeredgecolor=colour, markerfacecolor=colour, capsize=0.0, linestyle="none", label=label, zorder=6)
    ax.fill_between(radius_pc, low_velocity, high_velocity, color="tab:green", alpha=0.16, linewidth=0.0, label=r"$z \simeq 7$ stack 16-84\%")
    ax.fill_between(-radius_pc[::-1], -high_velocity[::-1], -low_velocity[::-1], color="tab:green", alpha=0.16, linewidth=0.0)
    signed_r, signed_median = _fig01_signed_profile(radius_pc, median_velocity)
    _, signed_mean = _fig01_signed_profile(radius_pc, mean_velocity)
    _, signed_best = _fig01_signed_profile(radius_pc, velocity_profiles[best_index])
    #ax.plot(signed_r, signed_median, color="tab:green", linewidth=1.8, label=r"$z \simeq 7$ median simulation")
    #ax.plot(signed_r, signed_mean, color="tab:green", linewidth=1.2, linestyle="--", label="z~7 mean simulation")
    for component, colour, linestyle, label in [
        ("central_bh", "tab:purple", "--", "Central BH only"),
        ("deposited_stars", "tab:blue", ":", "Deposited stars only"),
        ("sersic_stars", "tab:orange", "-.", "Sérsic stars only"),
        ("nfw", "tab:brown", (0, (5, 1, 1, 1)), "NFW only"),
    ]:
        _, signed_component = _fig01_signed_profile(
            radius_pc,
            profile_data["component_velocity_profiles"][component][best_index],
        )
        ax.plot(signed_r, signed_component, color=colour, linewidth=1.0, linestyle=linestyle, label=label, zorder=2)
    ax.plot(signed_r, signed_best, color="tab:red", linewidth=1.5, linestyle="-", label="Best Keplerian+UV halo")
    ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.set_xlim(-FIG01_RADIUS_MAX_PC, FIG01_RADIUS_MAX_PC)
    ax.set_ylim(-72.0, 72.0)
    ax.set_xlabel(r"Projected radius $r$ [pc]")
    ax.set_ylabel(r"Line-of-sight velocity $v$ [km/s]")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2, ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig
def _plot_fig02_logmass_marker(ax, x_value, y_value, *, marker, colour, label, size=7.0, zorder=6):
    xmin, xmax = FIG02_XLIM_LOGM
    if np.isfinite(x_value) and xmin <= float(x_value) <= xmax:
        ax.plot(float(x_value), y_value, marker=marker, ms=size, mfc=colour, mec=colour, color=colour, linestyle="none", label=label, zorder=zorder)
    elif np.isfinite(x_value):
        if float(x_value) > xmax:
            edge_x = xmax - 0.035
            edge_marker = ">"
            text_x = xmax - 0.72
            ha = "right"
        else:
            edge_x = xmin + 0.035
            edge_marker = "<"
            text_x = xmin + 0.72
            ha = "left"
        ax.plot(edge_x, y_value, marker=edge_marker, ms=size, mfc=colour, mec=colour, color=colour, linestyle="none", label=label, clip_on=False, zorder=zorder)
        ax.annotate(
            rf"$\log M={float(x_value):.2f}$",
            xy=(edge_x, y_value),
            xytext=(text_x, y_value + 0.20),
            ha=ha,
            va="bottom",
            fontsize=7.0,
            color=colour,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": colour},
        )
def plot_fig02_bh_masses(fig3_reference, best, uv_estimate):
    labels = fig3_reference["method"].tolist() + [
        r"Best Keplerian+UV halo NSC stellar mass ($<6$ pc)",
        "Best Keplerian+UV halo central BH",
    ]
    y_positions = np.arange(len(labels) - 1, -1, -1, dtype=float)
    y_by_label_index = {index: float(y_positions[index]) for index in range(len(labels))}

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.7))
    ax.axvspan(QSO1_MOKA3D_LOGMBH - QSO1_MOKA3D_LOGMBH_ERR, QSO1_MOKA3D_LOGMBH + QSO1_MOKA3D_LOGMBH_ERR, color="#7b3294", alpha=0.14, lw=0.0, label="MOKA3D 1 sigma")

    styles = {
        "virial": {"colour": "black", "marker": "o", "label": "Virial estimates", "size": 5.3},
        "scattering": {"colour": "#756bb1", "marker": "D", "label": "Scattering scenario", "size": 5.2},
        "bolometric": {"colour": "#d7301f", "marker": "s", "label": r"$L_{\rm bol}$ estimate", "size": 5.4},
        "direct": {"colour": "#238b45", "marker": "*", "label": "Direct estimates", "size": 8.0},
    }
    used_labels = set()
    for index, row in fig3_reference.iterrows():
        style = styles[str(row["group"])]
        label = None if style["label"] in used_labels else style["label"]
        used_labels.add(style["label"])
        x_value = float(row["log10_mass"])
        y_value = y_by_label_index[int(index)]
        colour = style["colour"]
        if _as_bool(row["is_lower_limit"]):
            ax.plot(x_value, y_value, marker=style["marker"], ms=style["size"], mfc=colour, mec=colour, color=colour, linestyle="none", label=label, zorder=6)
            ax.annotate("", xy=(min(FIG02_XLIM_LOGM[1] - 0.08, x_value + 0.42), y_value), xytext=(x_value, y_value), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": colour}, zorder=5)
        else:
            xerr = np.array([[float(row["err_low"])], [float(row["err_high"])]], dtype=float)
            xerr = None if np.all(xerr <= 0.0) else xerr
            ax.errorbar(x_value, y_value, xerr=xerr, fmt=style["marker"], ms=style["size"], color=colour, mfc=colour, mec=colour, ecolor=colour, elinewidth=1.0, capsize=2.2, linestyle="none", label=label, zorder=6)

    nsc_y = y_by_label_index[len(fig3_reference)]
    bh_y = y_by_label_index[len(fig3_reference) + 1]
    _plot_fig02_logmass_marker(ax, float(best["log10_nsc_mass"]), nsc_y, marker="P", colour="#1f9e89", label="Model NSC mass", size=7.0)
    _plot_fig02_logmass_marker(ax, float(best["log10_central_bh_mass"]), bh_y, marker="X", colour="#e6550d", label="Model central BH", size=7.0)

    m_uv = float(uv_estimate.get("M_UV", np.nan)) if isinstance(uv_estimate, dict) else np.nan
    if np.isfinite(m_uv):
        uv_text = rf"stellar old-NSC $M_{{UV}}={m_uv:.2f}$; QSO1 $M_{{UV}}={QSO1_MUV_AB:.2f}$"
        x_anchor = min(max(float(best["log10_nsc_mass"]), FIG02_XLIM_LOGM[0] + 0.1), FIG02_XLIM_LOGM[1] - 0.1)
        ax.annotate(
            uv_text,
            xy=(x_anchor, nsc_y),
            xytext=(FIG02_XLIM_LOGM[0] + 0.10, nsc_y + 0.70),
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="0.20",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.35"},
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlim(*FIG02_XLIM_LOGM)
    ax.set_ylim(-0.65, y_positions[0] + 0.65)
    ax.set_xlabel(r"$\log_{10}(M/M_{\odot})$")
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    ax.legend(frameon=False, loc="lower right", fontsize=7.0, ncol=2)
    return fig
def plot_fig03_uvmag(aperture_table):
    required = ["aperture_pc", "M_UV"]
    missing = [name for name in required if name not in aperture_table.columns]
    if missing:
        raise ValueError(f"Fig. 03 aperture table is missing required columns: {missing}")
    aperture_pc = aperture_table["aperture_pc"].to_numpy(dtype=float)
    m_uv = aperture_table["M_UV"].to_numpy(dtype=float)
    if len(aperture_pc) != 7 or np.any(~np.isfinite(aperture_pc)) or np.any(~np.isfinite(m_uv)):
        raise ValueError("Fig. 03 aperture table must contain seven finite aperture and UV-magnitude values.")

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.7))
    ax.plot(aperture_pc, m_uv, marker="o", ms=5.5, color="#1f77b4", label="Selected best halo")
    ax.axhline(QSO1_MUV_AB, c="black", ls=":", label=rf"QSO1 $M_{{\rm UV}}={QSO1_MUV_AB:.2f}$")
    ax.set_xlabel(r"UV aperture $R_{\rm UV}$ [pc]")
    ax.set_ylabel(r"Rest-frame $1500\,\AA$ absolute AB magnitude $M_{\rm UV}$")
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(np.arange(1.0, 8.0, 1.0))
    y_min = min(float(np.min(m_uv)), float(QSO1_MUV_AB))
    y_max = max(float(np.max(m_uv)), float(QSO1_MUV_AB))
    y_span = max(y_max - y_min, 0.5)
    y_margin = 0.08 * y_span
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig
def plot_fig04_halo_distribution(summary_by_z, best):
    distribution = _build_fig04_halo_distribution(summary_by_z, best)
    redshifts = distribution["redshifts"]
    log_edges = distribution["log_bin_edges"]
    mass_edges = np.power(10.0, log_edges)
    if np.any(~np.isfinite(mass_edges)) or np.any(mass_edges <= 0.0):
        raise ValueError("Fig. 04 generated non-positive or non-finite linear halo-mass bin edges.")
    if len(redshifts) == 1:
        norm = mpl.colors.Normalize(vmin=float(redshifts[0]) - 0.5, vmax=float(redshifts[0]) + 0.5)
    else:
        norm = mpl.colors.Normalize(vmin=float(redshifts.min()), vmax=float(redshifts.max()))
    cmap = mpl.cm.jet

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    maximum_count = 0
    for item in distribution["distributions"]:
        z_value = float(item["redshift"])
        counts = np.asarray(item["counts"], dtype=int)
        if len(counts) != len(mass_edges) - 1 or np.any(counts < 0):
            raise ValueError(f"Fig. 04 contains invalid integer histogram counts at z={z_value:.6g}.")
        maximum_count = max(maximum_count, int(np.max(counts)))
        ax.stairs(counts, mass_edges, baseline=0.0, fill=False, color=cmap(norm(z_value)), lw=1.35, alpha=0.92, zorder=2)

    for track in distribution["best_halo_track"]:
        ax.axvline(float(track["halo_mass_msun"]), color=cmap(norm(float(track["redshift"]))), ls="--", lw=1.0, alpha=0.45, zorder=5)

    ax.plot([], [], color="0.35", lw=1.5, label="Tracked halo distribution")
    ax.plot([], [], color="0.20", ls="--", lw=1.0, alpha=0.45, label=r"Best halo $M_{\rm h}(z)$")
    ax.set_xscale("log")
    ax.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
    ax.set_ylim(0.0, max(1.0, 1.15 * float(maximum_count)))
    ax.set_xlabel(r"Halo mass $M_{\rm h}(z)$ [$M_{\odot}$]")
    ax.set_ylabel("Halo number per mass bin")
    ax.text(
        0.03,
        0.96,
        rf"Tracked MPB final-halo sample; best $h_{{z=0}}$={int(distribution['best_halo_id_z0'])}" + "\n"
        r"coloured dashed lines: selected halo $M_{\rm h}(z)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="0.20",
    )
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="upper right", fontsize=7.4, ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(redshifts) == 1:
        colour_bar.set_ticks([float(redshifts[0])])
    return fig, distribution
def _fig05_redshift_tick_values(maximum_redshift):
    standard = np.asarray([7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0], dtype=float)
    ticks = standard[standard <= float(maximum_redshift) + FIG05_REDSHIFT_ROW_ATOL]
    if len(ticks) == 0:
        ticks = np.asarray([FIG05_TARGET_REDSHIFT], dtype=float)
    return ticks
def plot_fig05_assembly(selection):
    histories = list(selection.get("histories", ()))
    if len(histories) == 0 or len(histories) > FIG05_MAX_PANELS:
        raise ValueError(f"Fig. 05 requires between one and {FIG05_MAX_PANELS} selected histories.")
    if len({int(history["halo_id_z0"]) for history in histories}) != len(histories):
        raise ValueError("Fig. 05 selected histories contain repeated halo IDs.")

    t0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    x_right = float(t0 - Redshift2CosmicAge(FIG05_TARGET_REDSHIFT, time_unit="Gyr"))
    x_left = max(float(np.max(history["main"]["x_gyr"])) for history in histories)
    if not np.isfinite(x_left) or not np.isfinite(x_right) or x_left < x_right:
        raise ValueError("Fig. 05 selected histories do not share a valid high-z to z=7 time interval.")
    if x_left == x_right:
        x_left = float(np.nextafter(x_right, np.inf))

    mass_values = []
    maximum_redshift = FIG05_TARGET_REDSHIFT
    for history in histories:
        main = history["main"]
        mass_values.append(np.asarray(main["halo_mass_msun"], dtype=float))
        maximum_redshift = max(maximum_redshift, float(np.max(main["redshift"])))
        for satellite in history["satellites"]:
            mass_values.append(np.asarray(satellite["halo_mass_msun"], dtype=float))
    all_mass = np.concatenate(mass_values)
    if np.any(~np.isfinite(all_mass)) or np.any(all_mass <= 0.0):
        raise ValueError("Fig. 05 contains non-finite or non-positive plotted halo masses.")
    mass_min = float(np.min(all_mass))
    mass_max = float(np.max(all_mass))
    if mass_min == mass_max:
        mass_min = float(np.nextafter(mass_min, 0.0))
        mass_max = float(np.nextafter(mass_max, np.inf))
    redshift_ticks = _fig05_redshift_tick_values(maximum_redshift)
    redshift_x = np.asarray([t0 - Redshift2CosmicAge(float(z), time_unit="Gyr") for z in redshift_ticks], dtype=float)
    in_range = (redshift_x >= x_right - FIG05_REDSHIFT_ROW_ATOL) & (redshift_x <= x_left + FIG05_REDSHIFT_ROW_ATOL)
    redshift_ticks = redshift_ticks[in_range]
    redshift_x = redshift_x[in_range]
    if len(redshift_ticks) == 0:
        redshift_ticks = np.asarray([FIG05_TARGET_REDSHIFT], dtype=float)
        redshift_x = np.asarray([x_right], dtype=float)

    satellite_gc_counts = [
        int(satellite["n_gc_high_z"])
        for history in histories
        for satellite in history["satellites"]
    ]
    if any(count < 0 for count in satellite_gc_counts):
        raise ValueError("Fig. 05 satellite GC counts must be non-negative.")
    maximum_satellite_gc_count = max(satellite_gc_counts, default=0)
    satellite_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.0, float(maximum_satellite_gc_count)))
    satellite_cmap = plt.get_cmap(FIG05_SATELLITE_CMAP)
    colour_mappable = mpl.cm.ScalarMappable(norm=satellite_norm, cmap=satellite_cmap)
    colour_mappable.set_array(np.asarray(satellite_gc_counts, dtype=float))

    fig, axes = plt.subplots(3, 3, constrained_layout=True, dpi=STD_DPI, figsize=(14.2, 11.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(3, 3)
    legend_handles = [Line2D([], [], color="black", lw=1.35, label="Main progenitor")]
    if any(history["satellites"] for history in histories):
        legend_handles.extend([
            Line2D([], [], color="0.35", lw=0.9, label="Satellite branch"),
            Line2D([], [], marker="o", color="0.35", markerfacecolor="0.35", markeredgecolor="none", lw=0.0, label="Satellite maximum"),
        ])

    visible_index = 0
    for panel_index, ax in enumerate(axes.flat):
        row_index, column_index = divmod(panel_index, 3)
        if visible_index >= len(histories):
            ax.set_visible(False)
            continue
        history = histories[visible_index]
        visible_index += 1
        main = history["main"]
        ax.plot(main["x_gyr"], main["halo_mass_msun"], color="black", lw=1.35, zorder=4)
        for satellite in history["satellites"]:
            colour = satellite_cmap(satellite_norm(float(satellite["n_gc_high_z"])))
            ax.plot(satellite["x_gyr"], satellite["halo_mass_msun"], color=colour, lw=0.9, zorder=3)
            ax.plot(
                satellite["marker_x_gyr"],
                satellite["marker_halo_mass_msun"],
                marker="o",
                color=colour,
                markerfacecolor=colour,
                markeredgecolor="none",
                ms=3.8,
                linestyle="none",
                zorder=5,
            )
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(mass_min, mass_max)
        ax.set_xticks(redshift_x)
        ax.set_xticklabels([f"{float(z):g}" for z in redshift_ticks])
        ax.tick_params(direction="in", right=True, top=False, which="both", labelbottom=(row_index == 2), bottom=True)
        if row_index == 0:
            ax.tick_params(labeltop=False)
            lookback_axis = ax.twiny()
            lookback_axis.set_xlim(x_left, x_right)
            lookback_axis.set_xticks(redshift_x)
            lookback_axis.set_xticklabels([f"{float(x):.2f}" for x in redshift_x])
            lookback_axis.set_xlabel(r"Lookback time $t_{\rm lookback}$ [Gyr]")
            lookback_axis.tick_params(direction="in", top=True, bottom=False, which="both")
        if row_index == 2:
            ax.set_xlabel(r"Redshift $z$")
        if column_index == 0:
            ax.set_ylabel(r"Halo mass $M_{\rm h}$ [$M_{\odot}$]")
        score_annotation = _fig05_score_annotation(history)
        ax.text(
            0.04,
            0.96,
            f"{history['suite_label']}; $h_{{z=0}}$={int(history['halo_id_z0'])}\n"
            rf"$\log_{{10}}[M_{{\rm h,cat}}(z=7)/M_{{\odot}}]={float(history['catalogue_log10_halo_mass']):.2f}$; "
            f"$N_{{\\rm sat}}={int(history['n_satellites'])}$\n"
            f"{score_annotation}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            linespacing=1.0,
            color="0.15",
        )
        if visible_index == 1:
            ax.legend(handles=legend_handles, frameon=False, fontsize=7.0, loc="lower right", ncol=1)
        ax.grid(True, alpha=0.3, linestyle=":", which="both")

    colour_bar = fig.colorbar(colour_mappable, ax=axes, orientation="vertical", fraction=0.05, pad=0.025, aspect=25)
    colour_bar.set_label(r"GC count $N_{\rm GC}(z_{\rm form}\geq 7)$")
    colour_bar.locator = mpl.ticker.MaxNLocator(integer=True, nbins=6)
    colour_bar.update_ticks()
    return fig


# MAIN FUNCTION
def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Kong & Li 2026 suite b (rotation, BH-mass, UV-aperture, halo-distribution, and assembly figures) from one High-z SMBH Seeds output directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Model output directory.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Output plot directory. Default: <out_dir>/_plots_Kong&Li2026b.")
    parser.add_argument("--uv-table", type=Path, default=UV_CALIBRATION_PATH, help="FSPS-MIST/Chabrier pure-stellar 1500 Angstrom UV table per initially formed stellar mass; feh is log10(Z/Zsun).")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else out_dir / "_plots_Kong&Li2026b"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_run_metadata(out_dir)
    if "N_S" not in metadata:
        raise ValueError(f"run_metadata.json is missing required N_S: {out_dir / 'run_metadata.json'}")
    n_s = float(metadata["N_S"])
    if not np.isfinite(n_s) or n_s <= 0.0:
        raise ValueError(f"run_metadata N_S must be finite and positive, got {metadata['N_S']!r}.")
    summary_by_z = load_halo_summary_by_z(out_dir)
    final_gc = load_final_gc(out_dir)
    summary_by_z, final_gc, tng_volume_context = attach_tng_volume_weights(out_dir, summary_by_z, final_gc)
    _set_tng_volume_context(tng_volume_context)
    print(
        f"TNG volume normalisation: catalogue={TNG_CATALOGUE_ROOT}, "
        f"manifest_counts={tng_volume_context['manifest_counts']}, "
        f"output_counts={tng_volume_context['output_counts']}, "
        f"V_TNG50={tng_volume_context['volume_tng50_cmpc3']:.12g} cMpc^3, "
        f"V_TNG100={tng_volume_context['volume_tng100_cmpc3']:.12g} cMpc^3, "
        f"w100={tng_volume_context['tng100_weight']:.12g}."
    )
    final_redshift = float(metadata.get("final_redshift", 0.0))
    if not np.isfinite(final_redshift) or final_redshift < 0.0:
        raise ValueError(f"run_metadata final_redshift must be finite and non-negative, got {final_redshift!r}.")

    points, curves = load_juodzbalis2026_fig2()
    fig2_reference = load_juodzbalis2026_fig3_bh_masses()
    uv_calibration = load_uv_calibration(args.uv_table)
    all_fig01_z_rows = _select_fig01_z_rows(summary_by_z)
    all_deposit_profile = load_deposit_profile_for_redshift_summary(_deposit_path(out_dir), all_fig01_z_rows, final_redshift)
    missing_profile_halo_ids = np.asarray(all_deposit_profile["missing_halo_ids"], dtype=int)
    profile_max_radius_pc = np.asarray([float(np.asarray(rout, dtype=float)[-1]) * 1.0e3 for rout in all_deposit_profile["r_outer_kpc"]], dtype=float)
    keep_profile = np.isfinite(profile_max_radius_pc) & (profile_max_radius_pc >= FIG01_MATCH_RADIUS_RANGE_PC[1])
    all_halo_ids = _integer_values(all_deposit_profile["halo_ids"], "Fig. 01 all-profile halo IDs", non_negative=True)
    eligible_halo_ids = all_halo_ids[keep_profile]
    insufficient_profile_halo_ids = all_halo_ids[~keep_profile]
    eligible_fig01_z_rows = all_fig01_z_rows.loc[all_fig01_z_rows["halo_id_z0"].isin(eligible_halo_ids)].reset_index(drop=True)
    eligible_deposit_profile = {
        "halo_ids": eligible_halo_ids,
        "r_outer_kpc": [value for value, keep in zip(all_deposit_profile["r_outer_kpc"], keep_profile) if keep],
        "cumulative_mass_msun": [value for value, keep in zip(all_deposit_profile["cumulative_mass_msun"], keep_profile) if keep],
        "cumulative_formed_mass_msun": [value for value, keep in zip(all_deposit_profile["cumulative_formed_mass_msun"], keep_profile) if keep],
    }
    background_by_halo, background_exclusion_reasons = _fig01_build_background_parameters(
        eligible_fig01_z_rows,
        tng_volume_context["lookup"],
    )
    valid_background_halo_ids = np.asarray(sorted(background_by_halo), dtype=int)
    background_keep = np.isin(eligible_halo_ids, valid_background_halo_ids)
    background_excluded_halo_ids = eligible_halo_ids[~background_keep]
    eligible_fig01_z_rows = eligible_fig01_z_rows.loc[
        eligible_fig01_z_rows["halo_id_z0"].isin(valid_background_halo_ids)
    ].reset_index(drop=True)
    eligible_deposit_profile = _fig01_subset_deposit_profile(eligible_deposit_profile, background_keep)
    print(
        f"Fig. 01/Fig. 05 profile filter: missing deposited profiles={len(missing_profile_halo_ids)} "
        f"IDs={missing_profile_halo_ids.tolist()}; excluded {len(insufficient_profile_halo_ids)} halo(s) "
        f"with profile coverage <{FIG01_MATCH_RADIUS_RANGE_PC[1]:.0f} pc; "
        f"IDs={insufficient_profile_halo_ids.tolist()}; eligible profiles before background filter={len(eligible_halo_ids)}."
    )
    print(
        f"Fig. 01 background filter: fixed z={FIG01_TARGET_REDSHIFT:.2f}, N_S={FIG01_SERSIC_INDEX:.3g}, "
        f"excluded {len(background_excluded_halo_ids)} halo(s); IDs={background_excluded_halo_ids.tolist()}; "
        f"eligible velocity profiles={len(eligible_fig01_z_rows)}."
    )
    score_table, fig01_best = score_fig01_candidate_haloes(
        out_dir,
        points,
        eligible_fig01_z_rows,
        eligible_deposit_profile,
        final_gc,
        uv_calibration,
        background_by_halo=background_by_halo,
    )
    excluded_halo_ids = np.concatenate([
        missing_profile_halo_ids,
        insufficient_profile_halo_ids,
        background_excluded_halo_ids,
    ])
    if len(excluded_halo_ids) > 0:
        excluded_summary = all_fig01_z_rows.set_index("halo_id_z0").loc[excluded_halo_ids]
        excluded_table = pd.DataFrame(np.nan, index=np.arange(len(excluded_halo_ids)), columns=score_table.columns)
        excluded_table["index"] = -1
        excluded_table["halo_id_z0"] = excluded_halo_ids
        excluded_table["redshift"] = excluded_summary["redshift"].to_numpy(dtype=float)
        excluded_table["nsc_mass_msun"] = excluded_summary["nsc_mass_msun"].to_numpy(dtype=float)
        excluded_nsc = excluded_table["nsc_mass_msun"].to_numpy(dtype=float)
        excluded_table["log10_nsc_mass"] = np.nan
        valid_excluded_nsc = np.isfinite(excluded_nsc) & (excluded_nsc > 0.0)
        excluded_table.loc[valid_excluded_nsc, "log10_nsc_mass"] = np.log10(excluded_nsc[valid_excluded_nsc])
        excluded_bh = excluded_summary["central_bh_mass_final_msun"].to_numpy(dtype=float)
        excluded_table["central_bh_mass_msun"] = excluded_bh
        excluded_table["log10_central_bh_mass"] = np.nan
        valid_excluded_bh = np.isfinite(excluded_bh) & (excluded_bh > 0.0)
        excluded_table.loc[valid_excluded_bh, "log10_central_bh_mass"] = np.log10(excluded_bh[valid_excluded_bh])
        excluded_table["missing_reason"] = [
            f"no deposited stellar profile (depos.dat has no rows for halo_id_z0={hid})"
            for hid in missing_profile_halo_ids
        ] + [
            f"insufficient deposit radial coverage: {radius:.6g} pc < {FIG01_MATCH_RADIUS_RANGE_PC[1]:.6g} pc"
            for radius in profile_max_radius_pc[~keep_profile]
        ] + [
            background_exclusion_reasons[int(hid)]
            for hid in background_excluded_halo_ids
        ]
        score_table = pd.concat([score_table, excluded_table], ignore_index=True).sort_values("halo_id_z0").reset_index(drop=True)
    score_path = plot_dir / SCORE_FILENAME
    score_table.to_csv(score_path, index=False)
    print(f"Saved {score_path}")

    if fig01_best is None:
        raise ValueError(_candidate_no_score_error(out_dir, score_table))

    fig01 = plot_fig01_rotation_curve(
        points,
        curves,
        eligible_fig01_z_rows,
        eligible_deposit_profile,
        fig01_best,
        background_by_halo=background_by_halo,
    )
    z_values01 = eligible_fig01_z_rows["z_out"].to_numpy(dtype=float)
    print(f"Fig. 01 z selection: N={len(eligible_fig01_z_rows)}, z range={float(np.min(z_values01)):.3f}-{float(np.max(z_values01)):.3f}.")
    print(
        f"Fig. 01/Fig. 05 candidate pool: out_dir={out_dir}, metadata N_S={n_s:.3g}, "
        f"Fig. 01 N_S={FIG01_SERSIC_INDEX:.3g}, eligible profiles={len(eligible_fig01_z_rows)}, "
        f"CSV rows={len(score_table)}, finite Keplerian+UV scores={int(np.isfinite(score_table['score_keplerian_uv'].to_numpy(dtype=float)).sum())}."
    )
    print(
        "Fig. 01 velocity model: deterministic SMHM, deposited stars, NFW halo, and Sérsic background "
        f"at z={FIG01_TARGET_REDSHIFT:.2f} with fixed N_S={FIG01_SERSIC_INDEX:.3g}; "
        f"background exclusions={len(background_excluded_halo_ids)}."
    )
    print(f"UV mode: {uv_calibration['uv_mode']}.")
    print(f"UV table: {uv_calibration['path']} (age={uv_calibration['age_min_gyr']:.6g}-{uv_calibration['age_max_gyr']:.6g} Gyr, [Fe/H]={uv_calibration['feh_min']:.2f}-{uv_calibration['feh_max']:.2f}).")
    print(
        "Selected Keplerian+UV halo: "
        f"halo_id_z0={int(fig01_best['halo_id_z0'])}, z={float(fig01_best['redshift']):.3f}, "
        f"Keplerian term={float(fig01_best['keplerian_term']):.4f}, "
        f"weighted velocity chi2={float(fig01_best['keplerian_chi2_weighted']):.4f}, "
        f"velocity points={int(fig01_best['keplerian_n_points'])}, "
        f"M_UV={float(fig01_best['M_UV']):.2f}, target M_UV={QSO1_MUV_AB:.2f}, "
        f"UV term={float(fig01_best['uv_term']):.4f}, "
        f"score_keplerian_uv={float(fig01_best['score_keplerian_uv']):.4f}, "
        f"formed mass <{QSO1_NSC_APERTURE_PC:.1f} pc={float(fig01_best['formed_mass_6pc_msun']):.6g} Msun, "
        f"weighted age={float(fig01_best['weighted_age_gyr']):.3f} Gyr, "
        f"weighted [Fe/H]={float(fig01_best['weighted_feh']):.3f}, "
        f"nearest-grid counts(age, [Fe/H], any)=({int(fig01_best['n_uv_age_nearest_grid'])}, {int(fig01_best['n_uv_feh_nearest_grid'])}, {int(fig01_best['n_uv_any_nearest_grid'])}), "
        f"log10(M_NSC/Msun)={float(fig01_best['log10_nsc_mass']):.3f}, "
        f"log10(M_SMBH_final/Msun)={float(fig01_best['log10_central_bh_mass']):.3f}."
    )
    _save_figure(fig01, plot_dir / FIGURE_01_FILENAME)

    fig02 = plot_fig02_bh_masses(fig2_reference, fig01_best, fig01_best)
    _save_figure(fig02, plot_dir / FIGURE_02_FILENAME)

    if int(fig01_best["halo_id_z0"]) in set(insufficient_profile_halo_ids.tolist()):
        raise ValueError(f"Fig. 03 best halo_id_z0={int(fig01_best['halo_id_z0'])} is excluded for insufficient radial coverage.")
    aperture_table = estimate_uv_magnitude_apertures(all_deposit_profile, final_gc, int(fig01_best["halo_id_z0"]), uv_calibration)
    fig03 = plot_fig03_uvmag(aperture_table)
    _save_figure(fig03, plot_dir / FIGURE_03_FILENAME)

    fig04, inventory04 = plot_fig04_halo_distribution(summary_by_z, fig01_best)
    _save_figure(fig04, plot_dir / FIGURE_04_FILENAME)
    empty_redshifts04 = ", ".join(f"{float(z):.6g}" for z in inventory04["empty_redshifts"])
    missing_best_redshifts04 = ", ".join(f"{float(z):.6g}" for z in inventory04["best_halo_missing_redshifts"])
    print(
        f"Fig. 04 halo distribution: plotted redshifts={len(inventory04['redshifts'])}, "
        f"excluded unavailable rows={int(inventory04['excluded_unavailable_rows'])}, "
        f"empty redshifts=[{empty_redshifts04}], bin width={FIG04_DISTR_BIN_WIDTH_DEX:.2f} dex, "
        f"best halo_id_z0={int(inventory04['best_halo_id_z0'])}, "
        f"best-halo track lines={len(inventory04['best_halo_track'])}, "
        f"missing best-halo redshifts=[{missing_best_redshifts04}]."
    )

    fig05_selection = select_fig05_assembly_histories(out_dir, summary_by_z, final_gc, metadata, tng_volume_context, score_table, fig01_best)
    fig05 = plot_fig05_assembly(fig05_selection)
    _save_figure(fig05, plot_dir / FIGURE_05_FILENAME)
    print(
        "Fig. 05 assembly panels: "
        + ", ".join(
            f"{history['suite_label']} halo_id_z0={int(history['halo_id_z0'])} "
            f"log10M_h,cat(z=7)={float(history['catalogue_log10_halo_mass']):.4f} "
            f"raw-MPB-log10M_h(z=7)={float(history['main']['endpoint_log10_halo_mass']):.4f} "
            f"N_sat={int(history['n_satellites'])}; {_fig05_score_diagnostic(history)}"
            for history in fig05_selection["histories"]
        )
    )
    if fig05_selection["rejected"]:
        print(
            "Fig. 05 discarded comparison candidates: "
            + ", ".join(str(item["halo_id_z0"]) for item in fig05_selection["rejected"])
        )


if __name__ == "__main__":
    main()
