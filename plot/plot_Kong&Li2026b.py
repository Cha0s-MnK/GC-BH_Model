#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Kong & Li 2026 suite b figures for the High-z SMBH Seeds project."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import warnings

THREAD_CAP_DEFAULT = str(min(64, max(1, os.cpu_count() or 1)))
for env_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(env_name, THREAD_CAP_DEFAULT)

import matplotlib as mpl
mpl.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.default": "regular",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}",
})
import numpy as np
import pandas as pd
from scipy import special

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import *
from evo import STAT_SUNK_GC, read_haloevo_mpb  # noqa: E402
from run import (  # noqa: E402
    VALID_EVOLUTION_STATUS,
    _interpolate_mpb_logmh_at_redshift,
    _mpb_branch_id,
    _read_full_tree_numeric,
)

DATA_ROOT = PROJECT_ROOT / "data"
FSPS_ROOT = PROJECT_ROOT / "FSPS"
FIGURE_01_FILENAME = "Fig.01_RotationCurve.pdf"
FIGURE_02_FILENAME = "Fig.02_UVmag.pdf"
FIGURE_03_FILENAME = "Fig.03_distr.pdf"
FIGURE_04_FILENAME = "Fig.04_assembly.pdf"
SCORE_COLUMNS = (
    "halo_id_z0", "redshift", "nsc_mass_msun", "log10_nsc_mass",
    "central_bh_mass_msun", "log10_central_bh_mass", "formed_mass_6pc_msun",
    "weighted_age_gyr", "weighted_feh", "m1450_per_msun", "M_UV",
    "rotation_score", "rotation_n_points", "n_gc_contributors",
    "n_uv_valid_contributors", "n_uv_age_nearest_grid", "n_uv_feh_nearest_grid",
    "n_uv_any_nearest_grid", "min_raw_age_gyr", "max_raw_age_gyr",
    "min_raw_feh", "max_raw_feh", "min_eval_age_gyr", "max_eval_age_gyr",
    "min_eval_feh", "max_eval_feh", "missing_reason",
)

FIG01_SERSIC_INDEX = 2.2
FIG01_REDSHIFT_ATOL = 0.1
FIG01_MATCH_RADIUS_RANGE_PC = (10.0, 160.0)
FIG01_SCATTER_PERCENTILES = (16.0, 84.0)
FIG01_VELOCITY_SIN_I = np.sin(np.radians(52.0))
FIG01_RADIUS_MAX_PC = 160.0
FIG01_HALO_MASS_WINDOW_DEX = 0.1
FIG01_MOKA3D_LOG_MASS = 7.7
QSO1_MUV_AB = -16.98
QSO1_MUV_2026_AB = -15.60
QSO1_NSC_APERTURE_PC = 6.0
UV_APERTURES_PC = np.arange(1.0, 11.0, 1.0)
UV_MODEL_SPECS = (
    ("MIST+C3K_LR", FSPS_ROOT / "FSPS_MIST_C3KLR_Kroupa_m1450.csv"),
    ("PARSEC+C3K_LR", FSPS_ROOT / "FSPS_PARSEC_C3KLR_Kroupa_m1450.csv"),
    ("BPASS", FSPS_ROOT / "FSPS_BPASS_Kroupa_m1450.csv"),
)
UV_CALIBRATION_COLUMNS = ("age_gyr", "log10_age_gyr", "feh", "z_ratio", "M1450_AB_per_Msun")
FIG03_DISTR_BIN_WIDTH_DEX = 0.5
FIG04_TARGET_REDSHIFT = 7.0
FIG04_REDSHIFT_ROW_ATOL = 1.0e-10
FIG04_MAX_PANELS = 9
FIG04_SATELLITE_CMAP = "jet"
TNG_CATALOGUE_ROOT = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark")
TNG_TARGET_MANIFEST_FILENAME = "target_manifest_dark.csv"
TNG_TARGET_METADATA_FILENAME = "targets_z0_dark.json"
TNG_TREE_LOOKUP_FILENAME = "halo_tree_lookup.csv"
TNG_FIXED_TREE_ROOT = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark_Full/fixed_trees")


def _integer_values(values, name, non_negative=False):
    raw = np.asarray(values, dtype=object).reshape(-1)
    parsed = []
    for index, value in enumerate(raw):
        integer = parse_exact_int64(value, name=f"{name}[{index}]")
        if non_negative and integer < 0:
            raise ValueError(f"{name}[{index}] must be non-negative; got {integer}.")
        parsed.append(integer)
    return np.asarray(parsed, dtype=np.int64)


def _read_headered_whitespace_table(path):
    columns = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") and line[1:].strip():
                columns = line[1:].strip().split()
                break
    if columns is None:
        raise ValueError(f"Cannot find header columns in {path}")
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, dtype=str, engine="python")
    raw = raw.iloc[:, :len(columns)].copy()
    raw.columns = columns[:raw.shape[1]]
    return raw


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
    radius = np.asarray(deposit_profile["r_outer_kpc"][index], dtype=float) * 1.0e3
    cumulative = np.asarray(deposit_profile["cumulative_formed_mass_msun"][index], dtype=float)
    if (len(radius) == 0 or len(radius) != len(cumulative) or np.any(~np.isfinite(radius))
            or np.any(~np.isfinite(cumulative)) or np.any(cumulative < 0.0)
            or np.any(np.diff(radius) <= 0.0) or float(aperture_pc) < radius[0]
            or float(aperture_pc) > radius[-1]):
        return np.nan
    return float(np.interp(float(aperture_pc), radius, cumulative))


def select_qso1_gc_contributors(final_gc, halo_id, lookback_qso1_gyr):
    required = [
        "halo_id_z0", "status", "M_GC_final", "m_init_msun",
        "lookback_time_final_gyr", "lookback_time_init_gyr", "feh",
    ]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        return final_gc.iloc[0:0].copy(), f"final-GC catalogue missing {missing}"
    halo_ids = _integer_values(final_gc["halo_id_z0"], "final-GC parent halo IDs", non_negative=True)
    wanted = np.int64(parse_exact_int64(halo_id, name="QSO1 final-GC halo ID"))
    rows = final_gc.loc[halo_ids == wanted].copy()
    if len(rows) == 0:
        return rows, "no final-GC rows for halo"
    status = _integer_values(rows["status"], "selected final-GC statuses")
    lookback_final = pd.to_numeric(rows["lookback_time_final_gyr"], errors="coerce").to_numpy(dtype=float)
    lookback_init = pd.to_numeric(rows["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    mask = (
        (status == STAT_SUNK_GC) & np.isfinite(lookback_final) & np.isfinite(lookback_init)
        & (lookback_final >= float(lookback_qso1_gyr))
        & (lookback_init >= float(lookback_qso1_gyr))
    )
    contributors = rows.loc[mask].copy()
    if len(contributors) == 0:
        return contributors, "no clean STAT_SUNK_GC contributors before QSO1"
    return contributors, ""


def estimate_old_nsc_uv_mag(formed_mass_msun, contributors, lookback_eval_gyr, uv_calibration):
    formed_mass = float(formed_mass_msun) if np.isfinite(formed_mass_msun) else np.nan
    result = {
        "formed_mass_msun": formed_mass,
        "weighted_age_gyr": np.nan,
        "weighted_feh": np.nan,
        "m1450_per_msun": np.nan,
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
        "missing_reason": "",
    }
    if not (np.isfinite(formed_mass_msun) and float(formed_mass_msun) > 0.0):
        result["missing_reason"] = "formed aperture mass unavailable"
        return result
    if len(contributors) == 0:
        result["missing_reason"] = "no usable GC-origin age weights"
        return result

    age = pd.to_numeric(contributors["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float) - float(lookback_eval_gyr)
    weights = pd.to_numeric(contributors["M_GC_final"], errors="coerce").to_numpy(dtype=float)
    fallback = pd.to_numeric(contributors["m_init_msun"], errors="coerce").to_numpy(dtype=float)
    weights = weights.copy()
    bad_weights = ~np.isfinite(weights) | (weights <= 0.0)
    weights[bad_weights] = fallback[bad_weights]
    feh = pd.to_numeric(contributors["feh"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(age) & (age > 0.0) & np.isfinite(feh) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        result["missing_reason"] = "no finite positive GC-origin UV age, [Fe/H], and weight tuples"
        return result

    age, feh, weights = age[valid], feh[valid], weights[valid]
    age_grid = np.asarray(uv_calibration["age_gyr"], dtype=float)
    log_age_grid = np.asarray(uv_calibration["log10_age_gyr"], dtype=float)
    feh_grid = np.asarray(uv_calibration["feh"], dtype=float)
    m1450_grid = np.asarray(uv_calibration["m1450_grid"], dtype=float)
    age_nearest = (age < age_grid[0]) | (age > age_grid[-1])
    feh_nearest = (feh < feh_grid[0]) | (feh > feh_grid[-1])
    if np.any(feh_nearest):
        warnings.warn(
            f"{uv_calibration['uv_mode']} UV metallicity outside native grid clipped to {feh_grid[0]:.2f} <= feh <= {feh_grid[-1]:.2f}.",
            RuntimeWarning,
            stacklevel=2,
        )
    eval_age = np.clip(age, age_grid[0], age_grid[-1])
    eval_feh = np.clip(feh, feh_grid[0], feh_grid[-1])
    log_age = np.log10(eval_age)
    m1450 = np.full(age.shape, np.nan, dtype=float)
    for index, (x, y) in enumerate(zip(log_age, eval_feh)):
        i = int(np.clip(np.searchsorted(log_age_grid, x, side="right") - 1, 0, len(log_age_grid) - 2))
        j = int(np.clip(np.searchsorted(feh_grid, y, side="right") - 1, 0, len(feh_grid) - 2))
        x0, x1 = log_age_grid[i], log_age_grid[i + 1]
        y0, y1 = feh_grid[j], feh_grid[j + 1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        m1450[index] = ((1.0 - tx) * (1.0 - ty) * m1450_grid[i, j]
                        + tx * (1.0 - ty) * m1450_grid[i + 1, j]
                        + (1.0 - tx) * ty * m1450_grid[i, j + 1]
                        + tx * ty * m1450_grid[i + 1, j + 1])
    valid = np.isfinite(m1450)
    if not np.any(valid):
        result["missing_reason"] = "non-finite FSPS UV interpolation"
        return result
    age, feh, weights = age[valid], feh[valid], weights[valid]
    eval_age, eval_feh = eval_age[valid], eval_feh[valid]
    age_nearest, feh_nearest = age_nearest[valid], feh_nearest[valid]
    m1450 = m1450[valid]
    luminosity = np.power(10.0, -0.4 * m1450)
    weighted_luminosity = float(np.average(luminosity, weights=weights))
    if not (np.isfinite(weighted_luminosity) and weighted_luminosity > 0.0):
        result["missing_reason"] = "non-finite composite stellar UV luminosity"
        return result
    m1450_per_msun = float(-2.5 * np.log10(weighted_luminosity))
    result.update({
        "weighted_age_gyr": float(np.average(age, weights=weights)),
        "weighted_feh": float(np.average(feh, weights=weights)),
        "m1450_per_msun": m1450_per_msun,
        "M_UV": float(m1450_per_msun - 2.5 * np.log10(float(formed_mass_msun))),
        "n_uv_valid_contributors": int(len(age)),
        "n_uv_age_nearest_grid": int(np.sum(age_nearest)),
        "n_uv_feh_nearest_grid": int(np.sum(feh_nearest)),
        "n_uv_any_nearest_grid": int(np.sum(age_nearest | feh_nearest)),
        "min_raw_age_gyr": float(np.min(age)),
        "max_raw_age_gyr": float(np.max(age)),
        "min_raw_feh": float(np.min(feh)),
        "max_raw_feh": float(np.max(feh)),
        "min_eval_age_gyr": float(np.min(eval_age)),
        "max_eval_age_gyr": float(np.max(eval_age)),
        "min_eval_feh": float(np.min(eval_feh)),
        "max_eval_feh": float(np.max(eval_feh)),
    })
    return result


def estimate_uv_magnitude_apertures(deposit_profile, final_gc, halo_id, uv_calibrations):
    selection_lookback = _lookback_to_z0_gyr(7.04)
    age_lookback = _lookback_to_z0_gyr(7.04)
    contributors, reason = select_qso1_gc_contributors(final_gc, halo_id, selection_lookback)
    if reason:
        raise ValueError(f"Fig. 02 selected halo {int(halo_id)} has no usable 7.04-selected UV contributors: {reason}.")
    rows = []
    for aperture_pc in UV_APERTURES_PC:
        formed_mass = interpolate_formed_mass_inside_aperture(deposit_profile, halo_id, float(aperture_pc))
        if not np.isfinite(formed_mass) or formed_mass <= 0.0:
            raise ValueError(f"Fig. 02 selected halo {int(halo_id)} has no positive initially formed mass at R_UV={float(aperture_pc):.1f} pc.")
        row = {"aperture_pc": float(aperture_pc), "formed_mass_msun": float(formed_mass)}
        for uv_calibration in uv_calibrations:
            uv = estimate_old_nsc_uv_mag(formed_mass, contributors, age_lookback, uv_calibration)
            if uv["missing_reason"] or not np.isfinite(float(uv["M_UV"])):
                raise ValueError(f"Fig. 02 selected halo {int(halo_id)} cannot evaluate {uv_calibration['uv_mode']} at R_UV={float(aperture_pc):.1f} pc: {uv['missing_reason'] or 'non-finite UV magnitude'}.")
            row[uv_calibration["uv_mode"]] = float(uv["M_UV"])
        rows.append(row)
    return pd.DataFrame(rows)


def _velocity_profile_data(deposit_profile, z_rows, background_by_halo, best_halo_id=None):
    profile_halo_ids = _integer_values(deposit_profile["halo_ids"], "Fig. 01 profile halo IDs", non_negative=True)
    if best_halo_id is not None:
        mass = pd.to_numeric(z_rows["log10_halo_mass_at_redshift"], errors="coerce").to_numpy(dtype=float)
        ids = _integer_values(z_rows["halo_id_z0"], "Fig. 01 velocity-profile halo IDs", non_negative=True)
        matches = np.flatnonzero(ids == np.int64(parse_exact_int64(best_halo_id, name="Fig. 01 best halo ID")))
        if len(matches) != 1:
            raise ValueError(f"Fig. 01 best halo_id_z0={int(best_halo_id)} is not unique in z_rows.")
        best_index = int(matches[0])
        available = pd.to_numeric(z_rows.get("halo_mass_available", 1.0), errors="coerce").to_numpy(dtype=float) == 1.0
        if not available[best_index] or not np.isfinite(mass[best_index]):
            raise ValueError(f"Fig. 01 best halo_id_z0={int(best_halo_id)} has no valid redshift-resolved halo mass.")
        keep = available & np.isfinite(mass) & (np.abs(mass - mass[best_index]) <= FIG01_HALO_MASS_WINDOW_DEX)
        z_rows = z_rows.loc[keep].copy()
        profile_keep = np.isin(profile_halo_ids, _integer_values(z_rows["halo_id_z0"], "Fig. 01 filtered halo IDs", non_negative=True))
        if not np.any(profile_keep):
            raise ValueError("Fig. 01 halo-mass filter removed every deposit profile.")
        deposit_profile = dict(deposit_profile)
        deposit_profile["halo_ids"] = profile_halo_ids[profile_keep]
        for name in ("r_outer_kpc", "cumulative_mass_msun", "cumulative_formed_mass_msun"):
            if name in deposit_profile:
                deposit_profile[name] = [value for value, keep_value in zip(deposit_profile[name], profile_keep) if keep_value]

    selected_ids = _integer_values(z_rows["halo_id_z0"], "Fig. 01 selected halo IDs", non_negative=True)
    bh_by_halo = dict(zip(selected_ids, pd.to_numeric(z_rows["central_bh_mass_final_msun"], errors="coerce").to_numpy(dtype=float)))
    profile_halo_ids = _integer_values(deposit_profile["halo_ids"], "Fig. 01 profile halo IDs", non_negative=True)
    if set(bh_by_halo) != set(int(value) for value in profile_halo_ids):
        raise ValueError("Fig. 01 deposit-profile halo IDs do not match selected redshift rows.")
    first_radii = [float(value[0]) * 1.0e3 for value in deposit_profile["r_outer_kpc"]]
    last_radii = [float(value[-1]) * 1.0e3 for value in deposit_profile["r_outer_kpc"]]
    radius_min = max(1.0, max(first_radii))
    radius_max = min(FIG01_RADIUS_MAX_PC, min(last_radii))
    if radius_max < FIG01_MATCH_RADIUS_RANGE_PC[1]:
        raise ValueError(f"Fig. 01 deposit radial coverage is insufficient: {radius_min:.6g}-{radius_max:.6g} pc.")
    radius_pc = np.unique(np.concatenate([np.geomspace(radius_min, radius_max, 256), FIG01_MATCH_RADIUS_RANGE_PC]))
    component_profiles = {name: [] for name in ("deposited_stars", "sersic_stars", "nfw", "central_bh")}
    velocity_profiles = []
    for halo_id, outer_kpc, cumulative in zip(deposit_profile["halo_ids"], deposit_profile["r_outer_kpc"], deposit_profile["cumulative_mass_msun"]):
        outer_pc = np.asarray(outer_kpc, dtype=float) * 1.0e3
        deposited = np.asarray(cumulative, dtype=float)
        stellar = np.interp(radius_pc, outer_pc, deposited)
        parameters = background_by_halo[int(halo_id)]
        halo_mass = float(parameters["halo_mass_msun"])
        stellar_mass = float(parameters["mstar_msun"])
        effective_radius_pc = float(parameters["r_e_kpc"]) * 1.0e3
        scale_radius_pc = float(parameters["r_s_kpc"]) * 1.0e3
        c_vir = float(parameters["c_vir"])
        nfw_denominator = np.log1p(c_vir) - c_vir / (1.0 + c_vir)
        radius_over_scale = radius_pc / scale_radius_pc
        nfw = halo_mass * (np.log1p(radius_over_scale) - radius_over_scale / (1.0 + radius_over_scale)) / nfw_denominator
        sersic_p, sersic_b = Sersic_coefs(FIG01_SERSIC_INDEX)
        sersic_shape = FIG01_SERSIC_INDEX * (3.0 - sersic_p)
        sersic_argument = sersic_b * (radius_pc / effective_radius_pc) ** (1.0 / FIG01_SERSIC_INDEX)
        sersic = stellar_mass * special.gammainc(sersic_shape, sersic_argument)
        central_bh = float(bh_by_halo[int(halo_id)])
        #total = central_bh + stellar + sersic
        total = central_bh + stellar + sersic + nfw
        velocity_profiles.append(FIG01_VELOCITY_SIN_I * np.sqrt(G_Arepo * total / radius_pc))
        masses = {"deposited_stars": stellar, "sersic_stars": sersic, "nfw": nfw, "central_bh": np.full(radius_pc.shape, central_bh)}
        for name, values in masses.items():
            component_profiles[name].append(FIG01_VELOCITY_SIN_I * np.sqrt(G_Arepo * values / radius_pc))
    return {"halo_ids": profile_halo_ids, "radius_pc": radius_pc, "velocity_profiles": np.asarray(velocity_profiles, dtype=float), "component_velocity_profiles": {name: np.asarray(values, dtype=float) for name, values in component_profiles.items()}}


def score_fig01_candidate_haloes(z_rows, deposit_profile, final_gc, uv_calibration, background_by_halo):
    if len(z_rows) == 0:
        return pd.DataFrame(columns=SCORE_COLUMNS), None
    profile = _velocity_profile_data(deposit_profile, z_rows, background_by_halo)
    Moka3dVelocityKmS = FIG01_VELOCITY_SIN_I * np.sqrt(
        G_Arepo * 10.0 ** FIG01_MOKA3D_LOG_MASS / profile["radius_pc"]
    )
    rotation_scores = {}
    # The log-spaced grid gives the RMS equal weight per interval in log radius.
    for halo_id, velocity in zip(profile["halo_ids"], profile["velocity_profiles"]):
        valid = np.isfinite(velocity) & np.isfinite(Moka3dVelocityKmS)
        if not np.all(valid):
            rotation_scores[int(halo_id)] = {
                "rotation_score": np.nan,
                "rotation_n_points": int(np.sum(valid)),
                "missing_reason": "non-finite model/MOKA3D velocity",
            }
            continue
        residual = velocity - Moka3dVelocityKmS
        rotation_scores[int(halo_id)] = {
            "rotation_score": float(np.sqrt(np.mean(residual ** 2))),
            "rotation_n_points": len(residual),
            "missing_reason": "",
        }

    lookback_qso1 = _lookback_to_z0_gyr(7.04)
    rows = []
    for row in z_rows.sort_values("halo_id_z0").itertuples(index=False):
        halo_id = int(row.halo_id_z0)
        nsc_mass = float(row.nsc_mass_msun)
        central_bh = float(row.central_bh_mass_final_msun)
        formed_mass = interpolate_formed_mass_inside_aperture(deposit_profile, halo_id)
        contributors, contributor_reason = select_qso1_gc_contributors(final_gc, halo_id, lookback_qso1)
        uv = estimate_old_nsc_uv_mag(formed_mass, contributors, lookback_qso1, uv_calibration)
        missing = []
        if not np.isfinite(formed_mass) or formed_mass <= 0.0:
            missing.append("formed 6 pc mass unavailable")
        if contributor_reason:
            missing.append(contributor_reason)
        if uv["missing_reason"]:
            missing.append(uv["missing_reason"])
        M_UV = float(uv["M_UV"])
        uv_faint = np.isfinite(M_UV) and M_UV > QSO1_MUV_AB
        if np.isfinite(M_UV) and not uv_faint:
            missing.append(f"M_UV={M_UV:.2f} is not fainter than {QSO1_MUV_AB:.2f}")
        rotation_score = rotation_scores.get(halo_id, {"rotation_score": np.nan, "rotation_n_points": 0, "missing_reason": "MOKA3D rotation score unavailable"})
        if rotation_score["missing_reason"]:
            missing.append(rotation_score["missing_reason"])
        rows.append({
            "halo_id_z0": halo_id, "redshift": float(row.redshift), "nsc_mass_msun": nsc_mass,
            "log10_nsc_mass": float(np.log10(nsc_mass)) if np.isfinite(nsc_mass) and nsc_mass > 0.0 else np.nan,
            "central_bh_mass_msun": central_bh,
            "log10_central_bh_mass": float(np.log10(central_bh)) if np.isfinite(central_bh) and central_bh > 0.0 else np.nan,
            "formed_mass_6pc_msun": float(uv["formed_mass_msun"]), "weighted_age_gyr": float(uv["weighted_age_gyr"]),
            "weighted_feh": float(uv["weighted_feh"]) if np.isfinite(uv["weighted_feh"]) else np.nan,
            "m1450_per_msun": float(uv["m1450_per_msun"]) if np.isfinite(uv["m1450_per_msun"]) else np.nan,
            "M_UV": M_UV,
            "rotation_score": float(rotation_score["rotation_score"]),
            "rotation_n_points": int(rotation_score["rotation_n_points"]),
            "n_gc_contributors": int(uv["n_contributors"]), "n_uv_valid_contributors": int(uv["n_uv_valid_contributors"]),
            "n_uv_age_nearest_grid": int(uv["n_uv_age_nearest_grid"]), "n_uv_feh_nearest_grid": int(uv["n_uv_feh_nearest_grid"]),
            "n_uv_any_nearest_grid": int(uv["n_uv_any_nearest_grid"]), "min_raw_age_gyr": float(uv["min_raw_age_gyr"]),
            "max_raw_age_gyr": float(uv["max_raw_age_gyr"]), "min_raw_feh": float(uv["min_raw_feh"]),
            "max_raw_feh": float(uv["max_raw_feh"]), "min_eval_age_gyr": float(uv["min_eval_age_gyr"]),
            "max_eval_age_gyr": float(uv["max_eval_age_gyr"]), "min_eval_feh": float(uv["min_eval_feh"]),
            "max_eval_feh": float(uv["max_eval_feh"]), "missing_reason": "; ".join(dict.fromkeys(missing)),
        })
    score_table = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    rotation = score_table["rotation_score"].to_numpy(dtype=float)
    m_uv = score_table["M_UV"].to_numpy(dtype=float)
    finite = np.isfinite(rotation) & np.isfinite(m_uv) & (m_uv > QSO1_MUV_AB)
    if not np.any(finite):
        raise ValueError(f"No candidate halo has a finite MOKA3D rotation score and M_UV > {QSO1_MUV_AB:.2f} (fainter than the UV limit).")
    best = score_table.loc[finite].sort_values(["rotation_score", "halo_id_z0"], ascending=[True, True]).iloc[0].to_dict()
    return score_table, best


def plot_fig01_rotation_curve(points, curves, z_rows, deposit_profile, best, background_by_halo):
    profile = _velocity_profile_data(deposit_profile, z_rows, background_by_halo, best_halo_id=int(best["halo_id_z0"]))
    halo_ids, radius_pc, velocities = profile["halo_ids"], profile["radius_pc"], profile["velocity_profiles"]
    matches = np.flatnonzero(halo_ids == int(best["halo_id_z0"]))
    if len(matches) != 1:
        raise ValueError(f"Fig. 01 rotation-score best halo {int(best['halo_id_z0'])} is not present in the velocity-profile grid.")
    best_index = int(matches[0])
    low_velocity, high_velocity = np.percentile(velocities, FIG01_SCATTER_PERCENTILES, axis=0)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(5.4, 4.4))
    for curve_name, label, kwargs in (
        ("point_mass_keplerian", r"Keplerian point mass $\log M_\bullet = 6.75$", {"color": "black", "linewidth": 1.7, "zorder": 4}),
        ("mw_nsc", "MW-like NSC model", {"color": "0.45", "linestyle": "dashdot", "linewidth": 1.5, "zorder": 3}),
    ):
        used_label = False
        for sign in (-1.0, 1.0):
            rows = curves.loc[curves["curve"].eq(curve_name) & (np.sign(curves["r_pc"].to_numpy(dtype=float)) == sign)].sort_values("r_pc")
            if len(rows):
                ax.plot(rows["r_pc"].to_numpy(dtype=float), rows["v_km_s"].to_numpy(dtype=float), label=label if not used_label else None, **kwargs)
                used_label = True
    Moka3dRadiusPc = np.geomspace(1.0, FIG01_RADIUS_MAX_PC, 256)
    Moka3dVelocityKmS = FIG01_VELOCITY_SIN_I * np.sqrt(G_Arepo * 10.0 ** FIG01_MOKA3D_LOG_MASS / Moka3dRadiusPc)
    ax.plot(
        np.concatenate([-Moka3dRadiusPc[::-1], Moka3dRadiusPc]),
        np.concatenate([-Moka3dVelocityKmS[::-1], Moka3dVelocityKmS]),
        color="black",
        linestyle="--",
        linewidth=1.7,
        label="MOKA3D",
        zorder=4,
    )
    for component, colour, marker, size, label in (
        ("resolved_kinematics", "tab:blue", "o", 5.0, "Resolved kinematics"),
        ("spectroastrometry", "magenta", "X", 6.0, "Spectroastrometry"),
        ("spectroastrometry_fine", "orchid", "P", 5.5, "Spectroastrometry, fine split"),
    ):
        rows = points.loc[points["component"].eq(component)]
        if len(rows) == 0:
            continue
        xerr = rows[["r_err_low_pc", "r_err_high_pc"]].to_numpy(dtype=float).T
        yerr = rows[["v_err_low_km_s", "v_err_high_km_s"]].to_numpy(dtype=float).T
        ax.errorbar(rows["r_pc"], rows["v_km_s"], xerr=np.where(np.isfinite(xerr), xerr, 0.0), yerr=np.where(np.isfinite(yerr), yerr, 0.0), fmt=marker, ms=size, color=colour, ecolor=colour, elinewidth=1.0, markeredgecolor=colour, markerfacecolor=colour, capsize=0.0, linestyle="none", label=label, zorder=6)
    ax.fill_between(radius_pc, low_velocity, high_velocity, color="tab:green", alpha=0.16, linewidth=0.0, label=r"$z \simeq 7$ stack 16-84\%")
    ax.fill_between(-radius_pc[::-1], -high_velocity[::-1], -low_velocity[::-1], color="tab:green", alpha=0.16, linewidth=0.0)
    signed_radius = np.concatenate([-radius_pc[::-1], radius_pc])
    signed_best = np.concatenate([-velocities[best_index][::-1], velocities[best_index]])
    for component, colour, linestyle, label in (
        ("central_bh", "tab:purple", "--", "Central BH only"),
        ("deposited_stars", "tab:blue", ":", "Deposited stars only"),
        ("sersic_stars", "tab:orange", "-.", "Sérsic stars only"),
        ("nfw", "tab:brown", (0, (5, 1, 1, 1)), "NFW only"),
    ):
        component_velocity = profile["component_velocity_profiles"][component][best_index]
        ax.plot(signed_radius, np.concatenate([-component_velocity[::-1], component_velocity]), color=colour, linewidth=1.0, linestyle=linestyle, label=label, zorder=2)
    ax.plot(signed_radius, signed_best, color="tab:red", linewidth=1.5, label="Best rotation-score halo")
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


def plot_fig02_uvmag(aperture_table):
    model_names = [name for name, _ in UV_MODEL_SPECS]
    required = ["aperture_pc"] + model_names
    missing = [name for name in required if name not in aperture_table.columns]
    if missing:
        raise ValueError(f"Fig. 02 aperture table is missing required columns: {missing}")
    aperture_pc = aperture_table["aperture_pc"].to_numpy(dtype=float)
    if len(aperture_pc) != len(UV_APERTURES_PC) or np.any(~np.isfinite(aperture_pc)):
        raise ValueError("Fig. 02 aperture table must contain seventeen finite aperture values.")
    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.7))
    magnitudes = [QSO1_MUV_AB, QSO1_MUV_2026_AB]
    for uv_mode in model_names:
        m_uv = aperture_table[uv_mode].to_numpy(dtype=float)
        if np.any(~np.isfinite(m_uv)):
            raise ValueError(f"Fig. 02 aperture table must contain seventeen finite UV-magnitude values for {uv_mode}.")
        magnitudes.extend(m_uv.tolist())
        ax.plot(aperture_pc, m_uv, marker="o", ms=5.5, label=uv_mode)
    ax.axhline(QSO1_MUV_AB, c="black", ls=":", label=rf"QSO1 $M_{{\rm UV}}$ (2024) = {QSO1_MUV_AB:.2f}")
    ax.axhline(QSO1_MUV_2026_AB, c="black", ls="--", label=rf"QSO1 $M_{{\rm UV}}$ (2026) = {QSO1_MUV_2026_AB:.2f}")
    ax.set_xlabel(r"UV aperture $R_{\rm UV}$ [pc]")
    ax.set_ylabel(r"Rest-frame $1450\,\AA$ absolute AB magnitude $M_{\rm UV}$")
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(UV_APERTURES_PC)
    y_min, y_max = min(magnitudes), max(magnitudes)
    y_span = max(y_max - y_min, 0.5)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.08 * y_span)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def plot_fig03_halo_distribution(summary_by_z, best):
    table = summary_by_z.loc[:, ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift"]].copy()
    table["halo_id_z0"] = _integer_values(table["halo_id_z0"], "Fig. 03 haloSummaryByZ IDs", non_negative=True)
    for column in ("redshift", "halo_mass_available", "log10_halo_mass_at_redshift"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_ids = table["halo_id_z0"].to_numpy(dtype=np.int64)
    redshifts = table["redshift"].to_numpy(dtype=float)
    available = table["halo_mass_available"].to_numpy(dtype=float) == 1.0
    log_mass = table["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        halo_mass = np.power(10.0, log_mass)
    valid_mass = available & np.isfinite(log_mass) & np.isfinite(halo_mass) & (halo_mass > 0.0)
    values = log_mass[valid_mass]
    if len(values) == 0:
        raise ValueError("Fig. 03 cannot plot a halo distribution because no valid MPB halo masses remain.")
    lo = FIG03_DISTR_BIN_WIDTH_DEX * math.floor(float(values.min()) / FIG03_DISTR_BIN_WIDTH_DEX)
    hi = FIG03_DISTR_BIN_WIDTH_DEX * math.ceil(float(values.max()) / FIG03_DISTR_BIN_WIDTH_DEX)
    if hi <= lo:
        hi = lo + FIG03_DISTR_BIN_WIDTH_DEX
    log_edges = np.arange(lo, hi + 0.5 * FIG03_DISTR_BIN_WIDTH_DEX, FIG03_DISTR_BIN_WIDTH_DEX)
    distributions, empty_redshifts = [], []
    for redshift in np.sort(np.unique(redshifts)):
        valid = (redshifts == float(redshift)) & valid_mass
        if not np.any(valid):
            empty_redshifts.append(float(redshift))
            continue
        counts, _ = np.histogram(log_mass[valid], bins=log_edges)
        distributions.append({"redshift": float(redshift), "counts": counts.astype(int), "halo_mass_msun": halo_mass[valid]})
    best_id = int(parse_exact_int64(best["halo_id_z0"], name="Fig. 03 best halo_id_z0"))
    best_track, missing_best = [], []
    for redshift in np.sort(np.unique(redshifts)):
        selected = (halo_ids == best_id) & (redshifts == float(redshift))
        if not np.any(selected) or not valid_mass[np.flatnonzero(selected)[0]]:
            missing_best.append(float(redshift))
            continue
        index = int(np.flatnonzero(selected)[0])
        best_track.append({"redshift": float(redshift), "halo_mass_msun": float(halo_mass[index])})
    if not best_track:
        raise ValueError(f"Fig. 03 best halo_id_z0={best_id} has no valid MPB halo mass at any output redshift.")
    mass_edges = np.power(10.0, log_edges)
    unique_redshifts = np.unique(redshifts)
    norm = mpl.colors.Normalize(vmin=float(redshifts.min()) - 0.5, vmax=float(redshifts.max()) + 0.5) if len(unique_redshifts) == 1 else mpl.colors.Normalize(vmin=float(redshifts.min()), vmax=float(redshifts.max()))
    cmap = mpl.cm.jet
    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    maximum_count = 0
    for item in distributions:
        maximum_count = max(maximum_count, int(np.max(item["counts"])))
        ax.stairs(item["counts"], mass_edges, baseline=0.0, fill=False, color=cmap(norm(item["redshift"])), lw=1.35, alpha=0.92, zorder=2)
    for track in best_track:
        ax.axvline(track["halo_mass_msun"], color=cmap(norm(track["redshift"])), ls="--", lw=1.0, alpha=0.45, zorder=5)
    ax.plot([], [], color="0.35", lw=1.5, label="Tracked halo distribution")
    ax.plot([], [], color="0.20", ls="--", lw=1.0, alpha=0.45, label=r"Best halo $M_{\rm h}(z)$")
    ax.set_xscale("log")
    ax.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
    ax.set_ylim(0.0, max(1.0, 1.15 * maximum_count))
    ax.set_xlabel(r"Halo mass $M_{\rm h}(z)$ [$M_{\odot}$]")
    ax.set_ylabel("Halo number per mass bin")
    ax.text(0.03, 0.96, rf"Tracked MPB final-halo sample; best $h_{{z=0}}$={best_id}" + "\n" + r"coloured dashed lines: selected halo $M_{\rm h}(z)$", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color="0.20")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="upper right", fontsize=7.4, ncol=1)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(distributions) == 1:
        colour_bar.set_ticks([float(distributions[0]["redshift"])])
    return fig, {"redshifts": np.asarray([item["redshift"] for item in distributions]), "empty_redshifts": np.asarray(empty_redshifts), "best_halo_id_z0": best_id, "best_halo_track": best_track, "best_halo_missing_redshifts": np.asarray(missing_best), "excluded_unavailable_rows": int(np.count_nonzero(~available))}


def plot_fig04_assembly(out_dir, summary_by_z, final_gc, context, score_table, best):
    """Select raw-tree histories and draw the shifted Fig. 05 assembly figure."""
    best_id = int(parse_exact_int64(best["halo_id_z0"], name="Fig. 04 best halo ID"))
    target_scores = score_table.loc[np.abs(score_table["redshift"].to_numpy(dtype=float) - FIG04_TARGET_REDSHIFT) <= FIG04_REDSHIFT_ROW_ATOL]
    scores_by_halo = {}
    for row in target_scores.itertuples(index=False):
        rotation = float(row.rotation_score)
        scores_by_halo[int(row.halo_id_z0)] = {"score_rotation": rotation, "score_rotation_available": bool(np.isfinite(rotation))}
    allcat_paths = sorted(Path(out_dir).resolve().glob("allcat_*.txt"))
    if len(allcat_paths) != 1:
        raise ValueError(f"Fig. 04 requires exactly one allcat_*.txt catalogue in {out_dir}, found {len(allcat_paths)}")
    allcat_path = allcat_paths[0]
    header = None
    with allcat_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                fields = line[1:].strip().split()
                if {"hid_z0", "isMPB", "subfind_form"}.issubset(fields):
                    header = fields
                    break
    required = ["hid_z0", "logMh_form", "zform", "isMPB", "subfind_form"]
    formation = pd.read_csv(allcat_path, sep=r"\s+", comment="#", header=None, names=header, usecols=required, dtype=str)
    formation_halo = _integer_values(formation["hid_z0"], "allcat halo IDs", non_negative=True)
    formation_logmh = formation["logMh_form"].to_numpy(dtype=float)
    formation_z = formation["zform"].to_numpy(dtype=float)
    formation_mpb = formation["isMPB"].to_numpy(dtype=np.int64)
    formation_subfind = _integer_values(formation["subfind_form"], "allcat subfind_form", non_negative=True)
    formation_status = _integer_values(final_gc["status"], "finalGCs statuses")
    exact = np.isclose(summary_by_z["redshift"].to_numpy(dtype=float), FIG04_TARGET_REDSHIFT, rtol=0.0, atol=FIG04_REDSHIFT_ROW_ATOL)
    exact_table = summary_by_z.loc[exact, ["halo_id_z0", "redshift", "halo_mass_available", "log10_halo_mass_at_redshift"]].copy()
    exact_table = exact_table.rename(columns={"redshift": "catalogue_redshift", "halo_mass_available": "catalogue_mass_available", "log10_halo_mass_at_redshift": "catalogue_log10_halo_mass"})
    exact_table["halo_id_z0"] = _integer_values(exact_table["halo_id_z0"], "Fig. 04 catalogue halo IDs", non_negative=True)
    exact_table["catalogue_halo_mass_msun"] = np.power(10.0, exact_table["catalogue_log10_halo_mass"].to_numpy(dtype=float))
    exact_table["catalogue_mass_valid"] = (exact_table["catalogue_mass_available"].to_numpy(dtype=float) == 1.0) & np.isfinite(exact_table["catalogue_halo_mass_msun"])
    lookup = context["lookup"].copy()
    lookup["halo_id_z0"] = _integer_values(lookup["halo_id_z0"], "Fig. 04 TNG lookup halo IDs", non_negative=True)
    candidate_table = exact_table.merge(lookup[["halo_id_z0", "simulation_key", "simulation", "fixed_tree_basename"]], on="halo_id_z0", how="left", validate="one_to_one")
    comparisons = candidate_table.loc[candidate_table["catalogue_mass_valid"] & (candidate_table["halo_id_z0"] != best_id)].copy()
    best_mass = float(candidate_table.loc[candidate_table["halo_id_z0"] == best_id, "catalogue_halo_mass_msun"].iloc[0])
    comparisons["mass_delta_msun"] = np.abs(comparisons["catalogue_halo_mass_msun"] - best_mass)
    comparisons = comparisons.sort_values(["mass_delta_msun", "halo_id_z0", "fixed_tree_basename"], kind="mergesort")
    histories, rejected = [], []
    candidate_rows = pd.concat([candidate_table.loc[candidate_table["halo_id_z0"] == best_id], comparisons])
    for _, candidate_row in candidate_rows.iterrows():
        if len(histories) >= FIG04_MAX_PANELS:
            break
        halo_id = int(candidate_row["halo_id_z0"])
        try:
            basename = str(candidate_row["fixed_tree_basename"]).strip()
            tree_path = (TNG_FIXED_TREE_ROOT / basename).resolve()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Malformed fixed-tree row .*row=log10_mhalo_msun first_progenitor_id.*", category=RuntimeWarning)
                tree_rows = _read_full_tree_numeric(tree_path)
            mpb_branch = int(_mpb_branch_id(tree_rows))
            mpb_rows = np.asarray(read_haloevo_mpb(tree_path), dtype=float)
            logmh, redshift = mpb_rows[:, 0], mpb_rows[:, 1]
            exact_mpb = np.flatnonzero(np.abs(redshift - FIG04_TARGET_REDSHIFT) <= FIG04_REDSHIFT_ROW_ATOL)
            if len(exact_mpb):
                endpoint_logmh = float(logmh[int(exact_mpb[0])])
            else:
                endpoint_logmh, available_mpb = _interpolate_mpb_logmh_at_redshift(mpb_rows, FIG04_TARGET_REDSHIFT)
                if int(available_mpb) != 1:
                    raise ValueError(f"raw MPB cannot be interpolated to z={FIG04_TARGET_REDSHIFT:g}")
            visible = redshift >= FIG04_TARGET_REDSHIFT
            if len(exact_mpb):
                visible[int(exact_mpb[0])] = False
            visible_logmh, visible_redshift = logmh[visible], redshift[visible]
            visible_time = np.asarray([Redshift2CosmicAge(float(value), time_unit="Gyr") for value in visible_redshift])
            order = np.argsort(visible_time, kind="mergesort")
            visible_logmh, visible_redshift, visible_time = visible_logmh[order], visible_redshift[order], visible_time[order]
            t0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
            endpoint_x = t0 - float(Redshift2CosmicAge(FIG04_TARGET_REDSHIFT, time_unit="Gyr"))
            main = {"x_gyr": np.concatenate([t0 - visible_time, [endpoint_x]]), "redshift": np.concatenate([visible_redshift, [FIG04_TARGET_REDSHIFT]]), "halo_mass_msun": np.power(10.0, np.concatenate([visible_logmh, [endpoint_logmh]])), "endpoint_log10_halo_mass": endpoint_logmh}
            rows = np.flatnonzero(formation_halo == halo_id)
            branch_ids = np.empty(len(rows), dtype=np.int64)
            candidates_by_subfind = {}
            for tree_row in tree_rows:
                candidates_by_subfind.setdefault(int(tree_row[2]), []).append((int(tree_row[3]), float(tree_row[5]), float(tree_row[0])))
            for local_index, source_index in enumerate(rows):
                options = candidates_by_subfind.get(int(formation_subfind[source_index]), [])
                scored = sorted((abs(z - formation_z[source_index]) + abs(mass - formation_logmh[source_index]), branch, z, mass) for branch, z, mass in options)
                if not scored or scored[0][0] > 1.0e-3:
                    raise ValueError(f"formation subfind_form={int(formation_subfind[source_index])} is absent from the raw tree")
                branch_ids[local_index] = scored[0][1]
            expected_mpb = branch_ids == mpb_branch
            if len(rows) and not np.array_equal(formation_mpb[rows].astype(bool), expected_mpb):
                raise ValueError(f"allcat isMPB disagrees with raw-tree branch membership for halo {halo_id}")
            tree_branch = np.asarray(tree_rows[:, 3], dtype=np.int64)
            tree_logmh, tree_redshift = np.asarray(tree_rows[:, 0], dtype=float), np.asarray(tree_rows[:, 5], dtype=float)
            tree_mass = np.power(10.0, tree_logmh)
            high_z = np.isin(formation_status[rows], np.asarray(sorted(VALID_EVOLUTION_STATUS))) & (formation_z[rows] >= FIG04_TARGET_REDSHIFT)
            satellites = []
            for branch in sorted(set(int(value) for value in tree_branch if int(value) != mpb_branch)):
                visible = (tree_branch == branch) & (tree_redshift >= FIG04_TARGET_REDSHIFT)
                if not np.any(visible):
                    continue
                satellite_logmh, satellite_redshift, satellite_mass = tree_logmh[visible], tree_redshift[visible], tree_mass[visible]
                satellite_time = np.asarray([Redshift2CosmicAge(float(value), time_unit="Gyr") for value in satellite_redshift])
                order = np.argsort(satellite_time, kind="mergesort")
                satellite_logmh, satellite_redshift, satellite_mass, satellite_time = satellite_logmh[order], satellite_redshift[order], satellite_mass[order], satellite_time[order]
                maximum = float(np.max(satellite_logmh))
                marker = int(np.flatnonzero(satellite_logmh == maximum)[-1])
                satellites.append({"n_gc_high_z": int(np.count_nonzero(high_z & (branch_ids == branch))), "x_gyr": t0 - satellite_time, "redshift": satellite_redshift, "halo_mass_msun": satellite_mass, "marker_x_gyr": float(t0 - satellite_time[marker]), "marker_halo_mass_msun": float(satellite_mass[marker])})
            history = candidate_row.to_dict()
            history.update({"halo_id_z0": halo_id, "suite_label": {"tng50_1_dark": "TNG50", "tng100_1_dark": "TNG100"}.get(str(candidate_row["simulation_key"]), str(candidate_row["simulation_key"])), "catalogue_log10_halo_mass": float(candidate_row["catalogue_log10_halo_mass"]), "main": main, "satellites": satellites, "n_satellites": len(satellites)})
            history.update(scores_by_halo.get(halo_id, {"score_rotation": np.nan, "score_rotation_available": False}))
            histories.append(history)
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, OverflowError, IndexError) as error:
            if halo_id == best_id:
                raise
            rejected.append({"halo_id_z0": halo_id, "reason": str(error)})
    if not histories or histories[0]["halo_id_z0"] != best_id:
        raise ValueError(f"Fig. 04 best halo {best_id} could not be loaded as the first assembly panel.")

    t0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    x_right = t0 - float(Redshift2CosmicAge(FIG04_TARGET_REDSHIFT, time_unit="Gyr"))
    x_left = max(float(np.max(history["main"]["x_gyr"])) for history in histories)
    mass_values = [history["main"]["halo_mass_msun"] for history in histories] + [satellite["halo_mass_msun"] for history in histories for satellite in history["satellites"]]
    all_mass = np.concatenate(mass_values)
    mass_min, mass_max = float(np.min(all_mass)), float(np.max(all_mass))
    if mass_min == mass_max:
        mass_min, mass_max = float(np.nextafter(mass_min, 0.0)), float(np.nextafter(mass_max, np.inf))
    maximum_redshift = max(FIG04_TARGET_REDSHIFT, *(float(np.max(history["main"]["redshift"])) for history in histories))
    standard_ticks = np.asarray([7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])
    redshift_ticks = standard_ticks[standard_ticks <= maximum_redshift + FIG04_REDSHIFT_ROW_ATOL]
    if len(redshift_ticks) == 0:
        redshift_ticks = np.asarray([FIG04_TARGET_REDSHIFT])
    redshift_x = np.asarray([t0 - Redshift2CosmicAge(float(value), time_unit="Gyr") for value in redshift_ticks])
    keep_ticks = (redshift_x >= x_right - FIG04_REDSHIFT_ROW_ATOL) & (redshift_x <= x_left + FIG04_REDSHIFT_ROW_ATOL)
    redshift_ticks, redshift_x = redshift_ticks[keep_ticks], redshift_x[keep_ticks]
    if len(redshift_ticks) == 0:
        redshift_ticks, redshift_x = np.asarray([FIG04_TARGET_REDSHIFT]), np.asarray([x_right])
    counts = [int(satellite["n_gc_high_z"]) for history in histories for satellite in history["satellites"]]
    satellite_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(1.0, float(max(counts, default=0))))
    satellite_cmap = plt.get_cmap(FIG04_SATELLITE_CMAP)
    mappable = mpl.cm.ScalarMappable(norm=satellite_norm, cmap=satellite_cmap)
    mappable.set_array(np.asarray(counts, dtype=float))
    fig, axes = plt.subplots(3, 3, constrained_layout=True, dpi=STD_DPI, figsize=(14.2, 11.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(3, 3)
    legend_handles = [Line2D([], [], color="black", lw=1.35, label="Main progenitor")]
    if any(history["satellites"] for history in histories):
        legend_handles.extend([Line2D([], [], color="0.35", lw=0.9, label="Satellite branch"), Line2D([], [], marker="o", color="0.35", markerfacecolor="0.35", markeredgecolor="none", lw=0.0, label="Satellite maximum")])
    visible_index = 0
    for panel_index, ax in enumerate(axes.flat):
        row_index, column_index = divmod(panel_index, 3)
        if visible_index >= len(histories):
            ax.set_visible(False)
            continue
        history = histories[visible_index]
        visible_index += 1
        ax.plot(history["main"]["x_gyr"], history["main"]["halo_mass_msun"], color="black", lw=1.35, zorder=4)
        for satellite in history["satellites"]:
            colour = satellite_cmap(satellite_norm(float(satellite["n_gc_high_z"])))
            ax.plot(satellite["x_gyr"], satellite["halo_mass_msun"], color=colour, lw=0.9, zorder=3)
            ax.plot(satellite["marker_x_gyr"], satellite["marker_halo_mass_msun"], marker="o", color=colour, markerfacecolor=colour, markeredgecolor="none", ms=3.8, linestyle="none", zorder=5)
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(mass_min, mass_max)
        ax.set_xticks(redshift_x)
        ax.set_xticklabels([f"{float(value):g}" for value in redshift_ticks])
        ax.tick_params(direction="in", right=True, top=False, which="both", labelbottom=(row_index == 2), bottom=True)
        if row_index == 0:
            ax.tick_params(labeltop=False)
            lookback_axis = ax.twiny()
            lookback_axis.set_xlim(x_left, x_right)
            lookback_axis.set_xticks(redshift_x)
            lookback_axis.set_xticklabels([f"{float(value):.2f}" for value in redshift_x])
            lookback_axis.set_xlabel(r"Lookback time $t_{\rm lookback}$ [Gyr]")
            lookback_axis.tick_params(direction="in", top=True, bottom=False, which="both")
        if row_index == 2:
            ax.set_xlabel(r"Redshift $z$")
        if column_index == 0:
            ax.set_ylabel(r"Halo mass $M_{\rm h}$ [$M_{\odot}$]")
        score = history
        score_rotation = rf"$S_{{\rm rot}}={float(score['score_rotation']):.2f}$" if score["score_rotation_available"] else r"$S_{\rm rot}$=n/a"
        score_annotation = score_rotation
        ax.text(0.04, 0.96, f"{history['suite_label']}; $h_{{z=0}}$={int(history['halo_id_z0'])}\n$\\log_{{10}}[M_{{\\rm h,cat}}(z=7)/M_{{\\odot}}]={float(history['catalogue_log10_halo_mass']):.2f}$; $N_{{\\rm sat}}={int(history['n_satellites'])}$\n{score_annotation}", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, linespacing=1.0, color="0.15")
        if visible_index == 1:
            ax.legend(handles=legend_handles, frameon=False, fontsize=7.0, loc="lower right", ncol=1)
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
    colour_bar = fig.colorbar(mappable, ax=axes, orientation="vertical", fraction=0.05, pad=0.025, aspect=25)
    colour_bar.set_label(r"GC count $N_{\rm GC}(z_{\rm form}\geq 7)$")
    colour_bar.locator = mpl.ticker.MaxNLocator(integer=True, nbins=6)
    colour_bar.update_ticks()
    return fig, {"histories": histories, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description="Plot Kong & Li 2026 suite b (rotation, UV-aperture, halo-distribution, and assembly figures) from one High-z SMBH Seeds output directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Model output directory.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Output plot directory. Default: <out_dir>/_plots_Kong&Li2026b.")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else out_dir / "_plots_Kong&Li2026b"
    plot_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    n_s = float(metadata["N_S"])
    final_redshift = float(metadata.get("final_redshift", 0.0))
    summary = pd.read_csv(out_dir / "haloSummaryByZ.csv", dtype=str).rename(columns={
        "hid_z0": "halo_id_z0", "z_out": "redshift", "logMh_z_msun": "log10_halo_mass_at_redshift",
        "M_NSC": "nsc_mass_msun", "M_SMBH_init": "central_bh_mass_init_msun", "M_SMBH_final": "central_bh_mass_final_msun",
        "z_depos_sampled": "deposit_sample_redshift", "lookback_depos_sampled_gyr": "deposit_sample_lookback_gyr",
    })
    summary["halo_id_z0"] = _integer_values(summary["halo_id_z0"], "haloSummaryByZ halo IDs", non_negative=True)
    for column in summary.columns:
        if column != "halo_id_z0":
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    available = summary["halo_mass_available"].to_numpy(dtype=float) == 1.0
    log_mass = summary["log10_halo_mass_at_redshift"].to_numpy(dtype=float)
    summary["halo_mass_at_redshift_msun"] = np.where(available & np.isfinite(log_mass), np.power(10.0, log_mass), np.nan)
    summary["mstar_z_smhm_msun"] = np.asarray([Mstar_SMHM(float(mass), float(redshift), scatter=False) if np.isfinite(mass) and mass > 0.0 and np.isfinite(redshift) and redshift >= 0.0 else np.nan for mass, redshift in zip(summary["halo_mass_at_redshift_msun"], summary["redshift"])])
    final_gc = _read_headered_whitespace_table(out_dir / "finalGCs.dat")
    for column in ("halo_id_z0", "gc_index_halo", "status"):
        if column in final_gc:
            final_gc[column] = _integer_values(final_gc[column], f"finalGCs {column}", non_negative=column != "status")
    for column in final_gc.columns:
        if column not in ("halo_id_z0", "gc_index_halo", "status"):
            final_gc[column] = pd.to_numeric(final_gc[column], errors="coerce")

    with (TNG_CATALOGUE_ROOT / TNG_TARGET_METADATA_FILENAME).open("r", encoding="utf-8") as handle:
        catalogue_metadata = json.load(handle)
    volumes = catalogue_metadata["full_box_selection"]["volume_physical_cmpc3"]
    tng100_weight = float(volumes["tng50_1_dark"] / volumes["tng100_1_dark"])
    manifest = pd.read_csv(TNG_CATALOGUE_ROOT / TNG_TARGET_MANIFEST_FILENAME, dtype=str)
    manifest_counts = {key: int((manifest["simulation_key"] == key).sum()) for key in ("tng50_1_dark", "tng100_1_dark")}
    lookup = pd.read_csv(out_dir / TNG_TREE_LOOKUP_FILENAME, dtype=str).rename(columns={"hid_z0": "halo_id_z0"})
    lookup["halo_id_z0"] = _integer_values(lookup["halo_id_z0"], "TNG halo-tree lookup IDs", non_negative=True)
    lookup["simulation_key"] = lookup["simulation_key"].astype(str).str.strip()
    lookup["fixed_tree_basename"] = lookup["fixed_tree_basename"].astype(str).str.strip()
    lookup_ids = lookup["halo_id_z0"].to_numpy(dtype=np.int64)
    weight_by_halo = dict(zip(lookup_ids, np.where(lookup["simulation_key"].eq("tng100_1_dark"), tng100_weight, 1.0)))
    summary["volume_weight_tng50"] = [float(weight_by_halo[int(value)]) for value in summary["halo_id_z0"]]
    final_gc["volume_weight_tng50"] = [float(weight_by_halo[int(value)]) for value in final_gc["halo_id_z0"]]
    output_counts = {key: int(value) for key, value in lookup["simulation_key"].value_counts().to_dict().items()}
    context = {"lookup": lookup, "manifest_counts": manifest_counts, "output_counts": output_counts, "volume_tng50_cmpc3": float(volumes["tng50_1_dark"]), "volume_tng100_cmpc3": float(volumes["tng100_1_dark"]), "tng100_weight": tng100_weight}
    print(f"TNG volume normalisation: catalogue={TNG_CATALOGUE_ROOT}, manifest_counts={manifest_counts}, output_counts={output_counts}, V_TNG50={context['volume_tng50_cmpc3']:.12g} cMpc^3, V_TNG100={context['volume_tng100_cmpc3']:.12g} cMpc^3, w100={tng100_weight:.12g}.")

    points = pd.read_csv(DATA_ROOT / "Juodzbalis+2026Fig2" / "juodzbalis2026_fig2_points.csv")
    curves = pd.read_csv(DATA_ROOT / "Juodzbalis+2026Fig2" / "juodzbalis2026_fig2_curves.csv")
    for column in ("r_pc", "r_err_low_pc", "r_err_high_pc", "v_km_s", "v_err_low_km_s", "v_err_high_km_s"):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    for column in ("r_pc", "v_km_s"):
        curves[column] = pd.to_numeric(curves[column], errors="coerce")
    uv_calibrations = []
    for uv_mode, uv_path in UV_MODEL_SPECS:
        uv_table = pd.read_csv(uv_path, comment="#")
        uv_table = uv_table.loc[:, list(UV_CALIBRATION_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        age_by_log = uv_table.sort_values("log10_age_gyr").drop_duplicates("log10_age_gyr")
        age_axis = age_by_log["age_gyr"].to_numpy(dtype=float)
        log_age_axis = age_by_log["log10_age_gyr"].to_numpy(dtype=float)
        feh_axis = np.sort(uv_table["feh"].unique().astype(float))
        sorted_uv = uv_table.sort_values(["log10_age_gyr", "feh"])
        uv_calibrations.append({"path": uv_path.resolve(), "age_gyr": age_axis, "log10_age_gyr": log_age_axis, "feh": feh_axis, "m1450_grid": sorted_uv["M1450_AB_per_Msun"].to_numpy(dtype=float).reshape(len(log_age_axis), len(feh_axis)), "uv_mode": uv_mode, "age_min_gyr": float(age_axis[0]), "age_max_gyr": float(age_axis[-1]), "feh_min": float(feh_axis[0]), "feh_max": float(feh_axis[-1])})
    selection_uv_calibration = uv_calibrations[0]

    z_mask = np.isfinite(summary["redshift"]) & (np.abs(summary["redshift"] - 7.04) < FIG01_REDSHIFT_ATOL)
    z_rows = summary.loc[z_mask].sort_values("halo_id_z0").reset_index(drop=True)
    deposit_path = out_dir / "depos.dat"
    try:
        deposit_table = _read_headered_whitespace_table(deposit_path)
    except pd.errors.EmptyDataError:
        deposit_table = pd.DataFrame()
    if len(deposit_table):
        for column in ("halo_id_z0", "bin_index"):
            deposit_table[column] = _integer_values(deposit_table[column], f"{deposit_path} {column}", non_negative=True)
        for column in ("lookback_time_gyr", "r_inner_kpc", "r_outer_kpc", "m_star_no_evo_msun", "m_star_with_evo_msun"):
            deposit_table[column] = pd.to_numeric(deposit_table[column], errors="coerce")
    final_age = Redshift2CosmicAge(float(final_redshift))
    grouped = {} if not len(deposit_table) else {int(halo_id): group for halo_id, group in deposit_table.groupby("halo_id_z0", sort=True)}
    deposit = {"halo_ids": [], "r_outer_kpc": [], "cumulative_mass_msun": [], "cumulative_formed_mass_msun": [], "missing_halo_ids": []}
    for row in z_rows.itertuples(index=False):
        halo_id = int(row.halo_id_z0)
        if halo_id not in grouped:
            deposit["missing_halo_ids"].append(halo_id)
            continue
        group = grouped[halo_id]
        lookbacks = np.unique(np.sort(group["lookback_time_gyr"].to_numpy(dtype=float)))
        target_lookback = float(row.deposit_sample_lookback_gyr) if np.isfinite(row.deposit_sample_lookback_gyr) else np.nan
        if np.isfinite(target_lookback):
            block_lookback = float(lookbacks[np.argmin(np.abs(lookbacks - target_lookback))])
        else:
            target_z = float(row.deposit_sample_redshift) if np.isfinite(row.deposit_sample_redshift) else float(row.redshift)
            block_z = np.asarray([CosmicAge2Redshift(final_age - float(value)) for value in lookbacks])
            block_lookback = float(lookbacks[np.argmin(np.abs(block_z - target_z))])
        block = group.loc[np.isclose(group["lookback_time_gyr"], block_lookback, rtol=0.0, atol=1.0e-8)].sort_values("bin_index")
        deposit["halo_ids"].append(halo_id)
        deposit["r_outer_kpc"].append(block["r_outer_kpc"].to_numpy(dtype=float))
        deposit["cumulative_mass_msun"].append(np.cumsum(block["m_star_with_evo_msun"].to_numpy(dtype=float)))
        deposit["cumulative_formed_mass_msun"].append(np.cumsum(block["m_star_no_evo_msun"].to_numpy(dtype=float)))
    for key in ("halo_ids", "missing_halo_ids"):
        deposit[key] = np.asarray(deposit[key], dtype=np.int64)
    profile_radius = np.asarray([float(value[-1]) * 1.0e3 for value in deposit["r_outer_kpc"]], dtype=float)
    all_halo_ids = _integer_values(deposit["halo_ids"], "Fig. 01 all-profile halo IDs", non_negative=True)
    keep_profile = np.isfinite(profile_radius) & (profile_radius >= FIG01_MATCH_RADIUS_RANGE_PC[1])
    eligible_ids = all_halo_ids[keep_profile]
    insufficient_ids = all_halo_ids[~keep_profile]
    eligible_z_rows = z_rows.loc[z_rows["halo_id_z0"].isin(eligible_ids)].reset_index(drop=True)
    eligible_deposit = {"halo_ids": eligible_ids, "r_outer_kpc": [value for value, keep in zip(deposit["r_outer_kpc"], keep_profile) if keep], "cumulative_mass_msun": [value for value, keep in zip(deposit["cumulative_mass_msun"], keep_profile) if keep], "cumulative_formed_mass_msun": [value for value, keep in zip(deposit["cumulative_formed_mass_msun"], keep_profile) if keep]}

    background_by_halo, background_reasons = {}, {}
    for halo_id in eligible_ids:
        try:
            lookup_row = lookup.loc[lookup["halo_id_z0"] == int(halo_id)].iloc[0]
            tree_path = (TNG_FIXED_TREE_ROOT / str(lookup_row["fixed_tree_basename"]).strip()).resolve()
            mpb_rows = np.asarray(read_haloevo_mpb(tree_path), dtype=float)
            log_mass_bg, available_bg = _interpolate_mpb_logmh_at_redshift(mpb_rows, 7.04)
            if int(available_bg) != 1:
                raise ValueError("MPB cannot be interpolated to the target redshift")
            halo_mass = float(10.0 ** float(log_mass_bg))
            cosmic_time = np.asarray([Redshift2CosmicAge(float(value), time_unit="Gyr") for value in mpb_rows[:, 1]])
            order = np.argsort(cosmic_time, kind="mergesort")
            sorted_time, sorted_rows = cosmic_time[order], mpb_rows[order]
            keep_time = np.r_[sorted_time[1:] != sorted_time[:-1], True]
            unique_time, unique_spin = sorted_time[keep_time], sorted_rows[keep_time, 2:5]
            target_time = float(Redshift2CosmicAge(7.04, time_unit="Gyr"))
            spin = np.asarray([np.interp(target_time, unique_time, unique_spin[:, axis]) for axis in range(3)])
            j_pc_kms = float(np.linalg.norm(spin) * 1.0e3 / ReducedH0)
            effective_radius = float(calcRe(Mhalo_1e9Msun=halo_mass / 1.0e9, t_Gyr=target_time, j=j_pc_kms))
            sersic_p, sersic_b = Sersic_coefs(FIG01_SERSIC_INDEX)
            stellar_mass = float(Mstar_SMHM(halo_mass, 7.04, scatter=False))
            #c_vir = float(10.0 ** 0.971 / ((halo_mass / 1.0e9) * ReducedH0 / 1.0e3) ** 0.094)
            c_vir = halo_concn_IshiyamaP2021(Mhalo=halo_mass, z=7.04)
            virial_radius = float(Rv(Mhalo=halo_mass, z=7.04))
            background_by_halo[int(halo_id)] = {"halo_id_z0": int(halo_id), "tree_path": str(tree_path), "halo_mass_msun": halo_mass, "mstar_msun": stellar_mass, "r_e_kpc": effective_radius, "r_v_kpc": virial_radius, "r_s_kpc": virial_radius / c_vir, "c_vir": c_vir, "sersic_index": FIG01_SERSIC_INDEX, "sersic_p": float(sersic_p), "sersic_b": float(sersic_b), "j_pc_kms": j_pc_kms}
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, OverflowError, IndexError) as error:
            background_reasons[int(halo_id)] = f"Fig. 01 background unavailable: {error}"
    valid_background_ids = np.asarray(sorted(background_by_halo), dtype=np.int64)
    background_keep = np.isin(eligible_ids, valid_background_ids)
    background_excluded_ids = eligible_ids[~background_keep]
    eligible_z_rows = eligible_z_rows.loc[eligible_z_rows["halo_id_z0"].isin(valid_background_ids)].reset_index(drop=True)
    eligible_deposit = {**eligible_deposit, "halo_ids": eligible_ids[background_keep], "r_outer_kpc": [value for value, keep in zip(eligible_deposit["r_outer_kpc"], background_keep) if keep], "cumulative_mass_msun": [value for value, keep in zip(eligible_deposit["cumulative_mass_msun"], background_keep) if keep], "cumulative_formed_mass_msun": [value for value, keep in zip(eligible_deposit["cumulative_formed_mass_msun"], background_keep) if keep]}
    print(f"Fig. 01/Fig. 04 profile filter: missing deposited profiles={len(deposit['missing_halo_ids'])} IDs={deposit['missing_halo_ids'].tolist()}; excluded {len(insufficient_ids)} halo(s) with profile coverage <{FIG01_MATCH_RADIUS_RANGE_PC[1]:.0f} pc; IDs={insufficient_ids.tolist()}; eligible profiles before background filter={len(eligible_ids)}.")
    print(f"Fig. 01 background filter: fixed z = 7.04, N_S={FIG01_SERSIC_INDEX:.3g}, excluded {len(background_excluded_ids)} halo(s); IDs={background_excluded_ids.tolist()}; eligible velocity profiles={len(eligible_z_rows)}.")
    score_table, best = score_fig01_candidate_haloes(eligible_z_rows, eligible_deposit, final_gc, selection_uv_calibration, background_by_halo)
    fig01 = plot_fig01_rotation_curve(points, curves, eligible_z_rows, eligible_deposit, best, background_by_halo)
    print(f"Fig. 01 z selection: N={len(eligible_z_rows)}, z range={float(np.min(eligible_z_rows['redshift'])):.3f}-{float(np.max(eligible_z_rows['redshift'])):.3f}.")
    print(f"Fig. 01/Fig. 04 candidate pool: out_dir={out_dir}, metadata N_S={n_s:.3g}, Fig. 01 N_S={FIG01_SERSIC_INDEX:.3g}, eligible profiles={len(eligible_z_rows)}, finite rotation scores={int(np.isfinite(score_table['rotation_score']).sum())}, UV-faint candidates={int((np.isfinite(score_table['M_UV']) & (score_table['M_UV'] > QSO1_MUV_AB)).sum())}.")
    print(f"Fig. 01 velocity model: deterministic SMHM, deposited stars, NFW halo, and Sérsic background at z = 7.04 with fixed N_S={FIG01_SERSIC_INDEX:.3g}; background exclusions={len(background_excluded_ids)}.")
    print(f"UV selection mode: {selection_uv_calibration['uv_mode']}.")
    for uv_calibration in uv_calibrations:
        print(f"UV mode: {uv_calibration['uv_mode']}.")
        print(f"UV table: {uv_calibration['path']} (age={uv_calibration['age_min_gyr']:.6g}-{uv_calibration['age_max_gyr']:.6g} Gyr, [Fe/H]={uv_calibration['feh_min']:.2f}-{uv_calibration['feh_max']:.2f}).")
    print(f"Selected rotation-score halo: halo_id_z0={int(best['halo_id_z0'])}, z={float(best['redshift']):.3f}, MOKA3D RMS rotation score={float(best['rotation_score']):.4f} km/s over {int(best['rotation_n_points'])} radii, M_UV={float(best['M_UV']):.2f}, required M_UV>{QSO1_MUV_AB:.2f} (fainter), formed mass <{QSO1_NSC_APERTURE_PC:.1f} pc={float(best['formed_mass_6pc_msun']):.6g} Msun, weighted age={float(best['weighted_age_gyr']):.3f} Gyr, weighted [Fe/H]={float(best['weighted_feh']):.3f}, log10(M_NSC/Msun)={float(best['log10_nsc_mass']):.3f}, log10(M_SMBH_final/Msun)={float(best['log10_central_bh_mass']):.3f}.")
    print(f"BH-mass results: selected NSC={float(best['nsc_mass_msun']):.6g} Msun (log10={float(best['log10_nsc_mass']):.4f}), central BH={float(best['central_bh_mass_msun']):.6g} Msun (log10={float(best['log10_central_bh_mass']):.4f}).")

    aperture_table = estimate_uv_magnitude_apertures(deposit, final_gc, int(best["halo_id_z0"]), uv_calibrations)
    fig02 = plot_fig02_uvmag(aperture_table)
    fig03, inventory = plot_fig03_halo_distribution(summary, best)
    fig04, selection = plot_fig04_assembly(out_dir, summary, final_gc, context, score_table, best)
    for figure, filename in ((fig01, FIGURE_01_FILENAME), (fig02, FIGURE_02_FILENAME), (fig03, FIGURE_03_FILENAME), (fig04, FIGURE_04_FILENAME)):
        path = plot_dir / filename
        figure.savefig(path, dpi=STD_DPI, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {path}")
    empty_redshifts = ", ".join(f"{float(value):.6g}" for value in inventory["empty_redshifts"])
    missing_best = ", ".join(f"{float(value):.6g}" for value in inventory["best_halo_missing_redshifts"])
    print(f"Fig. 03 halo distribution: plotted redshifts={len(inventory['redshifts'])}, excluded unavailable rows={inventory['excluded_unavailable_rows']}, empty redshifts=[{empty_redshifts}], bin width={FIG03_DISTR_BIN_WIDTH_DEX:.2f} dex, best halo_id_z0={inventory['best_halo_id_z0']}, best-halo track lines={len(inventory['best_halo_track'])}, missing best-halo redshifts=[{missing_best}].")
    print("Fig. 04 assembly panels: " + ", ".join(f"{history['suite_label']} halo_id_z0={int(history['halo_id_z0'])} log10M_h,cat(z=7)={float(history['catalogue_log10_halo_mass']):.4f} raw-MPB-log10M_h(z=7)={float(history['main']['endpoint_log10_halo_mass']):.4f} N_sat={int(history['n_satellites'])}" for history in selection["histories"]))
    if selection["rejected"]:
        print("Fig. 04 discarded comparison candidates: " + ", ".join(str(item["halo_id_z0"]) for item in selection["rejected"]))


if __name__ == "__main__":
    main()
