#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Self-contained Kong & Li 2026 suite a plot suite for the High-z SMBH Seeds project."""

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
from matplotlib.patches import Patch
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CosmicAge2Redshift,
    Mstar_SMHM,
    parse_exact_int64,
    Redshift2CosmicAge,
    STD_DPI,
)

# input params
DATA_ROOT = PROJECT_ROOT / "data"
FIGURE_01_FILENAME = "Fig.01_Mbh-Mstar.pdf"
FIGURE_02_FILENAME = "Fig.02_BHseed_hist.pdf"
FIGURE_03_FILENAME = "Fig.03_BHMFs.pdf"
FIGURE_04_FILENAME = "Fig.04_BHSMF.pdf"
MBH_MSTAR_DATA_PATH = DATA_ROOT / "Mbh-Mstar.csv"
BHMF_DATA_PATH = DATA_ROOT / "BHMFs" / "BHMFs.csv"
CHEN2026_FIG05A_DATA_PATH = DATA_ROOT / "Chen+2026" / "chen2026_fig05a_seed_mass_functions.csv"
CHEN2026_FIG05A_REDSHIFT = 20.0
CHEN2026_FIG06_DATA_PATH = DATA_ROOT / "Chen+2026" / "chen2026_fig06_seeding_history.csv"
FIG02_CHEN_CURVE_ROLES = ("all_seeds", "popiii_subedd", "popiii_edd", "popii")
FIG02_CHEN_CURVE_LABELS = {
    "all_seeds": "All seeds",
    "popiii_subedd": "Pop-III (sub-Eddington)",
    "popiii_edd": "Pop-III (Eddington)",
    "popii": "Pop-II",
}
FIG02_DISPLAY_REDSHIFT_RANGE = (0.0, 50.0)
FIG02_XLIM_LOG1PZ = tuple(np.log10(1.0 + np.asarray(FIG02_DISPLAY_REDSHIFT_RANGE, dtype=float)))
FIG02_RATE_LOG1PZ_BIN_WIDTH = 0.02
FIG02_CHEN_REDSHIFT_ATOL = 1.0e-6
FIG02_MODEL_COLOUR = "#6a3d9a"
FIG02_MODEL_LINEWIDTH = 2.2
FIG02_REFERENCE_LINEWIDTH = 1.35
FIG02_FIGSIZE = (7.0, 7.0)
FIG04_SEED_LOGM_BIN_EDGES = np.round(np.arange(0.0, 5.0 + 0.05, 0.1), decimals=10)
FIG04_CENTRAL_LOGM_BIN_EDGES = np.round(np.arange(0.0, 8.0 + 0.05, 0.1), decimals=10)
FIG04_CHEN_CURVE_ROLES = (
    "all_seeds_central",
    "all_seeds_lower_envelope",
    "all_seeds_upper_envelope",
    "popiii_subedd",
    "popiii_edd",
    "fast_halo",
    "lw_halo",
    "popii",
)
FIG04_CHEN_CURVE_LABELS = {
    "all_seeds_central": "All seeds",
    "all_seeds_lower_envelope": "All seeds (lower envelope)",
    "all_seeds_upper_envelope": "All seeds (upper envelope)",
    "popiii_subedd": "Pop-III (sub-Eddington)",
    "popiii_edd": "Pop-III (Eddington)",
    "fast_halo": "Fast halo (gamma_v >= 3)",
    "lw_halo": "LW halo (J_LW,21 >= 7.5)",
    "popii": "Pop-II",
}
FIG04_CHEN_VISIBLE_CURVE_ROLES = (
    "popiii_subedd",
    "popiii_edd",
    "popii",
)
MBH_MSTAR_MARKER_STYLES = {
    "Carnall+2023": {"marker": "h", "marker_size": 7.0, "edgecolor": "#4682b4", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Ding+2023": {"marker": "X", "marker_size": 7.2, "edgecolor": "#ffd400", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Goulding+2023": {"marker": "P", "marker_size": 7.0, "edgecolor": "#0b84c9", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Harikane+2023": {"marker": "D", "marker_size": 6.4, "edgecolor": "#4b0082", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Ivey+2026": {"marker": "o", "marker_size": 6.5, "edgecolor": "white", "edgewidth": 0.7, "alpha": 1.0, "zorder": 7},
    "Juodzbalis+2025": {"marker": "o", "marker_size": 6.6, "edgecolor": "#ff9900", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Juodzbalis+2026": {"marker": "*", "marker_size": 11.0, "edgecolor": "black", "edgewidth": 0.7, "alpha": 1.0, "zorder": 9},
    "Kokorev+2023": {"marker": "^", "marker_size": 7.0, "edgecolor": "#191970", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Maiolino+2024": {"marker": "*", "marker_size": 8.0, "edgecolor": "#40d8cf", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Stone+2024": {"marker": "*", "marker_size": 7.6, "edgecolor": "#cd853f", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Ubler+2023": {"marker": "D", "marker_size": 6.4, "edgecolor": "#4169e1", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
    "Yue+2024": {"marker": "^", "marker_size": 7.4, "edgecolor": "#dca51d", "edgewidth": 0.0, "alpha": 1.0, "zorder": 7},
}
FIG01_DASHED_BH_THRESHOLD_MSUN = 100.0
FIG01_DASHED_SCATTER_ALPHA = 0.10
TNG_H = 0.6774
TNG50_NATIVE_SIDE_CMPC_H = 35.0
TNG50_FULL_BOX_SIDE_CMPC = TNG50_NATIVE_SIDE_CMPC_H / TNG_H
BHMF_REFERENCE_SIDE_CMPC = TNG50_FULL_BOX_SIDE_CMPC
BHMF_REFERENCE_VOLUME_CMPC3 = BHMF_REFERENCE_SIDE_CMPC**3
FIG03_BIN_EDGES = np.arange(4.0, 9.1, 0.25)
FIG03_MIN_MODEL_REDSHIFT_EXCLUSIVE = 3.0
TNG_CATALOGUE_ROOT = Path("/lingshan/disk3/subonan/TNG50+100-1-Dark")
TNG_TARGET_MANIFEST_FILENAME = "target_manifest_dark.csv"
TNG_TARGET_METADATA_FILENAME = "targets_z0_dark.json"
TNG_TREE_LOOKUP_FILENAME = "halo_tree_lookup.csv"
TNG_FIXED_TREE_DIRNAME = "fixed_trees_large_spin_dark"
TNG_ORIGINAL_LOOKUP_FILENAME = "id_lookup_original.csv"
TNG_SHIFTED_LOOKUP_FILENAME = "id_lookup_large_dark.csv"
TNG_SUITE_KEYS = ("tng50_1_dark", "tng100_1_dark")
TNG100_HALO_ID_OFFSET = 1_000_000
BH_TO_STELLAR_MASS_RATIOS = (0.01, 0.1, 1.0)
REINES_VOLONTERI_2015_NORM = 7.45
REINES_VOLONTERI_2015_SLOPE = 1.05
REINES_VOLONTERI_2015_SCATTER_DEX = 0.55
if len(FIG04_SEED_LOGM_BIN_EDGES) != 51 or not np.isclose(FIG04_SEED_LOGM_BIN_EDGES[-1], 5.0, rtol=0.0, atol=1.0e-12):
    raise RuntimeError("Fig. 04 seed-mass grid must contain the explicit 0.0--5.0 dex sequence in 0.1 dex steps.")
if (
    len(FIG04_CENTRAL_LOGM_BIN_EDGES) != 81
    or not np.isclose(FIG04_CENTRAL_LOGM_BIN_EDGES[0], 0.0, rtol=0.0, atol=1.0e-12)
    or not np.isclose(FIG04_CENTRAL_LOGM_BIN_EDGES[-1], 8.0, rtol=0.0, atol=1.0e-12)
    or np.any(~np.isfinite(FIG04_CENTRAL_LOGM_BIN_EDGES))
    or np.any(np.diff(FIG04_CENTRAL_LOGM_BIN_EDGES) <= 0.0)
    or not np.allclose(np.diff(FIG04_CENTRAL_LOGM_BIN_EDGES), 0.1, rtol=0.0, atol=1.0e-12)
):
    raise RuntimeError("Fig. 04 central-BH grid must contain the explicit 0.0--8.0 dex sequence in 0.1 dex steps.")


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
def _bin_track(track, edges, x_log_col):
    x_log = track[x_log_col].to_numpy(dtype=float)
    y = track["M_SMBH_final"].to_numpy(dtype=float)
    rows = []
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = np.isfinite(x_log) & np.isfinite(y) & (x_log >= left)
        mask &= x_log <= right if idx == len(edges) - 2 else x_log < right
        if np.any(mask):
            y_sel = y[mask]
            rows.append({"logx_center": 0.5 * (left + right), "mean_mass": float(np.mean(y_sel)), "std_mass": float(np.std(y_sel))})
    return pd.DataFrame(rows)
def _row_text(row, name, default=""):
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return str(value).strip()
def _reines_volonteri_2015_mbh(mstar_msun):
    return np.power(10.0, REINES_VOLONTERI_2015_NORM + REINES_VOLONTERI_2015_SLOPE * np.log10(np.asarray(mstar_msun) / 1.0e11))
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
    raw = np.asarray(values, dtype=object).reshape(-1)
    parsed = []
    for index, value in enumerate(raw):
        integer = parse_exact_int64(value, name=f"{name}[{index}]")
        if non_negative and integer < 0:
            raise ValueError(f"{name}[{index}] must be non-negative; got {integer}.")
        parsed.append(integer)
    return np.asarray(parsed, dtype=np.int64)
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
    for col in table.columns:
        if col != "halo_id_z0":
            table[col] = pd.to_numeric(table[col], errors="coerce")
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
    status = _integer_values(table["status"], "finalGCs statuses")
    table["status"] = status
    table["M_IMBH_final"] = pd.to_numeric(table["M_IMBH_final"], errors="coerce")
    if table["M_IMBH_final"].isna().any() or (table["M_IMBH_final"] < 0.0).any():
        raise ValueError("Final-GC table contains non-finite or negative M_IMBH_final values.")
    return table
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
def _mbh_mstar_base_label(label):
    return str(label).split("(", 1)[0].strip()
def load_mbh_mstar_observations():
    required = ["Mbh [M☉]", "Mstar [M☉]", "z", "label", "ADSABS"]
    table = _read_csv_required(MBH_MSTAR_DATA_PATH, required, numeric=["Mbh [M☉]", "Mstar [M☉]", "z"])
    if list(table.columns) != required:
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} must contain exactly these columns in this order: {required}.")
    table["label"] = table["label"].fillna("").astype(str).str.strip()
    table["ADSABS"] = table["ADSABS"].fillna("").astype(str).str.strip()
    if table["label"].eq("").any():
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains blank observational labels.")
    table["base_label"] = table["label"].map(_mbh_mstar_base_label)
    unknown = sorted(set(table["base_label"]) - set(MBH_MSTAR_MARKER_STYLES))
    if unknown:
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains labels without marker styles: {unknown}")
    urls = table["ADSABS"].to_numpy(dtype=str)
    invalid_urls = [url for url in urls if url and not (url.startswith("https://ui.adsabs.harvard.edu/abs/") or url.startswith("https://arxiv.org/abs/"))]
    if invalid_urls:
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains malformed ADSABS/arXiv URLs: {invalid_urls}")
    mass_columns = ["Mbh [M☉]", "Mstar [M☉]"]
    for column in mass_columns:
        values = table[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains non-finite or non-positive values in {column}.")
    redshifts = table["z"].to_numpy(dtype=float)
    if np.any(~np.isfinite(redshifts)) or np.any(redshifts < 0.0):
        raise ValueError(f"{MBH_MSTAR_DATA_PATH} contains non-finite or negative redshifts.")
    table["logMstar"] = np.log10(table["Mstar [M☉]"].to_numpy(dtype=float))
    table["logMBH"] = np.log10(table["Mbh [M☉]"].to_numpy(dtype=float))
    for name in ["marker", "marker_size", "edgecolor", "edgewidth", "alpha", "zorder"]:
        table[name] = table["base_label"].map(lambda label: MBH_MSTAR_MARKER_STYLES[label][name])
    return table
def load_bhmf_data():
    required = [
        "Phi [lgM☉⁻¹Mpc⁻³]", "sigma_Phi_low [lgM☉⁻¹Mpc⁻³]", "sigma_Phi_high [lgM☉⁻¹Mpc⁻³]",
        "Mbh [M☉]", "sigma_Mbh_low", "sigma_Mbh_high", "shape", "label", "z_low", "z_high", "ADSABS", "data",
    ]
    if not BHMF_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing BHMF catalogue: {BHMF_DATA_PATH}")
    table = pd.read_csv(BHMF_DATA_PATH, dtype=str, keep_default_na=False)
    if list(table.columns) != required:
        raise ValueError(f"BHMF catalogue columns must be exactly {required}, got {list(table.columns)}")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    numeric_columns = [
        "Phi [lgM☉⁻¹Mpc⁻³]", "sigma_Phi_low [lgM☉⁻¹Mpc⁻³]", "sigma_Phi_high [lgM☉⁻¹Mpc⁻³]",
        "Mbh [M☉]", "sigma_Mbh_low", "sigma_Mbh_high", "z_low", "z_high",
    ]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column].replace("", np.nan), errors="coerce")
    if table.empty:
        raise ValueError("BHMF catalogue is empty.")
    phi = table["Phi [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float)
    mass = table["Mbh [M☉]"].to_numpy(dtype=float)
    if np.any(~np.isfinite(phi)):
        raise ValueError("BHMF catalogue contains non-finite logarithmic Phi values.")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("BHMF catalogue contains non-positive or non-finite Mbh values.")
    allowed_shapes = {"h", "s", "o", "^"}
    if not set(table["shape"]).issubset(allowed_shapes):
        raise ValueError(f"BHMF catalogue contains unsupported shapes: {sorted(set(table['shape']) - allowed_shapes)}")
    if table["label"].eq("").any():
        raise ValueError("BHMF catalogue contains an empty label.")
    if table["ADSABS"].eq("").any() or table["data"].eq("").any():
        raise ValueError("BHMF catalogue contains an empty provenance URL.")
    for column in ["ADSABS", "data"]:
        invalid = []
        for value in table[column]:
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                invalid.append(value)
        if invalid:
            raise ValueError(f"BHMF catalogue contains invalid {column} URLs: {invalid[:5]}")
    phi_low = table["sigma_Phi_low [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float)
    phi_high = table["sigma_Phi_high [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float)
    mass_low = table["sigma_Mbh_low"].to_numpy(dtype=float)
    mass_high = table["sigma_Mbh_high"].to_numpy(dtype=float)
    if np.any(np.isnan(phi_low)) or np.any(phi_low < 0.0) or np.any(~np.isfinite(phi_high)) or np.any(phi_high < 0.0):
        raise ValueError("BHMF catalogue contains invalid logarithmic Phi uncertainties.")
    infinite_low = np.isposinf(phi_low)
    if np.any(np.isneginf(phi_low)) or np.any(infinite_low & ~table["label"].isin({"Wu+2022", "Lai+2024"}).to_numpy()):
        raise ValueError("Only Wu+2022 and Lai+2024 rows may have a positive infinite lower Phi error.")
    if np.any(~np.isfinite(mass_low)) or np.any(~np.isfinite(mass_high)) or np.any(mass_low < 0.0) or np.any(mass_high < 0.0):
        raise ValueError("BHMF catalogue contains invalid non-negative Mbh uncertainties in dex.")
    z_low = table["z_low"].to_numpy(dtype=float)
    z_high = table["z_high"].to_numpy(dtype=float)
    if np.any(~np.isfinite(z_low)) or np.any(~np.isfinite(z_high)) or np.any(z_low < 0.0) or np.any(z_low >= z_high):
        raise ValueError("BHMF catalogue contains invalid redshift limits.")
    expected_counts = {"Fei+2026": 4, "Matthee+2024": 2, "He+2024": 11, "Taylor+2025": 4, "Wu+2022": 11, "Lai+2024": 6}
    counts = table["label"].value_counts().to_dict()
    if len(table) != 38 or counts != expected_counts:
        raise ValueError(f"BHMF catalogue counts must total 38 with {expected_counts}, got {counts}")
    return table
def load_chen2026_fig05a_seed_mass_functions():
    """Load and validate the digitised Chen+2026 panel-5a curves."""

    required = [
        "curve_id", "curve_label", "curve_role", "redshift",
        "log10_mbh_seed_msun", "phi_mpc3_dex1", "colour", "linestyle", "source",
    ]
    if not CHEN2026_FIG05A_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing Chen+2026 Fig. 5a reference table: {CHEN2026_FIG05A_DATA_PATH}")
    table = pd.read_csv(CHEN2026_FIG05A_DATA_PATH, dtype=str, keep_default_na=False)
    if list(table.columns) != required:
        raise ValueError(f"Chen+2026 Fig. 5a columns must be exactly {required}, got {list(table.columns)}")
    if table.empty:
        raise ValueError("Chen+2026 Fig. 5a reference table is empty.")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    for column in ["curve_id", "curve_label", "curve_role", "colour", "linestyle", "source"]:
        if table[column].eq("").any():
            raise ValueError(f"Chen+2026 Fig. 5a contains empty {column} values.")

    numeric = ["redshift", "log10_mbh_seed_msun", "phi_mpc3_dex1"]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[numeric].isna().any().any() or not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Chen+2026 Fig. 5a contains non-finite numeric values.")
    if not np.allclose(table["redshift"].to_numpy(dtype=float), CHEN2026_FIG05A_REDSHIFT, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Chen+2026 Fig. 5a must contain only the reference redshift z={CHEN2026_FIG05A_REDSHIFT:g}.")
    log_mass = table["log10_mbh_seed_msun"].to_numpy(dtype=float)
    phi = table["phi_mpc3_dex1"].to_numpy(dtype=float)
    if np.any(log_mass < FIG04_SEED_LOGM_BIN_EDGES[0] - 1.0e-10) or np.any(log_mass > FIG04_SEED_LOGM_BIN_EDGES[-1] + 1.0e-10):
        raise ValueError("Chen+2026 Fig. 5a mass coordinates lie outside the calibrated 0--5 dex range.")
    if np.any(phi <= 0.0):
        raise ValueError("Chen+2026 Fig. 5a Phi values must be finite and positive.")
    invalid_colours = [value for value in table["colour"].unique() if not mpl.colors.is_color_like(value)]
    if invalid_colours:
        raise ValueError(f"Chen+2026 Fig. 5a contains invalid colours: {invalid_colours}")
    valid_linestyles = {"-", "--", "-.", ":", "None", "none", "solid", "dashed", "dashdot", "dotted"}
    invalid_linestyles = sorted(set(table["linestyle"]) - valid_linestyles)
    if invalid_linestyles:
        raise ValueError(f"Chen+2026 Fig. 5a contains invalid line styles: {invalid_linestyles}")
    if table.duplicated(["curve_role", "log10_mbh_seed_msun"]).any():
        raise ValueError("Chen+2026 Fig. 5a contains duplicate curve-coordinate pairs.")

    roles = set(table["curve_role"])
    if roles != set(FIG04_CHEN_CURVE_ROLES):
        raise ValueError(f"Chen+2026 Fig. 5a curve roles must be exactly {FIG04_CHEN_CURVE_ROLES}, got {sorted(roles)}")
    if table["curve_id"].nunique() != len(FIG04_CHEN_CURVE_ROLES):
        raise ValueError("Chen+2026 Fig. 5a must contain one unique curve identifier per curve role.")

    by_role = {}
    for role in FIG04_CHEN_CURVE_ROLES:
        rows = table.loc[table["curve_role"].eq(role)].sort_values("log10_mbh_seed_msun").reset_index(drop=True)
        if rows["curve_label"].nunique() != 1 or rows["curve_label"].iloc[0] != FIG04_CHEN_CURVE_LABELS[role]:
            raise ValueError(f"Chen+2026 Fig. 5a label for {role!r} is missing or unexpected.")
        if rows["curve_id"].nunique() != 1 or rows["curve_id"].iloc[0] == "":
            raise ValueError(f"Chen+2026 Fig. 5a role {role!r} must have one non-empty curve identifier.")
        if len(rows) < 2 or np.any(np.diff(rows["log10_mbh_seed_msun"].to_numpy(dtype=float)) <= 0.0):
            raise ValueError(f"Chen+2026 Fig. 5a role {role!r} is not a strictly increasing curve.")
        if rows["colour"].nunique() != 1 or rows["linestyle"].nunique() != 1 or rows["source"].nunique() != 1:
            raise ValueError(f"Chen+2026 Fig. 5a role {role!r} has inconsistent plotting metadata.")
        by_role[role] = rows

    central = by_role["all_seeds_central"]
    lower = by_role["all_seeds_lower_envelope"]
    upper = by_role["all_seeds_upper_envelope"]
    central_x = central["log10_mbh_seed_msun"].to_numpy(dtype=float)
    lower_x = lower["log10_mbh_seed_msun"].to_numpy(dtype=float)
    upper_x = upper["log10_mbh_seed_msun"].to_numpy(dtype=float)
    if not np.array_equal(central_x, lower_x) or not np.array_equal(central_x, upper_x):
        raise ValueError("Chen+2026 All-seeds envelope curves must share the central mass grid.")
    central_phi = central["phi_mpc3_dex1"].to_numpy(dtype=float)
    lower_phi = lower["phi_mpc3_dex1"].to_numpy(dtype=float)
    upper_phi = upper["phi_mpc3_dex1"].to_numpy(dtype=float)
    if np.any(lower_phi > central_phi + 1.0e-15) or np.any(central_phi > upper_phi + 1.0e-15):
        raise ValueError("Chen+2026 All-seeds envelope ordering must be lower <= central <= upper.")

    sorted_table = table.sort_values(["curve_role", "log10_mbh_seed_msun"]).reset_index(drop=True)
    return {
        "table": sorted_table,
        "by_role": by_role,
        "roles": FIG04_CHEN_CURVE_ROLES,
        "redshift": CHEN2026_FIG05A_REDSHIFT,
        "log10_mass_range": (float(log_mass.min()), float(log_mass.max())),
        "phi_range": (float(phi.min()), float(phi.max())),
    }
def load_chen2026_fig06_seed_history():
    """Load and validate the four retained Chen+2026 Fig. 6a/6d curves."""

    required = [
        "curve_id", "curve_label", "curve_role", "panel", "quantity",
        "x_log10_1pz", "redshift", "value_mpc3", "colour", "linestyle", "source",
    ]
    if not CHEN2026_FIG06_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing Chen+2026 Fig. 6 reference table: {CHEN2026_FIG06_DATA_PATH}")
    table = pd.read_csv(CHEN2026_FIG06_DATA_PATH, dtype=str, keep_default_na=False)
    if list(table.columns) != required:
        raise ValueError(f"Chen+2026 Fig. 6 columns must be exactly {required}, got {list(table.columns)}")
    if table.empty:
        raise ValueError("Chen+2026 Fig. 6 reference table is empty.")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    for column in ["curve_id", "curve_label", "curve_role", "panel", "quantity", "colour", "linestyle", "source"]:
        if table[column].eq("").any():
            raise ValueError(f"Chen+2026 Fig. 6 contains empty {column} values.")

    numeric = ["x_log10_1pz", "redshift", "value_mpc3"]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    numeric_values = table[numeric].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Chen+2026 Fig. 6 contains non-finite numeric values.")
    if np.any(table["value_mpc3"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("Chen+2026 Fig. 6 reference values must be finite and positive.")
    if set(table["panel"]) != {"a", "d"}:
        raise ValueError(f"Chen+2026 Fig. 6 panels must be exactly ['a', 'd'], got {sorted(set(table['panel']))}")
    if set(table["curve_role"]) != set(FIG02_CHEN_CURVE_ROLES):
        raise ValueError(
            f"Chen+2026 Fig. 6 visible curve roles must be exactly {FIG02_CHEN_CURVE_ROLES}, "
            f"got {sorted(set(table['curve_role']))}. Fast/LW rows cannot be plotted."
        )
    if table.duplicated(["curve_id", "panel", "x_log10_1pz"]).any():
        raise ValueError("Chen+2026 Fig. 6 contains duplicate curve-coordinate rows.")
    valid_linestyles = {"-", "--", "-.", ":", "None", "none", "solid", "dashed", "dashdot", "dotted"}
    invalid_colours = [value for value in table["colour"].unique() if not mpl.colors.is_color_like(value)]
    invalid_linestyles = sorted(set(table["linestyle"]) - valid_linestyles)
    if invalid_colours:
        raise ValueError(f"Chen+2026 Fig. 6 contains invalid colours: {invalid_colours}")
    if invalid_linestyles:
        raise ValueError(f"Chen+2026 Fig. 6 contains invalid line styles: {invalid_linestyles}")

    by_panel = {panel: {} for panel in ("a", "d")}
    expected_quantity = {"a": "rate", "d": "cumulative"}
    for panel in ("a", "d"):
        panel_rows = table.loc[table["panel"].eq(panel)]
        if set(panel_rows["curve_role"]) != set(FIG02_CHEN_CURVE_ROLES):
            raise ValueError(f"Chen+2026 Fig. 6 panel {panel} does not contain exactly the four visible roles.")
        if set(panel_rows["quantity"]) != {expected_quantity[panel]}:
            raise ValueError(f"Chen+2026 Fig. 6 panel {panel} must have quantity={expected_quantity[panel]!r}.")
        for role in FIG02_CHEN_CURVE_ROLES:
            rows = panel_rows.loc[panel_rows["curve_role"].eq(role)].reset_index(drop=True)
            if len(rows) < 2:
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} has fewer than two vertices.")
            if rows["curve_label"].nunique() != 1 or rows["curve_label"].iloc[0] != FIG02_CHEN_CURVE_LABELS[role]:
                raise ValueError(f"Chen+2026 Fig. 6 label for {panel}/{role} is missing or unexpected.")
            if rows["curve_id"].nunique() != 1:
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} must have one curve identifier.")
            x_values = rows["x_log10_1pz"].to_numpy(dtype=float)
            if np.any(np.diff(x_values) <= 0.0):
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} is not strictly increasing in x.")
            if rows["colour"].nunique() != 1 or rows["linestyle"].nunique() != 1 or rows["source"].nunique() != 1:
                raise ValueError(f"Chen+2026 Fig. 6 curve {panel}/{role} has inconsistent plotting metadata.")
            expected_redshift = np.power(10.0, x_values) - 1.0
            if not np.allclose(rows["redshift"].to_numpy(dtype=float), expected_redshift, rtol=0.0, atol=FIG02_CHEN_REDSHIFT_ATOL):
                raise ValueError(f"Chen+2026 Fig. 6 redshift calibration disagrees with x for {panel}/{role}.")
            by_panel[panel][role] = rows

    sorted_table = table.sort_values(["panel", "curve_role", "x_log10_1pz"]).reset_index(drop=True)
    return {
        "table": sorted_table,
        "by_panel": by_panel,
        "roles": FIG02_CHEN_CURVE_ROLES,
        "panels": ("a", "d"),
        "redshift_atol": FIG02_CHEN_REDSHIFT_ATOL,
    }
def _plot_mbh_mstar_observations(ax, observations, norm, cmap):
    seen_labels = set()
    for _, row in observations.iterrows():
        log_mstar = float(row["logMstar"])
        log_mbh = float(row["logMBH"])
        colour = cmap(norm(float(row["z"])))
        label_base = _row_text(row, "label")
        label = None if label_base in seen_labels else label_base
        seen_labels.add(label_base)
        x = 10.0**log_mstar
        y = 10.0**log_mbh
        ax.plot(x, y, marker=row["marker"], linestyle="None", ms=float(row["marker_size"]), mfc=colour, mec=row["edgecolor"], mew=float(row["edgewidth"]), color=colour, alpha=float(row["alpha"]), label=label, zorder=int(row["zorder"]))
def plot_fig01_mbh_mstar(summary_by_z, observations, mass_bin_width_dex):
    plot_rows = summary_by_z[np.isfinite(summary_by_z["logMstar_z_smhm_msun"].to_numpy(dtype=float)) & np.isfinite(summary_by_z["M_SMBH_final"].to_numpy(dtype=float))].copy()
    if len(plot_rows) == 0:
        raise ValueError("No finite rows are available for Fig. 01.")
    z_values = np.sort(plot_rows["z_out"].unique())
    edges = _regular_log_bin_edges(plot_rows["logMstar_z_smhm_msun"], mass_bin_width_dex)
    x_limit_values = plot_rows["logMstar_z_smhm_msun"].to_numpy(dtype=float)
    if observations is not None and len(observations) > 0:
        x_limit_values = np.concatenate([x_limit_values, observations["logMstar"].to_numpy(dtype=float)])
    x_limit_edges = _regular_log_bin_edges(x_limit_values, mass_bin_width_dex)
    norm = mpl.colors.Normalize(vmin=3.0, vmax=10.0, clip=True)
    cmap = mpl.cm.jet

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    n_tracks = 0
    for z_out in z_values:
        if z_out < 3.0:
            continue
        track = plot_rows[plot_rows["z_out"] == float(z_out)].copy()
        # Solid curves include every finite central-BH row; dashed curves and bands use the same statistic after strict M_SMBH_final > 100 M_sun selection.
        qualified_track = track.loc[track["M_SMBH_final"] > FIG01_DASHED_BH_THRESHOLD_MSUN].copy()
        if len(qualified_track) == 0:
            print(f"Fig. 01 qualified track omitted at z={float(z_out):.6g}: no central BH with M_SMBH_final > {FIG01_DASHED_BH_THRESHOLD_MSUN:g} M_sun.")
        binned = _bin_track(track, edges, "logMstar_z_smhm_msun")
        if len(binned) == 0:
            continue
        mean_mass = binned["mean_mass"].to_numpy(dtype=float)
        valid = np.isfinite(mean_mass) & (mean_mass > 0.0)
        if not np.any(valid):
            continue
        x = np.power(10.0, binned.loc[valid, "logx_center"].to_numpy(dtype=float))
        mean_mass = mean_mass[valid]
        std_mass = binned.loc[valid, "std_mass"].to_numpy(dtype=float)
        colour = cmap(norm(float(z_out)))
        ax.fill_between(x, np.maximum(mean_mass - std_mass, mean_mass * 1.0e-3), np.maximum(mean_mass + std_mass, mean_mass * 1.0e-3), color=colour, alpha=0.18, edgecolor="none")
        ax.plot(x, mean_mass, c=colour, ls="--", lw=1.5, zorder=2)
        n_tracks += 1
        if len(qualified_track) == 0:
            continue
        qualified_binned = _bin_track(qualified_track, edges, "logMstar_z_smhm_msun")
        if len(qualified_binned) == 0:
            print(f"Fig. 01 qualified track omitted at z={float(z_out):.6g}: no finite qualified stellar-mass bins.")
            continue
        qualified_bin_index = np.searchsorted(edges, qualified_binned["logx_center"].to_numpy(dtype=float), side="right") - 1
        qualified_mean = np.full(len(edges) - 1, np.nan, dtype=float)
        qualified_std = np.full(len(edges) - 1, np.nan, dtype=float)
        qualified_mean[qualified_bin_index] = qualified_binned["mean_mass"].to_numpy(dtype=float)
        qualified_std[qualified_bin_index] = qualified_binned["std_mass"].to_numpy(dtype=float)
        qualified_valid = np.isfinite(qualified_mean) & (qualified_mean > 0.0)
        if not np.any(qualified_valid):
            continue
        qualified_x = np.power(10.0, 0.5 * (edges[:-1] + edges[1:]))
        qualified_lower = np.full(len(qualified_x), np.nan, dtype=float)
        qualified_upper = np.full(len(qualified_x), np.nan, dtype=float)
        qualified_lower[qualified_valid] = np.maximum(qualified_mean[qualified_valid] - qualified_std[qualified_valid], qualified_mean[qualified_valid] * 1.0e-3)
        qualified_upper[qualified_valid] = np.maximum(qualified_mean[qualified_valid] + qualified_std[qualified_valid], qualified_mean[qualified_valid] * 1.0e-3)
        ax.fill_between(qualified_x, qualified_lower, qualified_upper, color=colour, alpha=FIG01_DASHED_SCATTER_ALPHA, edgecolor="none", zorder=1)
        ax.plot(qualified_x, np.where(qualified_valid, qualified_mean, np.nan), c=colour, ls="-", lw=2.0, zorder=2)
    if n_tracks == 0:
        raise ValueError("All binned central-BH tracks are empty or non-positive for Fig. 01.")
    ax.plot([], [], c="black", ls="--", lw=1.5, label="Model")
    ax.plot([], [], c="black", ls="-", lw=2.0, label=r"Model ($M_\bullet > 100~M_\odot$)")

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label(r"Redshift $z$")
    colour_bar.set_ticks(np.arange(3.0, 10.1, 1.0))
    x_line = np.logspace(x_limit_edges[0], x_limit_edges[-1], 256)
    rv15 = _reines_volonteri_2015_mbh(x_line)
    rv15_scatter = 10.0**REINES_VOLONTERI_2015_SCATTER_DEX
    ax.fill_between(x_line, rv15 / rv15_scatter, rv15 * rv15_scatter, color="#2ca25f", alpha=0.16, edgecolor="none", label="Reines+Volonteri 2015", zorder=0)
    ax.plot(x_line, rv15, c="#238b45", lw=1.8, zorder=1)
    for ratio in BH_TO_STELLAR_MASS_RATIOS:
        ax.plot(x_line, ratio * x_line, c="#31a354", ls="--", lw=1.0, alpha=0.75, zorder=2)
        label_x = 10.0**max(min(6.0, float(x_limit_edges[-1]) - 0.35), float(x_limit_edges[0]) + 0.45)
        label_y = ratio * label_x
        if 1.0e2 < label_y < 1.0e11:
            ax.text(label_x, label_y, rf"$M_{{\rm BH}}/M_\ast={ratio:g}$", color="#31a354", fontsize=8.5, rotation=33.0, ha="center", va="bottom", clip_on=True, zorder=3)
    if observations is not None and len(observations) > 0:
        _plot_mbh_mstar_observations(ax, observations, norm, cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Stellar mass $M_\star (z)$ [$M_\odot$]")
    ax.set_ylabel(r"Nuclear BH mass $M_\bullet$ [$M_{\odot}$]")
    ax.set_xlim(left=10.0**5, right=10.0**x_limit_edges[-1])
    ax.set_ylim(bottom=1.0e2, top=1.0e10)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    legend = ax.legend(loc="upper left", fontsize=6.2, frameon=False, framealpha=0.85, ncol=2)
    for legend_text in legend.get_texts():
        if legend_text.get_text() == r"Model ($M_\bullet > 100~M_\odot$)":
            legend_text.set_usetex(False)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig
def _weighted_density_inputs(masses, weights):
    masses = np.asarray(masses, dtype=float)
    if weights is None:
        weights = np.ones(len(masses), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    if masses.ndim != 1 or weights.ndim != 1 or len(masses) != len(weights):
        raise ValueError("BH density masses and volume weights must be one-dimensional arrays with equal lengths.")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("BH density masses must be finite and positive after population selection.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("BH density volume weights must be finite and non-negative.")
    return masses, weights
def _bhmf_density_per_dex(masses, weights=None):
    masses, weights = _weighted_density_inputs(masses, weights)
    counts, _ = np.histogram(masses, bins=10.0 ** FIG03_BIN_EDGES, weights=weights)
    bin_width_dex = np.diff(FIG03_BIN_EDGES)
    return counts.astype(float) / (BHMF_REFERENCE_VOLUME_CMPC3 * bin_width_dex)
def _fig03_project_densities(summary_by_z):
    required = ["z_out", "M_SMBH_final", "volume_weight_tng50"]
    missing = [name for name in required if name not in summary_by_z.columns]
    if missing:
        raise ValueError(f"Fig. 03 haloSummaryByZ is missing required columns: {missing}")
    z_out = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(z_out)):
        raise ValueError("Fig. 03 requires finite z_out values in haloSummaryByZ.")
    retained = z_out > FIG03_MIN_MODEL_REDSHIFT_EXCLUSIVE
    z_values = np.sort(np.unique(z_out[retained]))
    if len(z_values) == 0:
        raise ValueError("Fig. 03 has no model output redshifts strictly above z=3.")

    this_work_by_z = {}
    positive_raw_by_z = {}
    positive_effective_by_z = {}
    invalid_mass_by_z = {}
    out_of_range_raw_by_z = {}
    out_of_range_effective_by_z = {}
    omitted_no_positive_redshifts = []
    omitted_no_visible_density_redshifts = []
    lower_log_mass = float(FIG03_BIN_EDGES[0])
    upper_log_mass = float(FIG03_BIN_EDGES[-1])
    for z_out_value in z_values:
        rows = summary_by_z.loc[z_out == float(z_out_value)]
        masses_all = pd.to_numeric(rows["M_SMBH_final"], errors="coerce").to_numpy(dtype=float)
        weights_all = pd.to_numeric(rows["volume_weight_tng50"], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(weights_all)) or np.any(weights_all <= 0.0):
            raise ValueError(f"Fig. 03 volume weights must be finite and positive at z={float(z_out_value):.6g}.")
        positive = np.isfinite(masses_all) & (masses_all > 0.0)
        masses = masses_all[positive]
        weights = weights_all[positive]
        positive_raw_by_z[float(z_out_value)] = int(len(masses))
        positive_effective_by_z[float(z_out_value)] = float(np.sum(weights))
        invalid_mass_by_z[float(z_out_value)] = int(np.count_nonzero(~positive))
        if len(masses) == 0:
            omitted_no_positive_redshifts.append(float(z_out_value))
            continue

        log_masses = np.log10(masses)
        in_range = (log_masses >= lower_log_mass) & (log_masses <= upper_log_mass)
        out_of_range = ~in_range
        out_of_range_raw_by_z[float(z_out_value)] = int(np.count_nonzero(out_of_range))
        out_of_range_effective_by_z[float(z_out_value)] = float(np.sum(weights[out_of_range]))
        if not np.any(in_range):
            omitted_no_visible_density_redshifts.append(float(z_out_value))
            continue

        density = _bhmf_density_per_dex(masses[in_range], weights[in_range])
        if not np.any(density > 0.0):
            omitted_no_visible_density_redshifts.append(float(z_out_value))
            continue
        this_work_by_z[float(z_out_value)] = density

    inventory = {
        "this_work_by_z": positive_raw_by_z,
        "this_work_effective_by_z": positive_effective_by_z,
        "invalid_mass_by_z": invalid_mass_by_z,
        "out_of_range_raw_by_z": out_of_range_raw_by_z,
        "out_of_range_effective_by_z": out_of_range_effective_by_z,
        "omitted_no_positive_redshifts": np.asarray(omitted_no_positive_redshifts, dtype=float),
        "omitted_no_visible_density_redshifts": np.asarray(omitted_no_visible_density_redshifts, dtype=float),
    }
    return 0.5 * (FIG03_BIN_EDGES[:-1] + FIG03_BIN_EDGES[1:]), this_work_by_z, inventory
def _fig02_project_bhseed_events(final_gc, volume_cmpc3):
    """Return every final-GC row with a positive initial IMBH seed mass."""

    if not isinstance(final_gc, pd.DataFrame):
        raise ValueError("Fig. 02 seed-event projection requires finalGCs.dat as a pandas DataFrame.")
    required = ["M_IMBH_init", "lookback_time_init_gyr", "volume_weight_tng50", "halo_id_z0", "status"]
    missing = [name for name in required if name not in final_gc.columns]
    if missing:
        raise ValueError(f"Fig. 02 finalGCs.dat is missing required columns: {missing}")
    if len(final_gc) == 0:
        raise ValueError("Fig. 02 cannot project an empty finalGCs.dat catalogue.")

    initial_mass = pd.to_numeric(final_gc["M_IMBH_init"], errors="coerce").to_numpy(dtype=float)
    lookback_init = pd.to_numeric(final_gc["lookback_time_init_gyr"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(final_gc["volume_weight_tng50"], errors="coerce").to_numpy(dtype=float)
    status = _integer_values(final_gc["status"], "Fig. 02 finalGCs statuses")
    halo_id = _integer_values(final_gc["halo_id_z0"], "Fig. 02 finalGCs parent halo IDs", non_negative=True)
    if np.any(~np.isfinite(initial_mass)) or np.any(initial_mass < 0.0):
        raise ValueError("Fig. 02 M_IMBH_init values must be finite and non-negative.")
    if np.any(~np.isfinite(lookback_init)) or np.any(lookback_init < 0.0):
        raise ValueError("Fig. 02 initial lookback times must be finite and non-negative.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Fig. 02 inherited volume weights must be finite and strictly positive.")
    volume_cmpc3 = float(volume_cmpc3)
    if not np.isfinite(volume_cmpc3) or volume_cmpc3 <= 0.0:
        raise ValueError("Fig. 02 reference volume must be finite and strictly positive.")
    t0 = float(Redshift2CosmicAge(0.0, time_unit="Gyr"))
    formation_time = t0 - lookback_init
    if np.any(~np.isfinite(formation_time)) or np.any(formation_time <= 0.0) or np.any(formation_time > t0):
        raise ValueError("Fig. 02 contains a cosmic formation time outside 0 < t_form <= t_0.")
    formation_redshift = np.asarray(
        [CosmicAge2Redshift(float(value), time_unit="Gyr") for value in formation_time],
        dtype=float,
    )
    if np.any(~np.isfinite(formation_redshift)) or np.any(formation_redshift < 0.0):
        raise ValueError("Fig. 02 formation-time conversion produced an invalid formation redshift.")
    formation_x = np.log10(1.0 + formation_redshift)
    if np.any(~np.isfinite(formation_x)) or np.any(formation_x < 0.0):
        raise ValueError("Fig. 02 formation redshift conversion produced an invalid log10(1+z).")

    positive = initial_mass > 0.0
    if not np.any(positive):
        raise ValueError("Fig. 02 found no positive M_IMBH_init seed masses.")
    positive_indices = np.flatnonzero(positive)
    positive_status = status[positive]
    positive_weights = weights[positive]
    positive_x = formation_x[positive]
    positive_redshift = formation_redshift[positive]
    status_raw_counts = {
        int(code): int(np.count_nonzero(positive_status == code))
        for code in np.unique(positive_status)
    }
    status_effective_counts = {
        int(code): float(np.sum(positive_weights[positive_status == code]))
        for code in np.unique(positive_status)
    }
    display_mask = (positive_x >= FIG02_XLIM_LOG1PZ[0]) & (positive_x <= FIG02_XLIM_LOG1PZ[1])
    outside_low_mask = positive_x < FIG02_XLIM_LOG1PZ[0]
    outside_high_mask = positive_x > FIG02_XLIM_LOG1PZ[1]
    if np.any(display_mask & (outside_low_mask | outside_high_mask)):
        raise ValueError("Fig. 02 display-window event masks overlap unexpectedly.")
    return {
        "positive_indices": positive_indices,
        "M_IMBH_init": initial_mass[positive],
        "formation_time_gyr": formation_time[positive],
        "formation_redshift": positive_redshift,
        "formation_x": positive_x,
        "weights": positive_weights,
        "status": positive_status,
        "halo_id_z0": halo_id[positive],
        "volume_cmpc3": volume_cmpc3,
        "raw_positive_count": int(np.count_nonzero(positive)),
        "effective_positive_count": float(np.sum(positive_weights)),
        "status_raw_counts": status_raw_counts,
        "status_effective_counts": status_effective_counts,
        "display_mask": display_mask,
        "outside_low_mask": outside_low_mask,
        "outside_high_mask": outside_high_mask,
        "display_raw_count": int(np.count_nonzero(display_mask)),
        "display_effective_count": float(np.sum(positive_weights[display_mask])),
        "outside_low_raw_count": int(np.count_nonzero(outside_low_mask)),
        "outside_low_effective_count": float(np.sum(positive_weights[outside_low_mask])),
        "outside_high_raw_count": int(np.count_nonzero(outside_high_mask)),
        "outside_high_effective_count": float(np.sum(positive_weights[outside_high_mask])),
    }
def _fig04_project_central_bh_densities(summary_by_z, volume_cmpc3, mass_column="M_SMBH_init"):
    """Project one weighted central M_SMBH_init state per halo and snapshot."""

    if not isinstance(summary_by_z, pd.DataFrame):
        raise ValueError("Fig. 04 central-state projection requires haloSummaryByZ as a pandas DataFrame.")
    mass_aliases = {
        "M_SMBH_init": "central_bh_mass_init_msun",
        "central_bh_mass_init_msun": "M_SMBH_init",
    }
    source_mass_column = mass_column if mass_column in summary_by_z.columns else mass_aliases.get(mass_column)
    if source_mass_column not in summary_by_z.columns:
        raise ValueError(f"haloSummaryByZ is missing the requested central mass column {mass_column!r}.")
    if "redshift" in summary_by_z.columns:
        redshift_column = "redshift"
    elif "z_out" in summary_by_z.columns:
        redshift_column = "z_out"
    else:
        raise ValueError("Fig. 04 central-state projection requires redshift or z_out in haloSummaryByZ.")
    required = ["halo_id_z0", redshift_column, source_mass_column, "volume_weight_tng50"]
    missing = [name for name in required if name not in summary_by_z.columns]
    if missing:
        raise ValueError(f"haloSummaryByZ is missing Fig. 04 central-state columns: {missing}")
    if len(summary_by_z) == 0:
        raise ValueError("Fig. 04 central-state projection cannot use an empty haloSummaryByZ.")

    table = summary_by_z.loc[:, required].copy()
    table["halo_id_z0"] = _integer_values(table["halo_id_z0"], "Fig. 04 central-state halo IDs", non_negative=True)
    for column in required:
        if column != "halo_id_z0":
            table[column] = pd.to_numeric(table[column], errors="coerce")
    halo_id = table["halo_id_z0"].to_numpy(dtype=np.int64)
    redshift = table[redshift_column].to_numpy(dtype=float)
    central_mass = table[source_mass_column].to_numpy(dtype=float)
    weights = table["volume_weight_tng50"].to_numpy(dtype=float)
    if np.any(~np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("Fig. 04 central states require finite non-negative output redshifts.")
    if np.any(~np.isfinite(central_mass)) or np.any(central_mass < 0.0):
        raise ValueError("Fig. 04 central M_SMBH_init states must be finite and non-negative.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Fig. 04 central-state inherited volume weights must be finite and strictly positive.")
    volume_cmpc3 = float(volume_cmpc3)
    if not np.isfinite(volume_cmpc3) or volume_cmpc3 <= 0.0:
        raise ValueError("Fig. 04 central-state reference volume must be finite and strictly positive.")

    key_table = pd.DataFrame({"halo_id_z0": halo_id, "redshift": redshift})
    if key_table.duplicated().any():
        duplicate_keys = key_table.loc[key_table.duplicated(keep=False)].drop_duplicates().to_dict("records")
        raise ValueError(f"Fig. 04 central projection requires one state per (halo_id_z0, redshift); duplicates={duplicate_keys[:10]}.")

    bin_edges = np.asarray(FIG04_CENTRAL_LOGM_BIN_EDGES, dtype=float)
    if (
        len(bin_edges) != 81
        or not np.isclose(bin_edges[0], 0.0, rtol=0.0, atol=1.0e-12)
        or not np.isclose(bin_edges[-1], 8.0, rtol=0.0, atol=1.0e-12)
        or np.any(~np.isfinite(bin_edges))
        or np.any(np.diff(bin_edges) <= 0.0)
        or not np.allclose(np.diff(bin_edges), 0.1, rtol=0.0, atol=1.0e-12)
    ):
        raise RuntimeError("Fig. 04 central-state mass grid is not the validated 0.0--8.0 dex sequence.")
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = np.diff(bin_edges)
    positive = central_mass > 0.0
    if not np.any(positive):
        raise ValueError("Fig. 04 found no positive M_SMBH_init central states.")
    positive_log_mass = np.log10(central_mass[positive])
    if np.any(~np.isfinite(positive_log_mass)):
        raise ValueError("Fig. 04 positive central M_SMBH_init states have non-finite logarithms.")
    if np.any(positive_log_mass < bin_edges[0] - 1.0e-12) or np.any(positive_log_mass > bin_edges[-1] + 1.0e-12):
        offending = central_mass[positive][(positive_log_mass < bin_edges[0] - 1.0e-12) | (positive_log_mass > bin_edges[-1] + 1.0e-12)]
        raise ValueError(f"Fig. 04 positive central M_SMBH_init states lie outside the fixed 0--8 dex mass grid: {offending[:10].tolist()}")

    redshift_values = np.sort(np.unique(redshift))
    densities = []
    raw_counts = []
    effective_counts = []
    positive_raw_counts = []
    positive_effective_counts = []
    zero_raw_counts = []
    zero_effective_counts = []
    identity_tolerance = 1.0e-10
    for redshift_value in redshift_values:
        snapshot = redshift == float(redshift_value)
        positive_snapshot = snapshot & positive
        zero_snapshot = snapshot & (central_mass == 0.0)
        snapshot_log_mass = np.log10(central_mass[positive_snapshot]) if np.any(positive_snapshot) else np.asarray([], dtype=float)
        snapshot_weights = weights[positive_snapshot]
        snapshot_raw, _ = np.histogram(snapshot_log_mass, bins=bin_edges)
        snapshot_effective, _ = np.histogram(snapshot_log_mass, bins=bin_edges, weights=snapshot_weights)
        if int(np.sum(snapshot_raw)) != int(np.count_nonzero(positive_snapshot)):
            raise ValueError(f"Fig. 04 central histogram dropped positive states at z={float(redshift_value):.6g}.")
        positive_effective = float(np.sum(snapshot_weights))
        if not np.isclose(float(np.sum(snapshot_effective)), positive_effective, rtol=identity_tolerance, atol=identity_tolerance):
            raise ValueError(f"Fig. 04 central histogram weights failed at z={float(redshift_value):.6g}.")
        densities.append(snapshot_effective.astype(float) / (volume_cmpc3 * bin_width))
        raw_counts.append(snapshot_raw.astype(int, copy=False))
        effective_counts.append(snapshot_effective.astype(float, copy=False))
        positive_raw_counts.append(int(np.count_nonzero(positive_snapshot)))
        positive_effective_counts.append(positive_effective)
        zero_raw_counts.append(int(np.count_nonzero(zero_snapshot)))
        zero_effective_counts.append(float(np.sum(weights[zero_snapshot])))

    positive_raw_counts = np.asarray(positive_raw_counts, dtype=int)
    positive_effective_counts = np.asarray(positive_effective_counts, dtype=float)
    zero_raw_counts = np.asarray(zero_raw_counts, dtype=int)
    zero_effective_counts = np.asarray(zero_effective_counts, dtype=float)
    densities = np.asarray(densities, dtype=float)
    raw_counts = np.asarray(raw_counts, dtype=int)
    effective_counts = np.asarray(effective_counts, dtype=float)
    if int(np.sum(raw_counts)) != int(np.sum(positive_raw_counts)):
        raise ValueError("Fig. 04 central per-bin raw counts do not equal the positive-state inventory.")
    if not np.isclose(float(np.sum(effective_counts)), float(np.sum(positive_effective_counts)), rtol=identity_tolerance, atol=identity_tolerance):
        raise ValueError("Fig. 04 central per-bin effective counts do not equal the positive-state inventory.")
    positive_redshift_indices = np.flatnonzero(positive_raw_counts > 0)
    if len(positive_redshift_indices) == 0:
        raise ValueError("Fig. 04 found no output redshift with a positive M_SMBH_init central state.")
    return {
        "redshifts": redshift_values.astype(float),
        "plotted_redshifts": redshift_values[positive_redshift_indices].astype(float),
        "positive_redshift_indices": positive_redshift_indices.astype(int),
        "omitted_redshifts": redshift_values[positive_raw_counts == 0].astype(float),
        "log10_mass_bin_edges": bin_edges,
        "log10_mass_bin_centres": bin_centres,
        "mass_bin_centres_msun": np.power(10.0, bin_centres),
        "densities": densities,
        "raw_counts": raw_counts,
        "effective_counts": effective_counts,
        "positive_raw_counts": positive_raw_counts,
        "positive_effective_counts": positive_effective_counts,
        "zero_raw_counts": zero_raw_counts,
        "zero_effective_counts": zero_effective_counts,
        "volume_cmpc3": volume_cmpc3,
        "mass_column": str(mass_column),
    }
def _plot_values(values):
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out) | (out <= 0.0)] = np.nan
    return out
def _log10_plot_values(values):
    out = _plot_values(values)
    valid = np.isfinite(out)
    out[valid] = np.log10(out[valid])
    return out
def plot_fig03_bhmfs(bhmf_data, summary_by_z):
    required = [
        "Phi [lgM☉⁻¹Mpc⁻³]", "sigma_Phi_low [lgM☉⁻¹Mpc⁻³]", "sigma_Phi_high [lgM☉⁻¹Mpc⁻³]",
        "Mbh [M☉]", "sigma_Mbh_low", "sigma_Mbh_high", "shape", "label", "z_low", "z_high",
    ]
    missing = [name for name in required if name not in bhmf_data.columns]
    if missing:
        raise ValueError(f"Fig. 03 BHMF table is missing required columns: {missing}")
    x_project, this_work_by_z, inventory = _fig03_project_densities(summary_by_z)
    marker_rows = bhmf_data.copy()
    model_redshifts = np.asarray(sorted(this_work_by_z), dtype=float)
    observation_redshifts = 0.5 * (
        marker_rows["z_low"].to_numpy(dtype=float) + marker_rows["z_high"].to_numpy(dtype=float)
    )
    colour_redshifts = np.unique(np.concatenate((model_redshifts, observation_redshifts)))
    cmap = mpl.cm.jet
    if len(colour_redshifts) == 1:
        norm = mpl.colors.Normalize(vmin=float(colour_redshifts[0]) - 0.5, vmax=float(colour_redshifts[0]) + 0.5)
    else:
        norm = mpl.colors.Normalize(vmin=float(colour_redshifts.min()), vmax=float(colour_redshifts.max()))

    model_plot_values = {
        float(redshift): _log10_plot_values(density)
        for redshift, density in this_work_by_z.items()
    }
    y_values = [values[np.isfinite(values)] for values in model_plot_values.values()]
    for _, row in marker_rows.iterrows():
        phi = float(row["Phi [lgM☉⁻¹Mpc⁻³]"])
        sigma_low = float(row["sigma_Phi_low [lgM☉⁻¹Mpc⁻³]"])
        sigma_high = float(row["sigma_Phi_high [lgM☉⁻¹Mpc⁻³]"])
        y_values.append(np.asarray([phi, phi + sigma_high], dtype=float))
        if np.isfinite(sigma_low):
            y_values.append(np.asarray([phi - sigma_low], dtype=float))
    finite_y_values = np.concatenate([values[np.isfinite(values)] for values in y_values if len(values) > 0])
    if len(finite_y_values) == 0:
        raise ValueError("Fig. 03 cannot determine finite logarithmic density limits.")
    y_min = float(np.floor((np.min(finite_y_values) - 0.25) * 10.0) / 10.0)
    y_max = float(np.ceil((np.max(finite_y_values) + 0.25) * 10.0) / 10.0)
    x_values = np.log10(marker_rows["Mbh [M☉]"].to_numpy(dtype=float))
    x_min = min(4.2, float(np.min(x_project)))
    x_max = max(10.9, float(np.max(x_values) + 0.25))

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 5.0))
    plotted_labels = set()
    this_work_label_used = False
    for z_out, density in sorted(this_work_by_z.items()):
        ax.plot(
            x_project,
            model_plot_values[float(z_out)],
            c=cmap(norm(float(z_out))),
            lw=1.5,
            alpha=0.9,
            label="This work" if not this_work_label_used else None,
            zorder=2,
        )
        this_work_label_used = True
    for _, row in marker_rows.iterrows():
        label = str(row["label"])
        marker_label = label if label not in plotted_labels else None
        plotted_labels.add(label)
        x_value = float(np.log10(float(row["Mbh [M☉]"])))
        phi = float(row["Phi [lgM☉⁻¹Mpc⁻³]"])
        sigma_phi_low = float(row["sigma_Phi_low [lgM☉⁻¹Mpc⁻³]"])
        sigma_phi_high = float(row["sigma_Phi_high [lgM☉⁻¹Mpc⁻³]"])
        sigma_mbh_low = float(row["sigma_Mbh_low"])
        sigma_mbh_high = float(row["sigma_Mbh_high"])
        z_mid = 0.5 * (float(row["z_low"]) + float(row["z_high"]))
        colour = cmap(norm(z_mid))
        finite_sigma_phi_low = sigma_phi_low if np.isfinite(sigma_phi_low) else 0.0
        ax.errorbar(
            x_value,
            phi,
            xerr=np.asarray([[sigma_mbh_low], [sigma_mbh_high]], dtype=float),
            yerr=np.asarray([[finite_sigma_phi_low], [sigma_phi_high]], dtype=float),
            fmt=str(row["shape"]),
            ms=6.0,
            color=colour,
            ecolor=colour,
            markerfacecolor=colour,
            markeredgecolor=colour,
            markeredgewidth=0.8,
            elinewidth=0.8,
            capsize=0.0,
            label=marker_label,
            zorder=7,
        )
        if np.isposinf(sigma_phi_low):
            ax.plot([x_value, x_value], [phi, y_min], color=colour, lw=0.8, solid_capstyle="butt", zorder=6)

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label(r"Redshift $z$")
    if len(colour_redshifts) == 1:
        colour_bar.set_ticks([float(colour_redshifts[0])])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_{\odot})$")
    ax.set_ylabel(r"$\log \Phi\ [M_{\odot}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="lower left", fontsize=7.0, ncol=2)
    ax.tick_params(direction="in", right=True, top=True, which="both")
    inventory["model_redshifts"] = model_redshifts
    inventory["observational_redshift_midpoints"] = observation_redshifts
    inventory["colour_redshifts"] = colour_redshifts
    inventory["plot_y_limits"] = (y_min, y_max)
    inventory["infinite_lower_count"] = int(np.count_nonzero(np.isposinf(marker_rows["sigma_Phi_low [lgM☉⁻¹Mpc⁻³]"].to_numpy(dtype=float))))
    return fig, inventory
def plot_fig04_bhsmf(chen_data, summary_by_z, volume_cmpc3=BHMF_REFERENCE_VOLUME_CMPC3):
    """Plot the Chen+2026 reference curves and the central This work curve."""

    if not isinstance(chen_data, dict) or "by_role" not in chen_data:
        raise ValueError("Fig. 04 requires the structured output of load_chen2026_fig05a_seed_mass_functions().")
    chen_curves = chen_data["by_role"]
    missing_roles = [role for role in FIG04_CHEN_CURVE_ROLES if role not in chen_curves]
    if missing_roles:
        raise ValueError(f"Fig. 04 Chen+2026 data is missing curve roles: {missing_roles}")
    central_projection = _fig04_project_central_bh_densities(summary_by_z, volume_cmpc3=volume_cmpc3, mass_column="M_SMBH_init")
    central_x = central_projection["mass_bin_centres_msun"]
    central_z_values = central_projection["plotted_redshifts"]
    colour_redshifts = central_z_values
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=float(colour_redshifts[0]) - 0.5, vmax=float(colour_redshifts[0]) + 0.5) if len(colour_redshifts) == 1 else mpl.colors.Normalize(vmin=float(colour_redshifts.min()), vmax=float(colour_redshifts.max()))

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 5.0))
    central = chen_curves["all_seeds_central"]
    lower = chen_curves["all_seeds_lower_envelope"]
    upper = chen_curves["all_seeds_upper_envelope"]
    x_chen = np.power(10.0, central["log10_mbh_seed_msun"].to_numpy(dtype=float))
    y_central = central["phi_mpc3_dex1"].to_numpy(dtype=float)
    y_lower = lower["phi_mpc3_dex1"].to_numpy(dtype=float)
    y_upper = upper["phi_mpc3_dex1"].to_numpy(dtype=float)
    ax.fill_between(x_chen, y_lower, y_upper, color="#9e9e9e", alpha=0.35, linewidth=0.0, zorder=1)
    ax.plot(x_chen, y_lower, c="#9e9e9e", lw=0.65, zorder=2)
    ax.plot(x_chen, y_upper, c="#9e9e9e", lw=0.65, zorder=2)
    ax.plot(x_chen, y_central, c="#000000", lw=1.6, zorder=3)
    for role in FIG04_CHEN_VISIBLE_CURVE_ROLES:
        curve = chen_curves[role]
        x_curve = np.power(10.0, curve["log10_mbh_seed_msun"].to_numpy(dtype=float))
        y_curve = curve["phi_mpc3_dex1"].to_numpy(dtype=float)
        ax.plot(x_curve, y_curve, c=str(curve["colour"].iloc[0]), ls=str(curve["linestyle"].iloc[0]), lw=1.25, zorder=3)

    for redshift, row_index in zip(central_z_values, central_projection["positive_redshift_indices"]):
        density = central_projection["densities"][int(row_index)]
        if not np.any(density > 0.0):
            continue
        ax.plot(central_x, _plot_values(density), c=cmap(norm(float(redshift))), ls="-", lw=1.8, alpha=0.95, zorder=4)

    positive_x = [x_chen]
    positive_y = [y_central, y_lower, y_upper]
    for role in FIG04_CHEN_VISIBLE_CURVE_ROLES:
        curve = chen_curves[role]
        positive_x.append(np.power(10.0, curve["log10_mbh_seed_msun"].to_numpy(dtype=float)))
        positive_y.append(curve["phi_mpc3_dex1"].to_numpy(dtype=float))
    for row_index in central_projection["positive_redshift_indices"]:
        central_density = central_projection["densities"][int(row_index)]
        positive_mask = central_density > 0.0
        if np.any(positive_mask):
            positive_x.append(central_x[positive_mask])
            positive_y.append(central_density[positive_mask])
    x_positive = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in positive_x])
    y_positive = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in positive_y])
    if len(x_positive) == 0 or len(y_positive) == 0:
        raise ValueError("Fig. 04 cannot determine finite positive plot limits.")
    ax.set_xlim(float(np.min(x_positive) / 1.35), float(np.max(x_positive) * 1.35))
    y_min = 1.0e-6
    y_max = float(np.max(y_positive) * 1.6)
    if not np.isfinite(y_max) or y_max <= y_min:
        raise ValueError(f"Fig. 04 visible curves do not extend above the requested y-axis floor {y_min:.1e}.")
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Seed mass $M_{\rm BH,seed}\ [M_{\odot}]$")
    ax.set_ylabel(r"$\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.text(0.03, 0.97, "Central initial seed states\nTNG100 objects use approved parent-halo volume weights\nThis work: one central $M_{\\rm SMBH,init}$ state per halo per output snapshot\n(not an additional $M_{\\rm IMBH,init}$ event)", transform=ax.transAxes, fontsize=6.2, color="0.25", va="top")

    source_handles = [
        Line2D([], [], color="#000000", lw=1.6, label="All seeds"),
        Patch(facecolor="#9e9e9e", edgecolor="none", alpha=0.35, label="All seeds envelope"),
    ]
    for role in FIG04_CHEN_VISIBLE_CURVE_ROLES:
        curve = chen_curves[role]
        source_handles.append(Line2D([], [], color=str(curve["colour"].iloc[0]), ls=str(curve["linestyle"].iloc[0]), lw=1.25, label=str(curve["curve_label"].iloc[0])))
    source_legend = ax.legend(handles=source_handles, title="Chen+2026", frameon=False, fontsize=6.7, title_fontsize=7.4, loc="upper right", ncol=1, borderaxespad=0.3)
    ax.add_artist(source_legend)
    ax.legend(handles=[Line2D([], [], color="black", ls="-", lw=1.8, label="This work")], frameon=False, fontsize=7.0, loc="lower left", ncol=1, borderaxespad=0.3)

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(colour_redshifts) == 1:
        colour_bar.set_ticks([float(colour_redshifts[0])])
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig, {"central": central_projection, "colour_redshifts": colour_redshifts}
def plot_fig02_bhseed_history(chen_data, final_gc, volume_cmpc3=BHMF_REFERENCE_VOLUME_CMPC3):
    """Plot the unpartitioned final-GC IMBH-seed formation history."""

    if not isinstance(chen_data, dict) or "by_panel" not in chen_data:
        raise ValueError("Fig. 02 requires the structured output of load_chen2026_fig06_seed_history().")
    for panel in ("a", "d"):
        if panel not in chen_data["by_panel"]:
            raise ValueError(f"Fig. 02 Chen+2026 data is missing panel {panel!r}.")
        missing_roles = [role for role in FIG02_CHEN_CURVE_ROLES if role not in chen_data["by_panel"][panel]]
        if missing_roles:
            raise ValueError(f"Fig. 02 Chen+2026 panel {panel} is missing curve roles: {missing_roles}")

    events = _fig02_project_bhseed_events(final_gc, volume_cmpc3)
    volume_cmpc3 = float(events["volume_cmpc3"])
    display_x = events["formation_x"][events["display_mask"]]
    display_weights = events["weights"][events["display_mask"]]
    if len(display_x) == 0:
        raise ValueError("Fig. 02 has no positive IMBH seeds in the requested 0 <= z <= 50 display range.")

    x_left, x_right = map(float, FIG02_XLIM_LOG1PZ)
    rate_edges = np.arange(x_left, x_right, FIG02_RATE_LOG1PZ_BIN_WIDTH, dtype=float)
    if len(rate_edges) == 0 or not np.isclose(rate_edges[0], x_left, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Fig. 02 rate grid failed to start at the requested lower x limit.")
    if rate_edges[-1] < x_right - 1.0e-12:
        rate_edges = np.append(rate_edges, x_right)
    else:
        rate_edges[-1] = x_right
    if np.any(np.diff(rate_edges) <= 0.0):
        raise RuntimeError("Fig. 02 rate grid is not strictly increasing.")
    rate_bin_widths = np.diff(rate_edges)
    rate_x = 0.5 * (rate_edges[:-1] + rate_edges[1:])
    weighted_counts, _ = np.histogram(display_x, bins=rate_edges, weights=display_weights)
    raw_counts, _ = np.histogram(display_x, bins=rate_edges)
    if int(np.sum(raw_counts)) != int(events["display_raw_count"]):
        raise ValueError("Fig. 02 rate histogram dropped positive seed events in the display range.")
    rate_density = weighted_counts.astype(float) / (volume_cmpc3 * rate_bin_widths)
    rate_integral = float(np.sum(rate_density * rate_bin_widths))
    expected_rate_integral = float(events["display_effective_count"] / volume_cmpc3)
    if not np.isclose(rate_integral, expected_rate_integral, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("Fig. 02 weighted rate integral does not equal the displayed effective seed density.")

    # The survival count uses all positive events before the display cut and the
    # strict convention z_form > z, implemented equivalently in x.
    order = np.argsort(events["formation_x"], kind="mergesort")
    sorted_x = events["formation_x"][order]
    sorted_weights = events["weights"][order]
    suffix_weights = np.cumsum(sorted_weights[::-1], dtype=float)[::-1]
    survival_index = np.searchsorted(sorted_x, rate_x, side="right")
    cumulative_density = np.zeros_like(rate_x, dtype=float)
    has_survivors = survival_index < len(sorted_x)
    cumulative_density[has_survivors] = suffix_weights[survival_index[has_survivors]] / volume_cmpc3
    if np.any(~np.isfinite(cumulative_density)) or np.any(cumulative_density < 0.0):
        raise ValueError("Fig. 02 cumulative seed density is not finite and non-negative.")
    if np.any(np.diff(cumulative_density) > 1.0e-12 * max(1.0, float(np.max(cumulative_density)))):
        raise ValueError("Fig. 02 cumulative seed density is not non-increasing with redshift.")

    reference_values = {"a": [], "d": []}
    for panel in ("a", "d"):
        for role in FIG02_CHEN_CURVE_ROLES:
            reference_values[panel].append(chen_data["by_panel"][panel][role]["value_mpc3"].to_numpy(dtype=float))
    top_values = np.concatenate(reference_values["a"] + [rate_density[rate_density > 0.0]])
    bottom_values = np.concatenate(reference_values["d"] + [cumulative_density[cumulative_density > 0.0]])
    if len(top_values) == 0 or len(bottom_values) == 0:
        raise ValueError("Fig. 02 cannot determine finite positive y-axis limits.")
    top_values = top_values[np.isfinite(top_values) & (top_values > 0.0)]
    bottom_values = bottom_values[np.isfinite(bottom_values) & (bottom_values > 0.0)]
    if len(top_values) == 0 or len(bottom_values) == 0:
        raise ValueError("Fig. 02 has no finite positive values for logarithmic axes.")
    top_y_min = 10.0 ** (math.floor(math.log10(float(np.min(top_values)))) - 0.25)
    top_y_max = 10.0 ** (math.ceil(math.log10(float(np.max(top_values)))) + 0.25)
    bottom_y_min = 10.0 ** (math.floor(math.log10(float(np.min(bottom_values)))) - 0.25)
    bottom_y_max = 10.0 ** (math.ceil(math.log10(float(np.max(bottom_values)))) + 0.25)

    fig, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True, dpi=STD_DPI, figsize=FIG02_FIGSIZE)
    axes = np.asarray(axes).reshape(-1)
    for panel, ax, quantity, model_x, model_values in (
        ("a", axes[0], "rate", rate_x, rate_density),
        ("d", axes[1], "cumulative", rate_x, cumulative_density),
    ):
        for role in FIG02_CHEN_CURVE_ROLES:
            curve = chen_data["by_panel"][panel][role]
            x_curve = curve["x_log10_1pz"].to_numpy(dtype=float)
            y_curve = curve["value_mpc3"].to_numpy(dtype=float)
            # Extend only the panel-d plotting arrays with the native endpoint value.
            if panel == "d" and x_curve[0] > x_left:
                x_curve = np.concatenate(([x_left], x_curve))
                y_curve = np.concatenate(([y_curve[0]], y_curve))
            colour = str(curve["colour"].iloc[0])
            linestyle = str(curve["linestyle"].iloc[0])
            if role == "all_seeds":
                fill_floor = float(np.min(y_curve) * 0.75)
                ax.fill_between(x_curve, fill_floor, y_curve, color=colour, alpha=0.20, linewidth=0.0, zorder=1)
            elif role == "popii":
                fill_floor = float(np.min(y_curve) * 0.75)
                ax.fill_between(x_curve, fill_floor, y_curve, color=colour, alpha=0.50, linewidth=0.0, zorder=1)
            ax.plot(x_curve, y_curve, c=colour, ls=linestyle, lw=FIG02_REFERENCE_LINEWIDTH, zorder=3)
        model_plot_values = np.where(model_values > 0.0, model_values, np.nan)
        ax.plot(
            model_x,
            model_plot_values,
            c=FIG02_MODEL_COLOUR,
            lw=FIG02_MODEL_LINEWIDTH,
            drawstyle="steps-mid" if panel == "a" else "default",
            label="_nolegend_",
            zorder=5,
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.tick_params(direction="in", right=True, top=True, which="both")
        ax.text(0.965, 0.965, panel, transform=ax.transAxes, ha="right", va="top", fontsize=10.0, fontweight="bold")

    axes[0].set_ylim(top_y_min, top_y_max)
    axes[1].set_ylim(bottom_y_min, bottom_y_max)
    axes[0].set_ylabel(r"$d n_{\rm seed}/d\log_{10}(1+z)\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    axes[1].set_ylabel(r"$n_{\rm seed}(>z)\ [\mathrm{Mpc}^{-3}]$")
    axes[1].set_xlabel(r"$\log_{10}(1+z)$")
    axes[0].set_xlim(x_left, x_right)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xticks(np.arange(0.0, 1.61, 0.2))

    source_handles = {}
    for role in FIG02_CHEN_CURVE_ROLES:
        curve = chen_data["by_panel"]["a"][role]
        source_handles[role] = Line2D(
            [], [], color=str(curve["colour"].iloc[0]), ls=str(curve["linestyle"].iloc[0]),
            lw=FIG02_REFERENCE_LINEWIDTH, label=FIG02_CHEN_CURVE_LABELS[role],
        )
    axes[0].legend(
        handles=[source_handles[role] for role in FIG02_CHEN_CURVE_ROLES[:3]],
        title="Chen+2026", frameon=False, fontsize=6.9, title_fontsize=7.4,
        loc="best", ncol=1, borderaxespad=0.25,
    )
    axes[1].legend(
        handles=[source_handles["popii"], Line2D([], [], color=FIG02_MODEL_COLOUR, lw=FIG02_MODEL_LINEWIDTH, label="This work")],
        title="Chen+2026 / model", frameon=False, fontsize=6.9, title_fontsize=7.4,
        loc="best", ncol=1, borderaxespad=0.25,
    )

    secondary_axis = axes[0].secondary_xaxis(
        "top",
        functions=(lambda x: np.power(10.0, x) - 1.0, lambda z: np.log10(1.0 + z)),
    )
    secondary_axis.set_xticks([0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    secondary_axis.set_xticklabels(["0", "5", "10", "20", "30", "40", "50"])
    secondary_axis.set_xlabel(r"Redshift $z$")
    secondary_axis.tick_params(direction="in", which="both")

    return fig, {
        **events,
        "rate_x": rate_x,
        "rate_bin_edges": rate_edges,
        "rate_bin_widths": rate_bin_widths,
        "rate_density": rate_density,
        "cumulative_x": rate_x.copy(),
        "cumulative_density": cumulative_density,
        "rate_integral": rate_integral,
        "expected_rate_integral": expected_rate_integral,
        "rate_bin_width": float(FIG02_RATE_LOG1PZ_BIN_WIDTH),
        "cumulative_boundary": "strict z_form > z",
    }


# MAIN FUNCTION
def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Kong & Li 2026 suite a (MBH--stellar-mass, seed-history, BHMF, and central BHSMF figures) from one High-z SMBH Seeds output directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Model output directory.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Output plot directory. Default: <out_dir>/_plots_Kong&Li2026a.")
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.5, help="Log10 stellar-mass bin width for suite-a Fig. 01.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else out_dir / "_plots_Kong&Li2026a"
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

    observations = load_mbh_mstar_observations()
    fig01 = plot_fig01_mbh_mstar(summary_by_z, observations, float(args.mass_bin_width_dex))
    _save_figure(fig01, plot_dir / FIGURE_01_FILENAME)

    chen_fig02_data = load_chen2026_fig06_seed_history()
    fig02, inventory02 = plot_fig02_bhseed_history(
        chen_fig02_data,
        final_gc,
        volume_cmpc3=tng_volume_context["volume_tng50_cmpc3"],
    )
    status_inventory02 = ", ".join(
        f"status {int(status)}: raw={int(inventory02['status_raw_counts'][status])}, "
        f"effective={float(inventory02['status_effective_counts'][status]):.6g}"
        for status in sorted(inventory02["status_raw_counts"])
    )
    print(
        f"Fig. 02 This work positive IMBH-seed inventory: raw={int(inventory02['raw_positive_count'])}, "
        f"effective={float(inventory02['effective_positive_count']):.12g}; {status_inventory02}."
    )
    print(
        f"Fig. 02 display z range={FIG02_DISPLAY_REDSHIFT_RANGE[0]:g}--{FIG02_DISPLAY_REDSHIFT_RANGE[1]:g}: "
        f"raw={int(inventory02['display_raw_count'])}, effective={float(inventory02['display_effective_count']):.12g}; "
        f"outside low-z raw/effective={int(inventory02['outside_low_raw_count'])}/{float(inventory02['outside_low_effective_count']):.12g}, "
        f"outside high-z raw/effective={int(inventory02['outside_high_raw_count'])}/{float(inventory02['outside_high_effective_count']):.12g}; "
        f"unsmoothed Δlog10(1+z)={FIG02_RATE_LOG1PZ_BIN_WIDTH:g}."
    )
    native_first_d = ", ".join(
        f"{role}: x={float(chen_fig02_data['by_panel']['d'][role]['x_log10_1pz'].iloc[0]):.12g}, "
        f"y={float(chen_fig02_data['by_panel']['d'][role]['value_mpc3'].iloc[0]):.12g}"
        for role in FIG02_CHEN_CURVE_ROLES
    )
    print(
        f"Fig. 02 cumulative model grid/value summary: N={len(inventory02['cumulative_x'])}, "
        f"x={float(inventory02['cumulative_x'][0]):.12g}--{float(inventory02['cumulative_x'][-1]):.12g}, "
        f"n={float(inventory02['cumulative_density'][0]):.12g}--{float(inventory02['cumulative_density'][-1]):.12g}, "
        f"boundary={inventory02['cumulative_boundary']}; "
        f"Chen+2026 panel d native first points [{native_first_d}], "
        "endpoint-constant plotting extension to x=0; native CSV values unchanged."
    )
    _save_figure(fig02, plot_dir / FIGURE_02_FILENAME)

    bhmf_data = load_bhmf_data()
    fig03, inventory03 = plot_fig03_bhmfs(bhmf_data, summary_by_z)
    counts03 = ", ".join(f"z={z:.6g}: {n}" for z, n in sorted(inventory03["this_work_by_z"].items()))
    effective_counts03 = ", ".join(f"z={z:.6g}: {n:.6g}" for z, n in sorted(inventory03["this_work_effective_by_z"].items()))
    invalid_counts03 = ", ".join(f"z={z:.6g}: {n}" for z, n in sorted(inventory03["invalid_mass_by_z"].items()) if n > 0)
    out_of_range_counts03 = ", ".join(
        f"z={z:.6g}: raw={inventory03['out_of_range_raw_by_z'][z]}, effective={inventory03['out_of_range_effective_by_z'][z]:.6g}"
        for z in sorted(inventory03["out_of_range_raw_by_z"])
        if inventory03["out_of_range_raw_by_z"][z] > 0
    )
    summary_z_values03 = pd.to_numeric(summary_by_z["z_out"], errors="coerce").to_numpy(dtype=float)
    low_redshifts03 = np.sort(np.unique(summary_z_values03[summary_z_values03 <= FIG03_MIN_MODEL_REDSHIFT_EXCLUSIVE]))
    omitted_no_positive03 = ", ".join(f"{float(z):.6g}" for z in inventory03["omitted_no_positive_redshifts"])
    omitted_no_visible03 = ", ".join(f"{float(z):.6g}" for z in inventory03["omitted_no_visible_density_redshifts"])
    print(f"Fig. 03 excluded model output redshifts z<=3: [{', '.join(f'{float(z):.6g}' for z in low_redshifts03)}].")
    print(f"Fig. 03 This work positive-BH inventory: {counts03}.")
    print(f"Fig. 03 This work effective inventory: {effective_counts03}.")
    print(f"Fig. 03 invalid M_SMBH_final rows omitted: [{invalid_counts03}].")
    print(f"Fig. 03 positive masses outside plotted mass range: [{out_of_range_counts03}].")
    print(f"Fig. 03 omitted redshifts with no positive mass: [{omitted_no_positive03}].")
    print(f"Fig. 03 omitted redshifts with no visible density: [{omitted_no_visible03}].")
    _save_figure(fig03, plot_dir / FIGURE_03_FILENAME)

    chen_fig04_data = load_chen2026_fig05a_seed_mass_functions()
    fig04, inventory04 = plot_fig04_bhsmf(
        chen_fig04_data,
        summary_by_z,
        volume_cmpc3=tng_volume_context["volume_tng50_cmpc3"],
    )
    central_inventory04 = inventory04["central"]
    plotted_central_indices04 = set(int(value) for value in central_inventory04["positive_redshift_indices"])
    for index, redshift in enumerate(central_inventory04["redshifts"]):
        print(
            f"Fig. 04 This work inventory z={float(redshift):.6g}: "
            f"positive raw={int(central_inventory04['positive_raw_counts'][index])}, "
            f"positive effective={float(central_inventory04['positive_effective_counts'][index]):.6g}, "
            f"zero raw={int(central_inventory04['zero_raw_counts'][index])}, "
            f"zero effective={float(central_inventory04['zero_effective_counts'][index]):.6g}, "
            f"plotted positive curve={index in plotted_central_indices04}."
        )
    print(
        f"Fig. 04 reference volume: {inventory04['central']['volume_cmpc3']:.12g} cMpc^3; "
        "positive central M_SMBH_init states use inherited TNG parent-halo weights."
    )
    _save_figure(fig04, plot_dir / FIGURE_04_FILENAME)


if __name__ == "__main__":
    main()
