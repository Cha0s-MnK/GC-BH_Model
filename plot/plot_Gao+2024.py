#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""Reproduce the maintained Gao+2024 figure subset from one modern run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PLOT_DIR = PROJECT_ROOT / "plot"
for import_dir in [PLOT_DIR, SRC_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from config import STD_DPI, parse_exact_int64  # noqa: E402
from evo import (  # noqa: E402
    STAT_ALIVE,
    STAT_DISRUPT,
    STAT_SUNK_BH,
    STAT_SUNK_GC,
    STAT_WANDER,
)
from load_output import (  # noqa: E402
    DepositProfile,
    OutputPaths,
    load_allcat,
    load_deposit_profile,
    load_final_gcs,
    load_halo_summary,
    load_halo_summary_by_z,
    load_mpb,
    load_run_metadata,
    output_paths,
)
from plot_common import (  # noqa: E402
    apply_style,
    finish_axis,
    finish_log_axis,
    plot_dir as default_plot_dir,
    save_pdf,
    use_agg_backend,
)

use_agg_backend()

"""Local MW/M31 NSC+SMBH constants for Gao+2024 plotting."""

M_SMBH_MW = 4.297e6
M_SMBH_MW_err = 0.012e6

M_NSC_MW = 3.15e7
M_NSC_MW_err = 2.15e7
R_NSC_MW = 5.7
R_NSC_MW_err = 3.5

M_SMBH_M31 = 1.7e8
M_SMBH_M31_err = 0.6e8

M_NSC_M31 = 5.0e7
M_NSC_M31_err = 0.0
R_NSC_M31 = 8.0
R_NSC_M31_err = 4.0

RUN_METADATA_NAME = "run_metadata.json"


@dataclass
class ModelResult:
    """Container for one modern output model track."""

    ns_value: float
    r_init: np.ndarray
    r_final: np.ndarray
    m_final: np.ndarray
    status: np.ndarray
    deposit_profile: DepositProfile
    halo_summary: pd.DataFrame


@dataclass
class GaoOutput:
    """Validated modern Gao input tables and the aligned model arrays."""

    formed: pd.DataFrame
    final_gcs: pd.DataFrame
    halo_summary: pd.DataFrame
    halo_summary_by_z: pd.DataFrame
    mpb: pd.DataFrame
    deposit_profile: DepositProfile
    metadata: Dict[str, object]
    paths: OutputPaths
    final_redshift: float
    model: ModelResult


@dataclass(frozen=True)
class GalaxyObs:
    """Reference NSC/SMBH measurements used for overlays."""

    name: str
    m_smbh: float
    m_smbh_err: float
    m_nsc: float
    m_nsc_err: float
    r_nsc_pc: float
    r_nsc_err_pc: float
    color: str


def _safe_log10(arr: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Return log10 with a small floor to avoid -inf values."""

    return np.log10(np.clip(arr, floor, None))


def _log10_positive_or_nan(arr: np.ndarray) -> np.ndarray:
    """Return log10 for positive values and NaN elsewhere."""

    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    out[mask] = np.log10(arr[mask])
    return out


def _get_mw_m31_observations() -> Tuple[GalaxyObs, GalaxyObs]:
    """Read MW/M31 NSC and SMBH reference values from ``data.py``."""

    mw = GalaxyObs(
        name="MW",
        m_smbh=float(M_SMBH_MW),
        m_smbh_err=float(M_SMBH_MW_err),
        m_nsc=float(M_NSC_MW),
        m_nsc_err=float(M_NSC_MW_err),
        r_nsc_pc=float(R_NSC_MW),
        r_nsc_err_pc=float(R_NSC_MW_err),
        color="tab:red",
    )
    m31 = GalaxyObs(
        name="M31",
        m_smbh=float(M_SMBH_M31),
        m_smbh_err=float(M_SMBH_M31_err),
        m_nsc=float(M_NSC_M31),
        m_nsc_err=float(M_NSC_M31_err),
        r_nsc_pc=float(R_NSC_M31),
        r_nsc_err_pc=float(R_NSC_M31_err),
        color="tab:purple",
    )
    return mw, m31


def _add_nsc_smbh_points_pc(ax: plt.Axes, obs: GalaxyObs, show_labels: bool = True) -> None:
    """Overlay separate SMBH and NSC reference masses using radius in pc."""

    r_pc = max(obs.r_nsc_pc, 1.0e-2)
    xerr = obs.r_nsc_err_pc if obs.r_nsc_err_pc > 0 else None

    ax.errorbar(
        [r_pc],
        [obs.m_smbh],
        xerr=xerr,
        yerr=obs.m_smbh_err if obs.m_smbh_err > 0 else None,
        fmt="o",
        ms=6,
        mfc="white",
        mec=obs.color,
        color=obs.color,
        capsize=3,
        zorder=7,
        label=f"{obs.name} SMBH" if show_labels else None,
    )
    ax.errorbar(
        [r_pc],
        [obs.m_nsc],
        xerr=xerr,
        yerr=obs.m_nsc_err if obs.m_nsc_err > 0 else None,
        fmt="s",
        ms=6,
        mfc="white",
        mec=obs.color,
        color=obs.color,
        capsize=3,
        zorder=7,
        label=f"{obs.name} NSC" if show_labels else None,
    )


def _gao_observational_overlays() -> Dict[str, object]:
    """Return observational anchors used in Gao+2024 figure overlays.

    The points/curves below are compact digitized approximations from the
    original Gao+2024 figures and references listed in their captions.
    They are intentionally lightweight and self-contained so reproduction
    works without external catalog files.
    """

    # Fig. 2: observed m_GC / m_halo ratios from literature compilations.
    fig2_ratio_refs = {
        "S09": 7.0e-5,
        "G10": 5.5e-5,
        "H14": 2.8e-5,
        "H17": 4.0e-5,
    }

    # Fig. 3 / 7: digitized directly from Gao+2024 Fig. 3.
    fig3_G14 = {
        "r_kpc": np.array([
            0.010794385163805828,
            0.01230063722058812,
            0.014017056540677553,
            0.015885957941789649,
            0.01842923049791171,
            0.020638515286922114,
            0.023642304107814576,
            0.026812842640417521,
            0.03068065649923201,
            0.03481771932439584,
            0.039944728888434461,
            0.045330794603697426,
            0.051869648283687858,
            0.059153418073623108,
            0.066998307644800793,
            0.075030023455499988,
            0.08402457640539729,
            0.090262439565148898,
            0.10537418035304692,
            0.11800637212224759,
            0.13365630391219327,
            0.14799282057797614,
            0.16572852903919233,
            0.18350524251263223,
            0.2055002785561595,
            0.23013165132735636,
            0.25771101537826174,
            0.2886003948556277,
            0.32319219179891506,
            0.357847047984966,
            0.40073878563086688,
            0.44371618208541128,
            0.49689187543497038,
            0.55017198838376977,
            0.63394171535009246,
            0.71958255881831229,
            0.81535237778154514,
            0.9506437602765255,
            1.089868724595188,
            1.2434785508645683,
            1.4255369651226457,
            1.644812534271848,
            1.888645818909842,
            2.1673759695900916,
            2.544302336245633,
            2.8916691154160032,
            3.2956103293560832,
            3.7455550943770914,
            4.268785741747607,
            4.8594249503165067,
            5.4826699487870456,
            6.418497467163319,
            7.1112540348782005,
            8.6954469871221214,
            9.726932001959788,
            10.500459082798447,
            11.603912847605024,
            13.130690412962242,
            14.840205316722507,
            16.682616067302719,
            18.076889860627674,
            19.777484283556691,
            20.920007308913025,
            22.630583158099565,
            24.711647271833773,
            26.335538896278425,
            28.066103151268432,
            29.966053402457288,
            31.995353864081775,
            34.179033206174862,
            36.679661547674753,
            38.797207866128204,
            40.578438805557404,
            43.162118666779361], dtype=float),
        "Sigma": np.array([
            8356.6906981665297,
            7450.4877471336271,
            6630.5101644451859,
            5990.5625300056756,
            5184.5402755882734,
            4646.8853913068772,
            4164.9871911721458,
            3662.9109698697434,
            3240.6129273392885,
            2890.5106302927908,
            2515.8349213368992,
            2228.8150959209024,
            1958.4808373448052,
            1734.7118302628096,
            1512.8364376420081,
            1355.9500298640685,
            1215.3332890065183,
            1123.4469597770997,
            924.32492810547283,
            828.46921364628328,
            722.50525534194214,
            647.57894388426928,
            549.50327144061733,
            492.51786827772384,
            429.52320052341184,
            374.58575956445,
            317.85483802345385,
            277.20015074194092,
            241.74533270965966,
            205.13306128001383,
            178.89586284912812,
            156.01448905814297,
            132.38613290197867,
            112.33628549851414,
            91.787531311741657,
            75.715525722891437,
            63.006613557068478,
            46.026759963792099,
            36.732993025672044,
            29.844960002687827,
            22.416646421782012,
            16.942303760837735,
            12.485233615433351,
            9.6144877678573601,
            6.7950878638739867,
            5.0453518582363536,
            3.5296132212577125,
            2.6251998909511541,
            1.8427897663267763,
            1.3766103273666472,
            0.92361589529515589,
            0.56648337995115339,
            0.43739895848238897,
            0.21716432104578792,
            0.15386377948470603,
            0.11087908658297818,
            0.076722658163949368,
            0.045190853404211646,
            0.02793794213243249,
            0.017399163122002924,
            0.012356866785720583,
            0.0085701081379953789,
            0.0064746938468674902,
            0.0041620571539204979,
            0.0026201929496909872,
            0.0018900301538419758,
            0.0013602508999743794,
            0.00095009068739038265,
            0.00068875996529086335,
            0.00046761155104868584,
            0.00033222638607642097,
            0.00023672801748033864,
            0.00017699084920302209,
            0.00011954084738975554], dtype=float)}
    fig3_B21 = {
        "r_kpc": np.array([
            0.7540702727152646,
            2.160672441976534,
            3.8056650262561478,
            5.850232283203228,
            10.182134168792267,
            41.86707810315064,
            85.38374177603824], dtype=float),
        "Sigma": np.array([
            4.694827217503208,
            1.100537270475359,
            0.6026832936945054,
            0.2651410481520248,
            0.05570751921145461,
            0.0014619520722739484,
            0.0002606438897699442], dtype=float),
        "xerr": np.array([
            0.47264212016831164,
            0.7716571916216417,
            0.839539839261815,
            1.2386275436453413,
            3.096197916006476,
            27.75639854980916,
            11.693966387645133], dtype=float),
        "yerr": np.array([
            0.818592169595822,
            0.21642346456663253,
            0.1185192087147171,
            0.057891633068793746,
            0.012163334511347434,
            0.000287496608247744,
            0.00012916054965301022], dtype=float)}
    fig3_RBCver5 = {
        "r_kpc": np.array([
            0.9041384861689454,
            2.741559611696369,
            4.829126281225719,
            7.592894190931609,
            11.03249993157838,
            34.95270917442269,
            91.35702638558539], dtype=float),
        "Sigma": np.array([
            6.887135586430595,
            1.5284443417432307,
            0.9338611704098484,
            0.3486163061313284,
            0.24423941460507742,
            0.00501015788031781,
            0.00014669625243617185], dtype=float),
        "xerr": np.array([
            0.7254758069964147,
            1.0949316746346491,
            0.979065566523039,
            1.7416770358569362,
            1.6180869704551153,
            22.054883172778037,
            26.32302452418736], dtype=float),
        "yerr": np.array([
            0.714219948138501,
            0.19549309718602914,
            0.11944393888558147,
            0.04458918100834275,
            0.031239030635288323,
            0.000640815798563666,
            5.703792632013463e-05], dtype=float)}

    # Fig. 8: in-situ GC mass-function reference (Baumgardt+2021 trend).
    fig8_mass_obs = {
        "mass_msun": np.array([7.0e3, 1.5e4, 5.0e4, 1.0e5, 2.5e5, 8.0e5, 2.0e6, 4.0e6], dtype=float),
        "count": np.array([1.0, 2.0, 13.0, 25.0, 20.0, 6.5, 1.2, 0.2], dtype=float),
    }

    return {
        "fig2_ratio_refs": fig2_ratio_refs,
        "fig3_G14": fig3_G14,
        "fig3_B21": fig3_B21,
        "fig3_RBCver5": fig3_RBCver5,
        "fig8_mass_obs": fig8_mass_obs,
    }


def read_allcat(allcat_path: Path) -> pd.DataFrame:
    """Load one modern allcat table through the shared header-aware reader."""

    return load_allcat(Path(allcat_path))


def read_mpb(mpb_path: Path) -> pd.DataFrame:
    """Load one modern MPB table through the shared reader."""

    return load_mpb(Path(mpb_path))


def read_inputs(allcat_path: Path, mpb_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and standardise the modern formation and MPB tables."""

    return read_allcat(allcat_path), read_mpb(mpb_path)


def build_snap_to_z_map(gc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build a snapshot-to-redshift interpolation from formed GC rows."""

    snap_z = gc.groupby("snap_form")["zform"].median().sort_index()
    snap_arr = snap_z.index.to_numpy(dtype=float)
    z_arr = snap_z.to_numpy(dtype=float)
    return snap_arr, z_arr


def estimate_zhm(gc: pd.DataFrame, mpb: pd.DataFrame, *, final_redshift: float = 0.0) -> pd.DataFrame:
    """Estimate the half-mass assembly redshift for each target halo."""

    snap_arr, z_arr = build_snap_to_z_map(gc)
    rows: List[dict] = []

    for hid, grp in mpb.groupby("subhalo_id_z0", sort=True):
        g = grp.sort_values("SnapNum", ascending=False)
        if "Redshift" in g.columns:
            z_hist = pd.to_numeric(g["Redshift"], errors="coerce").to_numpy(dtype=float)
        else:
            z_hist = np.interp(
                g["SnapNum"].to_numpy(dtype=float),
                snap_arr,
                z_arr,
                left=z_arr[0],
                right=z_arr[-1],
            )

        if "SubhaloMass" in g.columns:
            mass_msun_h = pd.to_numeric(g["SubhaloMass"], errors="coerce").to_numpy(dtype=float) * 1.0e10
        elif "logMh_msun_h" in g.columns:
            logm = pd.to_numeric(g["logMh_msun_h"], errors="coerce").to_numpy(dtype=float)
            mass_msun_h = np.power(10.0, logm)
        else:
            raise ValueError("MPB table must provide either 'SubhaloMass' or 'logMh_msun_h'.")

        valid = np.isfinite(mass_msun_h) & (mass_msun_h > 0.0) & np.isfinite(z_hist)
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            usable_final = valid_idx[z_hist[valid_idx] >= (final_redshift - 1.0e-10)]
            if len(usable_final) > 0:
                idx_final = int(usable_final[np.argmin(z_hist[usable_final])])
            else:
                idx_final = int(valid_idx[-1])
            m0 = float(mass_msun_h[idx_final])
            half = 0.5 * m0
            hist_idx = valid_idx[valid_idx >= idx_final]
            crossed = hist_idx[mass_msun_h[hist_idx] <= half]
            hm_idx = int(crossed[0]) if len(crossed) > 0 else int(hist_idx[-1])
            snap_hm = int(g["SnapNum"].iloc[hm_idx])
            m_hm = float(mass_msun_h[hm_idx])
            m_halo_z0_mpb = m0
        else:
            sel_gc = gc["hid_z0"].to_numpy(dtype=int) == int(hid)
            if np.any(sel_gc):
                m_halo_z0_mpb = float(np.power(10.0, gc.loc[sel_gc, "logMh_z0"].iloc[0]))
            else:
                m_halo_z0_mpb = np.nan
            snap_hm = int(g["SnapNum"].min())
            m_hm = m_halo_z0_mpb
            idx_final = 0
            hm_idx = 0

        if np.any(valid):
            z_hm = float(z_hist[hm_idx])
        else:
            z_hm = float(np.interp(snap_hm, snap_arr, z_arr, left=z_arr[0], right=z_arr[-1]))
        spin_mag = float(g["spin_mag"].iloc[idx_final]) if "spin_mag" in g.columns else 500.0
        if not np.isfinite(spin_mag):
            spin_mag = 500.0

        rows.append(
            {
                "hid_z0": int(hid),
                "z_hm": z_hm,
                "snap_hm": snap_hm,
                "snap_final": int(g["SnapNum"].iloc[idx_final]),
                "z_final_used": float(z_hist[idx_final]) if np.any(valid) else np.nan,
                "M_halo_hm": m_hm,
                "spin_mag": spin_mag,
                "M_halo_z0_mpb": m_halo_z0_mpb,
                "M_halo_final_mpb": m_halo_z0_mpb,
            }
        )

    halo_meta = pd.DataFrame(rows).set_index("hid_z0").sort_index()
    return halo_meta


def lookback_time_gyr(z: np.ndarray) -> np.ndarray:
    """Approximate lookback time in Gyr."""

    z = np.asarray(z, dtype=float)
    try:
        from astropy.cosmology import Planck18  # type: ignore

        return Planck18.lookback_time(z).value
    except Exception:
        return 13.8 * (1.0 - 1.0 / np.sqrt(1.0 + np.clip(z, 0.0, None)))


def _integer_values(values: object, name: str) -> np.ndarray:
    """Parse identifiers as exact signed int64 values without a float round trip."""

    raw = np.asarray(values, dtype=object).reshape(-1)
    parsed = [parse_exact_int64(value, name=f"{name}[{index}]") for index, value in enumerate(raw)]
    return np.asarray(parsed, dtype=np.int64)


def _validate_identifier_set(actual: object, expected: set[int], name: str) -> None:
    """Require an output identifier set to match the formed catalogue."""

    actual_set = set(_integer_values(actual, name).tolist())
    if actual_set != expected:
        missing = sorted(expected - actual_set)
        extra = sorted(actual_set - expected)
        raise ValueError(f"{name} identifier set differs from formed allcat; missing={missing[:8]}, extra={extra[:8]}.")


def load_modern_output(out_dir: Path, final_redshift: float | None = None) -> GaoOutput:
    """Load and validate one flat modern output directory."""

    paths = output_paths(Path(out_dir))
    metadata = load_run_metadata(paths.out_dir)
    if not isinstance(metadata, dict):
        raise ValueError(f"{paths.run_metadata} must contain a JSON object.")
    if "N_S" not in metadata:
        raise ValueError(f"{RUN_METADATA_NAME} is missing required N_S: {paths.run_metadata}")

    try:
        ns_value = float(metadata["N_S"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{RUN_METADATA_NAME} contains a non-numeric N_S.") from exc
    if not np.isfinite(ns_value) or ns_value <= 0.0:
        raise ValueError(f"{RUN_METADATA_NAME} N_S must be finite and positive; got {metadata['N_S']!r}.")

    metadata_final_redshift = metadata.get("final_redshift", 0.0)
    try:
        metadata_final_redshift = float(metadata_final_redshift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{RUN_METADATA_NAME} final_redshift must be numeric.") from exc
    if not np.isfinite(metadata_final_redshift) or metadata_final_redshift < 0.0:
        raise ValueError(
            f"{RUN_METADATA_NAME} final_redshift must be finite and non-negative; "
            f"got {metadata.get('final_redshift')!r}."
        )
    if final_redshift is None:
        final_redshift = metadata_final_redshift
    try:
        final_redshift = float(final_redshift)
    except (TypeError, ValueError) as exc:
        raise ValueError("final redshift override must be numeric.") from exc
    if not np.isfinite(final_redshift) or final_redshift < 0.0:
        raise ValueError(f"final redshift override must be finite and non-negative; got {final_redshift!r}.")

    formed = load_allcat(paths.root_allcat)
    if formed.empty:
        raise ValueError(f"Formation catalogue is empty: {paths.root_allcat}")
    required_formed_columns = [
        "halo_id_z0",
        "log10_halo_mass_z0",
        "log10_halo_mass_form",
        "log10_stellar_mass_form",
        "log10_gc_mass_init",
        "redshift_form",
        "metallicity_feh",
        "is_mpb",
        "subhalo_id_form",
        "snapshot_form",
        "galaxy_radius_form_kpc",
        "gc_radius_form_pc",
        "gc_surface_density_msun_pc2",
        "imbh_mass_init_msun",
    ]
    missing_formed_columns = [column for column in required_formed_columns if column not in formed.columns]
    if missing_formed_columns:
        raise ValueError(f"{paths.root_allcat} is missing modern allcat columns: {missing_formed_columns}")
    for column in [
        "log10_halo_mass_z0",
        "log10_halo_mass_form",
        "log10_stellar_mass_form",
        "log10_gc_mass_init",
        "redshift_form",
        "metallicity_feh",
        "galaxy_radius_form_kpc",
        "gc_radius_form_pc",
        "gc_surface_density_msun_pc2",
        "imbh_mass_init_msun",
    ]:
        values = pd.to_numeric(formed[column], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{paths.root_allcat} contains non-finite {column} values.")
    if np.any(formed["redshift_form"].to_numpy(dtype=float) < 0.0):
        raise ValueError(f"{paths.root_allcat} contains negative redshift_form values.")
    for column in ["galaxy_radius_form_kpc", "gc_radius_form_pc", "gc_surface_density_msun_pc2"]:
        values = formed[column].to_numpy(dtype=float)
        if np.any(values <= 0.0):
            raise ValueError(f"{paths.root_allcat} contains non-positive {column} values.")
    if np.any(formed["imbh_mass_init_msun"].to_numpy(dtype=float) < 0.0):
        raise ValueError(f"{paths.root_allcat} contains negative imbh_mass_init_msun values.")
    formed_halo_ids = _integer_values(formed["halo_id_z0"], "allcat halo_id_z0")
    expected_halo_ids = set(np.unique(formed_halo_ids).tolist())

    final_gcs = load_final_gcs(paths.final_gcs, expected_halo_ids=formed_halo_ids)
    required_final_columns = [
        "radius_init_kpc",
        "radius_final_kpc",
        "status",
        "gc_mass_final_msun",
        "imbh_mass_init_msun",
        "imbh_mass_final_msun",
    ]
    missing_final_columns = [column for column in required_final_columns if column not in final_gcs.columns]
    if missing_final_columns:
        raise ValueError(f"{paths.final_gcs} is missing modern final-GC columns: {missing_final_columns}")
    final_halo_ids = _integer_values(final_gcs["halo_id_z0"], "finalGCs halo_id_z0")
    if not np.array_equal(final_halo_ids, formed_halo_ids):
        raise ValueError(f"{paths.final_gcs} does not align with the root allcat row order.")
    _validate_identifier_set(final_gcs["halo_id_z0"], expected_halo_ids, "finalGCs")

    halo_summary = load_halo_summary(paths.halo_summary)
    required_summary_columns = ["hid_z0", "M_IMBH_final_tot", "M_SMBH_final", "M_NSC", "n_sunk"]
    missing_summary_columns = [column for column in required_summary_columns if column not in halo_summary.columns]
    if missing_summary_columns:
        raise ValueError(f"{paths.halo_summary} is missing modern halo-summary columns: {missing_summary_columns}")
    _validate_identifier_set(halo_summary["hid_z0"], expected_halo_ids, "haloSummary")
    if halo_summary["hid_z0"].duplicated().any():
        raise ValueError(f"{paths.halo_summary} must contain one row per halo_id_z0.")
    for column in ["M_IMBH_final_tot", "M_SMBH_final", "M_NSC"]:
        values = pd.to_numeric(halo_summary[column], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{paths.halo_summary} contains invalid {column} values.")

    halo_summary_by_z = load_halo_summary_by_z(paths.halo_summary_by_z)
    _validate_identifier_set(halo_summary_by_z["halo_id_z0"], expected_halo_ids, "haloSummaryByZ")
    for column in ["nsc_mass_msun", "central_bh_mass_final_msun"]:
        values = pd.to_numeric(halo_summary_by_z[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if np.any(values[finite] < 0.0):
            raise ValueError(f"{paths.halo_summary_by_z} contains negative {column} values.")

    mpb = load_mpb(paths.mpb)
    mpb_halo_ids = _integer_values(mpb["subhalo_id_z0"], "MPB subhalo_id_z0")
    if not expected_halo_ids.issubset(set(mpb_halo_ids.tolist())):
        missing_mpb_ids = sorted(expected_halo_ids - set(mpb_halo_ids.tolist()))
        raise ValueError(f"{paths.mpb} is missing MPB histories for halo IDs: {missing_mpb_ids[:8]}")

    radius_init = pd.to_numeric(final_gcs["radius_init_kpc"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(radius_init)) or np.any(radius_init <= 0.0):
        raise ValueError(f"{paths.final_gcs} contains invalid radius_init_kpc values.")
    radius_final_raw = pd.to_numeric(final_gcs["radius_final_kpc"], errors="coerce").to_numpy(dtype=float)
    status = _integer_values(final_gcs["status"], "finalGCs status")
    valid_status = {STAT_ALIVE, STAT_DISRUPT, STAT_SUNK_GC, STAT_SUNK_BH, STAT_WANDER}
    invalid_status = sorted(set(status.tolist()).difference(valid_status))
    if invalid_status:
        raise ValueError(f"{paths.final_gcs} contains invalid status codes: {invalid_status}")
    radius_final = np.where(np.isfinite(radius_final_raw) & (radius_final_raw > 0.0), radius_final_raw, np.nan)
    survivor = status == STAT_ALIVE
    if np.any(survivor & ~np.isfinite(radius_final)):
        raise ValueError(f"{paths.final_gcs} contains invalid radius_final_kpc values for surviving GCs.")
    for column in ["imbh_mass_init_msun", "imbh_mass_final_msun"]:
        values = pd.to_numeric(final_gcs[column], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{paths.final_gcs} contains invalid {column} values.")

    deposit_profile = load_deposit_profile(paths.deposit)
    deposit_halo_ids = _integer_values(deposit_profile.halo_ids, "depos halo_id_z0")
    _validate_identifier_set(deposit_halo_ids, expected_halo_ids, "depos")
    if deposit_profile.cumulative_mass_msun is None:
        raise ValueError(f"{paths.deposit} did not provide cumulative deposited profiles.")
    for hid, radii, shell in zip(
        deposit_halo_ids,
        deposit_profile.r_outer_kpc,
        deposit_profile.shell_mass_msun,
    ):
        radii = np.asarray(radii, dtype=float)
        shell = np.asarray(shell, dtype=float)
        if len(radii) == 0 or len(radii) != len(shell):
            raise ValueError(f"{paths.deposit} has an invalid radial profile for halo_id_z0={int(hid)}.")
        if np.any(~np.isfinite(radii)) or np.any(~np.isfinite(shell)) or np.any(radii <= 0.0) or np.any(shell < 0.0):
            raise ValueError(f"{paths.deposit} has invalid radial or deposited-mass values for halo_id_z0={int(hid)}.")
        if np.any(np.diff(radii) <= 0.0):
            raise ValueError(f"{paths.deposit} has non-increasing r_outer_kpc for halo_id_z0={int(hid)}.")
    model = ModelResult(
        ns_value=ns_value,
        r_init=radius_init,
        r_final=radius_final,
        m_final=final_gcs["gc_mass_final_msun"].to_numpy(dtype=float),
        status=status,
        deposit_profile=deposit_profile,
        halo_summary=halo_summary,
    )
    return GaoOutput(
        formed=formed,
        final_gcs=final_gcs,
        halo_summary=halo_summary,
        halo_summary_by_z=halo_summary_by_z,
        mpb=mpb,
        deposit_profile=deposit_profile,
        metadata=metadata,
        paths=paths,
        final_redshift=final_redshift,
        model=model,
    )

def _surface_density_mean_by_halo(
    halo_ids: np.ndarray,
    radii: np.ndarray,
    bins: np.ndarray,
    *,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean surface density profile over halos, including empty-bin zeros."""

    halo_ids = np.asarray(halo_ids)
    radii = np.asarray(radii, dtype=float)
    if extra_mask is None:
        valid = np.isfinite(radii) & (radii > 0)
    else:
        valid = np.asarray(extra_mask, dtype=bool) & np.isfinite(radii) & (radii > 0)

    centers = np.sqrt(bins[:-1] * bins[1:])
    unique_halos = np.unique(halo_ids)
    if len(unique_halos) == 0:
        return centers, np.full(len(centers), np.nan, dtype=float)

    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    prof = np.zeros((len(unique_halos), len(centers)), dtype=float)
    for ii, hid in enumerate(unique_halos):
        counts, _ = np.histogram(radii[(halo_ids == hid) & valid], bins=bins)
        prof[ii] = counts / np.clip(area, 1e-20, None)

    density = np.mean(prof, axis=0)
    density = np.where(density > 0, density, np.nan)
    return centers, density


def _final_survivor_mask(model: ModelResult, extra_mask: np.ndarray | None = None) -> np.ndarray:
    """Return mask selecting only surviving GCs for final-state profiles."""

    mask = np.asarray(model.status, dtype=int) == STAT_ALIVE
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)
    return mask


def _mass_histograms_by_halo(
    halo_ids: np.ndarray,
    masses: np.ndarray,
    bins: np.ndarray,
    *,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-halo mass histograms on a common binning."""

    halo_ids = np.asarray(halo_ids, dtype=int)
    masses = np.asarray(masses, dtype=float)
    if extra_mask is None:
        valid = np.isfinite(masses) & (masses > 0)
    else:
        valid = np.asarray(extra_mask, dtype=bool) & np.isfinite(masses) & (masses > 0)

    centers = np.sqrt(bins[:-1] * bins[1:])
    unique_halos = np.unique(halo_ids)
    out = np.zeros((len(unique_halos), len(centers)), dtype=float)
    for ii, hid in enumerate(unique_halos):
        counts, _ = np.histogram(masses[(halo_ids == hid) & valid], bins=bins)
        out[ii] = counts
    return centers, out


def _cumulative_profile(radii: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Cumulative sum as a function of radius."""

    if len(radii) == 0 or len(values) == 0:
        return np.zeros_like(grid, dtype=float)

    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(radii) & np.isfinite(values) & (radii > 0) & (values > 0)
    if not np.any(valid):
        return np.zeros_like(grid, dtype=float)

    order = np.argsort(radii[valid])
    r = radii[valid][order]
    v = values[valid][order]
    c = np.cumsum(v)
    return np.interp(grid, r, c, left=c[0] if len(c) else 0.0, right=c[-1] if len(c) else 0.0)


def _cumulative_mean_by_halo(
    halo_ids: np.ndarray,
    radii: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    *,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Average cumulative radial profile over halos."""

    halo_ids = np.asarray(halo_ids, dtype=int)
    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=float)
    if extra_mask is None:
        base_mask = np.ones(len(halo_ids), dtype=bool)
    else:
        base_mask = np.asarray(extra_mask, dtype=bool)

    unique_halos = np.unique(halo_ids[base_mask])
    if len(unique_halos) == 0:
        return np.full(len(grid), np.nan, dtype=float)

    prof = np.zeros((len(unique_halos), len(grid)), dtype=float)
    for ii, hid in enumerate(unique_halos):
        hmask = base_mask & (halo_ids == hid)
        prof[ii] = _cumulative_profile(radii[hmask], values[hmask], grid)

    return np.mean(prof, axis=0)


def _deposit_mean_profile(
    profile: DepositProfile,
    *,
    grid_kpc: np.ndarray | None = None,
    halo_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average deposited cumulative profile for a halo subset.

    Per-halo deposit files carry halo-specific radial bins, so each halo is
    interpolated onto a shared grid before averaging.
    """

    if halo_ids is None:
        use = np.ones(len(profile.halo_ids), dtype=bool)
    else:
        use = np.isin(profile.halo_ids, np.asarray(halo_ids, dtype=int))
    idx = np.where(use)[0]
    if len(idx) == 0:
        grid = np.asarray(grid_kpc if grid_kpc is not None else np.array([1.0e-3]), dtype=float)
        return grid, np.full(len(grid), np.nan, dtype=float)

    if grid_kpc is None:
        r_min = min(max(float(profile.r_outer_kpc[ii][0]), 1.0e-6) for ii in idx)
        r_max = max(float(profile.r_outer_kpc[ii][-1]) for ii in idx)
        grid = np.logspace(np.log10(r_min), np.log10(r_max), 256)
    else:
        grid = np.asarray(grid_kpc, dtype=float)

    prof = np.zeros((len(idx), len(grid)), dtype=float)
    for jj, ii in enumerate(idx):
        radii = np.asarray(profile.r_outer_kpc[ii], dtype=float)
        cum = np.asarray(profile.cumulative_mass_msun[ii], dtype=float)
        # Deposit tables are already cumulative in radius, so only a 1D radial
        # interpolation is needed before taking the halo-average profile.
        prof[jj] = np.interp(grid, radii, cum, left=0.0, right=cum[-1])
    return grid, np.mean(prof, axis=0)


def _deposit_mass_within_radius(
    profile: DepositProfile,
    radius_kpc: float,
    *,
    halo_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deposited cumulative mass evaluated at `radius_kpc` for selected halos."""

    if halo_ids is None:
        use = np.ones(len(profile.halo_ids), dtype=bool)
    else:
        use = np.isin(profile.halo_ids, np.asarray(halo_ids, dtype=int))
    halo_use = profile.halo_ids[use]
    if len(halo_use) == 0:
        return halo_use, np.array([], dtype=float)

    vals = np.zeros(len(halo_use), dtype=float)
    use_idx = np.where(use)[0]
    for jj, ii in enumerate(use_idx):
        radii = np.asarray(profile.r_outer_kpc[ii], dtype=float)
        cum = np.asarray(profile.cumulative_mass_msun[ii], dtype=float)
        vals[jj] = float(np.interp(radius_kpc, radii, cum, left=0.0, right=cum[-1]))
    return halo_use, vals


def _select_halos_by_logmh(
    gc: pd.DataFrame,
    logmh_min: float,
    logmh_max: float,
    *,
    fallback_n: int = 3,
) -> np.ndarray:
    """Select halo IDs in a mass window, with nearest-mass fallback.

    Small demo subsets may not contain halos in the exact Gao+2024 mass
    windows; when that happens, use the nearest available halo masses so
    downstream figures remain reproducible.
    """

    halo_mass = gc.groupby("hid_z0", sort=True)["logMh_z0"].first()
    mask = (halo_mass >= logmh_min) & (halo_mass < logmh_max)
    selected = halo_mass.index[mask].to_numpy(dtype=int)
    if len(selected) > 0:
        return selected

    center = 0.5 * (logmh_min + logmh_max)
    nearest = (halo_mass - center).abs().sort_values().index.to_numpy(dtype=int)
    n_keep = min(max(1, int(fallback_n)), len(nearest))
    return nearest[:n_keep]


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient for finite inputs."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2 or np.ptp(x[mask]) == 0.0 or np.ptp(y[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _fit_band(
    x: np.ndarray,
    y: np.ndarray,
    *,
    logx: bool = False,
    logy: bool = False,
    n_grid: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit line with a symmetric 1-sigma band in transformed space."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    if np.sum(mask) < 2:
        xx = np.full(n_grid, np.nan, dtype=float)
        return xx, np.full_like(xx, np.nan), np.full_like(xx, np.nan), np.full_like(xx, np.nan)

    xt = np.log10(x[mask]) if logx else x[mask]
    yt = np.log10(y[mask]) if logy else y[mask]
    if np.ptp(xt) == 0.0:
        xx = np.full(n_grid, np.nan, dtype=float)
        return xx, np.full_like(xx, np.nan), np.full_like(xx, np.nan), np.full_like(xx, np.nan)

    aa, bb = np.polyfit(xt, yt, 1)
    xx_t = np.linspace(np.nanmin(xt), np.nanmax(xt), n_grid)
    yy_t = aa * xx_t + bb
    resid = yt - (aa * xt + bb)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0

    xx = np.power(10.0, xx_t) if logx else xx_t
    yy = np.power(10.0, yy_t) if logy else yy_t
    lo = np.power(10.0, yy_t - sigma) if logy else yy_t - sigma
    hi = np.power(10.0, yy_t + sigma) if logy else yy_t + sigma
    return xx, yy, lo, hi


def _mean_and_std(values: np.ndarray) -> Tuple[float, float]:
    """Mean and sample scatter for finite inputs."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _halo_level_table(
    gc: pd.DataFrame,
    halo_meta: pd.DataFrame,
    model: ModelResult,
    *,
    nsc_radius_kpc: float | None = None,
) -> pd.DataFrame:
    """Build one halo-level summary table used by correlation plots."""

    temp = gc[["hid_z0", "logMh_z0"]].copy()
    temp["M_form"] = gc["M_form"].to_numpy()
    temp["M_final"] = model.m_final
    temp["r_final"] = model.r_final

    rows = []
    for hid, g in temp.groupby("hid_z0", sort=True):
        rows.append(
            {
                "hid_z0": int(hid),
                "M_halo": float(np.power(10.0, g["logMh_z0"].iloc[0])),
                "M_gc_init": float(g["M_form"].sum()),
                "M_gc_final": float(g["M_final"].sum()),
            }
        )

    out = pd.DataFrame(rows).set_index("hid_z0").sort_index()
    if model.halo_summary is not None and len(model.halo_summary) > 0:
        hs = model.halo_summary[
            ["hid_z0", "M_IMBH_final_tot", "M_SMBH_final", "M_NSC", "n_sunk"]
        ].copy()
        hs["hid_z0"] = hs["hid_z0"].astype(int)
        hs = hs.set_index("hid_z0").sort_index()
        out = out.join(
            hs.rename(
                columns={
                    "M_IMBH_final_tot": "M_bh_total",
                    "M_SMBH_final": "M_smbh",
                    "M_NSC": "M_nsc",
                }
            ),
            how="left",
        )
    else:
        out["M_nsc"] = 0.0
        out["M_bh_total"] = 0.0
        out["M_smbh"] = 0.0
        out["n_sunk"] = 0
    out["M_nsc"] = pd.to_numeric(out["M_nsc"], errors="coerce").fillna(0.0)
    out["M_bh_total"] = pd.to_numeric(out["M_bh_total"], errors="coerce").fillna(0.0)
    out["M_smbh"] = pd.to_numeric(out["M_smbh"], errors="coerce").fillna(0.0)
    out["n_sunk"] = pd.to_numeric(out["n_sunk"], errors="coerce").fillna(0).astype(int)
    out = out.join(halo_meta[["z_hm"]], how="left")
    out["logM_halo"] = _safe_log10(out["M_halo"].to_numpy())
    out["logM_gc_init"] = _safe_log10(out["M_gc_init"].to_numpy())
    out["logM_gc_final"] = _safe_log10(out["M_gc_final"].to_numpy())
    out["logM_nsc"] = _safe_log10(out["M_nsc"].to_numpy())
    out["logM_bh_total"] = _safe_log10(out["M_bh_total"].to_numpy())
    out["logM_smbh"] = _safe_log10(out["M_smbh"].to_numpy())
    return out


def build_reproduction(
    gao_output: GaoOutput,
    output_dir: Path,
    *,
    include_observables: bool = True,
) -> List[Path]:
    """Write the maintained ten-figure Gao suite for one modern model run."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style(font="serif", tex=True, grid=False)

    gc = gao_output.formed.copy()
    model = gao_output.model
    halo_ids = gc["hid_z0"].to_numpy(dtype=int)
    expected_halo_ids = np.unique(halo_ids)
    halo_meta = estimate_zhm(
        gc=gc,
        mpb=gao_output.mpb,
        final_redshift=gao_output.final_redshift,
    )
    missing_halo_meta = sorted(set(expected_halo_ids.tolist()).difference(halo_meta.index.tolist()))
    if missing_halo_meta:
        raise ValueError(f"MPB assembly histories are missing halo IDs: {missing_halo_meta[:8]}")
    halo_meta = halo_meta.loc[expected_halo_ids].copy()
    z_hm_values = halo_meta["z_hm"].to_numpy(dtype=float)
    if np.any(~np.isfinite(z_hm_values)) or np.any(z_hm_values < 0.0):
        raise ValueError("Estimated halo assembly redshifts contain invalid values.")
    gc = gc.join(halo_meta[["z_hm"]], on="hid_z0")

    mw_obs, m31_obs = _get_mw_m31_observations()
    obs_overlay = _gao_observational_overlays() if include_observables else {}
    if include_observables and gao_output.final_redshift > 1.0e-12:
        print(
            "WARNING observational overlays are z=0 references while this run "
            f"ends at final_redshift={gao_output.final_redshift:g}"
        )

    q1, q2 = np.quantile(z_hm_values, [1 / 3, 2 / 3])
    halo_meta["zhm_bin"] = np.where(
        halo_meta["z_hm"] <= q1,
        "low",
        np.where(halo_meta["z_hm"] <= q2, "mid", "high"),
    )
    zhm_style = {
        "low": ("tab:blue", r"$z_{\rm h}<%.1f$" % q1),
        "mid": ("tab:green", r"$z_{\rm h}\in[%.1f,%.1f]$" % (q1, q2)),
        "high": ("tab:red", r"$z_{\rm h}>%.1f$" % q2),
    }
    halo_table_all = _halo_level_table(
        gc=gc,
        halo_meta=halo_meta,
        model=model,
        nsc_radius_kpc=float(mw_obs.r_nsc_pc) / 1000.0,
    )
    model_label = f"$N_{{\\rm S}}={model.ns_value:g}$"
    final_mask = _final_survivor_mask(model)
    r_bins = np.logspace(-2.0, 2.0, 24)
    written_paths: List[Path] = []

    def write_figure(fig_num: int, stem: str, fig: plt.Figure) -> Path:
        path = output_dir / f"Fig.{fig_num:02d}_{stem}.pdf"
        save_pdf(fig, path, STD_DPI)
        written_paths.append(path)
        return path

    # Figure 2: final M_GC/M_halo ratio with the existing percentile range.
    halo_ratio = (
        halo_table_all["M_gc_final"].to_numpy(dtype=float)
        / halo_table_all["M_halo"].to_numpy(dtype=float)
    )
    finite_ratio = halo_ratio[np.isfinite(halo_ratio)]
    if len(finite_ratio) == 0:
        raise ValueError("Cannot build Fig. 02 because no finite GC-to-halo ratios are available.")
    ratio_median = float(np.median(finite_ratio))
    ratio_q25, ratio_q75 = np.quantile(finite_ratio, [0.25, 0.75])
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(4.8, 3.6))
    ax.errorbar(
        [model.ns_value],
        [ratio_median / 1.0e-5],
        yerr=[[
            max((ratio_median - float(ratio_q25)) / 1.0e-5, 0.0),
        ], [
            max((float(ratio_q75) - ratio_median) / 1.0e-5, 0.0),
        ]],
        marker="D",
        color="black",
        mfc="black",
        mec="black",
        capsize=4,
        lw=1.0,
        label=f"High-z-SMBHs, {model_label}",
        zorder=3,
    )
    if include_observables:
        ref_colors = {"S09": "blue", "G10": "red", "H14": "c", "H17": "green"}
        for label, ratio in obs_overlay["fig2_ratio_refs"].items():
            ax.axhline(
                ratio / 1.0e-5,
                lw=1.0,
                ls="--",
                alpha=0.9,
                color=ref_colors.get(label, "gray"),
                label=label,
            )
    x_max = max(4.5, 1.15 * model.ns_value + 0.1)
    finish_axis(
        ax,
        xlabel=r"$N_{\rm S}$",
        ylabel=r"$M_{\rm GC} \, / \, M_{\rm halo} \times 10^{-5}$",
        xlim=(0.0, x_max),
        ylim=(1.0, 9.0),
        legend=True,
        legend_kwargs={"loc": "upper left", "ncol": 2},
    )
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([1, 3, 5, 7, 9])
    write_figure(2, "mgc_mhalo_ratio", fig)

    # Figure 3: global initial/final radial number-density profiles.
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(4.8, 3.2))
    centers_init, density_init = _surface_density_mean_by_halo(halo_ids, model.r_init, r_bins)
    centers_final, density_final = _surface_density_mean_by_halo(
        halo_ids,
        model.r_final,
        r_bins,
        extra_mask=final_mask,
    )
    ax.plot(centers_init, density_init, "--", lw=1.1, color="black", label=f"{model_label} init")
    ax.plot(centers_final, density_final, "-", lw=1.3, color="black", label=f"{model_label} final")
    if include_observables:
        f3_g14 = obs_overlay["fig3_G14"]
        f3_b21 = obs_overlay["fig3_B21"]
        f3_rbc = obs_overlay["fig3_RBCver5"]
        ax.plot(f3_g14["r_kpc"], f3_g14["Sigma"], color="red", ls=":", label="G14")
        ax.errorbar(
            f3_b21["r_kpc"],
            f3_b21["Sigma"],
            xerr=f3_b21["xerr"],
            yerr=f3_b21["yerr"],
            fmt="o",
            color="#d12ad1",
            ms=1.5,
            capsize=1.5,
            lw=0.5,
            label="B21",
        )
        ax.errorbar(
            f3_rbc["r_kpc"],
            f3_rbc["Sigma"],
            xerr=f3_rbc["xerr"],
            yerr=f3_rbc["yerr"],
            fmt="o",
            color="black",
            ms=1.5,
            capsize=1.5,
            lw=0.5,
            label="RBCver.5",
        )
    finish_log_axis(
        ax,
        xlabel=r"$r~[\mathrm{kpc}]$",
        ylabel=r"$\Sigma~[\mathrm{kpc}^{-2}]$",
        xlim=(0.01, 200.0),
        ylim=(1.0e-4, 2.0e4),
        legend=True,
        legend_kwargs={"loc": "upper right", "ncol": 1},
    )
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticks([1.0e-4, 1.0e-2, 1.0, 100.0, 10000.0])
    write_figure(3, "surface_number_density", fig)

    # Figure 6: z_hm histogram.
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(6.0, 4.2))
    ax.hist(z_hm_values, bins=22, alpha=0.85)
    finish_axis(
        ax,
        xlabel=r"$z_{\rm h}$",
        ylabel=r"\# halos",
        xlim=(0.0, 3.75),
    )
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    write_figure(6, "z_h_hist", fig)

    # Figure 7: radial profiles split into the existing z_hm terciles.
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(4.8, 3.6))
    for label_key in ["low", "mid", "high"]:
        hid_sel = halo_meta.index[halo_meta["zhm_bin"] == label_key].to_numpy(dtype=int)
        mask = np.isin(halo_ids, hid_sel)
        centers_init, density_init = _surface_density_mean_by_halo(
            halo_ids[mask],
            model.r_init[mask],
            r_bins,
        )
        centers_final, density_final = _surface_density_mean_by_halo(
            halo_ids[mask],
            model.r_final[mask],
            r_bins,
            extra_mask=final_mask[mask],
        )
        color, label = zhm_style[label_key]
        ax.plot(centers_init, density_init, "--", lw=1.5, color=color, label=label)
        ax.plot(centers_final, density_final, "-", lw=1.8, color=color)
    if include_observables:
        f3_b21 = obs_overlay["fig3_B21"]
        f3_rbc = obs_overlay["fig3_RBCver5"]
        ax.errorbar(
            f3_b21["r_kpc"],
            f3_b21["Sigma"],
            xerr=f3_b21["xerr"],
            yerr=f3_b21["yerr"],
            fmt="o",
            color="#d12ad1",
            ms=3.5,
            capsize=3.0,
            lw=1.0,
            label="B21",
        )
        ax.errorbar(
            f3_rbc["r_kpc"],
            f3_rbc["Sigma"],
            xerr=f3_rbc["xerr"],
            yerr=f3_rbc["yerr"],
            fmt="o",
            color="black",
            ms=3.5,
            capsize=3.0,
            lw=1.0,
            label="RBCver.5",
        )
    ax.plot([], [], "--", color="0.25", lw=1.2, label="init")
    ax.plot([], [], "-", color="0.25", lw=1.5, label="final")
    finish_log_axis(
        ax,
        xlabel=r"$r~[\mathrm{kpc}]$",
        ylabel=r"$\Sigma~[\mathrm{kpc}^{-2}]$",
        xlim=(0.1, 200.0),
        ylim=(1.0e-4, 2.0e2),
        legend=True,
        legend_kwargs={"loc": "best", "ncol": 2},
    )
    ax.set_yticks([1.0e-4, 1.0e-2, 1.0, 100.0])
    write_figure(7, "number_density_by_z_h", fig)

    # Figure 8: MW-like in-situ final GC mass function.
    insitu = gc["isMPB"].to_numpy().astype(bool)
    mw_hid_nominal = _select_halos_by_logmh(gc, 11.7, 11.9, fallback_n=3)
    mw_hid = mw_hid_nominal if len(mw_hid_nominal) >= 8 else halo_meta.index.to_numpy(dtype=int)
    mw_mask = np.isin(halo_ids, mw_hid) & insitu
    m_bins = np.logspace(3.8, 7.1, 12)
    centers, hist = _mass_histograms_by_halo(
        halo_ids[mw_mask],
        model.m_final[mw_mask],
        m_bins,
        extra_mask=final_mask[mw_mask],
    )
    if hist.shape[0] == 0:
        raise ValueError("Cannot build Fig. 08 because the selected MW-like sample has no in-situ GCs.")
    med = np.median(hist, axis=0)
    q25, q75 = np.quantile(hist, [0.25, 0.75], axis=0)
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(4.5, 3.0))
    ax.plot(centers, med, lw=1.0, label=model_label)
    ax.fill_between(
        centers,
        q25,
        q75,
        color="lightsteelblue",
        alpha=0.55,
        lw=0.0,
        label=f"{model_label} 25\%-75\% quantile",
    )
    if include_observables:
        f8 = obs_overlay["fig8_mass_obs"]
        ax.plot(f8["mass_msun"], f8["count"], color="black", lw=1.0, label="B21")
    finish_axis(
        ax,
        xlabel=r"$M_{\rm GC}~[M_\odot]$",
        ylabel=r"\# GCs per halo",
        xscale="log",
        ylim=(-0.5, 48.0),
        legend=True,
        legend_kwargs={"loc": "upper right", "ncol": 1},
    )
    write_figure(8, "mass_function_MW_insitu_GCs", fig)

    # Figure 10: cumulative initial and deposited stellar-mass profiles.
    r_grid_pc = np.logspace(0.0, 4.0, 120)
    c_init = _cumulative_mean_by_halo(
        halo_ids,
        1000.0 * model.r_init,
        gc["M_form"].to_numpy(dtype=float),
        r_grid_pc,
    )
    r_dep_kpc, c_dep = _deposit_mean_profile(
        gao_output.deposit_profile,
        grid_kpc=r_grid_pc / 1000.0,
    )
    r_dep_pc = 1000.0 * r_dep_kpc
    mean_smbh, std_smbh = _mean_and_std(halo_table_all["M_smbh"].to_numpy(dtype=float))
    mean_bh_total, std_bh_total = _mean_and_std(halo_table_all["M_bh_total"].to_numpy(dtype=float))
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(5.1, 3.6))
    ax.plot(r_grid_pc, c_init, "--", color="black", lw=1.0, label=f"{model_label} init")
    ax.plot(r_dep_pc, c_dep, "-", color="black", lw=1.2, label=f"{model_label} depo")
    if np.isfinite(mean_smbh) and mean_smbh > 0.0:
        ax.errorbar(
            [2.5],
            [mean_smbh],
            yerr=std_smbh if std_smbh > 0.0 else None,
            fmt="^",
            ms=3.5,
            mfc="white",
            mec="tab:blue",
            color="tab:blue",
            capsize=3.0,
            zorder=8,
            label="sunk BHs",
        )
    if np.isfinite(mean_bh_total) and mean_bh_total > 0.0:
        ax.errorbar(
            [2.5],
            [mean_bh_total],
            yerr=std_bh_total if std_bh_total > 0.0 else None,
            fmt="D",
            ms=3.5,
            mfc="white",
            mec="tab:orange",
            color="tab:orange",
            capsize=3.0,
            zorder=8,
            label="total BHs",
        )
    if include_observables:
        _add_nsc_smbh_points_pc(ax, mw_obs, show_labels=True)
        _add_nsc_smbh_points_pc(ax, m31_obs, show_labels=True)
    bh_means = [value for value in [mean_smbh, mean_bh_total] if np.isfinite(value) and value > 0.0]
    ylo10 = 1.0e6
    if bh_means:
        ylo10 = min(ylo10, 10.0 ** np.floor(np.log10(max(min(bh_means), 1.0e-6))))
    finish_log_axis(
        ax,
        xlabel=r"$r~[\mathrm{pc}]$",
        ylabel=r"$M_{\rm encl}~[M_\odot]$",
        xlim=(2.0, 1.0e4),
        ylim=(ylo10, 3.0e8),
        legend=True,
        legend_kwargs={"loc": "lower right", "ncol": 2},
    )
    write_figure(10, "cum_mass", fig)

    # Figure 11: deposited cumulative profiles split by z_hm tercile.
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(6.6, 4.8))
    mass_form_all = gc["M_form"].to_numpy(dtype=float)
    for hid in np.unique(halo_ids):
        hmask = halo_ids == hid
        c_init_h = _cumulative_profile(1000.0 * model.r_init[hmask], mass_form_all[hmask], r_grid_pc)
        ax.plot(r_grid_pc, c_init_h, "--", lw=0.45, color="0.78", alpha=0.10, zorder=1)
    for hid, radii, cum in zip(
        gao_output.deposit_profile.halo_ids,
        gao_output.deposit_profile.r_outer_kpc,
        gao_output.deposit_profile.cumulative_mass_msun,
    ):
        radii = np.asarray(radii, dtype=float)
        cum = np.asarray(cum, dtype=float)
        c_dep_h = np.interp(r_grid_pc / 1000.0, radii, cum, left=0.0, right=cum[-1])
        ax.plot(r_grid_pc, c_dep_h, "-", lw=0.45, color="0.62", alpha=0.10, zorder=1)
    for label_key in ["low", "mid", "high"]:
        hid_sel = halo_meta.index[halo_meta["zhm_bin"] == label_key].to_numpy(dtype=int)
        sel = np.isin(halo_ids, hid_sel)
        c_init = _cumulative_mean_by_halo(
            halo_ids,
            1000.0 * model.r_init,
            mass_form_all,
            r_grid_pc,
            extra_mask=sel,
        )
        r_dep_kpc, c_dep = _deposit_mean_profile(
            gao_output.deposit_profile,
            grid_kpc=r_grid_pc / 1000.0,
            halo_ids=hid_sel,
        )
        color, label = zhm_style[label_key]
        ax.plot(r_grid_pc, c_init, "--", lw=1.8, color=color, label=label, zorder=3)
        ax.plot(1000.0 * r_dep_kpc, c_dep, "-", lw=2.0, color=color, zorder=4)
    if include_observables:
        _add_nsc_smbh_points_pc(ax, mw_obs, show_labels=True)
        _add_nsc_smbh_points_pc(ax, m31_obs, show_labels=True)
    ax.plot([], [], "--", color="0.45", lw=1.2, label="init")
    ax.plot([], [], "-", color="0.45", lw=1.6, label="depo")
    ax.plot([], [], color="none", label=model_label)
    finish_log_axis(
        ax,
        xlabel=r"$r~[\mathrm{pc}]$",
        ylabel=r"$M_{\rm encl}~[M_\odot]$",
        xlim=(2.0, 1.0e4),
        ylim=(4.0e4, 2.0e9),
        legend=True,
        legend_kwargs={"loc": "lower right", "ncol": 2},
    )
    write_figure(11, "cum_mass_by_zhm", fig)

    # Figures 16--18 use the existing halo-level MW-like sample.
    halo_table = halo_table_all.loc[halo_table_all.index.intersection(mw_hid)].dropna(subset=["z_hm"])
    fig, axs = plt.subplots(4, 1, figsize=(5.4, 10.2), dpi=STD_DPI, sharex=True, constrained_layout=True)
    x = halo_table["z_hm"].to_numpy(dtype=float)
    panel_data = [
        (axs[0], halo_table["M_halo"].to_numpy(dtype=float), r"$m_{halo}\,(M_{\odot})$"),
        (axs[1], halo_table["M_gc_init"].to_numpy(dtype=float), r"$m_{GCi}\,(M_{\odot})$"),
        (axs[2], halo_table["M_nsc"].to_numpy(dtype=float), r"$m_{NSC}\,(M_{\odot})$"),
        (axs[3], halo_table["M_gc_final"].to_numpy(dtype=float), r"$m_{GCf}\,(M_{\odot})$"),
    ]
    for ax_panel, y_values, ylabel in panel_data:
        ax_panel.scatter(x, y_values, s=7, alpha=0.75, color="black", linewidths=0)
        fx, fy, flo, fhi = _fit_band(x, y_values)
        if np.any(np.isfinite(fx)):
            ax_panel.plot(fx, fy, color="gray", lw=1.3)
            ax_panel.fill_between(fx, flo, fhi, color="lightgray", alpha=0.75, lw=0.0)
        ax_panel.text(0.05, 0.82, f"r={_pearson_r(x, y_values):.2f}", transform=ax_panel.transAxes)
        finish_axis(ax_panel, ylabel=ylabel)
        ax_panel.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    x_max = float(np.max(x)) if len(x) else 0.0
    finish_axis(axs[-1], xlabel=r"$z_{hm}$", xlim=(0.0, max(3.75, x_max)))
    write_figure(16, "corr_zhm_panels", fig)

    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(6.0, 4.5))
    x_all = halo_table["M_gc_init"].to_numpy(dtype=float)
    y_all = np.clip(halo_table["M_gc_final"].to_numpy(dtype=float), 1.0, None)
    zhm_all = halo_table["z_hm"].to_numpy(dtype=float)
    scatter = ax.scatter(x_all, y_all, c=zhm_all, s=12, alpha=0.9, cmap="jet", linewidths=0)
    fx, fy, flo, fhi = _fit_band(x_all, y_all, logx=True, logy=True)
    if np.any(np.isfinite(fx)):
        ax.fill_between(fx, flo, fhi, color="lightgray", alpha=0.75, lw=0.0)
        ax.plot(fx, fy, color="black", lw=1.3)
    ax.text(0.20, 0.73, f"r={_pearson_r(np.log10(x_all), np.log10(y_all)):.2f}", transform=ax.transAxes)
    colourbar = fig.colorbar(scatter, ax=ax)
    colourbar.set_label(r"$z_{hm}$")
    finish_log_axis(
        ax,
        xlabel=r"$m_{GCi}\,(M_{\odot})$",
        ylabel=r"$m_{GCf}\,(M_{\odot})$",
    )
    write_figure(17, "m_init_vs_m_final", fig)

    fig, axs = plt.subplots(3, 1, figsize=(5.0, 9.0), dpi=STD_DPI, sharey=True, constrained_layout=True)
    y_nsc = halo_table["M_nsc"].to_numpy(dtype=float)
    x1 = halo_table["M_halo"].to_numpy(dtype=float) / 1.0e12
    x2 = np.clip(halo_table["M_gc_init"].to_numpy(dtype=float), 1.0, None)
    x3 = np.clip(halo_table["M_gc_final"].to_numpy(dtype=float), 1.0, None)
    configs = [
        (axs[0], x1, r"$m_{halo}\,(10^{12}M_{\odot})$", False, _pearson_r(x1, _log10_positive_or_nan(y_nsc))),
        (axs[1], x2, r"$m_{GCi}\,(M_{\odot})$", True, _pearson_r(_log10_positive_or_nan(x2), _log10_positive_or_nan(y_nsc))),
        (axs[2], x3, r"$m_{GCf}\,(M_{\odot})$", True, _pearson_r(_log10_positive_or_nan(x3), _log10_positive_or_nan(y_nsc))),
    ]
    for ax_panel, xx, xlabel, logx, correlation in configs:
        scatter_mask = np.isfinite(xx) & np.isfinite(y_nsc) & (y_nsc > 0.0)
        if logx:
            scatter_mask &= xx > 0.0
        ax_panel.scatter(xx[scatter_mask], y_nsc[scatter_mask], s=8, alpha=0.75, color="black", linewidths=0)
        fx, fy, flo, fhi = _fit_band(xx, y_nsc, logx=logx, logy=True)
        if np.any(np.isfinite(fx)):
            ax_panel.plot(fx, fy, color="gray", lw=1.3)
            ax_panel.fill_between(fx, flo, fhi, color="lightgray", alpha=0.75, lw=0.0)
        finish_axis(
            ax_panel,
            xlabel=xlabel,
            ylabel=r"$m_{NSC}\,(M_{\odot})$",
            xscale="log" if logx else None,
            yscale="log",
        )
        ax_panel.text(0.82, 0.12, f"r={correlation:.2f}", transform=ax_panel.transAxes)
    write_figure(18, "corr_nsc_panels", fig)

    return written_paths


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Reproduce the maintained Gao+2024 figure subset.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help=(
            "Finished modern output directory containing one allcat_s-*.txt, finalGCs.dat, "
            "depos.dat, haloSummary.csv, haloSummaryByZ.csv, mpb_from_fixed_trees.csv, and run_metadata.json."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for the ten PDFs; defaults to <out_dir>/_plots_Gao+2024.",
    )
    parser.add_argument("--final-z", "--final-redshift", dest="final_z", type=float, default=None)
    parser.add_argument(
        "--no-observables",
        action="store_true",
        help="Disable observational overlays.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    gao_output = load_modern_output(out_dir, final_redshift=args.final_z)
    plot_dir = (args.plot_dir if args.plot_dir is not None else default_plot_dir(out_dir, "Gao+2024")).resolve()

    figure_paths = build_reproduction(
        gao_output=gao_output,
        output_dir=plot_dir,
        include_observables=not args.no_observables,
    )
    print(f"FIGURES_WRITTEN {len(figure_paths)}")
    print(f"OUTPUT_DIR {plot_dir}")


if __name__ == "__main__":
    main()
