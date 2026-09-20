#!/usr/bin/env python3
"""Generate a project-compatible UV table with a precompiled FSPS driver."""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ProjectRoot = Path(__file__).resolve().parents[1]
DefaultFspsHome = Path("/home/czkong/software/fsps")
DefaultFspsExecutable = DefaultFspsHome / "src" / "genfsps_mist_c3k_lr.exe"
DefaultOutput = ProjectRoot / "data" / "UV" / "fsps_mist_c3k_lr_kroupa_m1450_grid.csv"
MinimumAgeGyr = 1.0e-4
NativeAbOffset = 32.365766193400795


ImfCodes = {
    "salpeter": 0,
    "chabrier": 1,
    "kroupa": 2,
    "van-dokkum": 3,
    "dave": 4,
}


def BuildParser() -> argparse.ArgumentParser:
    Parser = argparse.ArgumentParser(
        description=(
            "Generate an FSPS UV calibration CSV with a precompiled native "
            "FSPS driver; python-fsps is not required."
        )
    )
    Parser.add_argument(
        "--fsps-home",
        dest="FspsHome",
        type=Path,
        default=DefaultFspsHome,
        help=f"FSPS data root used by the driver (default: {DefaultFspsHome})",
    )
    Parser.add_argument(
        "--fsps-exe",
        dest="FspsExecutable",
        type=Path,
        default=DefaultFspsExecutable,
        help=(
            "precompiled native FSPS driver executable "
            f"(default: {DefaultFspsExecutable})"
        ),
    )
    Parser.add_argument(
        "--output",
        dest="Output",
        type=Path,
        default=DefaultOutput,
        help=f"CSV destination (default: {DefaultOutput})",
    )
    Parser.add_argument(
        "--force",
        dest="Force",
        action="store_true",
        help="replace an existing output CSV",
    )
    Parser.add_argument(
        "--max-age-gyr",
        dest="MaxAgeGyr",
        type=float,
        default=None,
        help="maximum population age in Gyr",
    )
    Parser.add_argument(
        "--max-redshift",
        dest="MaxRedshift",
        type=float,
        default=7.0,
        help="redshift used for the default maximum age (default: 7)",
    )
    Parser.add_argument(
        "--n-age",
        dest="NAge",
        type=int,
        default=80,
        help="number of logarithmically spaced age grid points (default: 80)",
    )
    Parser.add_argument(
        "--n-feh",
        dest="NFeh",
        type=int,
        default=81,
        help=(
            "number of linearly spaced metallicity grid points between "
            "the compiled driver's native limits (default: 81)"
        ),
    )
    Parser.add_argument(
        "--wavelength-a",
        dest="WavelengthA",
        type=float,
        default=1450.0,
        help="rest-frame vacuum wavelength in Angstrom (default: 1450)",
    )
    Parser.add_argument(
        "--IMF",
        dest="Imf",
        choices=tuple(ImfCodes),
        default="kroupa",
        help="initial mass function (default: kroupa)",
    )
    return Parser


def RedshiftToCosmicAgeGyr(Redshift: float) -> float:
    PcM = 3.0856775814913673e16
    MpcKm = 1.0e6 * PcM / 1.0e3
    GyrSeconds = 1.0e9 * 365.25 * 24.0 * 3600.0
    H0 = 67.97
    OmegaM = 0.307
    OmegaLambda = 0.693
    LambdaTime = 2.0 / (3.0 * H0 * math.sqrt(OmegaLambda)) * MpcKm / GyrSeconds
    return LambdaTime * math.asinh(
        math.sqrt(OmegaLambda / OmegaM) / (1.0 + Redshift) ** 1.5
    )


def BuildGrid(Minimum: float, Maximum: float, Count: int) -> list[float]:
    if not (math.isfinite(Minimum) and math.isfinite(Maximum)):
        raise ValueError("grid limits must be finite")
    if Minimum >= Maximum:
        raise ValueError("grid minimum must be smaller than grid maximum")
    if Count < 2:
        raise ValueError("grid point count must be at least 2")
    FractionGrid = [
        Index / (Count - 1)
        for Index in range(Count)
    ]
    return [
        Minimum + Fraction * (Maximum - Minimum)
        for Fraction in FractionGrid
    ]


def BuildLogGrid(Minimum: float, Maximum: float, Count: int) -> list[float]:
    if Minimum <= 0.0 or Maximum <= 0.0:
        raise ValueError("logarithmic grid limits must be positive")
    if not (math.isfinite(Minimum) and math.isfinite(Maximum)):
        raise ValueError("grid limits must be finite")
    if Minimum >= Maximum:
        raise ValueError("grid minimum must be smaller than grid maximum")
    if Count < 2:
        raise ValueError("grid point count must be at least 2")
    LogMinimum = math.log10(Minimum)
    LogMaximum = math.log10(Maximum)
    FractionGrid = [
        Index / (Count - 1)
        for Index in range(Count)
    ]
    return [
        10.0 ** (LogMinimum + Fraction * (LogMaximum - LogMinimum))
        for Fraction in FractionGrid
    ]


def ValidateArguments(Args: argparse.Namespace, MaxAgeGyr: float) -> None:
    if Args.MaxRedshift <= -1.0:
        raise ValueError("--max-redshift must be greater than -1")
    if not math.isfinite(MaxAgeGyr) or MaxAgeGyr <= MinimumAgeGyr:
        raise ValueError("maximum age must be finite and larger than minimum age")
    if Args.WavelengthA <= 0.0 or not math.isfinite(Args.WavelengthA):
        raise ValueError("--wavelength-a must be finite and positive")
    if Args.NAge > 10000:
        raise ValueError("--n-age cannot exceed 10000")
    if Args.NFeh < 2:
        raise ValueError("--n-feh must be at least 2")


def ValidateFspsHome(FspsHome: Path) -> None:
    if not FspsHome.is_dir():
        raise FileNotFoundError(f"FSPS root does not exist: {FspsHome}")
    RequiredPaths = [
        FspsHome / "data",
        FspsHome / "ISOCHRONES",
        FspsHome / "SPECTRA",
    ]
    MissingPaths = [str(Item) for Item in RequiredPaths if not Item.exists()]
    if MissingPaths:
        raise FileNotFoundError("FSPS checkout is missing: " + ", ".join(MissingPaths))


def ValidateFspsExecutable(FspsExecutable: Path) -> None:
    if not FspsExecutable.is_file():
        raise FileNotFoundError(
            f"precompiled FSPS driver does not exist: {FspsExecutable}"
        )
    if not os.access(FspsExecutable, os.X_OK):
        raise PermissionError(
            f"precompiled FSPS driver is not executable: {FspsExecutable}"
        )


def BuildConfiguration(Args: argparse.Namespace, Ages: list[float]) -> str:
    Lines = [
        "&config",
        f"n_age = {len(Ages)}",
        f"target_wavelength = {Args.WavelengthA:.17g}",
        f"imf_code = {ImfCodes[Args.Imf]}",
        "/",
    ]
    Lines.extend(f"{Age:.17g}" for Age in Ages)
    return "\n".join(Lines) + "\n"

def RunNativeDriver(
    Executable: Path,
    FspsHome: Path,
    Configuration: str,
    NAge: int,
) -> tuple[list[float], dict[int, list[float]], tuple[float, float]]:
    Executable = Executable.resolve()
    FspsHome = FspsHome.resolve()
    Environment = os.environ.copy()
    Environment["SPS_HOME"] = str(FspsHome)
    Completed = subprocess.run(
        [str(Executable)],
        cwd=Executable.parent,
        input=Configuration,
        capture_output=True,
        text=True,
        env=Environment,
        check=False,
    )
    if Completed.returncode != 0:
        DriverOutput = (Completed.stdout + "\n" + Completed.stderr).strip()
        raise RuntimeError(
            "native FSPS driver failed with exit code "
            f"{Completed.returncode}:\n{DriverOutput[-6000:]}"
        )
    NativeFeh: dict[int, float] = {}
    NativeValues: dict[int, list[float]] = {}
    Bracket: tuple[float, float] | None = None
    for Line in Completed.stdout.splitlines():
        Tokens = Line.split()
        if not Tokens:
            continue
        if Tokens[0] == "GRID" and len(Tokens) == 5:
            Bracket = (float(Tokens[3]), float(Tokens[4]))
        elif Tokens[0] == "NATIVE" and len(Tokens) == 3:
            NativeIndex = int(Tokens[1])
            NativeFeh[NativeIndex] = float(Tokens[2])
            NativeValues[NativeIndex] = [math.nan] * NAge
        elif Tokens[0] == "VALUE" and len(Tokens) == 5:
            NativeIndex = int(Tokens[1])
            AgeIndex = int(Tokens[2]) - 1
            if NativeIndex not in NativeValues or AgeIndex not in range(NAge):
                raise RuntimeError("native FSPS driver returned an invalid grid index")
            NativeValues[NativeIndex][AgeIndex] = float(Tokens[4])
    if not NativeFeh or Bracket is None:
        raise RuntimeError(
            "native FSPS driver returned no recognisable grid metadata"
        )
    if any(
        len(Values) != NAge
        or any(not math.isfinite(Value) or Value <= 0.0 for Value in Values)
        for Values in NativeValues.values()
    ):
        raise RuntimeError("native FSPS driver returned incomplete or invalid spectra")
    NativeOrder = sorted(NativeFeh)
    if NativeOrder != list(range(1, len(NativeOrder) + 1)):
        raise RuntimeError("native FSPS metallicity indices are not contiguous")
    NativeFehValues = [NativeFeh[Index] for Index in NativeOrder]
    if any(
        High <= Low
        for Low, High in zip(NativeFehValues, NativeFehValues[1:])
    ):
        raise RuntimeError("native FSPS metallicity values are not increasing")
    return (
        NativeFehValues,
        {Index: NativeValues[Index] for Index in NativeOrder},
        Bracket,
    )


def InterpolateLnu(
    TargetFeh: float,
    NativeFeh: list[float],
    NativeValues: dict[int, list[float]],
    AgeIndex: int,
) -> float:
    Tolerance = 1.0e-10
    if TargetFeh < NativeFeh[0] - Tolerance or TargetFeh > NativeFeh[-1] + Tolerance:
        raise ValueError(
            f"target [Fe/H]={TargetFeh} is outside native range "
            f"[{NativeFeh[0]}, {NativeFeh[-1]}]"
        )
    High = bisect.bisect_right(NativeFeh, TargetFeh)
    if High == 0:
        return NativeValues[1][AgeIndex]
    if High == len(NativeFeh):
        return NativeValues[len(NativeFeh)][AgeIndex]
    Low = High - 1
    Fraction = (TargetFeh - NativeFeh[Low]) / (NativeFeh[High] - NativeFeh[Low])
    return (
        (1.0 - Fraction) * NativeValues[Low + 1][AgeIndex]
        + Fraction * NativeValues[High + 1][AgeIndex]
    )


def MagnitudeColumn(WavelengthA: float) -> str:
    if WavelengthA.is_integer():
        WavelengthLabel = str(int(WavelengthA))
    else:
        WavelengthLabel = f"{WavelengthA:g}"
    return f"M{WavelengthLabel}_AB_per_Msun"


def WriteTable(
    Output: Path,
    Ages: list[float],
    FehGrid: list[float],
    NativeFeh: list[float],
    NativeValues: dict[int, list[float]],
    Args: argparse.Namespace,
    MaxAgeGyr: float,
    Bracket: tuple[float, float],
) -> int:
    if Output.exists() and not Args.Force:
        raise FileExistsError(
            f"{Output} exists; pass --force only after checking the requested settings"
        )
    Output.parent.mkdir(parents=True, exist_ok=True)
    MagColumn = MagnitudeColumn(Args.WavelengthA)
    TemporaryName: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=Output.parent,
            prefix=Output.name + ".",
            suffix=".tmp",
            delete=False,
        ) as Handle:
            TemporaryName = Handle.name
            Handle.write("# FSPS UV calibration generated by native FSPS v4.0.\n")
            Handle.write(f"# FSPS source: {Args.FspsHome.resolve()}\n")
            Handle.write(
                f"# Native driver: {Args.FspsExecutable.resolve()}; "
                f"IMF: {Args.Imf}; SFH: ssp.\n"
            )
            Handle.write(
                "# Spectrum is rest-frame vacuum L_nu per initially formed "
                "solar mass; no filter integration is used.\n"
            )
            Handle.write(
                f"# Wavelength: {Args.WavelengthA:.10g} Angstrom; native "
                f"bracket: {Bracket[0]:.10g}, {Bracket[1]:.10g} Angstrom.\n"
            )
            Handle.write(
                f"# Age grid: {len(Ages)} logarithmically spaced points, "
                f"min={Ages[0]:.10e} Gyr, max={MaxAgeGyr:.16g} Gyr.\n"
            )
            Handle.write(
                f"# Metallicity grid: {len(FehGrid)} points, uniform in "
                f"log10(Z/Zsun), range={FehGrid[0]:.10g} to {FehGrid[-1]:.10g}.\n"
            )
            Handle.write(
                "# Metallicity interpolation: linear in L_nu between native "
                f"[Fe/H] points; native range={NativeFeh[0]:.10g} to "
                f"{NativeFeh[-1]:.10g}.\n"
            )
            Handle.write(
                f"# M_AB = -2.5*log10(L_nu[Lsun/Hz]) - "
                f"{NativeAbOffset:.15f}.\n"
            )
            Writer = csv.writer(Handle, lineterminator="\n")
            Writer.writerow(
                ["age_gyr", "log10_age_gyr", "feh", "z_ratio", MagColumn]
            )
            RowCount = 0
            for AgeIndex, Age in enumerate(Ages):
                for Feh in FehGrid:
                    Lnu = InterpolateLnu(
                        Feh,
                        NativeFeh,
                        NativeValues,
                        AgeIndex,
                    )
                    if not math.isfinite(Lnu) or Lnu <= 0.0:
                        raise RuntimeError("non-positive interpolated spectrum")
                    Writer.writerow(
                        [
                            f"{Age:.10e}",
                            f"{math.log10(Age):.10e}",
                            f"{Feh:.10e}",
                            f"{10.0 ** Feh:.10e}",
                            f"{-2.5 * math.log10(Lnu) - NativeAbOffset:.10e}",
                        ]
                    )
                    RowCount += 1
        os.replace(TemporaryName, Output)
        TemporaryName = None
    finally:
        if TemporaryName is not None:
            Path(TemporaryName).unlink(missing_ok=True)
    return RowCount


def Main() -> int:
    Parser = BuildParser()
    Args = Parser.parse_args()
    if Args.Output.exists() and not Args.Force:
        raise FileExistsError(
            f"{Args.Output} exists; pass --force only after checking the requested settings"
        )
    MaxAgeGyr = (
        RedshiftToCosmicAgeGyr(Args.MaxRedshift)
        if Args.MaxAgeGyr is None
        else Args.MaxAgeGyr
    )
    ValidateArguments(Args, MaxAgeGyr)
    Ages = BuildLogGrid(MinimumAgeGyr, MaxAgeGyr, Args.NAge)
    ValidateFspsHome(Args.FspsHome)
    ValidateFspsExecutable(Args.FspsExecutable)
    Configuration = BuildConfiguration(Args, Ages)
    NativeFeh, NativeValues, Bracket = RunNativeDriver(
        Args.FspsExecutable,
        Args.FspsHome,
        Configuration,
        len(Ages),
    )
    # The compiled model defines the valid metallicity limits at runtime.
    FehGrid = BuildGrid(min(NativeFeh), max(NativeFeh), Args.NFeh)
    RowCount = WriteTable(
        Args.Output,
        Ages,
        FehGrid,
        NativeFeh,
        NativeValues,
        Args,
        MaxAgeGyr,
        Bracket,
    )
    print(f"native FSPS grid: {len(Ages)} ages x {len(FehGrid)} metallicities")
    print(f"wrote {RowCount} data rows to {Args.Output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(Main())
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as Error:
        print(f"ERROR: {Error}", file=sys.stderr)
        raise SystemExit(2)
