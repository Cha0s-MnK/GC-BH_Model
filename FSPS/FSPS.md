# FSPS table generation

This directory contains the project-specific wrapper used to generate UV tables from FSPS. Compilation and table generation are deliberately separate: `Makefile.genfsps` builds a small native FSPS driver, and `genFSPS.py` runs an already compiled driver to write one CSV table.

The three supported compiled models are:

| Table | Isochrone switch | Spectral-library switch |
| --- | --- | --- |
| MIST + C3K_LR | MIST=1 | C3K_LR=1 |
| PARSEC + C3K_LR | PARSEC=1 | C3K_LR=1 |
| BPASS | BPASS=1 | not used by BPASS |

Each variant must have its own build directory. FSPS Fortran module files are created with the compile-time switches, so reusing one build directory for different variants can mix incompatible objects.

## 1. Download FSPS

Use the upstream FSPS repository:

https://github.com/cconroy20/fsps

For example:

```bash
cd /home/czkong/software
git clone https://github.com/cconroy20/fsps
```

If the checkout already exists, use its path as `FSPS_HOME` and do not clone it again. The checkout must contain at least these directories:

```text
$FSPS_HOME/data
$FSPS_HOME/ISOCHRONES
$FSPS_HOME/SPECTRA
$FSPS_HOME/src
```

The three requested model families use the FSPS data already stored in ISOCHRONES/MIST, ISOCHRONES/PARSEC, ISOCHRONES/BPASS, and SPECTRA/C3K. The project was tested with the upstream checkout at commit `bd187a0`.

## 2. Install build requirements

The standalone build needs a Fortran compiler and GNU Make. Python table generation uses the Python standard library. Check the tools before compiling:

```bash
gfortran --version
make --version
python3 --version
```

The examples below use:

```bash
export FSPS_HOME=/home/czkong/software/fsps
```

`genFSPS.py` sets `SPS_HOME=$FSPS_HOME` automatically when it runs the native driver, so exporting `SPS_HOME` is optional for this workflow.

## 3. Compile one executable for each model

Run these commands from the project root, `/home/czkong/GC-BH_Model`. F90, F90FLAGS, `FSPS_HOME`, BuildDirectory, and OUTPUT are Make variables. The two flag variables below are passed as complete strings and explicitly turn every competing isochrone or spectral-library switch off.

### MIST + C3K_LR

```bash
make -f /home/czkong/GC-BH_Model/FSPS/Makefile \
  BuildDirectory="/home/czkong/software/fsps/build_MIST_C3KLR" \
  OUTPUT="/home/czkong/GC-BH_Model/FSPS/FSPS_MIST_C3KLR" \
  IsochroneFlags='-DMIST=1 -DPADOVA=0 -DPARSEC=0 -DBASTI=0 -DGENEVA=0 -DBPASS=0' \
  SpectralFlags='-DMILES=0 -DC3K_LR=1 -DC3K_HR=0' \
  -j1
```

### PARSEC + C3K_LR

```bash
make -f /home/czkong/GC-BH_Model/FSPS/Makefile \
  BuildDirectory="/home/czkong/software/fsps/build_PARSEC_C3KLR" \
  OUTPUT="/home/czkong/GC-BH_Model/FSPS/FSPS_PARSEC_C3KLR" \
  IsochroneFlags='-DMIST=0 -DPADOVA=0 -DPARSEC=1 -DBASTI=0 -DGENEVA=0 -DBPASS=0' \
  SpectralFlags='-DMILES=0 -DC3K_LR=1 -DC3K_HR=0' \
  -j1
```

### BPASS

```bash
make -f /home/czkong/GC-BH_Model/FSPS/Makefile \
  BuildDirectory="/home/czkong/software/fsps/build_BPASS" \
  OUTPUT="/home/czkong/GC-BH_Model/FSPS/FSPS_BPASS" \
  IsochroneFlags='-DMIST=0 -DPADOVA=0 -DPARSEC=0 -DBASTI=0 -DGENEVA=0 -DBPASS=1' \
  SpectralFlags='-DMILES=0 -DC3K_LR=0 -DC3K_HR=0' \
  -j1
```

The default flags in `/home/czkong/GC-BH_Model/FSPS/Makefile` are MIST + C3K_LR. Changing the FSPS source checkout or any compile-time isochrone/spectral switch requires rebuilding the corresponding executable.

## 4. Generate the tables separately

Use one invocation of `genFSPS.py` for each executable. The following commands write the full default grids to `data/UV/`:

```bash
python3 /home/czkong/GC-BH_Model/FSPS/genFSPS.py \
  --force --fsps-exe "/home/czkong/GC-BH_Model/FSPS/FSPS_MIST_C3KLR" \
  --IMF kroupa --n-age 99 --n-feh 99 \
  --output /home/czkong/GC-BH_Model/FSPS/FSPS_MIST_C3KLR_Kroupa_m1450.csv

python3 /home/czkong/GC-BH_Model/FSPS/genFSPS.py \
  --force --fsps-exe "/home/czkong/GC-BH_Model/FSPS/FSPS_PARSEC_C3KLR" \
  --IMF kroupa --n-age 99 --n-feh 99 \
  --output /home/czkong/GC-BH_Model/FSPS/FSPS_PARSEC_C3KLR_Kroupa_m1450.csv

python3 /home/czkong/GC-BH_Model/FSPS/genFSPS.py \
  --force --fsps-exe "/home/czkong/GC-BH_Model/FSPS/FSPS_BPASS" \
  --IMF kroupa --n-age 99 --n-feh 99 \
  --output /home/czkong/GC-BH_Model/FSPS/FSPS_BPASS_Kroupa_m1450.csv
```

A successful run reports the native age/metallicity grid size and the number of rows written.

## 5. Fixed behaviour of `genFSPS.py`

The Python script is intentionally narrower than the general FSPS interface:

- Ages use a logarithmic grid. The minimum is fixed at 1e-4 Gyr (0.1 Myr). The maximum is --max-age-gyr, or the cosmic age at --max-redshift when no maximum age is supplied. --n-age controls the number of age points.
- The native driver reports its valid [Fe/H] grid. The script uses that reported minimum and maximum as the table limits, with --n-feh uniformly spaced target points between them.
- Metallicity is always interpolated in the Python code: the script performs linear interpolation in L_nu between neighbouring native [Fe/H] points. There is no --zcontinuous option.
- The default wavelength is 1450 Angstrom. The output column is always M<wavelength>_AB_per_Msun, for example M1450_AB_per_Msun.
- The IMF option is uppercase --IMF and defaults to kroupa. Other supported runtime choices are salpeter, chabrier, van-dokkum, and dave.
- The population is always an SSP (sfh=0). There are no SFH, tau, burst, or star-formation start/truncation options.
- Dust emission (including AGN and AGB dust), nebular emission and continuum, IGM absorption, and X-ray-binary emission are hard-coded off in genfsps_driver.f90. There are no command-line options for these components.
- Isochrone and spectral-library choices are compile-time Make flags, not Python options. Use a separate executable for each of the three requested model families.

BPASS is a special case in upstream FSPS: its SSP spectra are precomputed and the upstream BPASS data set has its own fixed IMF (the bundled models are the -bin-imf135all_100 Salpeter models). The Python interface still uses --IMF kroupa by default for a consistent project command, but changing --IMF does not change the underlying BPASS precomputed IMF. For MIST and PARSEC, the runtime IMF setting is applied by FSPS.

The CSV contains age_gyr, log10_age_gyr, feh, z_ratio, and the M<wavelength>_AB_per_Msun column. The magnitude is computed from the rest-frame vacuum L_nu per initially formed solar mass; no filter integration is performed.

## 6. Inspect the available options

The final command-line interface can be inspected with:

```bash
python3 FSPS/genFSPS.py --help
```

The only runtime controls are the FSPS paths, output/overwrite handling, age and metallicity point counts, maximum age or redshift, wavelength, and IMF.
