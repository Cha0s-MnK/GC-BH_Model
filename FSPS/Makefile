# Build the standalone driver without modifying the FSPS source checkout.
F90 ?= gfortran
FSPS_HOME ?= /home/czkong/software/fsps
OUTPUT ?= /home/czkong/GC-BH_Model/FSPS/FSPS_MIST_C3KLR

MakefileDirectory := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DriverSource := $(MakefileDirectory)/genfsps_driver.f90
BuildDirectory ?= $(FSPS_HOME)/.genfsps-build

F90FLAGS ?= -O2 -cpp -fPIC -ffree-line-length-none
# The default requested model is MIST + C3K_LR. Override both variables on
# the make command line for PARSEC + C3K_LR or BPASS (see FSPS.md).
IsochroneFlags ?= -DMIST=1 -DPADOVA=0 -DPARSEC=0 -DBASTI=0 -DGENEVA=0 -DBPASS=0
SpectralFlags ?= -DMILES=0 -DC3K_LR=1 -DC3K_HR=0
CompileFlags := $(F90FLAGS) $(IsochroneFlags) $(SpectralFlags) \
                -J$(BuildDirectory) -I$(BuildDirectory)

CommonNames := \
	sps_vars sps_utils compsp csp_gen ssp_gen getmags locate funcint sps_setup pz_convol \
	get_tuniv intsfwght imf imf_weight add_dust getspec sbf add_bs \
	mod_hb add_remnants getindx smoothspec mod_gb add_nebular add_xrb \
	write_isochrone sfhstat linterp tsum add_agb_dust linterparr \
	ztinterp vacairconv igm_absorb get_lumdist attn_curve sfh_weight \
	sfhlimit sfhinfo setup_tabular_sfh agn_dust

CommonObjects := $(addprefix $(BuildDirectory)/,$(addsuffix .o,$(CommonNames)))
OtherObjects := $(filter-out $(BuildDirectory)/sps_vars.o $(BuildDirectory)/sps_utils.o,$(CommonObjects))
DriverObject := $(BuildDirectory)/genfsps_driver.o

.PHONY: all clean

all: $(OUTPUT)

$(BuildDirectory):
	mkdir -p $@

$(BuildDirectory)/sps_vars.o: $(FSPS_HOME)/src/sps_vars.f90 | $(BuildDirectory)
	$(F90) $(CompileFlags) -o $@ -c $<

$(BuildDirectory)/sps_utils.o: $(FSPS_HOME)/src/sps_utils.f90 $(BuildDirectory)/sps_vars.o | $(BuildDirectory)
	$(F90) $(CompileFlags) -o $@ -c $<

$(OtherObjects): $(BuildDirectory)/sps_utils.o

$(BuildDirectory)/%.o: $(FSPS_HOME)/src/%.f90 | $(BuildDirectory)
	$(F90) $(CompileFlags) -o $@ -c $<

$(DriverObject): $(DriverSource) $(BuildDirectory)/sps_utils.o | $(BuildDirectory)
	$(F90) $(CompileFlags) -o $@ -c $<

$(OUTPUT): $(DriverObject) $(CommonObjects)
	mkdir -p $(dir $@)
	$(F90) $(CompileFlags) -o $@ $^

clean:
	rm -rf $(BuildDirectory)
