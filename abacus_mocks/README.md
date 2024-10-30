
Documentation of BGS Abacus mock creation for PV
------------------------------------------------

This is an adapted version of Alex Smith's code 
https://github.com/amjsmith/shared_code/tree/main/abacus_mocks 

- we do not need the cubicbox part of the code
- we only need the low-z resolved and unresolved parts of the code



1. Creating mocks from Abacus

The following script creates z < 0.11 cutsky mocks by shifting 
the observer inside the 2 Gpc/h box. 

Based on zmax, converted to rmax, the observer positions are 
such to maximize the number of non-overlapping realisations.

For realisation #0, simply run:

python -u make_mocks_script.py 0 

Or for in batch job script: 

./sbatch_make_mocks.sh 0 


2. Apply DESI footprint 

The following script reads one mock and adds a new column 
containing the DESI mask, for Y5 and Y1. 

python -u apply_desi_footprint_hdf5.py /global/cfs/cdirs/desi/users/bautista/bgs/Abacus_mocks/v0.3/AbacusSummit_base_c000_ph000/galaxy_catalogue_lowz_z0.11_r000/BGS_PV_AbacusSummit_base_c000_ph000_r000_z0.11.dat.hdf5 

3. Make randoms 

Assigns redshift and all associated properties from the data to a 
sample of randomly distributed RA, DEC points. 

Need to specify the seed for each random catalog. 

python -u make_randoms_hdf5.py /global/cfs/cdirs/desi/users/bautista/bgs/Abacus_mocks/v0.3/AbacusSummit_base_c000_ph000/galaxy_catalogue_lowz_z0.11_r000/BGS_PV_AbacusSummit_base_c000_ph000_r000_z0.11.dat.hdf5 0 

The output random catalog has the same name as the input catalog 
but with an .ran.fits extension 

Also need to apply footprint to randoms

python -u apply_desi_footprint_hdf5.py /global/cfs/cdirs/desi/users/bautista/bgs/Abacus_mocks/v0.3/AbacusSummit_base_c000_ph000/galaxy_catalogue_lowz_z0.11_r000/BGS_PV_AbacusSummit_base_c000_ph000_r000_z0.11.ran.hdf5

4. Copy to DESI folders

for i in {00..26}; do cp /global/cfs/cdirs/desi/users/bautista/bgs/Abacus_mocks/AbacusSummit_base_c000_ph000/galaxy_catalogue_lowz_z0.11_r0${i}/final/galaxy_full_sky.fits /global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/v0.1/AbacusSummit_base_c000_ph000_r0${i}_z0.11.dat.fits; done

for i in {00..26}; do cp /global/cfs/cdirs/desi/users/bautista/bgs/Abacus_mocks/AbacusSummit_base_c000_ph000/galaxy_catalogue_lowz_z0.11_r0${i}/final/galaxy_full_sky_ran.fits /global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/v0.1/AbacusSummit_base_c000_ph000_r0${i}_z0.11.ran.fits; done









