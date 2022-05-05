# FLAMINGO simulation mocks

Code for creating lightcone mocks from the FLAMINGO simulations. 

This is based on a stripped-down version of my HOD code, which can be found on github here
https://github.com/amjsmith/hodpy

The nbodykit package is used for cosmological calculations
https://nbodykit.readthedocs.io/en/latest/getting-started/install.html

## full_sky_evolution.py

This is the main program, which when run will create a full-sky mock from one of the FLAMINGO simulation snapshot, when run directly. 
By default, it reads in the z=0.2 snapshot of the 1 Gpc intermediate resolution box.

The function `make_cut_sky` loops through periodic replications of the box, assigning each galaxy an r-band magnitude
and g-r colour (from SDSS and GAMA measurements, both k-corrected to z=0.1, with no dust correction). The magnitudes
are assigned by ranking galaxies based on Vmax, and colours are assigned randomly.

The outputs from each periodic replication are stored in a temporary file, with a separate dataset for each replication.
The function `merge_files` combines these into the final output file. 

The function `cut_sky_replication` will need to be modified if the colour assignent is changed to include star formation
rate information

## Other files

### hodpy/catalogue.py

The `Catalogue` class contains useful methods for converting between Cartesian and equitorial coordinates, and for projecting 
the velocity along the line of sight.

### hodpy/colour.py

The class `ColourNew` is used for assigning ^{0.1}(g-r) colours to each galaxy, based on fits to the GAMA colour-magnitude 
diagram (Smith et al., in prep). The colour distributions from this class should be used, which are in good agreement with the
data.

The class `Colour` uses the original colour assignment from the old MXXL mock 
(Smith et al. 2017, https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.4646S/abstract).

### hodpy/cosmology.py

Cosmology class, which uses nbodykit to do the cosmological calculations. The `CosmologyFlamingo` class contains the FLAMINGO
simulation cosmology.

### hodpy/k_correction.py

The class `GAMA_KCorrection` contains the colour-dependent k-corrections from GAMA. These are used to convert the absolute
r-band magnitude assigned to each galaxy to the observed apparent magnitude. The polynomial k-corrections are measured from
the GAMA data in 7 bins of g-r colour, and are interpolated between bins (see Smith et al. 2017).

### hodpy/lookup.py

This file contains the location of several lookup files, which are used when creating the mock.

### hodpy/luminosity_function.py

The class `LuminosityFunctionTargetBGS` is the target luminosity function used when making the mock (and is the same LF
that was used in the MXXL mock). This transitions from the tabulated SDSS LF at low redshift (Blanton et al. 2003), 
to an evolving Schechter fit to the GAMA LF at high redshift (Loveday et al. 2012), where the transition is at z=0.15.

### lookup/colour_fits_v1.npy

File containing the fits to the GAMA colour-magnitude diagram, used for colour assignment

### lookup/k_corr_gband_z01.dat

g-band colour-dependent k-corrections from GAMA

### lookup/k_corr_rband_z01.dat

r-band colour-dependent k-corrections from GAMA

### lookup/lf_params.dat

Evolving Schechter fit to GAMA luminosity function

### lookup/target_lf.dat

Tabluated file of SDSS luminosity function







lookup/

