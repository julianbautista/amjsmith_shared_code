#! /usr/bin/env python
import numpy as np
import os
import hodpy

def get_lookup_dir():
    """
    Returns the directory containing the lookup files
    """
    path = os.path.abspath(hodpy.__file__)
    path = path.split("/")[:-1]
    path[-1] = "lookup"
    return "/".join(path)

path = get_lookup_dir()

# k-corrections
kcorr_file_rband = path+"/k_corr_rband_z01.dat"
kcorr_file_gband = path+"/k_corr_gband_z01.dat"

# SDSS/GAMA luminosity functions
gama_lf_fits      = path+"/lf_params.dat"
target_lf         = path+"/target_lf.dat"

# new colour fits
colour_fits = path+"/colour_fits_v1.npy"


