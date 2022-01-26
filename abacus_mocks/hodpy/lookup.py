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


def read_hod_param_file_mxxl(hod_param_file, new=False):
    """
    Read the HOD parameter file
    """
    params = np.loadtxt(hod_param_file, skiprows=8, delimiter=",")
    
    if new:
        Mmin_Ls, Mmin_Mt, Mmin_am, Mmin_bm = params[0,:4]
        M1_Ls,   M1_Mt,   M1_am,   M1_bm   = params[1,:4]
        M0_A,    M0_B,    M0_C,    M0_D    = params[2,:4]
        alpha_A, alpha_B, alpha_C          = params[3,:3]
        sigma_A, sigma_B, sigma_C, sigma_D = params[4,:4]
        
        return Mmin_Ls, Mmin_Mt, Mmin_am, Mmin_bm, M1_Ls, M1_Mt, M1_am, M1_bm, M0_A, M0_B, M0_C, M0_D, \
                alpha_A, alpha_B, alpha_C, sigma_A, sigma_B, sigma_C, sigma_D

    else:
        Mmin_Ls, Mmin_Mt, Mmin_am          = params[0,:3]
        M1_Ls,   M1_Mt,   M1_am            = params[1,:3]
        M0_A,    M0_B                      = params[2,:2]
        alpha_A, alpha_B, alpha_C          = params[3,:3]
        sigma_A, sigma_B, sigma_C, sigma_D = params[4,:4]
        
        return Mmin_Ls, Mmin_Mt, Mmin_am, M1_Ls, M1_Mt, M1_am, M0_A, M0_B,\
                alpha_A, alpha_B, alpha_C, sigma_A, sigma_B, sigma_C, sigma_D
    
    
def read_hod_param_file_abacus(hod_param_file):
    """
    Read the HOD parameter file
    """
    params = np.loadtxt(hod_param_file)
    
    Mmin_A, Mmin_B, Mmin_C, Mmin_D, \
    sigma_A, sigma_B, sigma_C, sigma_D, \
    M0_A, M0_B, \
    M1_A, M1_B, M1_C, M1_D, \
    alpha_A, alpha_B, alpha_C = params
    
    return Mmin_A, Mmin_B, Mmin_C, Mmin_D, sigma_A, sigma_B, sigma_C, sigma_D, \
                            M0_A, M0_B, M1_A, M1_B, M1_C, M1_D, alpha_A, alpha_B, alpha_C
    
    
    
        

path = get_lookup_dir()

# MXXL simulation
mxxl_mass_function = path+"/mf_fits.dat"
mxxl_snapshots     = path+"/mxxl_snapshots.dat"

# Abacus simulations
abacus_cosmologies = path+"/abacus_cosmologies.csv"

# HOD parameters for MXXL BGS mock
bgs_hod_parameters    = path+"/hod_params.dat"
bgs_hod_slide_factors = path+"/slide_factors.dat" # will be created if doesn't exist

# lookup files for central/satellite magnitudes
central_lookup_file   = path+"/central_magnitudes.npy"   # will be created if doesn't exist
satellite_lookup_file = path+"/satellite_magnitudes.npy" # will be created if doesn't exist

# k-corrections
kcorr_file = path+"/k_corr_rband_z01.dat"

# SDSS/GAMA luminosity functions
sdss_lf_tabulated = path+"/sdss_cumulative_lf.dat"
gama_lf_fits      = path+"/lf_params.dat"
target_lf         = path+"/target_lf.dat" # will be created if doesn't exist

# colours
colour_fits = path+"/colour_fits_v1.npy"

# tabulated file of fraction of centrals (from mass function fits + HODs)
fraction_central = path+"/f_central.dat"