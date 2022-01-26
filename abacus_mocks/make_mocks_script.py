### Script for making cubic box and cut-sky mocks from the AbacusSummit simulations

import numpy as np
import sys
import h5py
from hodpy.cosmology import CosmologyAbacus, CosmologyMXXL
from make_catalogue_snapshot import *

###################################################
# various variables to set, depending on the simulation we are using

cosmo=4   # AbacusSummit cosmology number
simulation="base"
phase=0
Lbox = 2000. # box size (Mpc/h)
snapshot_redshift = 0.2
Nfiles = 34  # number of files the simulation is split into

zmax_low = 0.15 # maximum redshift of low-z faint lightcone
zmax = 0.6      # maximum redshift of lightcone
mass_cut = 11   # mass cut between unresolved+resolved low z lightcone

# observer position in box (in Mpc/h)
# Note that in Abacus, position coordinates in box go from -Lbox/2 < x < Lbox/2
# so an observer at the origin is in the centre of the box
observer=(0,0,0) 

cosmology = CosmologyAbacus(cosmo)
cosmology_mxxl = CosmologyMXXL()
SODensity=304.64725494384766

mag_faint_snapshot  = -18 # faintest absolute mag when populating snapshot
mag_faint_lightcone = -10 # faintest absolute mag when populating low-z faint lightcone
app_mag_faint = 20.2 # faintest apparent magnitude for cut-sky mock

### locations of various files
lookup_path = "lookup/"
hod_param_file        = lookup_path+"hod_fits/hod_c%03d.txt"%cosmo # Results of HOD fitting

# these files will be created if they don't exists
# (lookup files for efficiently assigning cen/sat magnitudes, and fit to halo mass function)
# and files of number of field particles in shells around the observer
central_lookup_file   = lookup_path+"central_magnitudes_c%03d_test.npy"%cosmo
satellite_lookup_file = lookup_path+"satellite_magnitudes_c%03d_test.npy"%cosmo
mf_fit_file           = lookup_path+"mass_functions/mf_c%03d.txt"%cosmo
Nparticle             = lookup_path+"particles/N_c%03d_ph%03d.dat"%(cosmo, phase)
Nparticle_shell       = lookup_path+"particles/Nshells_c%03d_ph%03d.dat"%(cosmo, phase)

# input path of the simulation
mock = "AbacusSummit_base_c%03d_ph%03d"%(cosmo, phase)
abacus_path = "/global/cfs/cdirs/desi/cosmosim/Abacus/"

# output path to save temporary files
output_path = "/global/cscratch1/sd/amjsmith/Abacus_mocks/%s/galaxy_catalogue/"%mock
# output path to save the final cubic box and cut-sky mocks
output_path_final = "/global/cscratch1/sd/amjsmith/Abacus_mocks/%s/galaxy_catalogue/final/"%mock

# file names
galaxy_snapshot_file   = "galaxy_snapshot_%i.hdf5"
halo_lightcone_unres   = "halo_lightcone_unresolved_%i.hdf5"
galaxy_lightcone_unres = "galaxy_lightcone_unresolved_%i.hdf5"
galaxy_lightcone_res   = "galaxy_lightcone_resolved_%i.hdf5"
galaxy_cutsky_low      = "galaxy_cut_sky_low_%i.hdf5"
galaxy_cutsky          = "galaxy_cut_sky_%i.hdf5"

galaxy_cutsky_final   = "galaxy_full_sky.hdf5"
galaxy_snapshot_final = "galaxy_snapshot.hdf5"


# how many periodic replications do we need for full cubic box to get to zmax?
# n_rep=0 is 1 replication (i.e. just the original box)
# n=1 is replicating at the 6 faces (total of 7 replications)
# n=2 is a 3x3x3 cube of replications, but omitting the corners (19 in total)
# n=3 is the full 3x3x3 cube of replications (27 in total)
rmax = cosmology.comoving_distance(zmax)
n_rep=0
if rmax >= Lbox/2.: n_rep=1
if rmax >= np.sqrt(2)*Lbox/2.: n_rep=2
if rmax >= np.sqrt(3)*Lbox/2.: n_rep=3


################################################
print("MAKING CUBIC BOX MOCK")

for file_number in range(Nfiles):
    print("FILE NUMBER", file_number)
    
    input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"%(snapshot_redshift, file_number)
    output_file = output_path + galaxy_snapshot_file%file_number
    
    main(input_file, output_file, snapshot_redshift, mag_faint_snapshot, cosmology, hod_param_file,
             central_lookup_file, satellite_lookup_file)
    
    
    
###############################################    
print("MAKING LOW Z UNRESOLVED HALO LIGHTCONE")

# this function will loop through the 34 particle files
# will find minimum (rescaled by cosmology) halo mass needed to make mock to faint app mag limit
# this includes calculating a fit to the halo mass function, and counting the available number
# of field particles in shells. 
# These will be read from files (mf_fit_file, Nparticle, Nparticle_shell) if they exist
# If the files don't exist yet, they will be automatically created (but this is fairly slow)
# app mag limit is also shifted slightly fainter than is needed

output_file = output_path+halo_lightcone_unres

halo_lightcone_unresolved(output_file, snapshot_redshift, cosmology, hod_param_file, 
                          central_lookup_file, satellite_lookup_file, mf_fit_file, Nparticle, 
                          Nparticle_shell, box_size=Lbox, SODensity=SODensity, 
                          simulation=simulation, cosmo=cosmo, ph=phase, observer=observer, 
                          app_mag_faint=app_mag_faint+0.05, cosmology_orig=cosmology_mxxl)



###############################################    
print("MAKING UNRESOLVED LOW Z GALAXY LIGHTCONE")

# this will populate the unresolved halo lightcone with galaxies

for file_number in range(Nfiles):
    print("FILE NUMBER", file_number)
    input_file = output_path+halo_lightcone_unres%file_number
    output_file = output_path+galaxy_lightcone_unres%file_number
    
    main_unresolved(input_file, output_file, snapshot_redshift, mag_faint_lightcone, 
                cosmology, hod_param_file, central_lookup_file, 
                satellite_lookup_file, SODensity=SODensity, 
                zmax=zmax_low+0.01, log_mass_max=mass_cut)
    
    
###############################################    
print("MAKING RESOLVED LOW Z GALAXY LIGHTCONE")

# this will populate the resolved haloes in the lightcone with faint galaxies, for making
# the lightcone at low redshifts

for file_number in range(Nfiles):
    print("FILE NUMBER", file_number)
    
    input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"%(snapshot_redshift, file_number)
    output_file = output_path+galaxy_lightcone_res%file_number

    main(input_file, output_file, snapshot_redshift, mag_faint_lightcone, cosmology, 
         hod_param_file, central_lookup_file, satellite_lookup_file, zmax=zmax_low+0.01, 
         observer=observer, log_mass_min=mass_cut)
    
    
    
###############################################    
print("RESCALING MAGNITUDES (FOR CREATING CUT-SKY MOCK)")

# rescale magnitudes of snapshot to match target LF exactly
input_file = output_path+galaxy_snapshot_file
rescale_snapshot_magnitudes(input_file, Lbox, snapshot_redshift, cosmology_mxxl, cosmology)

# rescale magnitudes of low z lightcone to match target LF exactly
input_file_res = output_path   + galaxy_lightcone_res
input_file_unres = output_path + galaxy_lightcone_unres
rescale_lightcone_magnitudes(input_file_res, input_file_unres, zmax_low, snapshot_redshift, 
                             cosmology_mxxl, cosmology)


###############################################    
print("MAKE CUT-SKY FROM LOW Z LIGHTCONE")

# make cut-sky mock, with evolving LF, from faint, low z lightcone
# this will use the magnitudes rescaled to match target LF exactly

for file_number in range(Nfiles):
    print("FILE NUMBER", file_number)

    unresolved_file = output_path+galaxy_lightcone_unres%file_number
    resolved_file = output_path+galaxy_lightcone_res%file_number

    output_file = output_path+galaxy_cutsky_low%file_number

    make_lightcone_lowz(resolved_file, unresolved_file, output_file, 
                        snapshot_redshift, app_mag_faint+0.05, cosmology, box_size=Lbox, 
                        observer=observer, zmax=zmax_low, cosmology_orig=cosmology_mxxl)



###############################################    
print("MAKE CUT-SKY FROM SNAPSHOT")

# make cut-sky mock, with evolving LF, from the snapshot files
# this will use the magnitudes rescaled to match target LF exactly

for file_number in range(Nfiles):
    print("FILE NUMBER", file_number)
    
    input_file = output_path+galaxy_snapshot_file%file_number
    output_file = output_path+galaxy_cutsky%file_number

    make_lightcone(input_file, output_file, snapshot_redshift, app_mag_faint+0.05, 
                   cosmology, box_size=Lbox, observer=observer,
                   zmax=zmax, cosmology_orig=cosmology_mxxl)
    
    
    
    
###############################################
print("MERGE CUBIC BOX FILES INTO FINAL MOCK")

# join snapshot files together into single file
# use original (not rescaled) magnitudes

abs_mag   = [None]*Nfiles
col       = [None]*Nfiles
halo_mass = [None]*Nfiles
is_cen    = [None]*Nfiles
pos       = [None]*Nfiles
vel       = [None]*Nfiles

for i in range(Nfiles):
    print(i)
    f = h5py.File(output_path+galaxy_snapshot_file%i,"r")
    abs_mag[i]   = f["abs_mag"][...]
    col[i]       = f["col"][...]
    halo_mass[i] = f["halo_mass"][...]
    is_cen[i]    = f["is_cen"][...]
    pos[i]       = f["pos"][...]
    vel[i]       = f["vel"][...]
    f.close()

abs_mag = np.concatenate(abs_mag)
col = np.concatenate(col)
halo_mass = np.concatenate(halo_mass)
is_cen = np.concatenate(is_cen)
pos = np.concatenate(pos)
vel = np.concatenate(vel)

gtype = np.zeros(len(is_cen), dtype="i")
gtype[is_cen==False] = 1

f = h5py.File(galaxy_snapshot_final+galaxy_snapshot_final,"a")
f.create_dataset("Data/abs_mag", data=abs_mag, compression="gzip")
f.create_dataset("Data/g_r", data=col, compression="gzip")
f.create_dataset("Data/halo_mass", data=halo_mass/1e10, compression="gzip")
f.create_dataset("Data/galaxy_type", data=gtype, compression="gzip")
f.create_dataset("Data/pos", data=pos, compression="gzip")
f.create_dataset("Data/vel", data=vel, compression="gzip")
f.close()




###############################################
print("MERGE CUT-SKY FILES INTO FINAL MOCK")


# join lightcone files together

abs_mag   = [None]*(Nfiles*2)
app_mag   = [None]*(Nfiles*2)
col       = [None]*(Nfiles*2)
col_obs   = [None]*(Nfiles*2)
dec       = [None]*(Nfiles*2)
halo_mass = [None]*(Nfiles*2)
is_cen    = [None]*(Nfiles*2)
is_res    = [None]*(Nfiles*2)
ra        = [None]*(Nfiles*2)
zcos      = [None]*(Nfiles*2)
zobs      = [None]*(Nfiles*2)


for file_number in range(Nfiles):
    print(file_number)
    filename1 = output_path+galaxy_cutsky%file_number
    filename2 = output_path+galaxy_cutsky_low%file_number
    
    zobs[file_number] = read_dataset_cut_sky_rep(filename1, "zobs", n=n_rep)
    zobs[file_number+Nfiles] = read_dataset_cut_sky(filename2, "zobs")
    
    app_mag[file_number] = read_dataset_cut_sky_rep(filename1, "app_mag", n=n_rep)
    app_mag[file_number+Nfiles] = read_dataset_cut_sky(filename2, "app_mag")
    
    keep1 = np.logical_and(zobs[file_number]   >  zmax_low, app_mag[file_number] <=app_mag_faint)
    keep2 = np.logical_and(zobs[file_number+Nfiles] <= zmax_low, app_mag[file_number+Nfiles]<=app_mag_faint)
    
    zobs[file_number]   = zobs[file_number][keep1]
    zobs[file_number+Nfiles] = zobs[file_number+Nfiles][keep2]
    
    app_mag[file_number]   = app_mag[file_number][keep1]
    app_mag[file_number+Nfiles] = app_mag[file_number+Nfiles][keep2]
    
    abs_mag[file_number] = read_dataset_cut_sky_rep(filename1, "abs_mag", n=n_rep)[keep1]
    abs_mag[file_number+Nfiles] = read_dataset_cut_sky(filename2, "abs_mag")[keep2]
    
    col[file_number] = read_dataset_cut_sky_rep(filename1, "col", n=n_rep)[keep1]
    col[file_number+Nfiles] = read_dataset_cut_sky(filename2, "col")[keep2]
    
    col_obs[file_number] = read_dataset_cut_sky_rep(filename1, "col_obs", n=n_rep)[keep1]
    col_obs[file_number+Nfiles] = read_dataset_cut_sky(filename2, "col_obs")[keep2]
    
    dec[file_number] = read_dataset_cut_sky_rep(filename1, "dec", n=n_rep)[keep1]
    dec[file_number+Nfiles] = read_dataset_cut_sky(filename2, "dec")[keep2]
    
    halo_mass[file_number] = read_dataset_cut_sky_rep(filename1, "halo_mass", n=n_rep)[keep1]
    halo_mass[file_number+Nfiles] = read_dataset_cut_sky(filename2, "halo_mass")[keep2]
    
    is_cen[file_number] = read_dataset_cut_sky_rep(filename1, "is_cen", n=n_rep)[keep1]
    is_cen[file_number+Nfiles] = read_dataset_cut_sky(filename2, "is_cen", dtype="bool")[keep2]
    
    is_res[file_number] = read_dataset_cut_sky_rep(filename1, "is_res", n=n_rep)[keep1]
    is_res[file_number+Nfiles] = read_dataset_cut_sky(filename2, "is_res", dtype="bool")[keep2]
    
    ra[file_number] = read_dataset_cut_sky_rep(filename1, "ra", n=n_rep)[keep1]
    ra[file_number+Nfiles] = read_dataset_cut_sky(filename2, "ra")[keep2]
    
    zcos[file_number] = read_dataset_cut_sky_rep(filename1, "zcos", n=n_rep)[keep1]
    zcos[file_number+Nfiles] = read_dataset_cut_sky(filename2, "zcos")[keep2]
    
    
abs_mag   = np.concatenate(abs_mag)
app_mag   = np.concatenate(app_mag)
col       = np.concatenate(col)
col_obs   = np.concatenate(col_obs)
dec       = np.concatenate(dec)
halo_mass = np.concatenate(halo_mass)
is_cen    = np.concatenate(is_cen)
is_res    = np.concatenate(is_res)
ra        = np.concatenate(ra)
zcos      = np.concatenate(zcos)
zobs      = np.concatenate(zobs)

gtype = np.zeros(len(is_cen), dtype="i")
gtype[is_cen==False] = 1 # set to 1 if satellite
gtype[is_res==False] += 2 # add 2 to unresolved

f = h5py.File(output_path_final+galaxy_cutsky_final,"a")
f.create_dataset("Data/abs_mag",   data=abs_mag, compression="gzip")
f.create_dataset("Data/app_mag",   data=app_mag, compression="gzip")
f.create_dataset("Data/g_r",       data=col,     compression="gzip")
f.create_dataset("Data/g_r_obs",   data=col_obs, compression="gzip")
f.create_dataset("Data/dec",       data=dec,     compression="gzip")
f.create_dataset("Data/halo_mass", data=halo_mass/1e10, compression="gzip")
f.create_dataset("Data/galaxy_type", data=gtype, compression="gzip")
f.create_dataset("Data/ra", data=ra, compression="gzip")
f.create_dataset("Data/z_cos", data=zcos, compression="gzip")
f.create_dataset("Data/z_obs", data=zobs, compression="gzip")
f.close()




