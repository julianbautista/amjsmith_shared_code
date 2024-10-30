### Script for making cubic box and cut-sky mocks from the AbacusSummit simulations

import numpy as np
import sys
import os
#import h5py
from hodpy.cosmology import CosmologyMXXL, CosmologyAbacus
from hodpy.hod_bgs_snapshot_abacus import HOD_BGS
from make_catalogue_snapshot import *
#import fitsio
import time

#--
realisation = int(sys.argv[1])
phase = int(sys.argv[2])

###################################################
# various variables to set, depending on the simulation we are using

version = 'v0.3'

cosmo = 0   # AbacusSummit cosmology number
simulation = "base"
#phase = 0
Lbox = 2000. # box size (Mpc/h)
snapshot_redshift = 0.2
Nfiles = 34  # number of files the simulation is split into

zmax_low = 0.11 # maximum redshift of low-z faint lightcone
zmax = 0.11      # maximum redshift of lightcone
mass_cut = 11   # mass cut between unresolved+resolved low z lightcone

cosmology = CosmologyAbacus(cosmo)
cosmology_mxxl = CosmologyMXXL()
SODensity = 304.64725494384766

mag_faint_snapshot  = -18 # faintest absolute mag when populating snapshot
mag_faint_lightcone = -10 # faintest absolute mag when populating low-z faint lightcone
app_mag_faint = 20.2 # faintest apparent magnitude for cut-sky mock

# how many periodic replications do we need for full cubic box to get to zmax?
# n_rep=0 is 1 replication (i.e. just the original box)
# n=1 is replicating at the 6 faces (total of 7 replications)
# n=2 is a 3x3x3 cube of replications, but omitting the corners (19 in total)
# n=3 is the full 3x3x3 cube of replications (27 in total)
rmax = cosmology.comoving_distance(zmax)
rmax_low = cosmology.comoving_distance(zmax_low)
n_rep = replications(Lbox, rmax)


# observer position in box (in Mpc/h)
# Note that in Abacus, position coordinates in box go from -Lbox/2 < x < Lbox/2
# so an observer at the origin is in the centre of the box
#observer=(0,0,0) 
def n_boxes(box_size, dist_max):
    subbox_size = 2*dist_max 
    n_sub =  np.floor( box_size / subbox_size ).astype(int)
    return n_sub

def get_centers(box_size, dist_max):
    n_sub = n_boxes(box_size, dist_max)
    sub_box_size = box_size / n_sub 
    centers = (np.arange(n_sub)+0.5)*sub_box_size - box_size/2
    return centers

centers = get_centers(Lbox, rmax_low)
n_observers = len(centers)**3
if realisation >= n_observers:
    print(f'ERROR: Realisation number is beyond number of possible realisations: {realisation} {n_observers}')
    sys.exit()

n_o = centers.size
i = realisation // (n_o*n_o)
j = (realisation -  i*n_o*n_o ) // n_o
k = (realisation -  i*n_o*n_o - j*n_o ) 

observer = (centers[i], centers[j], centers[k]) 
print(n_observers, realisation, observer)





### locations of various files
# these files will be created if they don't exists
# (lookup files for efficiently assigning cen/sat magnitudes, and fit to halo mass function)
# and files of number of field particles in shells around the observer
lookup_path = f"lookups/{version}/z{zmax_low:.2f}_r{realisation:03d}"
os.makedirs(lookup_path+"/mass_functions", exist_ok=True)
os.makedirs(lookup_path+"/particles", exist_ok=True)

#hod_param_file = lookup_path+"hod_fits/hod_c%03d.txt"%cosmo # Results of HOD fitting
#hod_param_file = ("/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/BGS_v2/"+
#                f"z{snapshot_redshift:.3f}/AbacusSummit_base_c{cosmo:03d}_ph{phase:03d}/"+
#                "fitting_data/best_params.txt")
hod_param_file = "lookup/hod_fits/hod_Y1.txt"

central_lookup_file   = lookup_path+"/central_magnitudes_c%03d_test.npy"%cosmo
satellite_lookup_file = lookup_path+"/satellite_magnitudes_c%03d_test.npy"%cosmo

mf_fit_file           = lookup_path+"/mass_functions/mf_c%03d.txt"%cosmo

Nparticle             = lookup_path+"/particles/N_c%03d_ph%03d.dat"%(cosmo,phase)
Nparticle_shell       = lookup_path+"/particles/Nshells_c%03d_ph%03d.dat"%(cosmo, phase)


# input path of the simulation
mock = f"AbacusSummit_{simulation}_c{cosmo:03d}_ph{phase:03d}"
if simulation=="small":
    abacus_path = "/global/cfs/cdirs/desi/cosmosim/Abacus/small/"
else:
    abacus_path = "/global/cfs/cdirs/desi/cosmosim/Abacus/"

# output path to save temporary files
output_path = f"/global/cfs/cdirs/desi/users/bautista/bgs/Abacus_mocks/{version}/{mock}/galaxy_catalogue_lowz_z{zmax_low:.2f}_r{realisation:03d}/"
# output path to save the final cubic box and cut-sky mocks
#output_path_final = f"/global/cfs/cdirs/desi/users/bautista/bgs/Abacus_mocks/{version}/{mock}/galaxy_catalogue_lowz_z{zmax_low:.2f}_r{realisation:03d}/final/"
os.makedirs(output_path, exist_ok=True)
#os.makedirs(output_path_final, exist_ok=True)

# file names
#galaxy_snapshot_file   = "galaxy_snapshot_%i.hdf5" #-- not needed for PV

halo_lightcone_unres   = "halo_lightcone_unresolved_%02d.hdf5.v2"
galaxy_lightcone_unres = "galaxy_lightcone_unresolved_%02d.hdf5"
galaxy_lightcone_res   = "galaxy_lightcone_resolved_%02d.hdf5"
galaxy_cutsky_low      = "galaxy_cut_sky_low_%02d.hdf5"
galaxy_cutsky          = "galaxy_cut_sky_%02d.hdf5"

galaxy_cutsky_final   = f"BGS_PV_AbacusSummit_{simulation}_c{cosmo:03d}_ph{phase:03d}_r{realisation:03d}_z{zmax:.2f}.dat.hdf5"

#galaxy_snapshot_final = "galaxy_snapshot.fits"  #-- not needed for PV

overwrite = False 

do_cubic_box = False #- not needed for PV

do_unresolved_halos = True #- about 400 secs now with new code
do_unresolved_galaxies = True #- about 160 secs
do_resolved_galaxies = True  #- about 400 secs

do_rescale_magnitudes = False  #- Not need anymore

#do_cutsky_snapshot = False #- not needed for PV

do_cutsky_lightcone = True #- 400 secs 
do_merge = True #- 80 secs 


t00 = time.time()

# get fit to halo mass function
print("getting fit to halo mass function")
input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"
mass_function = get_mass_function(input_file, mf_fit_file,
                    redshift=snapshot_redshift, box_size=Lbox,
                                  cosmology=cosmology, Nfiles=Nfiles)


################################################
#if do_cubic_box:
#    print("\n==== MAKING CUBIC BOX MOCK\n")
#
#    for file_number in range(Nfiles):
#        print("\n== FILE NUMBER \n", file_number)

#        input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"%(snapshot_redshift, file_number)
#        output_file = output_path + galaxy_snapshot_file%file_number
        
#        if os.path.exists(output_file):
#            print(f' Output file exists : {output_file}')
#            if overwrite:
#                print(' Removing existing file !')
#                os.remove(output_file)
#            else:
#                continue
        
#        main(input_file, output_file, snapshot_redshift, mag_faint_snapshot,
#                cosmology, hod_param_file, central_lookup_file, satellite_lookup_file,
#                mass_function, cosmology_old=cosmology)


###############################################    
if do_unresolved_halos:
    print("\n==== MAKING LOW Z UNRESOLVED HALO LIGHTCONE\n")

    # this function will loop through the 34 particle files
    # will find minimum halo mass needed to make mock to faint app mag limit 
    # (taking into account scaling of magnitudes by cosmology)
    # this includes calculating a fit to the halo mass function, and counting the
    # available number of field particles in shells. These will be read from files
    # (mf_fit_file, Nparticle, Nparticle_shell) if they exist
    # If the files don't exist yet, they will be automatically created
    # (but this is fairly slow)
    # app mag limit is also shifted slightly fainter than is needed

    output_file = output_path+halo_lightcone_unres

    t0 = time.time()
    np.random.seed(realisation)
    halo_lightcone_unresolved_jb(output_file, abacus_path, snapshot_redshift,
    #halo_lightcone_unresolved(output_file, abacus_path, snapshot_redshift,
            cosmology, hod_param_file, central_lookup_file, satellite_lookup_file, 
            mass_function, Nparticle, Nparticle_shell, box_size=Lbox,
            SODensity=SODensity, simulation=simulation, cosmo=cosmo, ph=phase,
            observer=observer, app_mag_faint=app_mag_faint+0.05,
            cosmology_orig=cosmology, Nfiles=Nfiles,
            overwrite=overwrite, rmax=rmax_low)
    t1 = time.time()
    print(f'Elapsed time: {t1-t0} sec')


###############################################  
if do_unresolved_galaxies:  
    print("\n==== MAKING UNRESOLVED LOW Z GALAXY LIGHTCONE \n")

    # this will populate the unresolved halo lightcone with galaxies
    t0 = time.time()
    for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)
        input_file = output_path+halo_lightcone_unres%file_number
        output_file = output_path+galaxy_lightcone_unres%file_number
        
        if not os.path.exists(input_file): 
            continue 
        if os.path.exists(output_file):
            if overwrite == False: 
                continue
            else:
                print(f'Overwritting {output_file}')
                os.remove(output_file)
        
        main_unresolved(input_file, output_file, snapshot_redshift,
                mag_faint_lightcone, cosmology, hod_param_file, central_lookup_file,
                satellite_lookup_file, mass_function, SODensity=SODensity, 
                zmax=zmax_low, log_mass_max=mass_cut, 
                cosmology_old=cosmology, observer=observer, box_size=Lbox)
    t1 = time.time()
    print(f'Elapsed time: {t1-t0} sec')
    

###############################################   
if do_resolved_galaxies: 
    print("\n==== MAKING RESOLVED LOW Z GALAXY LIGHTCONE\n")

    # this will populate the resolved haloes in the lightcone with faint galaxies,
    # for making the lightcone at low redshifts
    t0 = time.time()
    for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)
        
        input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"%(snapshot_redshift, file_number)
        output_file = output_path+galaxy_lightcone_res%file_number

        if not os.path.exists(input_file): 
            continue 
        if os.path.exists(output_file) and overwrite == False: 
            continue 

        if rmax_low < Lbox/2.:
            # no replications required (e.g. for base resolution)
            main(input_file, output_file, snapshot_redshift, mag_faint_lightcone,
                cosmology, hod_param_file, central_lookup_file,
                satellite_lookup_file, mass_function, zmax=zmax_low,
                observer=observer, log_mass_min=mass_cut, box_size=Lbox,
                cosmology_old=cosmology, replication=(0,0,0), overwrite=overwrite)
        
        else:
            # need replications, (e.g. for small boxes)
            # 27 replications should be enough
            for i in range(-1,2,1):
                for j in range(-1,2,1):
                    for k in range(-1,2,1):

                        main(input_file, output_file+"%i%i%i"%(i,j,k), 
                            snapshot_redshift, mag_faint_lightcone, cosmology, 
                            hod_param_file, central_lookup_file,
                            satellite_lookup_file, mass_function, 
                            zmax=zmax_low, observer=observer,
                            log_mass_min=mass_cut, box_size=Lbox,
                            cosmology_old=cosmology, replication=(i,j,k))

            # merge the replications into a single file
            merge_galaxy_lightcone_res(output_file)
    t1 = time.time()
    print(f'Elapsed time: {t1-t0} sec')    

###############################################    
if do_rescale_magnitudes: 
    print("\n==== RESCALING MAGNITUDES (FOR CREATING CUT-SKY MOCK)\n")

    t0 = time.time()
    # rescale magnitudes of snapshot to match target LF exactly
    # this will create a new dataset called "abs_mag_rescaled" in the input file
    
    #input_file = output_path+galaxy_snapshot_file
    #rescale_snapshot_magnitudes(input_file, Lbox, snapshot_redshift, cosmology_mxxl,
    #                            cosmology, Nfiles=Nfiles,
    #                            mag_dataset="abs_mag_rescaled")

    # rescale magnitudes of low z lightcone to match target LF exactly
    input_file_res = output_path   + galaxy_lightcone_res
    input_file_unres = output_path + galaxy_lightcone_unres

    rescale_lightcone_magnitudes(input_file_res, input_file_unres, zmax_low,
                                snapshot_redshift, cosmology, cosmology,
                                Nfiles=Nfiles, mag_dataset="abs_mag_rescaled",
                                observer=observer)
    t1 = time.time()
    print(f'Elapsed time: {t1-t0} sec')

###############################################
if do_cutsky_lightcone:
    print("\n==== MAKE CUT-SKY FROM LOW Z LIGHTCONE\n")

    # initialize the HOD
    hod = HOD_BGS(cosmology, mag_faint_lightcone, hod_param_file,
                central_lookup_file, satellite_lookup_file,
                replace_central_lookup=True, replace_satellite_lookup=True,
                mass_function=mass_function)

    # make cut-sky mock, with evolving LF, from faint, low z lightcone
    # this will use the magnitudes rescaled to match target LF exactly
    # note that the low z resolved/unresolved lightcones are already shifted 
    # to have observer at origin
    t0 = time.time()
    for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)

        unresolved_file = output_path+galaxy_lightcone_unres%file_number
        resolved_file = output_path+galaxy_lightcone_res%file_number

        output_file = output_path+galaxy_cutsky_low%file_number

        #make_lightcone_lowz(resolved_file, unresolved_file, output_file, 
        #        snapshot_redshift, app_mag_faint+0.05, cosmology, hod=hod,
        #        box_size=Lbox, observer=observer, zmax=zmax_low,
        #        cosmology_orig=cosmology_mxxl, mag_dataset="abs_mag_rescaled")
        make_lightcone_lowz(resolved_file, unresolved_file, output_file, 
                snapshot_redshift, app_mag_faint+0.05, cosmology, hod=hod,
                box_size=Lbox, observer=observer, zmax=zmax_low,
                cosmology_orig=cosmology, mag_dataset="abs_mag",
                overwrite=overwrite, return_vel=True)
    t1 = time.time()
    print(f'Elapsed time: {t1-t0} sec')


###############################################
#if do_cutsky_snapshot:
#    print("\n==== MAKE CUT-SKY FROM SNAPSHOT\n")

    # make cut-sky mock, with evolving LF, from the snapshot files
    # this will use the magnitudes rescaled to match target LF exactly

    #for file_number in range(Nfiles):
    #    print("FILE NUMBER", file_number)
        
    #    input_file = output_path+galaxy_snapshot_file%file_number
    #    output_file = output_path+galaxy_cutsky%file_number

    #    make_lightcone(input_file, output_file, snapshot_redshift,
    #            app_mag_faint+0.05, cosmology, hod=hod, box_size=Lbox,
    #            observer=observer, zmax=zmax, cosmology_orig=cosmology,
    #            mag_dataset="abs_mag")
        


    ###############################################
    #print("\n==== MERGE CUBIC BOX FILES INTO FINAL MOCK \n")

    # join snapshot files together into single file
    # use original (not rescaled) magnitudes
    #t0 = time.time()
    #merge_box(output_path, galaxy_snapshot_file, output_path_final,
    #        galaxy_snapshot_final, fmt="fits", Nfiles=Nfiles, offset=Lbox/2.)

    #t1 = time.time()
    #print(f'Elapsed time: {t1-t0} sec')

###############################################
if do_merge:
    print("\n==== MERGE CUT-SKY FILES INTO FINAL MOCK\n")


    # join files together 
    t0 = time.time()
    #merge_lightcone(output_path, galaxy_cutsky, galaxy_cutsky_low, 
    #                output_path_final, galaxy_cutsky_final, fmt='fits',
    #                Nfiles=Nfiles, zmax_low=zmax_low, app_mag_faint=app_mag_faint)
    merge_lightcone_jb(output_path+galaxy_cutsky_low, output_path+galaxy_cutsky_final)
    t1 = time.time()
    print(f'Elapsed time: {t1-t0} sec')

t11 = time.time()

print(f'Elapsed time: {(t11-t00)/60} min')
