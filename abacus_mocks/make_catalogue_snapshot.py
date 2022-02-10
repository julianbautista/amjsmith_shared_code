#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import gc
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.read_abacus import read_asdf
import h5py
from os.path import exists

from cut_sky_evolution import cut_sky

from hodpy.halo_catalogue import AbacusSnapshot, AbacusSnapshotUnresolved
from hodpy.galaxy_catalogue_snapshot import GalaxyCatalogueSnapshot
from hodpy.cosmology import CosmologyAbacus
from hodpy.hod_bgs_snapshot_abacus import HOD_BGS
from hodpy.colour import ColourNew
from hodpy import lookup
from hodpy.mass_function import MassFunction
from hodpy.k_correction import GAMA_KCorrection
from hodpy.luminosity_function import LuminosityFunctionTargetBGS
from hodpy.hod_bgs import HOD_BGS_Simple



def main(input_file, output_file, snapshot_redshift, mag_faint, cosmology, 
         hod_param_file, central_lookup_file, satellite_lookup_file, box_size=2000.,
        zmax=None, observer=(0,0,0), log_mass_min=None, log_mass_max=None, cosmology_old=None):
    """
    Create a HOD mock catalogue by populating the AbacusSummit simulation snapshot
    with galaxies. The output galaxy catalogue is in Cartesian coordinates
    
    Args:
        input_file:        string, containting the path to the AbacusSummit snapshot file
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        mag_faint:         float, faint absolute magnitude limit
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        hod_param_file:    string, path to file containing HOD hyperparameter fits
        central_lookup_file: lookup file of central magnitudes, will be created if the file
                                doesn't already exist
        satellite_lookup_file: lookup file of satellite magnitudes, will be created if the file
                                doesn't already exist
        box_size:          float, simulation box size (Mpc/h)
        zmax:              float, maximum redshift. If provided, will cut the box to only haloes
                                that are within a comoving distance to the observer that 
                                corresponds to zmax. By default, is None
        observer:          3D position vector of the observer, in units Mpc/h. By default 
                                observer is at the origin (0,0,0) Mpc/h
        log_mass_min:      float, log10 of minimum halo mass cut, in Msun/h
        log_mass_max:      float, log10 of maximum halo mass cut, in Msun/h
    """

    import warnings
    warnings.filterwarnings("ignore")
    
    # create halo catalogue
    print("read halo catalogue")
    halo_cat = AbacusSnapshot(input_file, snapshot_redshift, cosmology=cosmology, 
                              box_size=box_size, particles=False, clean=True)

    
    # apply cuts to halo mass, if log_mass_min or log_mass_max is provided
    # this cut is applied to make sure there no overlap in masses between the unresolved/resolved
    # halo lightcones
    if not log_mass_min is None or not log_mass_max is None:
        log_mass = halo_cat.get("log_mass")
        
        keep = np.ones(len(log_mass), dtype="bool")
        
        if not log_mass_min is None:
            keep = np.logical_and(keep, log_mass >= log_mass_min)
            
        if not log_mass_max is None:
            keep = np.logical_and(keep, log_mass <= log_mass_max)
            
        halo_cat.cut(keep)
    
    
    # cut to haloes that are within a comoving distance corresponding to the redshift zmax
    # this is done is we are making a lightcone, to remove any high redshift haloes we don't need
    # Note that if we are making a lightcone, the output file of this function will still be
    # in Cartesian coordinates, and will need to be converted to cut-sky
    if not zmax is None:
        print("cutting to zmax")
        pos = halo_cat.get("pos")
        
        # make sure observer is at origin
        for i in range(3):
            pos[:,i] -= observer[i]
            
        #apply periodic boundary conditions, so -Lbox/2 < pos < Lbox/2
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        
        dist = np.sum(pos**2, axis=1)**0.5
        dist_max = cosmology.comoving_distance(np.array([zmax,]))[0]
        halo_cat.cut(dist<dist_max)
        
        if len(halo_cat.get("zcos")) == 0:
            print("No haloes in lightcone, skipping file")
            return
    
    # empty galaxy catalogue
    print("create galaxy catalogue")
    gal_cat  = GalaxyCatalogueSnapshot(halo_cat, cosmology=cosmology, box_size=box_size)
    
    # use hods to populate galaxy catalogue
    print("read HODs")
    hod = HOD_BGS(cosmology, mag_faint, hod_param_file, central_lookup_file, satellite_lookup_file,
                  replace_central_lookup=True, replace_satellite_lookup=True)
    
    print("add galaxies")
    gal_cat.add_galaxies(hod)
    
    # position galaxies around their haloes
    print("position galaxies")
    gal_cat.position_galaxies(particles=False, conc="conc")

    # add g-r colours
    print("assigning g-r colours")
    col = ColourNew()
    if cosmology_old is None:
        gal_cat.add_colours(col)
    else:
        # use magnitudes in original cosmology
        gal_cat.add_colours(col, cosmology_old, cosmology)

    # cut to galaxies brighter than absolute magnitude threshold
    gal_cat.cut(gal_cat.get("abs_mag") <= mag_faint)
    
    if not zmax is None:
        # if we are making a lightcone and the observer is not at the origin,
        # this will shift the coords back to the original Cartesian coordinates of the snapshot
        pos = gal_cat.get("pos")
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        gal_cat.add("pos", pos)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])
    
    
    
def main_unresolved(input_file, output_file, snapshot_redshift, mag_faint, 
                    cosmology, hod_param_file, central_lookup_file, 
                    satellite_lookup_file, box_size=2000., SODensity=200,
                    zmax=0.6, observer=(0,0,0), log_mass_min=None, log_mass_max=None,
                    cosmology_old=None):
    """
    Create a HOD mock catalogue by populating a hdf5 file of unresolved haloes. 
    The output galaxy catalogue is in Cartesian coordinates
    
    Args:
        input_file:        string, containting the path to the hdf5 file of unresolved haloes
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        mag_faint:         float, faint absolute magnitude limit
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        hod_param_file:    string, path to file containing HOD hyperparameter fits
        central_lookup_file: lookup file of central magnitudes, will be created if the file
                                doesn't already exist
        satellite_lookup_file: lookup file of satellite magnitudes, will be created if the file
                                doesn't already exist
        box_size:          float, simulation box size (Mpc/h)
        zmax:              float, maximum redshift. If provided, will cut the box to only haloes
                                that are within a comoving distance to the observer that 
                                corresponds to zmax. By default, is None
        observer:          3D position vector of the observer, in units Mpc/h. By default 
                                observer is at the origin (0,0,0) Mpc/h
        log_mass_min:      float, log10 of minimum halo mass cut, in Msun/h
        log_mass_max:      float, log10 of maximum halo mass cut, in Msun/h
    """
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # create halo catalogue
    print("read halo catalogue")
    halo_cat = AbacusSnapshotUnresolved(input_file, snapshot_redshift, cosmology=cosmology,
                                        box_size=box_size, SODensity=SODensity)

    # apply cuts to halo mass, if log_mass_min or log_mass_max is provided
    # this cut is applied to make sure there no overlap in masses between the unresolved/resolved
    # halo lightcones
    if not log_mass_min is None or not log_mass_max is None:
        log_mass = halo_cat.get("log_mass")
        
        keep = np.ones(len(log_mass), dtype="bool")
        
        if not log_mass_min is None:
            keep = np.logical_and(keep, log_mass >= log_mass_min)
            
        if not log_mass_max is None:
            keep = np.logical_and(keep, log_mass <= log_mass_max)
            
        halo_cat.cut(keep)
    
    
    # cut to haloes that are within a comoving distance corresponding to the redshift zmax
    # this is done is we are making a lightcone, to remove any high redshift haloes we don't need
    # Note that if we are making a lightcone, the output file of this function will still be
    # in Cartesian coordinates, and will need to be converted to cut-sky
    if not zmax is None:
        pos = halo_cat.get("pos")
        
        # make sure observer is at origin
        for i in range(3):
            pos[:,i] -= observer[i]
            
        #apply periodic boundary conditions, so -Lbox/2 < pos < Lbox/2
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        
        dist = np.sum(pos**2, axis=1)**0.5
        dist_max = cosmology.comoving_distance(np.array([zmax,]))[0]
        halo_cat.cut(dist<dist_max)
        
        if len(halo_cat.get("zcos")) == 0:
            print("No haloes in lightcone, skipping file")
            return
    
    
    # use hods to populate galaxy catalogue
    print("read HODs")
    hod = HOD_BGS(cosmology, mag_faint, hod_param_file, central_lookup_file, 
                  satellite_lookup_file,
                  replace_central_lookup=True, replace_satellite_lookup=True)
    
    # empty galaxy catalogue
    print("create galaxy catalogue")
    gal_cat  = GalaxyCatalogueSnapshot(halo_cat, cosmology=cosmology, box_size=box_size)
    
    print("add galaxies")
    gal_cat.add_galaxies(hod)

    # position galaxies around their haloes
    print("position galaxies")
    gal_cat.position_galaxies(particles=False, conc="conc")

    # add g-r colours
    print("assigning g-r colours")
    col = ColourNew()
    if cosmology_old is None:
        gal_cat.add_colours(col)
    else:
        # use magnitudes in original cosmology
        gal_cat.add_colours(col, cosmology_old, cosmology)

    # cut to galaxies brighter than absolute magnitude threshold
    gal_cat.cut(gal_cat.get("abs_mag") <= mag_faint)
    
    if not zmax is None:
        # if we are making a lightcone and the observer is not at the origin,
        # this will shift the coords back to the original Cartesian coordinates of the snapshot
        pos = gal_cat.get("pos")
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        gal_cat.add("pos", pos)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])
    
    
    
def get_mass_function(input_file, fit_file, redshift=0.2, box_size=2000, cosmology=None,
                     Nfiles=34):
    """
    Get smooth fit to the mass function of an Abacus snapshot
    Args:
        input_file:  path to AbacusSummit halo catalogues
        fit_file:    path of file of mass function fit. Will be created if it doesn't exist
        redshift:    Redshift of simulation snapshot
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmology:   Abacus cosmology, object of class hodpy.cosmology.Cosmology
        Nfiles:      Number of AbacusSummit files for this snapshot. Default is 34
    Returns:
        the halo mass function, hodpy.mass_function.MassFunction object
    """
    
    # read mass function file if it already exists
    if exists(fit_file):
        fit_params = np.loadtxt(fit_file)
        mf = MassFunction(cosmology=cosmology, redshift=redshift, 
                      fit_params=fit_params)
        return mf

    
    # if it doesn't exist, read in all haloes to get the fit, and save it to file
    log_mass = [None]*Nfiles
    for file_number in range(Nfiles):

        halo_cat = CompaSOHaloCatalog(input_file%(redshift, file_number), 
                                      cleaned=True, fields=['N'])
        m_par = halo_cat.header["ParticleMassHMsun"]
        log_mass[file_number] = np.log10(np.array(halo_cat.halos["N"])*m_par)

        print(file_number, len(log_mass[file_number]))

    log_mass = np.concatenate(log_mass)

    # get number densities in mass bins  
    bin_size = 0.02
    mass_bins = np.arange(10,16,bin_size)
    mass_binc = mass_bins[:-1]+bin_size/2.
    hist, bins = np.histogram(log_mass, bins=mass_bins)
    n_halo = hist/bin_size/box_size**3

    # remove bins with zero haloes
    keep = n_halo > 0
    measured_mass_function = np.array([mass_binc[keep], n_halo[keep]])

    # create mass function object
    mf = MassFunction(cosmology=cosmology, redshift=redshift, 
                      measured_mass_function=measured_mass_function)

    # get fit to mass function
    fit_params = mf.get_fit()

    # save file
    np.savetxt(fit_file, fit_params)
    
    return mf
    
    
def get_min_mass(rfaint, hod, cosmo_orig, cosmo_new):
    """
    Get the minimum halo masses, in shells of comoving distance, needed to create a lightcone
    down to a faint apparent magnitude limit. This takes into account the rescaling of
    magnitudes by cosmology. For making Abacus mocks that were fit to MXXL, cosmo_orig
    is the MXXL cosmology, and cosmo_new is the Abacus cosmology.
    Args:
        rfaint:     float, minimum r-band apparent magnitude
        hod:        HOD_BGS object
        cosmo_orig: hodpy.cosmology.Cosmology object, the original cosmology
        cosmo_new:  hodpy.cosmology.Cosmology object, the new cosmology
    """
    
    # bins of comoving distance to find minimum masses in
    rcom = np.arange(25,1001,25)
    z = cosmo_new.redshift(rcom)
    
    kcorr = GAMA_KCorrection(cosmo_new)
    lf = LuminosityFunctionTargetBGS(lookup.target_lf, lookup.sdss_lf_tabulated, 
                        lookup.gama_lf_fits, HOD_BGS_Simple(lookup.bgs_hod_parameters))

    # absolute magnitude corresponding to rfaint depends on colour
    # calculate for red and blue galaxies, and choose the faintest mag
    mag_faint1 = kcorr.absolute_magnitude(np.ones(len(z))*rfaint, z, np.ones(len(z))*-10)
    mag_faint2 = kcorr.absolute_magnitude(np.ones(len(z))*rfaint, z, np.ones(len(z))*10)
    mag_faint = np.maximum(mag_faint1, mag_faint2)
    
    
    # now do scaling of magnitudes by cosmology
    mags = np.arange(-23,-8,0.01)
    mags_unscaled = np.zeros(len(z))

    for i in range(len(z)):
        # loop through the different shells of distance
        mags_new = lf.rescale_magnitude(mags, np.ones(len(mags))*0.2, np.ones(len(mags))*z[i],
                                       cosmo_orig, cosmo_new)
        idx = np.where(mags_new>=mag_faint[i])[0][0]
        mags_unscaled[i] = mags[idx]
        
        
    # Now get minimum halo masses needed to add central galaxies brighter than the faint abs mag
    log_mass = np.arange(9,15,0.001)
    mass_limit = np.zeros(len(z))

    for i in range(len(z)):
        N = hod.number_centrals_mean(log_mass,np.ones(len(log_mass))*mags_unscaled[i])
        idx = np.where(N>0)[0][0]
        mass_limit[i] = log_mass[idx]
        
    for i in range(len(mass_limit)):
        mass_limit[i] = np.min(mass_limit[i:])
        
    return rcom, mass_limit
    
    
    
def halo_lightcone_unresolved(output_file, snapshot_redshift, cosmology, hod_param_file, 
                              central_lookup_file, satellite_lookup_file, mf_fit_file,
                              Nparticle, Nparticle_shell, box_size=2000., SODensity=200,
                              simulation="base", cosmo=0, ph=0, observer=(0,0,0), 
                              app_mag_faint=20.25, cosmology_orig=None, Nfiles=34):
    """
    Create a lightcone of unresolved AbacusSummit haloes, using the field particles 
    (not in haloes) as tracers. The output galaxy catalogue is in Cartesian coordinates
    
    Args:
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        hod_param_file:    string, path to file containing HOD hyperparameter fits
        central_lookup_file: lookup file of central magnitudes, will be created if the file
                                doesn't already exist
        satellite_lookup_file: lookup file of satellite magnitudes, will be created if the file
                                doesn't already exist
        mf_fit_file:       path of file of mass function. Will be created if it doesn't exist
        Nparticle:         file containing total number of field particles in each Abacus file.
                                Will be created if it doesn't exist
        Nparticle_shell:   file containing total number of field particles in shells of comoving
                                distance, ineach Abacus file. Will be created if it doesn't exist
        box_size:          float, simulation box size (Mpc/h)
        SODensity:         float, spherical overdensity of L1 haloes
        simulation:        string, the AbacusSummit simulation, default is "base"
        cosmo:             integer, the AbacusSummit cosmology number, default is 0 (Planck LCDM)
        ph:                integer, the AbacusSummit simulation phase, default is 0
        observer:          3D position vector of the observer, in units Mpc/h. By default 
        app_mag_faint:     float, faint apparent magnitude limit
        cosmology_orig:    object of class hodpy.cosmology.Cosmology, if provided, this is the
                                original cosmology when doing cosmology rescaling.
        Nfiles:      Number of AbacusSummit files for this snapshot. Default is 34
    """
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # get fit to halo mass function
    print("getting fit to halo mass function")
    path = "/global/cfs/cdirs/desi/cosmosim/Abacus/"
    mock = "AbacusSummit_%s_c%03d_ph%03d"%(simulation, cosmo, ph)
    input_file = path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"
    mf = get_mass_function(input_file, mf_fit_file, redshift=snapshot_redshift, 
                           box_size=box_size, cosmology=cosmology, Nfiles=Nfiles)
    
    # read HOD files
    mag_faint=-10
    hod = HOD_BGS(cosmology, mag_faint, hod_param_file, central_lookup_file, satellite_lookup_file,
                  replace_central_lookup=True, replace_satellite_lookup=True)
    
    # get min mass
    if cosmology_orig is None:
        # no rescaling of cosmology
        rcom, logMmin = get_min_mass(app_mag_faint, hod, cosmology, cosmology)
    else:
        # apply cosmology rescaling
        rcom, logMmin = get_min_mass(app_mag_faint, hod, cosmology_orig, cosmology)
    
    
    # get total number of field particles (using A particles)
    
    if exists(Nparticle) and exists(Nparticle_shell):
        print("Reading total number of field particles")
        N = np.loadtxt(Nparticle)
        Nshells = np.loadtxt(Nparticle_shell)
    
    else:
        print("File doesn't exist yet, finding number of field particles")
        N = np.zeros(Nfiles, dtype="i")
        Nshells = np.zeros((Nfiles,len(rcom)),dtype="i")
        for file_number in range(Nfiles):
            # this loop is slow. Is there a faster way to get total number of field particles in each file?
            file_name = path+mock+"/halos/z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(snapshot_redshift, file_number)
            data = read_asdf(file_name, load_pos=True, load_vel=False)
            p = data["pos"]
            for i in range(3):
                p[:,i] -= observer[i]
            p[p>box_size/2.] -= box_size
            p[p<-box_size/2.] += box_size
            dist = np.sum(p**2, axis=1)**0.5
            del data
            N[file_number] = p.shape[0]
    
            for j in range(len(rcom)):
                keep = np.logical_and(dist>=rcom[j]-25, dist<rcom[j])
                Nshells[file_number,j] = np.count_nonzero(keep)
                print(file_number, j, Nshells[file_number,j])
            
            gc.collect() # need to run garbage collection to release memory
        
        # save files
        np.savetxt(Nparticle, N)
        np.savetxt(Nparticle_shell, Nshells)
    
    
    # Now make lightcone of unresolved haloes
    for file_number in range(Nfiles):
        file_name = path+mock+"/halos/z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(snapshot_redshift, file_number)
        print(file_name)

        # read file
        data = read_asdf(file_name, load_pos=True, load_vel=True)
        vel = data["vel"]
        pos = data["pos"]
        for i in range(3):
            pos[:,i] -= observer[i]
        pos[pos>box_size/2.] -= box_size
        pos[pos<-box_size/2.] += box_size
        
        dist = np.sum(pos**2, axis=1)**0.5
        
        pos_bins = [None]*len(rcom)
        vel_bins = [None]*len(rcom)
        mass_bins = [None]*len(rcom)
        
        for j in range(len(rcom)):
            
            rmin_bin, rmax_bin = rcom[j]-25, rcom[j]
            vol_bin = 4/3.*np.pi*(rmax_bin**3 - rmin_bin**3)
            if j==0:
                logMmin_bin, logMmax_bin = logMmin[j], logMmin[-1]
            else:
                logMmin_bin, logMmax_bin = logMmin[j-1], logMmin[-1]
                
            
            # cut to particles in shell
            keep_shell = np.logical_and(dist>=rmin_bin, dist<rmax_bin)
            N_shell = np.count_nonzero(keep_shell)
            print(file_number, j, N_shell)
            
            if N_shell==0: 
                pos_bins[j] = np.zeros((0,3))
                vel_bins[j] = np.zeros((0,3))
                mass_bins[j] = np.zeros(0)
                continue
                
            Npar =  np.sum(Nshells[:,j]) # total number of field particles in shell
            
            # number of randoms to generate in shell
            Nrand = mf.number_density_in_mass_bin(logMmin_bin, logMmax_bin) * vol_bin
            
            if Nrand==0: 
                pos_bins[j] = np.zeros((0,3))
                vel_bins[j] = np.zeros((0,3))
                mass_bins[j] = np.zeros(0)
                continue
            
            print(Npar, Nrand, np.count_nonzero(keep_shell))
            
            # probability to keep a particle
            prob = Nrand*1.0 / Npar
            print(prob)
        
            keep = np.random.rand(N_shell) <= prob
            pos_bins[j] = pos[keep_shell][keep]
            vel_bins[j] = vel[keep_shell][keep]
        
            mass_bins[j] = 10**mf.get_random_masses(np.count_nonzero(keep), logMmin_bin, logMmax_bin) / 1e10
            
        del data
        gc.collect() # need to run garbage collection to release memory
        
        pos_bins = np.concatenate(pos_bins)
        vel_bins = np.concatenate(vel_bins)
        mass_bins = np.concatenate(mass_bins)
        
        # shift positions back
        for i in range(3):
            pos_bins[:,i] += observer[i]
        pos_bins[pos_bins>box_size/2.] -= box_size
        pos_bins[pos_bins<-box_size/2.] += box_size
        
        # save halo lightcone file
        f = h5py.File(output_file%file_number, "a")
        f.create_dataset("mass", data=mass_bins, compression="gzip")
        f.create_dataset("position", data=pos_bins, compression="gzip")
        f.create_dataset("velocity", data=vel_bins, compression="gzip")
        f.close()

        
        
def read_galaxy_file(filename, resolved=True, mag_name="abs_mag"):
    """
    Read the galaxy mock, which is in Cartesian coordinates
    """
    if exists(filename):
        f = h5py.File(filename, "r")
        abs_mag = f[mag_name][...]
        col = f["col"][...]
        pos = f["pos"][...]
        vel = f["vel"][...]
        is_cen = f["is_cen"][...]
        log_mass = np.log10(f["halo_mass"][...])
        is_res = np.ones(len(is_cen), dtype="bool") * resolved
        f.close()
    else:
        # if there is no file, return empty arrays
        abs_mag = np.zeros(0)
        col = np.zeros(0)
        pos = np.zeros((0,3))
        vel = np.zeros((0,3))
        is_cen = np.zeros(0, dtype="bool")
        log_mass = np.zeros(0)
        is_res = np.zeros(0, dtype="bool")
    return pos, vel, log_mass, abs_mag, col, is_cen, is_res
        
    
    
    
def make_lightcone(input_file, output_file, snapshot_redshift, mag_faint, 
                   cosmology, box_size=2000., observer=(0,0,0),
                   zmax=0.6, cosmology_orig=None, mag_dataset="abs_mag_rescaled"):
    """
    Create a cut-sky mock (with ra, dec, z), from the cubic box galaxy mock,
    also rescaling magnitudes by cosmology if the original cosmology is provided
    Args:
        input_file:        string, containing the path of the input galaxy mock
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        mag_faint:         float, faint apparent magnitude limit
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        box_size:          float, simulation box size (Mpc/h)
        observer:          3D position vector of the observer, in units Mpc/h. By default
        zmax:              float, maximum redshift cut. 
        cosmology_orig:    object of class hodpy.cosmology.Cosmology, if provided, this is the
                                original cosmology when doing cosmology rescaling.
        mag_dataset:       string, name of the dataset of absolute magnitudes to read from
                                the input mock file
    """
    pos, vel, log_mass, abs_mag, col, is_cen, is_res = \
                        read_galaxy_file(input_file, resolved=True, mag_name=mag_dataset)
    
    # shift coordinates so observer at origin
    for i in range(3):
        pos[:,i] -= observer[i]
    pos[pos >  box_size/2.] -= box_size
    pos[pos < -box_size/2.] += box_size
        
        
    kcorr_r = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_rband_z01.dat", 
                               cubic_interpolation=True)
    kcorr_g = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_gband_z01.dat", 
                               cubic_interpolation=True)
    
    # find how many periodic replications we need
    rmax = cosmology.comoving_distance(zmax)
    n=0
    if rmax >= box_size/2.: n=1
    if rmax >= np.sqrt(2)*box_size/2.: n=2
    if rmax >= np.sqrt(3)*box_size/2.: n=3
    
    # loop through periodic replications
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                
                n_i = abs(i) + abs(j) + abs(k)
                if n_i > n: continue
                
                rep = (i,j,k)
                print(rep)
    
                ra, dec, zcos, zobs, magnitude_new, app_mag, col_new, col_obs, index = \
                        cut_sky(pos, vel, abs_mag, is_cen, cosmology, Lbox=box_size, 
                            zsnap=snapshot_redshift, kcorr_r=kcorr_r, kcorr_g=kcorr_g, 
                            replication=rep, 
                            zcut=zmax, mag_cut=mag_faint, cosmology_orig=cosmology_orig)

                print("NGAL:", np.count_nonzero(ra))

                f = h5py.File(output_file,"a")
                f.create_dataset("%i%i%i/ra"%(i,j,k),data=ra, compression="gzip")
                f.create_dataset("%i%i%i/dec"%(i,j,k), data=dec, compression="gzip")
                f.create_dataset("%i%i%i/zcos"%(i,j,k), data=zcos, compression="gzip")
                f.create_dataset("%i%i%i/zobs"%(i,j,k), data=zobs, compression="gzip")
                f.create_dataset("%i%i%i/app_mag"%(i,j,k), data=app_mag,compression="gzip")
                f.create_dataset("%i%i%i/abs_mag"%(i,j,k), 
                                         data=magnitude_new,compression="gzip")
                f.create_dataset("%i%i%i/col"%(i,j,k), data=col_new, compression="gzip")
                f.create_dataset("%i%i%i/col_obs"%(i,j,k), data=col_obs,compression="gzip")
                f.create_dataset("%i%i%i/is_cen"%(i,j,k), data=is_cen[index],
                                                             compression="gzip")
                f.create_dataset("%i%i%i/is_res"%(i,j,k), data=is_res[index], 
                                                             compression="gzip")
                f.create_dataset("%i%i%i/halo_mass"%(i,j,k), data=10**log_mass[index], 
                                                             compression="gzip")
                f.close()


        
    
def make_lightcone_lowz(resolved_file, unresolved_file, output_file, 
                        snapshot_redshift, mag_faint, cosmology, box_size=2000., 
                        observer=(0,0,0), zmax=0.15, cosmology_orig=None, 
                        mag_dataset="abs_mag_rescaled"):
    """
    Create a cut-sky mock (with ra, dec, z), from the cubic box galaxy mock,
    also rescaling magnitudes by cosmology if the original cosmology is provided
    Args:
        resolved_file:     string, containing the path of the input resolved galaxy mock
        unresolved_file:   string, containing the path of the input unresolved galaxy mock
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        mag_faint:         float, faint apparent magnitude limit
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        box_size:          float, simulation box size (Mpc/h)
        observer:          3D position vector of the observer, in units Mpc/h. By default
        zmax:              float, maximum redshift cut. 
        cosmology_orig:    object of class hodpy.cosmology.Cosmology, if provided, this is the
                                original cosmology when doing cosmology rescaling.
        mag_dataset:       string, name of the dataset of absolute magnitudes to read from
                                the input mock file
    """
    
    # read resolved
    pos_r, vel_r, log_mass_r, abs_mag_r, col_r, is_cen_r, is_res_r = \
                        read_galaxy_file(resolved_file, resolved=True, mag_name=mag_dataset)
    
    # read unresolved
    pos_u, vel_u, log_mass_u, abs_mag_u, col_u, is_cen_u, is_res_u = \
                        read_galaxy_file(unresolved_file, resolved=False, mag_name=mag_dataset)
    
    # combine into single array
    pos      = np.concatenate([pos_r, pos_u])
    vel      = np.concatenate([vel_r, vel_u])
    log_mass = np.concatenate([log_mass_r, log_mass_u])
    abs_mag  = np.concatenate([abs_mag_r, abs_mag_u])
    col      = np.concatenate([col_r, col_u])
    is_cen   = np.concatenate([is_cen_r, is_cen_u])
    is_res   = np.concatenate([is_res_r, is_res_u])
    
    # shift coordinates so observer at origin
    for i in range(3):
        pos[:,i] -= observer[i]
    pos[pos >  box_size/2.] -= box_size
    pos[pos < -box_size/2.] += box_size
    
    kcorr_r = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_rband_z01.dat", 
                               cubic_interpolation=True)
    kcorr_g = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_gband_z01.dat", 
                               cubic_interpolation=True)
    
    # don't need to apply multiple replications
    ra, dec, zcos, zobs, magnitude_new, app_mag, col_new, col_obs, index = \
            cut_sky(pos, vel, abs_mag, is_cen, cosmology, Lbox=box_size, 
                    zsnap=snapshot_redshift, kcorr_r=kcorr_r, kcorr_g=kcorr_g, 
                    replication=(0,0,0), zcut=zmax, mag_cut=mag_faint, 
                    cosmology_orig=cosmology_orig)

    print("NGAL:", np.count_nonzero(ra))

    f = h5py.File(output_file,"a")
    f.create_dataset("ra",      data=ra,      compression="gzip")
    f.create_dataset("dec",     data=dec,     compression="gzip")
    f.create_dataset("zcos",    data=zcos,    compression="gzip")
    f.create_dataset("zobs",    data=zobs,    compression="gzip")
    f.create_dataset("app_mag", data=app_mag, compression="gzip")
    f.create_dataset("abs_mag",data=magnitude_new,compression="gzip")
    f.create_dataset("col",     data=col_new, compression="gzip")
    f.create_dataset("col_obs", data=col_obs, compression="gzip")
    f.create_dataset("is_cen", data=is_cen[index], compression="gzip")
    f.create_dataset("is_res", data=is_res[index], compression="gzip")
    f.create_dataset("halo_mass", data=10**log_mass[index], compression="gzip")
    f.close()

    
    
    
def rescale_snapshot_magnitudes(filename, box_size, zsnap, cosmo_orig, cosmo_new, Nfiles=34,
                               mag_dataset="abs_mag_rescaled"):
    """
    Rescale the magnitudes in the cubic box mock to match target LF exactly
    Args:
        filename:     string, containing the path of the input galaxy mock
        box_size:     float, simulation box size (Mpc/h)
        zsnap:        integer, the redshift of the snapshot
        cosmo_orig:   object of class hodpy.cosmology.Cosmology, the cosmology of the 
                            original simulation
        cosmo_new:    object of class hodpy.cosmology.Cosmology, the cosmology of the
                            new simultion
        Nfiles:       Number of AbacusSummit files for this snapshot. Default is 34
        mag_dataset:  string, name of the dataset of absolute magnitudes to read from
                                the input mock file
    """
    # rescale the magnitudes of the snapshot to match target LF exactly
    
    lf = LuminosityFunctionTargetBGS(lookup.target_lf, lookup.sdss_lf_tabulated, 
                        lookup.gama_lf_fits, HOD_BGS_Simple(lookup.bgs_hod_parameters))
    
    abs_mag = [None]*Nfiles
    filenum = [None]*Nfiles
    
    for i in range(Nfiles):
        print(i)
        f = h5py.File(filename%i,"r")
        abs_mag[i] = f["abs_mag"][...]
        filenum[i] = np.ones(len(abs_mag[i]), dtype="i") * i
        f.close()
        
    abs_mag = np.concatenate(abs_mag)
    filenum = np.concatenate(filenum)
    
    volume = box_size**3
    magnitude_new = lf.rescale_magnitude_to_target_box(abs_mag, zsnap, volume,
                                    cosmo_orig=cosmo_orig, cosmo_new=cosmo_new)
    
    for i in range(Nfiles):
        print(i)
        keep = filenum==i
        f = h5py.File(filename%i,"a")
        f.create_dataset(mag_dataset, data=magnitude_new[keep], compression="gzip")
        f.close()

        
def rescale_lightcone_magnitudes(filename_resolved, filename_unresolved, 
                                 zmax, zsnap, cosmo_orig, cosmo_new, Nfiles=34,   
                                 mag_dataset="abs_mag_rescaled"):
    """
    Rescale the magnitudes in the cubic box mock to match target LF exactly
    Args:
        filename:     string, containing the path of the input galaxy mock
        box_size:     float, simulation box size (Mpc/h)
        zsnap:        integer, the redshift of the snapshot
        cosmo_orig:   object of class hodpy.cosmology.Cosmology, the cosmology of the 
                            original simulation
        cosmo_new:    object of class hodpy.cosmology.Cosmology, the cosmology of the
                            new simultion
        Nfiles:       Number of AbacusSummit files for this snapshot. Default is 34
        mag_dataset:  string, name of the dataset of absolute magnitudes to read from
                                the input mock file
    """
    # rescale the magnitudes of the snapshot to match target LF exactly
    
    lf = LuminosityFunctionTargetBGS(lookup.target_lf, lookup.sdss_lf_tabulated, 
                        lookup.gama_lf_fits, HOD_BGS_Simple(lookup.bgs_hod_parameters))
    
    # read the resolved and unresolved lightcone files
    abs_mag = [None]*Nfiles*2
    pos = [None]*Nfiles*2
    filenum = [None]*Nfiles*2
    is_res = [None]*Nfiles*2
    
    for i in range(Nfiles):
        print(i)
        # resolved files
        if exists(filename_resolved%i):
            f = h5py.File(filename_resolved%i,"r")
            abs_mag[i] = f["abs_mag"][...]
            pos[i] = f["pos"][...]
            filenum[i] = np.ones(len(abs_mag[i]), dtype="i") * i
            is_res[i] = np.ones(len(abs_mag[i]), dtype="bool")
            f.close()
        else:
            abs_mag[i] = np.zeros(0)
            pos[i] = np.zeros((0,3))
            filenum[i] = np.zeros(0, dtype="i")
            is_res[i] = np.ones(0, dtype="bool")
        
        # unresolved files
        if exists(filename_unresolved%i):
            f = h5py.File(filename_unresolved%i,"r")
            abs_mag[i+Nfiles] = f["abs_mag"][...]
            pos[i+Nfiles] = f["pos"][...]
            filenum[i+Nfiles] = np.ones(len(abs_mag[i+Nfiles]), dtype="i") * i
            is_res[i+Nfiles] = np.zeros(len(abs_mag[i+Nfiles]), dtype="bool")
            f.close()
        else:
            abs_mag[i+Nfiles] = np.zeros(0)
            pos[i+Nfiles] = np.zeros((0,3))
            filenum[i+Nfiles] = np.zeros(0, dtype="i")
            is_res[i+Nfiles] = np.zeros(0, dtype="bool")
        
    abs_mag = np.concatenate(abs_mag)
    filenum = np.concatenate(filenum)
    pos = np.concatenate(pos)
    is_res = np.concatenate(is_res)
    
    # do the magnitude rescaling in shells of comoving distance
    # these shells match up with the shells of unresolved haloes
    r = np.sum(pos**2, axis=1)**0.5
    rmax = cosmo_new.comoving_distance(zmax)
    r_bins = np.arange(25,rmax+25,25)
    r_bins[0]=0
    r_bins[-1] = rmax

    magnitude_new = abs_mag.copy()
    for i in range(len(r_bins)-1):
        # volume of the spherical shell
        volume = 4./3 * np.pi * ((r_bins[i+1])**3 - (r_bins[i])**3)
    
        keep = np.logical_and(r>=r_bins[i], r<r_bins[i+1])
        magnitude_new[keep] = lf.rescale_magnitude_to_target_box(abs_mag[keep], zsnap, volume,
                                    cosmo_orig=cosmo_orig, cosmo_new=cosmo_new)
    
    
    # save new magnitudes to files
    for i in range(Nfiles):
        print(i)
        keep = np.logical_and(filenum==i, is_res)
        if np.count_nonzero(keep) > 0:
            f = h5py.File(filename_resolved%i,"a")
            f.create_dataset(mag_dataset, data=magnitude_new[keep], compression="gzip")
            f.close()
            
        keep = np.logical_and(filenum==i, np.invert(is_res))
        if np.count_nonzero(keep) > 0:
            f = h5py.File(filename_unresolved%i,"a")
            f.create_dataset(mag_dataset, data=magnitude_new[keep], compression="gzip")
            f.close()
    
    
    
    
def read_dataset_cut_sky_rep(filename, dataset, n=2):
    """
    Read a dataset from the cut-sky mock files, with periodic replictions
    """
    length=[1,7,19,27][n]
    data = [None]*length
    
    f = h5py.File(filename,"r")
    
    idx=0
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                
                if abs(i)+abs(j)+abs(k) > n: continue
                
                data[idx] = f["%i%i%i/%s"%(i,j,k,dataset)][...]
                idx+=1
    f.close()
    
    data = np.concatenate(data)
    
    return data


def read_dataset_cut_sky(filename, dataset, dtype="f"):
    """
    Read a dataset from the cut-sky mock files, with no replications
    """
    if exists(filename):
        f = h5py.File(filename,"r")
        data = f[dataset][...]
        f.close()
    else:
        data = np.zeros(0,dtype=dtype)
        
    return data