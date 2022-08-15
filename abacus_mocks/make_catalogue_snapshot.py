#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import gc
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.read_abacus import read_asdf
import h5py
import fitsio
from os.path import exists

from cut_sky_evolution import cut_sky

from hodpy.halo_catalogue import AbacusSnapshot, AbacusSnapshotUnresolved
from hodpy.galaxy_catalogue_snapshot import GalaxyCatalogueSnapshot
from hodpy.hod_bgs_snapshot_abacus import HOD_BGS
from hodpy.colour import ColourNew
from hodpy import lookup
from hodpy.mass_function import MassFunction
from hodpy.k_correction import GAMA_KCorrection
from hodpy.luminosity_function import LuminosityFunctionTargetBGS
from hodpy.hod_bgs import HOD_BGS_Simple


def main(input_file, output_file, snapshot_redshift, mag_faint, cosmology, 
         hod_param_file, central_lookup_file, satellite_lookup_file,
         mass_function, box_size=2000., zmax=None, observer=(0,0,0),
         log_mass_min=None, log_mass_max=None, cosmology_old=None,
         replication=(0,0,0)):
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
        mass_function:     object of class hodpy.mass_function.MassFunction
        box_size:          float, simulation box size (Mpc/h)
        zmax:              float, maximum redshift. If provided, will cut the box to only haloes
                                that are within a comoving distance to the observer that 
                                corresponds to zmax. By default, is None
        observer:          3D position vector of the observer, in units Mpc/h. By default 
                                observer is at the origin (0,0,0) Mpc/h
        log_mass_min:      float, log10 of minimum halo mass cut, in Msun/h
        log_mass_max:      float, log10 of maximum halo mass cut, in Msun/h
        cosmology_old:     the original cosmology, used when applying cosmology rescaling
        replication:       tuple of length 3 indicating the replication, (i,j,k), where the positions
                           in the x direction are shifted by i*box_size, etc. By default (0,0,0)
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
        #if apply_periodic==True:
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        
        # replicate the box
        pos[:,0] += replication[0]*box_size
        pos[:,1] += replication[1]*box_size
        pos[:,2] += replication[2]*box_size
        
        halo_cat.add("pos", pos)
        
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
                  replace_central_lookup=True, replace_satellite_lookup=True, 
                  mass_function=mass_function)
    
    print("add galaxies")
    gal_cat.add_galaxies(hod)
    
    # position galaxies around their haloes
    print("position galaxies")
    gal_cat.position_galaxies(particles=False, conc="conc")
    
    # add g-r colours
    print("assigning g-r colours")
    col = ColourNew(hod=hod)
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
        for i in range(3):
            pos[:,i] += observer[i]
        
        gal_cat.add("pos", pos)
        
    else:
        pos = gal_cat.get("pos")
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        gal_cat.add("pos", pos)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])
    
    
    
def main_unresolved(input_file, output_file, snapshot_redshift, mag_faint, 
                    cosmology, hod_param_file, central_lookup_file, 
                    satellite_lookup_file, mass_function, box_size=2000., SODensity=200,
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
        mass_function:     object of class hodpy.mass_function.MassFunction
        box_size:          float, simulation box size (Mpc/h)
        SODensity:         spherical overdensity
        zmax:              float, maximum redshift. If provided, will cut the box to only haloes
                                that are within a comoving distance to the observer that 
                                corresponds to zmax. By default, is None
        observer:          3D position vector of the observer, in units Mpc/h. By default 
                                observer is at the origin (0,0,0) Mpc/h
        log_mass_min:      float, log10 of minimum halo mass cut, in Msun/h
        log_mass_max:      float, log10 of maximum halo mass cut, in Msun/h
        cosmology_old:     the original cosmology, used when applying cosmology rescaling
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
        
        halo_cat.add("pos", pos)
        
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
                  replace_central_lookup=True, replace_satellite_lookup=True,
                  mass_function=mass_function)
    
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
    col = ColourNew(hod=hod)
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
        for i in range(3):
            pos[:,i] += observer[i]
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
    
    
    
def replications(box_size, rmax):
    """
    returns number of periodic replications needed. 
    0 = no replications, 1 box total
    1 = replicate at each of the 6 faces, 7 boxes in total
    2 = replicate at faces and edges, 19 boxes in total
    3 = replicate at faces, edges and corners, 27 boxes in total

    Args:
        box_size: Simulation box size, in Mpc/h
        rmax:     Comoving distance in Mpc/h corresponding to the maximum redshift of the lightcone
    """

    if rmax < box_size/2.: return 0
    
    for i in range(100):
        if rmax >= (box_size*(2*i+1))/2.:
            n = 1+i*3
        if rmax >= np.sqrt(2)*(box_size*(2*i+1))/2.:
            n = 2+i*3
        if rmax >= np.sqrt(3)*(box_size*(2*i+1))/2.:
            n = 3+i*3

    return n
    

def num_in_rmax(p, rmax, box_size):
    '''
    Return number of particles within a cube of side length 2*rmax, with observer in
    centre, applying the necessary number of periodic replications

    Args:
        p:        3d position vectors of particles
        rmax:     Comoving distance in Mpc/h corresponding to the maximum redshift of the lightcone
        box_size: Simulation box size, in Mpc/h
    '''

    nrep = 0
    for i in range(100):
        if rmax >= (box_size*(2*i+1))/2.:
            nrep = i+1
        else: 
            break
            
    number=0
    
    for i in range(-nrep, nrep+1):
        for j in range(-nrep, nrep+1):
            for k in range(-nrep, nrep+1):
                p_i = p.copy()
                p_i[:,0] += box_size*i
                p_i[:,1] += box_size*j
                p_i[:,2] += box_size*k
                
                keep = np.logical_and.reduce([p_i[:,0] <= rmax, p_i[:,0] >= -rmax,
                                              p_i[:,1] <= rmax, p_i[:,1] >= -rmax,
                                              p_i[:,2] <= rmax, p_i[:,2] >= -rmax])
                
                number += np.count_nonzero(keep)
                
    return number
    
    
def num_in_shell(p, rmin, rmax, box_size=2000):
    
    '''
    Return number of particles within a shell rmin < r < rmax with observer at
    the origin, and applying the necessary number of periodic replications

    Args:
        p:        3d position vectors of particles
        rmin:     inner radius of shell, in Mpc/h
        rmax:     outer radius of shell, in Mpc/h
        box_size: Simulation box size, in Mpc/h
    '''
  
    nrep = 0
    for i in range(100):
        if rmax >= (box_size*(2*i+1))/2.:
            nrep = i+1
        else: 
            break
            
    number=0
    
    rmin2 = rmin**2
    rmax2 = rmax**2
    
    for i in range(-nrep, nrep+1):
        for j in range(-nrep, nrep+1):
            for k in range(-nrep, nrep+1):
                p_i = p.copy()
                p_i[:,0] += box_size*i
                p_i[:,1] += box_size*j
                p_i[:,2] += box_size*k
                
                dist2 = np.sum(p_i**2, axis=1)
                
                keep = np.logical_and(dist2>=rmin2, dist2<rmax2)
                
                number += np.count_nonzero(keep)
                
    return number



def particles_in_shell(pos, vel, box_size, rmin, rmax):
    '''
    Cuts to the particles in a shell rmin < r < rmax, with observer at
    the origin, and applying the necessary number of periodic replications

    Args:
        pos:      3d position vectors of particles
        vel:      3d velocity vectors of particles
        box_size: Simulation box size, in Mpc/h
        rmin:     inner radius of shell, in Mpc/h
        rmax:     outer radius of shell, in Mpc/h
    Returns:
        pos_shell: 3d position vectors of particles in the shell
        vel_shell: 3d velocity vectors of particles in the shell
    '''
    nrep = 0
    for i in range(100):
        if rmax >= (box_size*(2*i+1))/2.:
            nrep = i+1
        else: 
            break
            
    number=0
    
    rmin2 = rmin**2
    rmax2 = rmax**2
    
    pos_shell = [None]*(nrep*2+1)**3
    vel_shell = [None]*(nrep*2+1)**3
    idx=0
    
    for i in range(-nrep, nrep+1):
        for j in range(-nrep, nrep+1):
            for k in range(-nrep, nrep+1):
                p_i = pos.copy()
                p_i[:,0] += box_size*i
                p_i[:,1] += box_size*j
                p_i[:,2] += box_size*k
                
                dist2 = np.sum(p_i**2, axis=1)
                
                keep = np.logical_and(dist2>=rmin2, dist2<rmax2)
                
                if np.count_nonzero(keep) > 0:
                    pos_shell[idx] = p_i[keep]
                    vel_shell[idx] = vel[keep]
                else:
                    pos_shell[idx] = np.zeros((0,3))
                    vel_shell[idx] = np.zeros((0,3))
                idx += 1

    pos_shell = np.concatenate(pos_shell)
    vel_shell = np.concatenate(vel_shell)
    
    return pos_shell, vel_shell
            
            
    
def halo_lightcone_unresolved(output_file, abacus_path, snapshot_redshift, cosmology, hod_param_file, 
                              central_lookup_file, satellite_lookup_file, mass_function,
                              Nparticle, Nparticle_shell, box_size=2000., SODensity=200,
                              simulation="base", cosmo=0, ph=0, observer=(0,0,0), 
                              app_mag_faint=20.25, cosmology_orig=None, Nfiles=34):
    """
    Create a lightcone of unresolved AbacusSummit haloes, using the field particles 
    (not in haloes) as tracers. The output galaxy catalogue is in Cartesian coordinates
    
    Args:
        output_file:       string, containing the path of hdf5 file to save outputs
        abacus_path:       path to the abacus mocks    
        snapshot_redshift: integer, the redshift of the snapshot
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        hod_param_file:    string, path to file containing HOD hyperparameter fits
        central_lookup_file: lookup file of central magnitudes, will be created if the file
                                doesn't already exist
        satellite_lookup_file: lookup file of satellite magnitudes, will be created if the file
                                doesn't already exist
        mass_function:     object of class hodpy.mass_function.MassFunction
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
    
    mock = "AbacusSummit_%s_c%03d_ph%03d"%(simulation, cosmo, ph)
    mf = mass_function
    
    # read HOD files
    mag_faint=-10
    hod = HOD_BGS(cosmology, mag_faint, hod_param_file, central_lookup_file, satellite_lookup_file,
                  replace_central_lookup=True, replace_satellite_lookup=True)
    
    # get min mass
    # rcom bins go up to 1000 Mpc/h
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
            file_name = abacus_path+mock+"/halos/z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(snapshot_redshift, file_number)
            data = read_asdf(file_name, load_pos=True, load_vel=False)
            p = data["pos"]
            for i in range(3):
                p[:,i] -= observer[i]
            p[p>box_size/2.] -= box_size
            p[p<-box_size/2.] += box_size
            del data
            
            rmax = 1000
            N[file_number] = num_in_rmax(p, rmax, box_size)
    
            for j in range(len(rcom)):
                Nshells[file_number,j] = num_in_shell(p, rmin=rcom[j]-25, rmax=rcom[j], box_size=box_size)
                print(file_number, j, Nshells[file_number,j])
            
            gc.collect() # need to run garbage collection to release memory
        
        # save files
        np.savetxt(Nparticle, N)
        np.savetxt(Nparticle_shell, Nshells)
    
    
    # Now make lightcone of unresolved haloes
    for file_number in range(Nfiles):
        file_name = abacus_path+mock+"/halos/z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(snapshot_redshift, file_number)
        print(file_name)

        # read file
        data = read_asdf(file_name, load_pos=True, load_vel=True)
        vel = data["vel"]
        pos = data["pos"]
        for i in range(3):
            pos[:,i] -= observer[i]
        pos[pos>box_size/2.] -= box_size
        pos[pos<-box_size/2.] += box_size
        
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
            
            pos_shell, vel_shell = particles_in_shell(pos, vel, box_size, 
                                                      rmin=rmin_bin, rmax=rmax_bin)
            N_shell = pos_shell.shape[0]
            
            print(file_number, j, N_shell)
            
            if N_shell==0: 
                pos_bins[j] = np.zeros((0,3))
                vel_bins[j] = np.zeros((0,3))
                mass_bins[j] = np.zeros(0)
                continue
                
            try:
                Npar =  np.sum(Nshells[:,j]) # total number of field particles in shell
            except:
                Npar =  Nshells[j]
            
            # number of randoms to generate in shell
            Nrand = mf.number_density_in_mass_bin(logMmin_bin, logMmax_bin) * vol_bin
            
            if Nrand==0: 
                pos_bins[j] = np.zeros((0,3))
                vel_bins[j] = np.zeros((0,3))
                mass_bins[j] = np.zeros(0)
                continue
            
            
            # probability to keep a particle
            prob = Nrand*1.0 / Npar
            
            keep = np.random.rand(N_shell) <= prob
            pos_bins[j] = pos_shell[keep]
            vel_bins[j] = vel_shell[keep]

            # generate random masses if number of particles > 0
            if np.count_nonzero(keep)>0:
                mass_bins[j] = 10**mf.get_random_masses(np.count_nonzero(keep), logMmin_bin, logMmax_bin) / 1e10
            else:
                mass_bins[j] = np.zeros(0)
            
        del data
        gc.collect() # need to run garbage collection to release memory
        
        pos_bins = np.concatenate(pos_bins)
        vel_bins = np.concatenate(vel_bins)
        mass_bins = np.concatenate(mass_bins)
        
        # shift positions back
        for i in range(3):
            pos_bins[:,i] += observer[i]
        
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
                   cosmology, hod, box_size=2000., observer=(0,0,0),
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
        hod:               object of class hodpy.hod_bgs_snapshot_abacus.HOD_BGS
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
    n = replications(box_size, rmax)
    
    # loop through periodic replications
    for i in range(-5,6):
        for j in range(-5,6):
            for k in range(-5,6):
                
                rmin = 0
                if abs(i) + abs(j) + abs(k) > 0:
                    rmin = ((abs(i)-0.5)**2 + (abs(j)-0.5)**2 + (abs(k)-0.5)**2)**0.5 * box_size
                if rmax<rmin: continue
                
                rep = (i,j,k)
                print(rep)
    
                ra, dec, zcos, zobs, magnitude_new, app_mag, col_new, col_obs, index = \
                        cut_sky(pos, vel, abs_mag, is_cen, cosmology, Lbox=box_size, 
                            zsnap=snapshot_redshift, kcorr_r=kcorr_r, kcorr_g=kcorr_g, 
                            hod=hod, replication=rep, 
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
                        snapshot_redshift, mag_faint, cosmology, hod, box_size=2000., 
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
        hod:               object of class hodpy.hod_bgs_snapshot_abacus.HOD_BGS
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

    kcorr_r = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_rband_z01.dat", 
                               cubic_interpolation=True)
    kcorr_g = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_gband_z01.dat", 
                               cubic_interpolation=True)
    
    # don't need to apply multiple replications
    ra, dec, zcos, zobs, magnitude_new, app_mag, col_new, col_obs, index = \
            cut_sky(pos, vel, abs_mag, is_cen, cosmology, Lbox=box_size, 
                    zsnap=snapshot_redshift, kcorr_r=kcorr_r, kcorr_g=kcorr_g, 
                    hod=hod, replication=(0,0,0), zcut=zmax, mag_cut=mag_faint, 
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
                                 mag_dataset="abs_mag_rescaled", observer=(0,0,0)):
    """
    Rescale the magnitudes in the lightcone mock to match target LF exactly
    Args:
        filename_resolved:     string, containing the path of the input resolved galaxy lightcone
        filename_unresolved:     string, containing the path of the input unresolved galaxy lightcone
        zmax:
        zsnap:        integer, the redshift of the snapshot
        cosmo_orig:   object of class hodpy.cosmology.Cosmology, the cosmology of the 
                            original simulation
        cosmo_new:    object of class hodpy.cosmology.Cosmology, the cosmology of the
                            new simultion
        Nfiles:       Number of AbacusSummit files for this snapshot. Default is 34
        mag_dataset:  string, name of the dataset of absolute magnitudes to read from
                                the input mock file
        observer:     observer position, default (0,0,0)
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
            p = f["pos"][...]
            for j in range(3):
                p[:,j] = p[:,j] - observer[j]
            pos[i] = p 
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
            p = f["pos"][...]
            for j in range(3):
                p[:,j] = p[:,j] - observer[j]
            pos[i+Nfiles] = p
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
    
    
    
def read_dataset_cut_sky_rep(filename, dataset):
    """
    Read a dataset from the cut-sky mock files, with periodic replictions
    """

    length = 1331
    data = [None]*length
    
    f = h5py.File(filename,"r")
    
    idx=0
    for i in range(-5,6):
        for j in range(-5,6):
            for k in range(-5,6):
                
                try:
                    data[idx] = f["%i%i%i/%s"%(i,j,k,dataset)][...]
                except:
                    data[idx] = np.zeros(0)
                    
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



def merge_galaxy_lightcone_res(filename):
    '''
    Merge the periodic replications for the low redshift resolved lightcone
    '''
    
    Nfiles = 27
    
    abs_mag = [None]*Nfiles
    cen_ind = [None]*Nfiles
    col     = [None]*Nfiles
    halo_ind = [None]*Nfiles
    halo_mass = [None]*Nfiles
    is_cen  = [None]*Nfiles
    pos     = [None]*Nfiles
    vel     = [None]*Nfiles
    zcos    = [None]*Nfiles
    
    idx=0
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            for k in range(-1,2,1):

                try:
                    f = h5py.File(filename+"%i%i%i"%(i,j,k),"r")
                    abs_mag[idx]   = f["abs_mag"][...]
                    cen_ind[idx]   = f["cen_ind"][...]
                    col[idx]       = f["col"][...]
                    halo_ind[idx] = f["halo_ind"][...]
                    halo_mass[idx] = f["halo_mass"][...]
                    is_cen[idx]    = f["is_cen"][...]
                    pos[idx]       = f["pos"][...]
                    vel[idx]       = f["vel"][...]
                    zcos[idx]       = f["zcos"][...]
                    f.close()
                    
                except:
                    abs_mag[idx]   = np.zeros(0, dtype='f')
                    cen_ind[idx]   = np.zeros(0, dtype='i')
                    col[idx]       = np.zeros(0, dtype='f')
                    halo_ind[idx]  = np.zeros(0, dtype='i')
                    halo_mass[idx] = np.zeros(0, dtype='f')
                    is_cen[idx]    = np.zeros(0, dtype='bool')
                    pos[idx]       = np.zeros((0,3), dtype='f')
                    vel[idx]       = np.zeros((0,3), dtype='f')
                    zcos[idx]      = np.zeros(0, dtype='f')
                
                idx += 1

    abs_mag = np.concatenate(abs_mag)
    cen_ind = np.concatenate(cen_ind)
    col = np.concatenate(col)
    halo_ind = np.concatenate(halo_ind)
    halo_mass = np.concatenate(halo_mass)
    is_cen = np.concatenate(is_cen)
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    zcos = np.concatenate(zcos)
    
    
    f = h5py.File(filename, "a")
    f.create_dataset("abs_mag", data=abs_mag, compression="gzip")
    f.create_dataset("cen_ind", data=cen_ind, compression="gzip")
    f.create_dataset("col", data=col, compression="gzip")
    f.create_dataset("halo_ind", data=halo_ind, compression="gzip")
    f.create_dataset("halo_mass", data=halo_mass, compression="gzip")
    f.create_dataset("is_cen", data=is_cen, compression="gzip")
    f.create_dataset("pos", data=pos, compression="gzip")
    f.create_dataset("vel", data=vel, compression="gzip")
    f.create_dataset("zcos", data=zcos, compression="gzip")
    f.close()


def merge_box(output_path, galaxy_snapshot_file, output_path_final, galaxy_snapshot_final,
             Nfiles, fmt="fits", offset=0):
    '''
    Merge the cubic box files
    '''
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
        pos[i]       = f["pos"][...] + offset
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

    if fmt=="hdf5":
        f = h5py.File(output_path_final+galaxy_snapshot_final,"a")
        f.create_dataset("Data/abs_mag", data=abs_mag, compression="gzip")
        f.create_dataset("Data/g_r", data=col, compression="gzip")
        f.create_dataset("Data/halo_mass", data=halo_mass/1e10, compression="gzip")
        f.create_dataset("Data/galaxy_type", data=gtype, compression="gzip")
        f.create_dataset("Data/pos", data=pos, compression="gzip")
        f.create_dataset("Data/vel", data=vel, compression="gzip")
        f.close()
        
    elif fmt=="fits":
        
        N = len(abs_mag)
        
        data_fits = np.zeros(N, dtype=[('R_MAG_ABS', 'f4'), ('G_R_REST', 'f4'), 
                                       ('HALO_MASS', 'f4'), ('cen', 'i4'), 
                               ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4')])

        data_fits['R_MAG_ABS']   = abs_mag
        del abs_mag
        data_fits['G_R_REST']    = col
        del col
        data_fits['HALO_MASS']   = halo_mass/1e10
        del halo_mass
        data_fits['cen']         = 1 - gtype
        del gtype
        data_fits['x']           = pos[:,0]
        data_fits['y']           = pos[:,1]
        data_fits['z']           = pos[:,2]
        del pos
        data_fits['vx']           = vel[:,0]
        data_fits['vy']           = vel[:,1]
        data_fits['vz']           = vel[:,2]
        del vel
        
        fits = fitsio.FITS(output_path_final+galaxy_snapshot_final, "rw")
        fits.write(data_fits)
        fits.close()




def merge_lightcone(output_path, galaxy_cutsky, galaxy_cutsky_low, 
                    output_path_final, galaxy_cutsky_final, Nfiles, 
                    zmax_low, app_mag_faint, fmt='fits'):

    '''
    Merge the low and high redshift lightcones
    '''
    
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

        zobs[file_number] = read_dataset_cut_sky_rep(filename1, "zobs")
        zobs[file_number+Nfiles] = read_dataset_cut_sky(filename2, "zobs")

        app_mag[file_number] = read_dataset_cut_sky_rep(filename1, "app_mag")
        app_mag[file_number+Nfiles] = read_dataset_cut_sky(filename2, "app_mag")

        keep1 = np.logical_and(zobs[file_number] > zmax_low, app_mag[file_number] <=app_mag_faint)
        keep2 = np.logical_and(zobs[file_number+Nfiles] <= zmax_low, app_mag[file_number+Nfiles]<=app_mag_faint)

        zobs[file_number]   = zobs[file_number][keep1]
        zobs[file_number+Nfiles] = zobs[file_number+Nfiles][keep2]

        app_mag[file_number]   = app_mag[file_number][keep1]
        app_mag[file_number+Nfiles] = app_mag[file_number+Nfiles][keep2]

        abs_mag[file_number] = read_dataset_cut_sky_rep(filename1, "abs_mag")[keep1]
        abs_mag[file_number+Nfiles] = read_dataset_cut_sky(filename2, "abs_mag")[keep2]

        col[file_number] = read_dataset_cut_sky_rep(filename1, "col")[keep1]
        col[file_number+Nfiles] = read_dataset_cut_sky(filename2, "col")[keep2]

        col_obs[file_number] = read_dataset_cut_sky_rep(filename1, "col_obs")[keep1]
        col_obs[file_number+Nfiles] = read_dataset_cut_sky(filename2, "col_obs")[keep2]

        dec[file_number] = read_dataset_cut_sky_rep(filename1, "dec")[keep1]
        dec[file_number+Nfiles] = read_dataset_cut_sky(filename2, "dec")[keep2]

        halo_mass[file_number] = read_dataset_cut_sky_rep(filename1, "halo_mass")[keep1]
        halo_mass[file_number+Nfiles] = read_dataset_cut_sky(filename2, "halo_mass")[keep2]

        is_cen[file_number] = read_dataset_cut_sky_rep(filename1, "is_cen")[keep1]
        is_cen[file_number+Nfiles] = read_dataset_cut_sky(filename2, "is_cen", dtype="bool")[keep2]

        is_res[file_number] = read_dataset_cut_sky_rep(filename1, "is_res")[keep1]
        is_res[file_number+Nfiles] = read_dataset_cut_sky(filename2, "is_res", dtype="bool")[keep2]

        ra[file_number] = read_dataset_cut_sky_rep(filename1, "ra")[keep1]
        ra[file_number+Nfiles] = read_dataset_cut_sky(filename2, "ra")[keep2]

        zcos[file_number] = read_dataset_cut_sky_rep(filename1, "zcos")[keep1]
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

    cen = np.array(is_cen, dtype="i")
    res = np.array(is_res, dtype="i")
    
    if fmt=="hdf5":
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

    elif fmt=="fits":

        N = len(ra)

        hdict = {'SV3_AREA': 207.5, 'Y5_AREA':14850.4}

        data_fits = np.zeros(N, dtype=[('R_MAG_APP', 'f4'), ('R_MAG_ABS', 'f4'),
                                   ('G_R_REST', 'f4'), ('G_R_OBS', 'f4'),
                                   ('DEC', 'f4'), ('HALO_MASS', 'f4'),
                                   ('CEN', 'i4'), ('RES', 'i4'), ('RA', 'f4'),  
                                   ('Z_COSMO', 'f4'), ('Z', 'f4'),
                                   ('STATUS', 'i4')])

        data_fits['R_MAG_APP']   = app_mag
        data_fits['R_MAG_ABS']   = abs_mag
        data_fits['G_R_REST']    = col
        data_fits['G_R_OBS']     = col_obs
        data_fits['DEC']         = dec
        data_fits['HALO_MASS']   = halo_mass/1e10
        #data_fits['GALAXY_TYPE'] = gtype
        data_fits['CEN']         = cen
        data_fits['RES']         = res
        data_fits['RA']          = ra
        data_fits['Z_COSMO']     = zcos
        data_fits['Z']           = zobs

        fits = fitsio.FITS(output_path_final+galaxy_cutsky_final, "rw")
        fits.write(data_fits, header=hdict)
        fits.close()




