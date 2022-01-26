#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import gc
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.read_abacus import read_asdf
import h5py

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
        zmax=None, observer=(0,0,0), log_mass_min=None, log_mass_max=None):

    import warnings
    warnings.filterwarnings("ignore")
    
    # create halo catalogue
    print("read halo catalogue")
    halo_cat = AbacusSnapshot(input_file, snapshot_redshift, cosmology=cosmology, 
                              box_size=box_size, particles=False, clean=True)

    
    # apply cuts to halo mass
    if not log_mass_min is None or not log_mass_max is None:
        log_mass = halo_cat.get("log_mass")
        
        keep = np.ones(len(log_mass), dtype="bool")
        
        if not log_mass_min is None:
            keep = np.logical_and(keep, log_mass >= log_mass_min)
            
        if not log_mass_max is None:
            keep = np.logical_and(keep, log_mass <= log_mass_max)
            
        halo_cat.cut(keep)
    
    # if lightcone
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
    col = ColourNew()
    gal_cat.add_colours(col)

    # cut to galaxies brighter than absolute magnitude threshold
    gal_cat.cut(gal_cat.get("abs_mag") <= mag_faint)
    
    if not zmax is None:
        #shift back to original coordinates
        pos = gal_cat.get("pos")
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        gal_cat.add("pos", pos)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])
    
    
def main_unresolved(input_file, output_file, snapshot_redshift, mag_faint, 
                    cosmology, hod_param_file, central_lookup_file, 
                    satellite_lookup_file, box_size=2000., SODensity=200,
                    zmax=0.6, observer=(0,0,0), log_mass_min=None, log_mass_max=None):

    import warnings
    warnings.filterwarnings("ignore")
    
    # create halo catalogue
    print("read halo catalogue")
    halo_cat = AbacusSnapshotUnresolved(input_file, snapshot_redshift, cosmology=cosmology,
                                        box_size=box_size, SODensity=SODensity)

    # apply cuts to halo mass
    if not log_mass_min is None or not log_mass_max is None:
        log_mass = halo_cat.get("log_mass")
        
        keep = np.ones(len(log_mass), dtype="bool")
        
        if not log_mass_min is None:
            keep = np.logical_and(keep, log_mass >= log_mass_min)
            
        if not log_mass_max is None:
            keep = np.logical_and(keep, log_mass <= log_mass_max)
            
        halo_cat.cut(keep)
    
    # if zmax is set, lightcone
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
    col = ColourNew()
    gal_cat.add_colours(col)

    # cut to galaxies brighter than absolute magnitude threshold
    gal_cat.cut(gal_cat.get("abs_mag") <= mag_faint)
    
    if not zmax is None:
        #shift back to original coordinates
        pos = gal_cat.get("pos")
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        gal_cat.add("pos", pos)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])
    
    
    
def get_mass_function(input_file, fit_file, redshift=0.2, box_size=2000, cosmology=None):
    """
    Get smooth fit to the mass function of an Abacus snapshot
    Args:
        clean:       use cleaned Abacus halo catalogue? Default is True
        redshift:    snapshot redshift. Default z=0.2
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        abacus_cosmologies_file: file of Abacus cosmological parameters
    """
    
    file_name = input_file
    
    try:
        fit_params = np.loadtxt(fit_file)
        
        mf = MassFunction(cosmology=cosmology, redshift=redshift, 
                      fit_params=fit_params)

    except:
        # loop through all 34 files, reading in halo masses
        log_mass = [None]*34
        for file_number in range(34):
            input_file = file_name%(redshift, file_number)

            halo_cat = CompaSOHaloCatalog(input_file, cleaned=True, fields=['N'])
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
        
        np.savetxt(fit_file, fit_params)
    
    return mf
    
    
def get_min_mass(rfaint, hod, cosmo_orig, cosmo_new):
    
    rcom = np.arange(25,1001,25)
    z = cosmo_new.redshift(rcom)
    
    kcorr = GAMA_KCorrection(cosmo_new)
    lf = LuminosityFunctionTargetBGS(lookup.target_lf, lookup.sdss_lf_tabulated, 
                        lookup.gama_lf_fits, HOD_BGS_Simple(lookup.bgs_hod_parameters))

    mag_faint1 = kcorr.absolute_magnitude(np.ones(len(z))*rfaint, z, np.ones(len(z))*-10)
    mag_faint2 = kcorr.absolute_magnitude(np.ones(len(z))*rfaint, z, np.ones(len(z))*10)
    mag_faint = np.maximum(mag_faint1, mag_faint2)
    
    mags = np.arange(-23,-8,0.01)
    mags_unscaled = np.zeros(len(z))

    for i in range(len(z)):

        mags_new = lf.rescale_magnitude(mags, np.ones(len(mags))*0.2, np.ones(len(mags))*z[i],
                                       cosmo_orig, cosmo_new)
        idx = np.where(mags_new>=mag_faint[i])[0][0]
        mags_unscaled[i] = mags[idx]
        
    log_mass = np.arange(9,15,0.001)

    mass_limit = np.zeros(len(z))

    for i in range(len(z)):
        N = hod.number_centrals_mean(log_mass,np.ones(len(log_mass))*mags_unscaled[i])
        idx = np.where(N>0)[0][0]
        mass_limit[i] = log_mass[idx]
        
    for i in range(len(mass_limit)):
        mass_limit[i] = np.min(mass_limit[i:])
        
    return rcom, mass_limit
    
    
    
def halo_lightcone_unresolved(output_file, snapshot_redshift, 
                              cosmology, hod_param_file, central_lookup_file, 
                              satellite_lookup_file, mf_fit_file,
                              Nparticle, Nparticle_shell,
                              box_size=2000., SODensity=200,
                              simulation="base", cosmo=0, ph=0, 
                              observer=(0,0,0), app_mag_faint=20.25,
                              cosmology_orig=None):

    import warnings
    warnings.filterwarnings("ignore")
    
    # get fit to halo mass function
    print("getting fit to halo mass function")
    path = "/global/cfs/cdirs/desi/cosmosim/Abacus/"
    mock = "AbacusSummit_%s_c%03d_ph%03d"%(simulation, cosmo, ph)
    input_file = path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"
    mf = get_mass_function(input_file, mf_fit_file, redshift=snapshot_redshift, 
                           box_size=box_size, cosmology=cosmology)
    
    # read HOD files
    mag_faint=-10
    hod = HOD_BGS(cosmology, mag_faint, hod_param_file, central_lookup_file, satellite_lookup_file,
                  replace_central_lookup=True, replace_satellite_lookup=True)
    
    # get min mass
    rcom, logMmin = get_min_mass(app_mag_faint, hod, cosmology_orig, cosmology)
    
    
    # get total number of field particles (using A particles)
    
    try:
        print("Reading total number of field particles")
        N = np.loadtxt(Nparticle)
        Nshells = np.loadtxt(Nparticle_shell)
    
    except:
        print("File doesn't exist yet, finding number of field particles")
        N = np.zeros(34, dtype="i")
        Nshells = np.zeros((34,len(rcom)),dtype="i")
        for file_number in range(34):
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
        
        np.savetxt(Nparticle, N)
        np.savetxt(Nparticle_shell, Nshells)
    
    
    for file_number in range(34):
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
        ##pos = pos + box_size/2.
        
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
                
            #pos_bin = pos[keep_shell]
            #vel_bin = vel[keep_shell]
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
        
        print(mass_bins)
        print(np.min(mass_bins), np.max(mass_bins))
        
        f = h5py.File(output_file%file_number, "a")
        f.create_dataset("mass", data=mass_bins, compression="gzip")
        f.create_dataset("position", data=pos_bins, compression="gzip")
        f.create_dataset("velocity", data=vel_bins, compression="gzip")
        f.close()

        
        
def read_galaxy_file(filename, resolved=True, mag_name="abs_mag"):
    
    try:
        f = h5py.File(filename, "r")
        abs_mag = f[mag_name][...]
        col = f["col"][...]
        pos = f["pos"][...]
        vel = f["vel"][...]
        is_cen = f["is_cen"][...]
        log_mass = np.log10(f["halo_mass"][...])
        is_res = np.ones(len(is_cen), dtype="bool") * resolved
        f.close()
    except:
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
                   zmax=0.6, cosmology_orig=None):
        
    pos, vel, log_mass, abs_mag, col, is_cen, is_res = \
                        read_galaxy_file(input_file, resolved=True, mag_name="abs_mag_rescaled")
    
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
                        observer=(0,0,0), zmax=0.15, cosmology_orig=None):
    
    # read resolved
    pos_r, vel_r, log_mass_r, abs_mag_r, col_r, is_cen_r, is_res_r = \
                        read_galaxy_file(resolved_file, resolved=True, mag_name="abs_mag_rescaled")
    
    # read unresolved
    pos_u, vel_u, log_mass_u, abs_mag_u, col_u, is_cen_u, is_res_u = \
                        read_galaxy_file(unresolved_file, resolved=False, mag_name="abs_mag_rescaled")
    
    # combine into single array
    pos      = np.concatenate([pos_r, pos_u])
    vel      = np.concatenate([vel_r, vel_u])
    log_mass = np.concatenate([log_mass_r, log_mass_u])
    abs_mag  = np.concatenate([abs_mag_r, abs_mag_u])
    col      = np.concatenate([col_r, col_u])
    is_cen   = np.concatenate([is_cen_r, is_cen_u])
    is_res   = np.concatenate([is_res_r, is_res_u])
    
    for i in range(3):
        pos[:,i] -= observer[i]
    pos[pos >  box_size/2.] -= box_size
    pos[pos < -box_size/2.] += box_size
    
    kcorr_r = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_rband_z01.dat", 
                               cubic_interpolation=True)
    kcorr_g = GAMA_KCorrection(cosmology, k_corr_file="lookup/k_corr_gband_z01.dat", 
                               cubic_interpolation=True)
    
    ra, dec, zcos, zobs, magnitude_new, app_mag, col_new, col_obs, index = \
            cut_sky(pos, vel, abs_mag, is_cen, cosmology, Lbox=box_size, 
                            zsnap=snapshot_redshift, kcorr_r=kcorr_r, kcorr_g=kcorr_g, 
                            replication=(0,0,0), 
                            zcut=zmax, mag_cut=mag_faint, cosmology_orig=cosmology_orig)

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


    
    
    
def rescale_snapshot_magnitudes(filename, box_size, zsnap, cosmo_orig, cosmo_new):
    
    # rescale the magnitudes of the snapshot to match target LF exactly
    
    lf = LuminosityFunctionTargetBGS(lookup.target_lf, lookup.sdss_lf_tabulated, 
                        lookup.gama_lf_fits, HOD_BGS_Simple(lookup.bgs_hod_parameters))
    
    abs_mag = [None]*34
    filenum = [None]*34
    
    for i in range(34):
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
    
    for i in range(34):
        print(i)
        keep = filenum==i
        f = h5py.File(filename%i,"a")
        f.create_dataset("abs_mag_rescaled", data=magnitude_new[keep], compression="gzip")
        f.close()

        
def rescale_lightcone_magnitudes(filename_resolved, filename_unresolved, 
                                 zmax, zsnap, cosmo_orig, cosmo_new):
    
    # rescale the magnitudes of the snapshot to match target LF exactly
    
    lf = LuminosityFunctionTargetBGS(lookup.target_lf, lookup.sdss_lf_tabulated, 
                        lookup.gama_lf_fits, HOD_BGS_Simple(lookup.bgs_hod_parameters))
    
    # read the resolved and unresolved lightcone files
    N = 34
    abs_mag = [None]*N*2
    pos = [None]*N*2
    filenum = [None]*N*2
    is_res = [None]*N*2
    
    for i in range(N):
        print(i)
        # resolved files
        try:
            f = h5py.File(filename_resolved%i,"r")
            abs_mag[i] = f["abs_mag"][...]
            pos[i] = f["pos"][...]
            filenum[i] = np.ones(len(abs_mag[i]), dtype="i") * i
            is_res[i] = np.ones(len(abs_mag[i]), dtype="bool")
            f.close()
        except:
            abs_mag[i] = np.zeros(0)
            pos[i] = np.zeros((0,3))
            filenum[i] = np.zeros(0, dtype="i")
            is_res[i] = np.ones(0, dtype="bool")
        
        # unresolved files
        try:
            f = h5py.File(filename_unresolved%i,"r")
            abs_mag[i+N] = f["abs_mag"][...]
            pos[i+N] = f["pos"][...]
            filenum[i+N] = np.ones(len(abs_mag[i+N]), dtype="i") * i
            is_res[i+N] = np.zeros(len(abs_mag[i+N]), dtype="bool")
            f.close()
        except:
            abs_mag[i+N] = np.zeros(0)
            pos[i+N] = np.zeros((0,3))
            filenum[i+N] = np.zeros(0, dtype="i")
            is_res[i+N] = np.zeros(0, dtype="bool")
        
    abs_mag = np.concatenate(abs_mag)
    filenum = np.concatenate(filenum)
    pos = np.concatenate(pos)
    is_res = np.concatenate(is_res)
    
    # print(len(abs_mag))
    # print(len(filenum))
    # print(len(is_res))
    # print(np.count_nonzero(is_res))
    # print(np.count_nonzero(is_res==False))

    
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
    
    ###return r, abs_mag, magnitude_new, is_res
    
    # save new magnitudes to files
    for i in range(34):
        print(i)
        keep = np.logical_and(filenum==i, is_res)
        if np.count_nonzero(keep) > 0:
            f = h5py.File(filename_resolved%i,"a")
            f.create_dataset("abs_mag_rescaled", data=magnitude_new[keep], compression="gzip")
            f.close()
            
        keep = np.logical_and(filenum==i, np.invert(is_res))
        if np.count_nonzero(keep) > 0:
            f = h5py.File(filename_unresolved%i,"a")
            f.create_dataset("abs_mag_rescaled", data=magnitude_new[keep], compression="gzip")
            f.close()
    
    
    
    
    
    
    
    
    
    
    
    
def read_dataset_cut_sky_rep(filename, dataset, n=2):
    
    #print(filename)
    
    length=[1,7,19,27][n]
    #print(length)
    data = [None]*length
    
    f = h5py.File(filename,"r")
    
    idx=0
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                #print(i,j,k,idx)
                
                try:
                    data[idx] = f["%i%i%i/%s"%(i,j,k,dataset)][...]
                    idx+=1
                except:
                    continue
    
    f.close()
    
    data = np.concatenate(data)
    
    return data


def read_dataset_cut_sky(filename, dataset, dtype="f"):
    
    try:
        f = h5py.File(filename,"r")
        data = f[dataset][...]
        f.close()
    except:
        data = np.zeros(0,dtype=dtype)
        
    return data