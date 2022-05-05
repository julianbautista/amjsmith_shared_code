#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d

from hodpy.cosmology import CosmologyFlamingo
from hodpy.k_correction import GAMA_KCorrection
from hodpy.luminosity_function import LuminosityFunctionTargetBGS
from hodpy import lookup
from hodpy.catalogue import Catalogue
from hodpy.colour import ColourNew


def cut_sky_replication(position, velocity, magnitude, is_cen, cosmology, 
                        Lbox, zsnap, kcorr_r, kcorr_g, fcen_func, 
                        luminosity_function, replication=(0,0,0), 
                        zcut=0.6, mag_cut=20.2):
    """
    Creates a cut sky mock from a single periodic replication, by converting 
    the cartesian coordiantes of a cubic box mock to ra, dec, z
    Adds evolution to the magnitudes and colours of the mock
    Args:
        position:  array of comoving position vectors (Mpc/h), 
                   in the range -Lbox/2 < pos < Lbox/2
        velocity:  array of proper velocity vectors (km/s)
        magnitude: array of absolute magnitude
        is_cen:    boolean array indicating if galaxy is central (True) or 
                   satellite (False)
        cosmology: instance of cosmology.Cosmology class
        Lbox:      comoving box length of simulation (Mpc/h)
        zsnap:     redshift of simulation snapshot
        kcorr_r:   GAMA_KCorrection object with r-band k-correction
        kcorr_g:   GAMA_KCorrection object with g-band k-correction
        fcen_func: function returning the fraction of central galaxies,
                   as a function of number density
        luminosity_function: target luminosity function, instance of class
                             luminosity_function.LuminosityFunctionTargetBGS
        replication: tuple indicating which periodic replication to use. 
                     Default value is (0,0,0) (no replications).
                     E.g. (1,-1,0) would shift x coordinates by +Lbox and 
                     y coordinates by -Lbox
        zcut:    Maximum redshift cut, default value is 0.6
        mag_cut: Apparent magnitude cut, default value is 20.2

    Returns:
        ra:   array of ra (deg)
        dec:  array of dec (deg)
        zcos: array of cosmological redshift, which does not include the 
              effect of peculiar velocities
        zobs: array of observed redshift, which includes peculiar velocities.
        magnitude_new: array of new absolute magnitude, rescaled to match the
                       target luminosity function at each redshift
        app_mag: array of apparent magnitudes (calculated from rescaled 
                 magnitudes and colours)
        colour_new: array of g-r colours, which are re-assigned to add evolution
        index: array of indices. Used to match galaxies between the input and 
               output arrays of this function
    """

    # index in original arrays
    index = np.arange(position.shape[0])

    cat = Catalogue(cosmology)
    
    position_rep = position.copy()
    if replication==(0,0,0):
        print("No periodic replications")
    else:
        print("Applying periodic replications")
        for i in range(3):
            print("%.1f < %s < %.1f"%((-1+2*replication[i])*Lbox/2., chr(120+i), (1+2*replication[i])*Lbox/2.))
            position_rep[:,i] += Lbox*replication[i]
    
    # get ra, dec, z coordinates of each galaxy
    ra, dec, zcos = cat.pos3d_to_equitorial(position_rep)
    vlos = cat.vel_to_vlos(position_rep, velocity)
    zobs = cat.vel_to_zobs(zcos, vlos)
    
    print("Applying redshift cut z < %.1f"%zcut)
    keep = zobs <= zcut
    ra, dec, zcos, zobs, magnitude, is_cen, index = \
                ra[keep], dec[keep], zcos[keep], zobs[keep], magnitude[keep], \
                is_cen[keep], index[keep]
        
    print("Rescaling magnitudes")
    magnitude_new = luminosity_function.rescale_magnitude(magnitude, \
                                            np.ones(len(zobs))*zsnap, zobs)
    
    print("Assigning colours")
    is_sat = np.invert(is_cen)
    colour_new = np.zeros(len(magnitude_new))

    col = ColourNew(frac_cen_func=fcen_func, lf=luminosity_function)
    
    # randomly assign colours to centrals and satellites
    # this will need to be modified if SSFRs are used
    colour_new[is_cen] = col.get_central_colour(magnitude_new[is_cen], zobs[is_cen])
    colour_new[is_sat] = col.get_satellite_colour(magnitude_new[is_sat], zobs[is_sat])
    
    # get apparent magnitude
    app_mag = kcorr_r.apparent_magnitude(magnitude_new, zobs, colour_new)
    
    # observer frame colours
    colour_obs = colour_new + kcorr_g.k(zobs, colour_new) - kcorr_r.k(zobs, colour_new)
    
    print("Applying magnitude cut r < %.1f"%mag_cut)
    keep = app_mag <= mag_cut
    ra, dec, zcos, zobs, magnitude_new, app_mag, colour_new, \
        colour_obs, index = ra[keep], dec[keep], zcos[keep], zobs[keep], \
                    magnitude_new[keep], app_mag[keep], colour_new[keep], \
                    colour_obs[keep], index[keep]
        
    
    return ra, dec, zcos, zobs, magnitude_new, app_mag, colour_new, \
            colour_obs, index



def read_flamingo(snap, zsnap, input_path, cosmo):
    """
    Reads galaxy properties from a FLAMINGO simulation snapshot

    Args:
        snap: snapshot number
        zsnap: snapshot redshift
        input_path: path of FLAMINGO snpahot
        cosmo: cosmology, cosmology.Cosmology class

    Returns:
        pos: array of comoving position vectors (Mpc/h)
        vel: array of velocity vectors (km/s)
        Mhalo: halo mass (Msun/h)
        Mstar: stellar masses (Msun/h^2)
        Vmax: maximum circular velocity (km/s)
        SSFR: specific star formation rate (1/yr)
        is_cen: array indicating if galaxies are centrals (True) or 
                satellites (False)
    """

    path = input_path+ "catalogue_00%i/"%snap
    h = cosmo.h0
    SFRunit = 97.780000 # for converting SFR to Msun/yr

    f = h5py.File(path+"vr_catalogue_00%i.properties.0"%snap, "r")
    Mhalo = f["Mass_200mean"][...] * 1e10 * h
    Mstar = f["Aperture_mass_star_50_kpc"][...] * 1e10 * h**2 # in Msun/h2
    Vmax = f["Vmax"][...] # in km/s
    SFR = f["Aperture_SFR_gas_50_kpc"][...] * h / SFRunit # in Msun/h/yr
    SSFR = SFR / Mstar * h # in 1/yr

    # convert pos to comoving Mpc/h
    pos=np.array([f["Xc"][...],f["Yc"][...],f["Zc"][...]]).transpose()*(1+zsnap)*h
    vel=np.array([f["VXc"][...],f["VYc"][...], f["VZc"][...]]).transpose()

    # central/satellite galaxies?
    is_cen = f["hostHaloID"][...]==-1

    return pos, vel, Mhalo, Mstar, Vmax, SSFR, is_cen


def match_magnitudes(prop, zsnap, lf, Lbox, prop2=None, threshold=None):
    """
    Assign magnitudes to each galaxy, based on a ranking of the
    property prop. It will also optionally rank galaxies by a second property, 
    prop2, if the initial property prop < threshold

    Args:
        prop: array of the main property to rank galaxies by
        zsnap: snapshot redshift
        lf: luminosity function
        Lbox: box size, Mpc/h
        prop2: secondary property to rank galaxy by, if provided
               (default is None)
        threshold: value of prop below which galaxies are ranked by prop2,
                   if provided (default is None)
    Returns:
        magnitudes: array of r-band absolute magnitudes
    """

    # array of indices, used to sort back to the original order at the end
    index = np.arange(len(prop), dtype="i")
    
    # sort arrays based on property 'prop'
    idx = np.argsort(prop)
    prop = prop[idx]
    index = index[idx] 
    
    if not prop2 is None:
        prop2 = prop2[idx]

        # sort by second property 'prop2', for galaxies with prop < transition
        to_sort = prop2 <= threshold # galaxies to be sorted 
        idx = np.argsort(prop[to_sort])
        prop[to_sort] = prop[to_sort][idx]
        prop2[to_sort] = prop2[to_sort][idx]
        index[to_sort] = index[to_sort][idx]

    # get index needed to sort back to original order
    index = np.argsort(index)

    # number density of all galaxies in the simulation box
    n_total = len(prop) / Lbox**3

    # function to convert n to mag
    mag_bins = np.arange(-10,-24,-0.01)
    n_cum = lf.Phi_cumulative(mag_bins, zsnap)
    mag_func = interp1d(np.log10(n_cum), mag_bins, kind="cubic")

    # get random values of number density, then sort from highest to lowest
    n_rand = np.random.rand(len(prop))*n_total
    n_rand.sort()
    n_rand=n_rand[::-1]

    # convert n to mag to get the magnitude for each galaxy
    magnitudes = mag_func(np.log10(n_rand))

    # return magnitudes, sorted back to the original galaxy order in the mock
    return magnitudes[index]


def number_replications(zmax, cosmology, Lbox):
    """
    Returns the number of periodic replications needed to create a mock
    that extends to a certain maximum redshift
    Args:
        zmax: maximum redshift
        cosmology: cosmology, cosmology.Cosmology class
        Lbox: box size (Mpc/h)
    Returns:
        n: number of replications
    """

    rmax = cosmology.comoving_distance(zmax)
    n=0
    for i in range(10):
        if rmax > (Lbox*(1 + 2*(i//3))/2.)*np.sqrt(1+(i%3)):
            n += 1
    return n


def get_fcen_func(mag, is_cen, zsnap, lf):
    """
    Returns a function for getting the fraction of central galaxies as
    a function of number density
    Args:
        mag: array of r-band absolute magnitude bins
        is_cen: array indicating if a galaxy is a central (True) or 
                satellite (False)
        zsnap: snapshot redshift
        lf: luminosity function
    """
    # convert the magnitudes to the corresponding number density at zsnap
    logn = np.log10(lf.Phi_cumulative(mag, zsnap))

    # find central fraction in bins of log number density
    bw=0.2
    bins = np.arange(-8, -1, bw)
    binc = bins[:-1]+bw/2.

    f_cen = np.zeros(len(bins)-1)
    for i in range(len(f_cen)):
        keep = np.logical_and(logn>=bins[i], logn<bins[i+1])
        if np.count_nonzero(keep)==0:continue
        f_cen[i] = np.count_nonzero(is_cen[keep])/np.count_nonzero(keep)

    # get function that interpolates between bins
    func = interp1d(binc[f_cen>0], f_cen[f_cen>0], kind="cubic", 
                    fill_value="extrapolate")

    return func


def make_cut_sky(snap, zsnap, Lbox, cosmo, simulation, input_path,
                 output_file, obs_pos=(0,0,0), zmax=0.6):
    """
    Make a cut-sky mock from a single snapshot, looping through periodic 
    replications of the box
    Args:
        snap: snapshot number
        zsnap: snapshot redshift
        Lbox: box size (Mpc/h)
        cosmo: cosmology, cosmology.Cosmology class
        simulation: name of the simulation
        input_path: path of simultion snapshot
        output_file: location of file to save output
        obs_pos: tuple, 3D position vector of observer, in Mpc/h
        zmax: maximum redshift of mock
    """

    # the target LF used when making the mock
    lf = LuminosityFunctionTargetBGS(lookup.target_lf,lookup.gama_lf_fits)


    # read from the simulation
    if simulation=="FLAMINGO":
        pos, vel, halo_mass, Mstar, Vmax, SSFR, is_cen = \
                            read_flamingo(snap, zsnap, input_path, cosmo)
    else:
        # add new functions to read from different simulations,
        # e.g. for Sibelius simulations
        raise ValueError("Invalid simulation, %s"%simulation)

    # shift observer position to the origin
    for i in range(3):
        pos[:,0] -= obs_pos[i]
    pos[pos >  Lbox/2.] -= Lbox #apply periodic replications
    pos[pos < -Lbox/2.] += Lbox

    # assign magnitudes to each galaxy from target LF
    # rank galaxies by Vmax
    M_r = match_magnitudes(Vmax, zsnap, lf, Lbox)

    # or rank by Mstar, then rank by Vmax below some mass threshold
    #M_r = match_magnitudes(Mstar, zsnap, lf, Lbox, Vmax, 1e11)

    # Function returning fraction of galaxies that are centrals
    # This is needed for the (g-r) colour assignment
    fcen_func = get_fcen_func(M_r, is_cen, zsnap, lf)
    
    # Colour-dependent k-corrections, needed to get apparent magnitudes
    kcorr_r = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file_rband)
    kcorr_g = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file_gband)

    # how many replications do we need?
    n = number_replications(zmax, cosmo, Lbox)
    print("N REP", n)
    
    # loop through periodic replications
    for i in range(-2,3):
        for j in range(-2,3):
            for k in range(-2,3):
        
                rep = (i,j,k)
                print(rep)

                # skip the replication if it is beyond zmax
                X = np.array([abs(i), abs(j), abs(k)])
                for ii in range(3):
                    if X[ii] > 1: X[ii] = 3*X[ii]-2

                skip_replication = np.sum(X) > n
                if skip_replication:
                    print("Skipping")
                    continue

                ra, dec, zcos, zobs, magnitude_new, app_mag, \
                    col_new, col_obs, index = \
                    cut_sky_replication(pos, vel, M_r, is_cen, cosmo, Lbox=Lbox,
                            zsnap=zsnap, kcorr_r=kcorr_r, kcorr_g=kcorr_g, 
                            replication=rep, zcut=zmax, mag_cut=20.2,
                            fcen_func=fcen_func, luminosity_function=lf)

                print("NGAL:", np.count_nonzero(ra))

                # save to a temporary file
                # more datasets can be added
                f = h5py.File(output_file+"_temp","a")
                f.create_dataset("%i%i%i/ra"%(i,j,k),      data=ra,      compression="gzip")
                f.create_dataset("%i%i%i/dec"%(i,j,k),     data=dec,     compression="gzip")
                f.create_dataset("%i%i%i/zcos"%(i,j,k),    data=zcos,    compression="gzip")
                f.create_dataset("%i%i%i/zobs"%(i,j,k),    data=zobs,    compression="gzip")
                f.create_dataset("%i%i%i/app_mag"%(i,j,k), data=app_mag, compression="gzip")
                f.create_dataset("%i%i%i/abs_mag"%(i,j,k),data=magnitude_new, compression="gzip")
                f.create_dataset("%i%i%i/col"%(i,j,k),     data=col_new, compression="gzip")
                f.create_dataset("%i%i%i/col_obs"%(i,j,k), data=col_obs, compression="gzip")
                f.create_dataset("%i%i%i/is_cen"%(i,j,k), data=is_cen[index], compression="gzip")
                f.create_dataset("%i%i%i/halo_mass"%(i,j,k), data=halo_mass[index], compression="gzip")
                f.create_dataset("%i%i%i/Mstar"%(i,j,k), data=Mstar[index], compression="gzip")
                f.create_dataset("%i%i%i/pos"%(i,j,k),   data=pos[index], compression="gzip")
                f.create_dataset("%i%i%i/vel"%(i,j,k),   data=vel[index], compression="gzip")
                f.close()
                


def merge_files(output_file, cosmo, Lbox, zmax=0.6):
    """
    Combines the outputs from each periodic replication into a single
    dataset in the final output file
    Args:
        output_file: location of the output file
        cosmo: cosmology, cosmology.Cosmology
        Lbox: box size (Mpc/h)
        zmax: maximum redshift
    """
    n = number_replications(zmax, cosmo, Lbox)
    print("N REP", n)

    # datasets to add to the new file
    properties = "ra", "dec", "zcos", "zobs", "app_mag", "abs_mag", "col", \
                 "col_obs", "is_cen", "halo_mass", "Mstar", "pos", "vel"

    for p in range(len(properties)):

        data = [None]*100
        idx=0

        for i in range(-2,3):
            for j in range(-2,3):
                for k in range(-2,3):
        
                    rep = (i,j,k)
                    print(rep)

                    X = np.array([abs(i), abs(j), abs(k)])
                    for ii in range(3):
                        if X[ii] > 1: X[ii] = 3*X[ii]-2


                    corner = np.sum(X) > n
                    if corner:
                        print("Skipping")
                        continue

                    f = h5py.File(output_file+"_temp","r")
                    data[idx] = f["%i%i%i/%s"%(i,j,k,properties[p])][...]
                    f.close()
                    
                    idx += 1

        data = np.concatenate(data[:idx])

        f = h5py.File(output_file,"a")
        f.create_dataset("Data/%s"%properties[p], data=data, compression="gzip")
        f.close()


if __name__ == "__main__":

    # initialize cosmology and get box size
    simulation = "FLAMINGO"
    cosmo = CosmologyFlamingo()
    Lbox = 1000*cosmo.h0 #box size in Mpc/h

    # array of redshifts of each snapshot
    # maybe this can be read from a file instead
    scale_factors = np.ones(78)*0.5
    scale_factors[70:] = 0.740741, 0.769231, 0.800000, 0.833333, \
                         0.869565, 0.909091, 0.952381, 1.0
    redshifts = 1/scale_factors - 1


    snap = 73 # snapshot number we are using
    zsnap = redshifts[snap] # redshift of this snapshot
    obs_pos = np.ones(3)*Lbox/2. #observer placed at centre of box

    # maximum redshift of lightcone
    zmax = 0.6

    # location of mock, and where to save output
    input_path="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/"
    output_file="/cosma5/data/durham/tccb49/FLAMINGO/test.hdf5"

    # make cut-sky mock, looping through periodic replications of box
    make_cut_sky(snap=snap, zsnap=zsnap, obs_pos=obs_pos, Lbox=Lbox, zmax=zmax,
                 cosmo=cosmo, input_path=input_path, output_file=output_file,
                 simulation=simulation)

    # combine into a single file
    merge_files(output_file, cosmo, Lbox, zmax=zmax)
