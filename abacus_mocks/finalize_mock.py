# finalize the mock by finding galaxies in DESI footprint, and making randoms
import numpy as np
import h5py
import pandas as pd
from astropy.table import Table
from desimodel.io import load_tiles
from desimodel.footprint import is_point_in_desi


def get_status(ra, dec):
    """
    Get the status of the galaxies, given the ra, dec coordinates
    
    For each galaxy, the status is an integer
    bit0 = 1
    bit1 = 1 if in final DESI footprint
    bit2 = 1 if in SV3 footprint
    """
    STATUS = np.zeros(len(ra), dtype="i")

    # bit0, set to 1 for now
    STATUS += 2**0

    # bit1, set this to 1 if in final DESI footprint
    tiles = load_tiles()
    mask = tiles["PROGRAM"] == "BRIGHT"
    tiles_bright = tiles[mask]
    keep = is_point_in_desi(tiles=tiles_bright, ra=ra, dec=dec)
    STATUS[keep] += 2**1


    # bit2, set this to 1 if in SV3 footprint
    tiles = load_tiles(tilesfile="tiles-sv3.fits")
    mask = tiles["PROGRAM"] == "BRIGHT"
    tiles_bright = tiles[mask]
    keep = is_point_in_desi(tiles=tiles_bright, ra=ra, dec=dec)
    STATUS[keep] += 2**2

    return STATUS



# the SV3 tiles file is a ecsv file
# convert this to fits file so it can be read by the load_tiles() function
file_sv3 = "/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-sv3.ecsv"
tiles = pd.read_csv(file_sv3, skiprows=17, delimiter=" ")
tiles = Table.from_pandas(tiles)
tiles["OBSCONDITIONS"] = np.zeros(len(tiles["RA"]), dtype="i")
tiles.write('tiles-sv3.fits', overwrite=True)


#######################################################
print("FINDING GALAXIES IN DESI FOOTPRINT")


# path of the mock 
path = "/global/cscratch1/sd/amjsmith/Uchuu/full_sky_evolution/lightcone/observer/final/"
input_file = path+"bgs_uchuu_mock_v0.4.hdf5"

# read in mock and get STATUS of each galaxy
f = h5py.File(input_file, "r")
ra = f["Data/ra"][...]
dec = f["Data/dec"][...]
f.close()
N = len(ra)

STATUS = get_status(ra, dec)

# save STATUS dataset to mock
f = h5py.File(input_file, "a")
f.create_dataset("Data/STATUS", data=STATUS, compression="gzip")
f.close()


#######################################################
print("MAKING RANDOMS")

# make random catalogues
# each random file has same number of galaxies as the mock
# each random galaxy is a galaxy sampled from the mock, with the same magnitude, colour, etc
# but with random ra, dec coordinates

# these are the datasets to copy from mock to randoms
datasets="abs_mag", "app_mag", "g_r", "g_r_obs", "halo_mass", "galaxy_type", "z_cos", "z_obs"

# loop to make multiple random files
for i in range(10):
    print("RANDOM FILE %i"%i)
    
    # generate uniform random ra, dec on full sky, and get status
    ra_r = np.random.rand(N)*360
    dec_r = np.arcsin(np.random.rand(N)*2 - 1) * 180/np.pi
    STATUS_r = get_status(ra_r, dec_r)

    # randomly sample galaxies from mock
    # idx is the index of the galaxy in the mock
    idx = np.random.randint(0,N,N)
    
    f1 = h5py.File(input_file, "r")
    f2 = h5py.File(path+"random_S%i_1X.hdf5"%((i+1)*100), "a")
    
    # save ra, dec and status
    f2.create_dataset("Data/ra", data=ra_r, compression="gzip")
    f2.create_dataset("Data/dec", data=dec_r, compression="gzip")
    f2.create_dataset("Data/STATUS", data=STATUS_r, compression="gzip")
    
    # then loop through the other datasets
    for d in range(len(datasets)):
        data = f1["Data/%s"%datasets[d]][...]
        f2.create_dataset("Data/%s"%datasets[d], data=data[idx], compression="gzip")
    
    f1.close()
    f2.close()

