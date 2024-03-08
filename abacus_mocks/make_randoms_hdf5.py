import sys
import numpy as np
import fitsio
import os.path
import desimodel.io
import desimodel.footprint
import h5py 
import os
 
def produce_randoms(cat, nran=1, seed=0, columns_to_get=['Z']):
    
    np.random.seed(seed)
    nran = np.floor(nran * len(cat['ra'][...])).astype(int)
    index_cat = np.arange(nran, dtype=int)
    index_ran = np.random.choice(index_cat, size=nran, replace=True)
    
    #-- Full sky random RA and DEC
    ra = np.random.rand(nran)*2*np.pi
    theta = np.arccos( 2*np.random.rand(nran) - 1 )
    dec = np.pi/2 - theta

    ra = np.degrees(ra)
    dec = np.degrees(dec) 
    
    ran = {}
    ran['ra'] = ra 
    ran['dec'] = dec 
    for col in columns_to_get:
        ran[col] = cat[col][...][index_ran] 
    return ran 
   
'''From Andrei Variu'''
tiles = desimodel.io.load_tiles(programs=['DARK', 'BRIGHT'])
def apply_footprint(ra, dec):
    """ apply desi footprint """

    mask = desimodel.footprint.is_point_in_desi(tiles, ra, dec)    
    print(f"DESI footprint: Selected {np.sum(mask)} out of {mask.size} galaxies")
    newbits = np.zeros(len(ra), dtype=np.int32)
    newbits[mask] = 2

    return newbits



cat_file = sys.argv[1]
seed = int(sys.argv[2])
ran_file = sys.argv[1].replace('.dat.hdf5', '.ran.hdf5')

cat = h5py.File(cat_file, 'r')

ran_columns = ['zobs', 'zcos', 'col', 'col_obs', 'app_mag', 'abs_mag', 'halo_mass'] 
ran = produce_randoms(cat, nran=1, seed=seed, columns_to_get=ran_columns) 
ran_columns += ['ra', 'dec']


if os.path.exists(ran_file):
    os.remove(ran_file)

f = h5py.File(ran_file, 'a')
for col in ran_columns:
    f.create_dataset(col, data=ran[col], compression='gzip')
#-- desi footprint
f.create_dataset('STATUS', data=apply_footprint(f['ra'][...], f['dec'][...]), compression='gzip')
f.close()
