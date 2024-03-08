import sys
import numpy as np
import fitsio
import os.path
from astropy.table import Table
import desimodel.io
import desimodel.footprint
 
def produce_randoms(cat, nran=1, seed=0, columns_to_get=['Z']):
    
    np.random.seed(seed)
    nran = np.floor(nran * len(cat)).astype(int)
    index_cat = np.arange(nran, dtype=int)
    index_ran = np.random.choice(index_cat, size=nran, replace=True)
    
    #-- Full sky random RA and DEC
    ra = np.random.rand(nran)*2*np.pi
    theta = np.arccos( 2*np.random.rand(nran) - 1 )
    dec = np.pi/2 - theta

    ra = np.degrees(ra)
    dec = np.degrees(dec) 
    
    ran = Table()
    ran['RA'] = ra 
    ran['DEC'] = dec 
    for col in columns_to_get:
        ran[col] = cat[col][index_ran] 
    return ran 
    
cat_file = sys.argv[1]
seed = int(sys.argv[2])
ran_file = sys.argv[1].replace('.fits', '_ran.fits')

cat = fitsio.read(cat_file)

ran = produce_randoms(cat, nran=1, seed=seed, columns_to_get=['Z', 'R_MAG_APP', 'R_MAG_ABS', 'G_R_REST', 'G_R_OBS', 'HALO_MASS', 'CEN', 'RES']) 

ran.write(ran_file, overwrite=True)

