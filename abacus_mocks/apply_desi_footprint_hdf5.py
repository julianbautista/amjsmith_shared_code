import numpy as np
import desimodel.io
import desimodel.footprint 
import sys
import h5py 
import os

fin = sys.argv[1]


'''From Andrei Variu'''
tiles = desimodel.io.load_tiles(programs=['DARK', 'BRIGHT'])
def apply_footprint(ra, dec):
    """ apply desi footprint """

    mask = desimodel.footprint.is_point_in_desi(tiles, ra, dec)    
    print(f"DESI footprint: Selected {np.sum(mask)} out of {mask.size} galaxies")
    newbits = np.zeros(len(ra), dtype=np.int32)
    newbits[mask] = 2

    return newbits

tab = h5py.File(fin, 'r+')
status = apply_footprint(tab['ra'][...], tab['dec'][...])
if 'STATUS' in tab.keys():
    del tab['STATUS']
tab.create_dataset('STATUS', data=status, compression='gzip')
tab.close()


