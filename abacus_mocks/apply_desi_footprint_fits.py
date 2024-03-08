import numpy as np
from astropy.table import Table
import desimodel.io
import desimodel.footprint 
import sys

fin = sys.argv[1]


'''From Andrei Variu'''
tiles = desimodel.io.load_tiles()
def apply_footprint(ra, dec):
    """ apply desi footprint """

    mask = desimodel.footprint.is_point_in_desi(tiles, ra, dec)    
    print(f"DESI footprint: Selected {np.sum(mask)} out of {mask.size} galaxies")
    newbits = np.zeros(len(ra), dtype=np.int32)
    newbits[mask] = 2

    return newbits

tab = Table.read(fin)
tab['STATUS'] = apply_footprint(tab['RA'], tab['DEC'])
tab.write(fin, overwrite=True)


