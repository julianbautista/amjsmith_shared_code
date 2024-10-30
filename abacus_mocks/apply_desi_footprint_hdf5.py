import numpy as np
import desimodel.io
import desimodel.footprint 
import sys
import h5py 
import os
import healpy as hp

fin = sys.argv[1]

def apply_y5_footprint(ra, dec):
    ''' Y5 flag. Code from Andrei Variu'''

    tiles = desimodel.io.load_tiles(programs=['DARK', 'BRIGHT'])
    
    mask = desimodel.footprint.is_point_in_desi(tiles, ra, dec)    
    print(f"DESI Y5 footprint: {np.sum(mask)} out of {mask.size} galaxies")
    return mask 

def apply_y1_footprint(ra, dec):
    ''' Y1 completeness map'''
    comp_map = '/global/cfs/cdirs/desi/users/bautista/bgs/BGS_BRIGHT_Y1_v1.2_full_comp.npy'

    print('Assigning Y1 completeness from')
    print(comp_map)

    comp_map = np.load(comp_map, allow_pickle=True)
    nside = hp.npix2nside(comp_map.size)
    pix = hp.ang2pix(nside, np.pi/2-np.radians(dec), np.radians(ra))
    comp_gal = comp_map[pix]

    w = comp_gal > 0
    print(f'DESI Y1 footprint: {np.sum(w)} our of {comp_gal.size} galaxies')

    return comp_gal

def get_completeness(ra, dec, mask):
    nside = hp.npix2nside(mask.size)
    pix = hp.ang2pix(nside, np.pi/2-np.radians(dec), np.radians(ra), nest=True)
    comp_gal = mask[pix]
    return comp_gal


def get_y1_completeness(ra, dec):
    ''' Apply Y1 completeness as defined in 
        /global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/Y1_footprint.ipynb
    '''
    mask = hp.read_map(f"/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/healpix_map_ran_comp_Y1_BGS_BRIGHT.fits")
    comp_y1 = get_completeness(ra, dec, mask) 
    return comp_y1

def get_y3_completeness(ra, dec):
    ''' Apply Y3 completeness as defined in 
        /global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/Y3_footprint.ipynb
    '''
    mask = hp.read_map(f"/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/healpix_map_ran_comp_Y3_BGS_BRIGHT.fits")
    comp_y3 = get_completeness(ra, dec, mask) 
    return comp_y3



tab = h5py.File(fin, 'r+')
ra = tab['ra'][...]
dec = tab['dec'][...]
y5_flag = apply_y5_footprint(ra, dec)
if 'Y5' in tab.keys():
    del tab['Y5']
tab.create_dataset('Y5', data=y5_flag, compression='gzip')


y1_comp = get_y1_completeness(ra, dec)
if 'Y1_COMP' in tab.keys():
    del tab['Y1_COMP']
tab.create_dataset('Y1_COMP', data=y1_comp, compression='gzip')

y3_comp = get_y3_completeness(ra, dec)
if 'Y3_COMP' in tab.keys():
    del tab['Y3_COMP']
tab.create_dataset('Y3_COMP', data=y3_comp, compression='gzip')
tab.close()


