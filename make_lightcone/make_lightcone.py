import numpy as np
import h5py
from cosmology import CosmologyUchuu
from catalogue import Catalogue


def read_mock(filename, Lbox=2000.):
    '''
    Read the halo/galaxy snapshot
    '''
    # Read in halo or galaxy snapshot, return position and velocities
    # Probably better to read in a fraction of the snapshot
    
    # generate some random values to test the code
    N = 1000000
    x = np.random.rand(N)*Lbox
    y = np.random.rand(N)*Lbox
    z = np.random.rand(N)*Lbox
    vx = np.random.normal(loc=0, scale=500, size=N)
    vy = np.random.normal(loc=0, scale=500, size=N)
    vz = np.random.normal(loc=0, scale=500, size=N)
    
    # 2D arrays of position and velocity
    pos = np.array([x,y,z]).transpose()
    vel = np.array([vx,vy,vz]).transpose()
    
    # position observer in centre of box
    # shift coordinates so that observer is at (0,0,0)
    pos -= Lbox/2.
    
    # index of each halo or galaxy in the file
    index = np.arange(len(x))
    
    return pos, vel, index


def save_lightcone(filename, ra, dec, z_cos, z_obs, index):
    '''
    Saves a shell of the lightcone to a hdf5 file
    '''
    f = h5py.File(filename, "a")
    f.create_dataset("ra",    data=ra,    compression="gzip")
    f.create_dataset("dec",   data=ra,    compression="gzip")
    f.create_dataset("z_cos", data=z_cos, compression="gzip")
    f.create_dataset("z_obs", data=z_obs, compression="gzip")
    f.create_dataset("index", data=index, compression="gzip")
    f.close()



def get_shell_from_box(pos, vel, index, cat, rmin, rmax, Lbox=2000.):
    '''
    Finds all haloes/galaxies in a shell of the lightcone, applying periodic replications
    if necessary. Saves this to a hdf5 file
    
    Args:
        pos: 3D numpy array of positions (comoving Mpc/h)
        vel: 3D numpy array of velocities (proper km/s)
        index: array containing index of each halo/galaxy in the original snapshot file
        cat: Catalogue object, used for coordinate conversion
        rmin: Minimum comoving distance of this shell (Mpc/h)
        rmax: Maximum comoving distance of this shell (Mpc/h)
        Lbox: Simulation box size (Mpc/h)
    '''
    # find how many periodic replications we need
    n=0
    if rmax >= Lbox/2.: n=1
    if rmax >= np.sqrt(2)*Lbox/2.: n=2
    if rmax >= np.sqrt(3)*Lbox/2.: n=3

    Nperiod = [1,7,19,27][n]
    print(Nperiod, "periodic replications needed")
    
    # lists to save the haloes/galaxies in the shell
    pos_shell = [None]*Nperiod
    vel_shell = [None]*Nperiod
    index_shell = [None]*Nperiod
    
    # loop through periodic replications
    idx=0
    for ii in range(-1,2):
        for jj in range(-1,2):
            for kk in range(-1,2):
                
                #skip replications where all haloes/galaxies farther away than rmax
                n_i = abs(ii) + abs(jj) + abs(kk)
                if n_i > n: continue
                
                # apply periodic replication to position vector
                pos_i = pos.copy()
                pos_i[:,0] += Lbox*ii
                pos_i[:,1] += Lbox*jj
                pos_i[:,2] += Lbox*kk
                
                # find haloes/galaxies that are in the shell
                dist = np.sum(pos_i**2, axis=1)**0.5
                 
                keep = np.logical_and(dist>=rmin, dist<rmax)
                
                pos_shell[idx] = pos_i[keep]
                vel_shell[idx] = vel[keep]
                index_shell[idx] = index[keep]
                idx+=1
                
    # merge values together into single array
    pos_shell = np.concatenate(pos_shell)
    vel_shell = np.concatenate(vel_shell)
    index_shell = np.concatenate(index_shell)
    
    # convert pos and vel into ra, dec, z
    ra, dec, z_cos = cat.pos3d_to_equitorial(pos_shell)
    v_los = cat.vel_to_vlos(pos_shell, vel_shell)
    z_obs = cat.vel_to_zobs(z_cos, v_los)
    
    return ra, dec, z_cos, z_obs, index



def make_lightcone():
    '''
    Loop through snapshots to make the lightcone
    '''

    z_snaps = np.array([0, 0.093, 0.19, 0.30, 0.43, 0.49])
    Lbox=2000.
    
    cosmo = CosmologyUchuu()
    cat = Catalogue(cosmo) # class with methods for converting Cartesian coords to ra,dec,z

    # get comoving distances where snapshots are joined together
    # this corresponds to the redshift half-way between each snapshot
    z_cuts = np.zeros(len(z_snaps)+1)
    z_cuts[1:-1] = (z_snaps[1:]+z_snaps[:-1])/2.
    z_cuts[-1]=0.6 # set maximum redshift to z=0.6
    dist_cuts = cosmo.comoving_distance(z_cuts)

    
    # loop through each snapshot
    for i in range(len(z_snaps)):
        z_i = z_snaps[i] 
        
        filename = "" # filename of box at z_i
        pos, vel, index = read_mock(filename, Lbox=Lbox)
        
        ra, dec, z_cos, z_obs, index = \
                            get_shell_from_box(pos, vel, index, cat, 
                                           dist_cuts[i], dist_cuts[i+1], Lbox=Lbox)
        
        savename = "test_z%.2f.hdf5"%z_i # file to save this shell
        save_lightcone(savename, ra, dec, z_cos, z_obs, index)
    
        # the shells will then need to be joined together
        
        # the index can then be used to add other halo/galaxy properties to the 
        # lightcone from the snapshots
        
                
if __name__ == "__main__":
    
    make_lightcone()
    
    
