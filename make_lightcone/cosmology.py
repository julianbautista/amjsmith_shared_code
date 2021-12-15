#! /usr/bin/env python
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import nbodykit.cosmology as cosmology_nbodykit


class Cosmology(object):
    """
    Class containing useful cosmology methods. Assumes flat LCDM Universe.

    Args:
        cosmo: nbodykit cosmology object
    """
    def __init__(self, cosmo):

        self.h0     = cosmo.h
        self.OmegaM = cosmo.Om0

        self.cosmo_nbodykit = cosmo

        self.__interpolator = self.__initialize_interpolator()


    def __initialize_interpolator(self):
        # create RegularGridInterpolator for converting comoving
        # distance to redshift
        z = np.arange(0, 3, 0.0001)
        rcom = self.comoving_distance(z)
        return RegularGridInterpolator((rcom,), z,
                                       bounds_error=False, fill_value=None)

    
    def H(self, redshift):
        """
        Hubble parameter

        Args:
            redshift: array of redshift
        Returns:
            array of H(z) in units km/s/Mpc
        """
        return 100 * self.h0 * self.cosmo_nbodykit.efunc(redshift)
    

    def critical_density(self, redshift):
        """
        Critical density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        rho_crit = self.cosmo_nbodykit.rho_crit(redshift) * 1e10

        return rho_crit


    def mean_density(self, redshift):
        """
        Mean matter density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        # mean density at z=0
        rho_mean0 = self.critical_density(0) * self.OmegaM

        # evolve to redshift z
        return  rho_mean0 * (1+redshift)**3


    def comoving_distance(self, redshift):
        """
        Comoving distance to redshift

        Args:
            redshift: array of redshift
        Returns:
            array of comoving distance in units [Mpc/h]
        """
        return self.cosmo_nbodykit.comoving_distance(redshift)


    def redshift(self, distance):
        """
        Redshift to comoving distance

        Args:
            distance: comoving distance in units [Mpc/h]
        Returns:
            array of redshift
        """
        return self.__interpolator(distance)
    
    
    def growth_factor(self, z):
        """
        Linear growth factor D(a), as a function of redshift

        Args:
            z: array of redshift
        Returns:
            Linear growth factor
        """
        return self.cosmo_nbodykit.scale_independent_growth_factor(z)
       
        
    def growth_rate(self, z):
        """
        Returns the growth rate, f = dln(D)/dln(a)

        Args:
            z: array of redshift
        Returns:
            Growth rate
        """
        return self.cosmo_nbodykit.scale_independent_growth_rate(z)
    
    def dVdz(self, z):
        """
        Returns comoving volume element (multiplied by solid angle of full sky)
        
        Args:
            z: array of redshift
        Returns:
            Comoving volume element
        """
        c    = 3e5 # km/s
        H100 = 100 # km/s/Mpc
        return 4*np.pi*(c/H100) * self.comoving_distance(z)**2 / \
                                self.cosmo_nbodykit.efunc(z)
    

class CosmologyMXXL(Cosmology):
    '''
    MXXL Cosmolology
    '''
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.WMAP5
        cosmo_nbody = cosmo_nbody.clone(Omega0_b=0.045, Omega0_cdm=0.25-0.045, h=0.73, n_s=1)
        cosmo_nbody = cosmo_nbody.match(sigma8=0.9)
        super().__init__(cosmo_nbody)
        
        
class CosmologyOR(Cosmology):
    '''
    OuterRim Cosmolology
    '''
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.WMAP7
        super().__init__(cosmo_nbody)
        
        
class CosmologyUNIT(Cosmology):
    '''
    UNIT Cosmolology
    '''
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.Planck15
        cosmo_nbody = cosmo_nbody.clone(Omega0_b=0.04860, Omega0_cdm=0.3089-0.04860, h=0.6774, n_s=0.9667)
        cosmo_nbody = cosmo_nbody.match(sigma8=0.8147)
        super().__init__(cosmo_nbody)
        
        
class CosmologyUchuu(Cosmology):
    '''
    Uchuu Cosmology
    '''
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.Planck15
        cosmo_nbody = cosmo_nbody.clone(Omega0_b=0.04860, Omega0_cdm=0.3089-0.04860, h=0.6774, n_s=0.9667)
        cosmo_nbody = cosmo_nbody.match(sigma8=0.8159)
        super().__init__(cosmo_nbody)
    

class CosmologyAbacus(Cosmology):
    '''
    Abacus Cosmolology
    '''
    def __init__(self, cosmo, abacus_cosmologies_file):
        
        self.__param_array = pd.read_csv(abacus_cosmologies_file, sep=",").to_numpy()
        omega_b, omega_cdm, h, A_s, n_s, alpha_s, N_ur, N_ncdm, omega_ncdm, w0_fld, \
                            wa_fld, sigma8_m, sigma8_cb = self.__get_params(cosmo)
        Omega_b = omega_b/h**2
        Omega_cdm = omega_cdm/h**2
        
        cosmo_nbody = cosmology_nbodykit.cosmology.Cosmology(h=h, T0_cmb=2.7255, 
                        Omega0_b=Omega_b, Omega0_cdm=Omega_cdm, N_ur=N_ur, 
                                            m_ncdm=[0.06], n_s=n_s, A_s=A_s)
        
        super().__init__(cosmo_nbody)
        
        
    def __get_params(self, cosmo):
        
        cosmo_number = np.zeros(len(self.__param_array[:,0]), dtype="i")
        for i in range(len(self.__param_array[:,0])):
            cosmo_number[i] = int(self.__param_array[i,0][11:])
        
        idx = np.where(cosmo_number==cosmo)[0][0]
        
        return np.array(self.__param_array[idx,2:], dtype="f")
    