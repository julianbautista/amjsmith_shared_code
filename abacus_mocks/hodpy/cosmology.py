#! /usr/bin/env python
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import nbodykit.cosmology as cosmology_nbodykit
import nbodykit.cosmology.cosmology

from hodpy import lookup

class Cosmology(object):
    """
    Class containing useful cosmology methods. Assumes flat LCDM Universe.

    Args:
        cosmo: nbodykit cosmology object
    """
    def __init__(self, cosmo):

        self.c = 299792.458 #km/s
        
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
        H100 = 100 # km/s/Mpc
        return 4*np.pi*(self.c/H100) * self.comoving_distance(z)**2 / \
                                self.cosmo_nbodykit.efunc(z)
    
    def get_xi_scaling_factor(self, cosmo_new, r_bins, pimax=120,
                           correlation_function="xi", scale=8, 
                           power_spectrum="zel", z=0.2):
        """
        Returns the cosmology rescaling factors for the correlation
        function

        Args:
            cosmo_new: new cosmology to rescale to
            r_bins:    array of r bins that xi is evaluated at (Mpc/h)
            pimax:     maximum value of pi in integral, for wp only
            correlation_function: "xi" or "wp"
            scale:     scale below which the scaling factor is set to 1
            power_spectrum: "lin", "nl" or "zel"
            z:         redshift
        """

        from hodpy.power_spectrum import PowerSpectrum

        Pk1 = PowerSpectrum(self)
        Pk2 = PowerSpectrum(cosmo_new)
        
        if correlation_function=="xi":
            xi_c1_8, xi_c1 = Pk1.get_xi(r_bins, scale=scale,
                                        power_spectrum=power_spectrum, z=z)
            xi_c2_8, xi_c2 = Pk2.get_xi(r_bins, scale=scale,
                                        power_spectrum=power_spectrum, z=z)
            scaling_factor = xi_c2/xi_c1 * (xi_c1_8/xi_c2_8)
        elif correlation_function=="wp":
            wp_c1_8, wp_c1 = Pk1.get_wp(r_bins, pimax=pimax, scale=scale,
                                        power_spectrum=power_spectrum, z=z)
            wp_c2_8, wp_c2 = Pk2.get_wp(r_bins, pimax=pimax,scale=scale,
                                        power_spectrum=power_spectrum, z=z)
            scaling_factor = wp_c2/wp_c1 * (wp_c1_8/wp_c2_8)
        else: 
            raise ValueError("Invalid correlation function", correlation_function)

        print(xi_c2)
        print(xi_c1)
        print(xi_c1_8)
        print(xi_c2_8)
        
        # keep scaling_factor fixed to 1 below scale
        scaling_factor[r_bins<scale] = 1.0
    
        return scaling_factor

    
class CosmologyMXXL(Cosmology):
    
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.WMAP5
        cosmo_nbody = cosmo_nbody.clone(Omega0_b=0.045, Omega0_cdm=0.25-0.045, h=0.73, n_s=1)
        cosmo_nbody = cosmo_nbody.match(sigma8=0.9)
        super().__init__(cosmo_nbody)
        
        
class CosmologyOR(Cosmology):
    
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.WMAP7
        super().__init__(cosmo_nbody)
        
        
class CosmologyUNIT(Cosmology):
    
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.Planck15
        cosmo_nbody = cosmo_nbody.clone(Omega0_b=0.04860, Omega0_cdm=0.3089-0.04860, h=0.6774, n_s=0.9667)
        cosmo_nbody = cosmo_nbody.match(sigma8=0.8147)
        super().__init__(cosmo_nbody)
        
        
class CosmologyUchuu(Cosmology):
    
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.Planck15
        cosmo_nbody = cosmo_nbody.clone(Omega0_b=0.04860, Omega0_cdm=0.3089-0.04860, h=0.6774, n_s=0.9667)
        cosmo_nbody = cosmo_nbody.match(sigma8=0.8159)
        super().__init__(cosmo_nbody)
    

class CosmologyAbacus(Cosmology):
    
    def __init__(self, cosmo):
        
        filename = lookup.path+"/Cosmologies/CLASS_c%03d.ini"%cosmo

        # when reading the file, it complains that it didn't read w0_fld, wa_fld
        # cloning it seems to fix the issue, and set w0_fld, wa_fld correctly
        cosmo_nbody = nbodykit.cosmology.cosmology.Cosmology.from_file(filename)
        cosmo_nbody = cosmo_nbody.clone()
    
        super().__init__(cosmo_nbody)
