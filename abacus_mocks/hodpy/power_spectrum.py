#! /usr/bin/env python
import numpy as np
from scipy.integrate import simps, quad
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize
from nbodykit.lab import cosmology as nbodykit_cosmology

from hodpy.cosmology import Cosmology

class PowerSpectrum(object):
    """
    Class containing the linear power spectrum and useful methods

    Args:
        filename: Tabulated file of linear P(k) at z=0
        h0:       Hubble parameter at z=0, in units [100 km/s/Mpc]
        OmegaM:   Omega matter at z=0
    """
    def __init__(self, cosmo):
        
        self.cosmo = cosmo   # this is my cosmology class
        self.__p_lin = nbodykit_cosmology.LinearPower(cosmo.cosmo_nbodykit,
                                                redshift=0, transfer="CLASS")
        
        self.__k = 10**np.arange(-6,6,0.01)
        self.__P = self.P_lin(self.__k, z=0)
        self.__tck = self.__get_sigma_spline() #spline fit to sigma(M,z=0)

        
    def P_lin(self, k, z):
        """
        Returns the linear power spectrum at redshift z

        Args:
            k: array of k in units [h/Mpc]
            z: array of z
        Returns:
            array of linear power spectrum in units [Mpc/h]^-3
        """
        return self.__p_lin(k) * self.cosmo.growth_factor(z)**2

    
    def Delta2_lin(self, k, z):
        """
        Returns the dimensionless linear power spectrum at redshift z,
        defined as Delta^2(k) = 4pi * (k/2pi)^3 * P(k)

        Args:
            k: array of k in units [h/Mpc]
            z: array of z
        Returns:
            array of dimensionless linear power spectrum
        """
        return self.P_lin(k, z) * k**3 / (2*np.pi**2)

    
    def W(self, k, R):
        """
        Window function in k-space (Fourier transform of top hat window)

        Args:
            k: array of k in units [h/Mpc]
            z: array of R in units [Mpc/h]
        Returns:
            window function
        """
        return 3 * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3

    
    def R_to_M(self, R):
        """
        Average mass enclosed by a sphere of comoving radius R

        Args:
            R: array of comoving radius in units [Mpc/h]
        Returns:
            array of mass in units [Msun/h]
        """
        return 4./3 * np.pi * R**3 * self.cosmo.mean_density(0)
    
    
    def M_to_R(self, M):
        """
        Comoving radius of a sphere which encloses on average mass M

        Args:
            M: array of mass in units [Msun/h]
        Returns:
            array of comoving radius in units [Mpc/h]
        """
        return (3*M / (4 * np.pi * self.cosmo.mean_density(0)))**(1./3)

    
    def __func(self, k, R):
        # function to integrate to get sigma(M)
        return self.__k**2 * self.__P * self.W(k,R)**2

    
    def __get_sigma_spline(self):
        # spline fit to sigma(R) at z=0
        logR = np.arange(-2,2,0.01)
        sigma = np.zeros(len(logR))
        R = 10**logR
        for i in range(len(R)):
            sigma[i] = simps(self.__func(self.__k, R[i]), self.__k)

        sigma = sigma / (2 * np.pi**2)
        sigma = np.sqrt(sigma)
        
        return splrep(logR, np.log10(sigma))

    
    def sigmaR_z0(self, R):
        """
        Returns sigma(R), the rms mass fluctuation in spheres of radius R,
        at redshift 0

        Args:
            R: array of comoving distance in units [Mpc/h]
        Returns:
            array of sigma
        """
        return 10**splev(np.log10(R), self.__tck)

    
    def sigmaR(self, R, z):
        """
        Returns sigma(R,z), the rms mass fluctuation in spheres of radius R,
        at redshift z

        Args:
            R: array of comoving distance in units [Mpc/h]
            z: array of redshift
        Returns:
            array of sigma
        """
        return self.sigmaR_z0(R) * self.delta_c(0) / self.delta_c(z)

    
    def sigma_z0(self, M):
        """
        Returns sigma(M), the rms mass fluctuation in spheres of mass M,
        at redshift 0

        Args:
            M: array of mass in units [Msun/h]
        Returns:
            array of sigma
        """
        R = self.M_to_R(M)
        return self.sigmaR_z0(R)

    
    def sigma(self, M, z):
        """
        Returns sigma(M), the rms mass fluctuation in spheres of mass M,
        at redshift z

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of sigma
        """
        return self.sigma_z0(M) * self.delta_c(0) / self.delta_c(z)
    
    
    def nu(self, M, z):
        """
        Returns nu = delta_c(z=0) / (sigma(M,z=0) * D(z))

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of nu
        """
        return self.delta_c(z) / self.sigma_z0(M)


    def delta_c(self, z):
        """
        Returns delta_c, the linear density threshold for collapse, 
        at redshift z

        Args:
            z: redshift
        Returns:
            delta_c
        """
        return 1.686 / self.cosmo.growth_factor(z)

    
    def get_xi(self, r_bins, scale=8, power_spectrum="zel", z=0.2):
        """
        Returns the correlation function xi(r)
    
        Args:
            cosmo: nbodykit cosmology class
            r_bins: array of bins in r to calculate xi(r)
            scale: scale for normalisation (default 8 Mpc/h)
            power_spectrum: can be "lin", "nl" or "zel" (default is "zel")
            z: redshift

        Returns:
            xi at the normalisation scale
            array of xi evaluated in r_bins
        """

        # get power spectra at z=0, then evolve xi to the right redshift after
        
        if power_spectrum=="lin":
            # linear power spectrum
            Pk = nbodykit_cosmology.LinearPower(self.cosmo.cosmo_nbodykit,
                                        redshift=0, transfer='CLASS')
        elif power_spectrum=="nl":
            # non-linear power spectrum
            Pk = nbodykit_cosmology.HalofitPower(self.cosmo.cosmo_nbodykit,
                                                 redshift=0)
        elif power_spectrum=="zel":
            # Zel'dovich power spectrum 
            Pk = nbodykit_cosmology.ZeldovichPower(self.cosmo.cosmo_nbodykit,
                                                   redshift=0)
        else:
            raise ValueError("Invalid power spectrum", power_spectrum)
    
        xi = nbodykit_cosmology.CorrelationFunction(Pk)

        xi_scale = xi(scale)*self.cosmo.growth_factor(z)**2
        xi_bins = xi(r_bins)*self.cosmo.growth_factor(z)**2
        
        return xi_scale, xi_bins

    
    def get_wp(self, rp_bins, pimax=120, scale=8, power_spectrum="zel", z=0.2):
        """
        Returns the projected correlation function wp(rp)
    
        Args:
            cosmo: nbodykit cosmology class
            rp_bins: array of bins in rp to calculate wp(rp)
            pimax: maximum value of pi in integral
            scale: scale for normalisation (default 8 Mpc/h)
            power_spectrum: can be "lin", "nl" or "zel" (default is "zel")
            z: redshift

        Returns:
            wp at the normalisation scale
            array of wp evaluated in rp_bins
        """
        pi_bins = np.arange(0,pimax+0.01,0.1)
        rp_grid, pi_grid = np.meshgrid(np.append(rp_bins,scale), pi_bins)
        r_bins = (rp_grid**2 + pi_grid**2)**0.5
    
        xi0, xi = self.get_xi(r_bins=r_bins, scale=scale,
                              power_spectrum=power_spectrum, z=z)
        wp = np.sum(xi,axis=0)
    
        return wp[-1], wp[:-1]
