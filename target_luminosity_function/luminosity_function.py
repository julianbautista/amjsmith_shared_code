#! /usr/bin/env python
import numpy as np
from scipy.special import gamma, gammaincc
from scipy.interpolate import RegularGridInterpolator, splrep, splev
from cosmology import *


class LuminosityFunction(object):
    """
    Luminsity function base class
    """
    def __init__(self):
        pass

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__Phi_z01(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)
        
        return 10**log_lf

    def __Phi_z01(self, magnitude):
        # returns a spline fit to the LF at z=0.1 (using the cumulative LF)
        mags = np.arange(0, -25, -0.001)
        phi_cums = self.Phi_cumulative(mags, 0.1)
        phi = (phi_cums[:-1] - phi_cums[1:]) / 0.001
        tck = splrep((mags[1:]+0.0005)[::-1], np.log10(phi[::-1]))
        return splev(magnitude, tck)
        
    def Phi_cumulative(self, magnitude, redshift):
        raise NotImplementedError


    def Phi_rescaled(self, magnitude, redshift, original_cosmology, 
                     new_cosmology):
        """
        Luminosity function as a function of absoulte magnitude and redshift,
        in a rescaled cosmology
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift:  array of redshift
            original_cosmology: original cosmology
            new_cosmology:      new cosmology to rescale to
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        # adjust magnitudes to take into account different luminosity distance
        rcom_orig = original_cosmology.comoving_distance(redshift)
        rcom_new = new_cosmology.comoving_distance(redshift)
        magnitude_new = magnitude - 5*np.log10(rcom_new/rcom_orig)

        # get LF at new magnitudes
        lf = self.Phi(magnitude_new, redshift)

        # adjust number densities to take into accound volume differences
        # calculate volume in spherical shells (factors of 4/3 pi cancel out)
        dz = 0.001
        vol_orig = original_cosmology.comoving_distance(redshift+dz)**3 - \
                   original_cosmology.comoving_distance(redshift-dz)**3 
        vol_new = new_cosmology.comoving_distance(redshift+dz)**3 - \
                  new_cosmology.comoving_distance(redshift-dz)**3

        return lf*(vol_orig/vol_new)


    def Phi_cumulative_rescaled(self, magnitude, redshift, original_cosmology, 
                                new_cosmology):
        """
        Cumulative luminosity function as a function of absoulte 
        magnitude and redshift, in a rescaled cosmology
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift:  array of redshift
            original_cosmology: original cosmology
            new_cosmology:      new cosmology to rescale to
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        # adjust magnitudes to take into account different luminosity distance
        rcom_orig = original_cosmology.comoving_distance(redshift)
        rcom_new = new_cosmology.comoving_distance(redshift)
        magnitude_new = magnitude - 5*np.log10(rcom_new/rcom_orig)

        # get LF at new magnitudes
        lf = self.Phi_cumulative(magnitude_new, redshift)

        # adjust number densities to take into accound volume differences
        # calculate volume in spherical shells (factors of 4/3 pi cancel out)
        dz = 0.001
        vol_orig = original_cosmology.comoving_distance(redshift+dz)**3 - \
                   original_cosmology.comoving_distance(redshift-dz)**3 
        vol_new = new_cosmology.comoving_distance(redshift+dz)**3 - \
                  new_cosmology.comoving_distance(redshift-dz)**3

        return lf*(vol_orig/vol_new)

        


class LuminosityFunctionSchechter(LuminosityFunction):
    """
    Schecter luminosity function with evolution
    Args:
        Phi_star: LF normalization [h^3/Mpc^3]
        M_star: characteristic absolute magnitude [M-5logh]
        alpha: faint end slope
        P: number density evolution parameter
        Q: magnitude evolution parameter
    """
    def __init__(self, Phi_star, M_star, alpha, P, Q):

        # Evolving Shechter luminosity function parameters
        self.Phi_star = Phi_star
        self.M_star = M_star
        self.alpha = alpha
        self.P = P
        self.Q = Q

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
    
        # evolve M_star and Phi_star to redshift
        M_star = self.M_star - self.Q * (redshift - 0.1)
        Phi_star = self.Phi_star * 10**(0.4*self.P*redshift)

        # calculate luminosity function
        lf = 0.4 * np.log(10) * Phi_star
        lf *= (10**(0.4*(M_star-magnitude)))**(self.alpha+1)
        lf *= np.exp(-10**(0.4*(M_star-magnitude)))
        
        return lf

    
    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude 
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # evolve M_star and Phi_star to redshift
        M_star = self.M_star - self.Q * (redshift - 0.1)
        Phi_star = self.Phi_star * 10**(0.4*self.P*redshift)

        # calculate cumulative luminosity function
        t = 10**(0.4 * (M_star-magnitude))
        lf = Phi_star*(gammaincc(self.alpha+2, t)*gamma(self.alpha+2) - \
                           t**(self.alpha+1)*np.exp(-t)) / (self.alpha+1)

        return lf


class LuminosityFunctionTabulated(LuminosityFunction):
    """
    Luminosity function from tabulated file, with evolution
    Args:
        filename: path to ascii file containing tabulated values of cumulative
                  luminsity function
        P: number density evolution parameter
        Q: magnitude evolution parameter
    """
    def __init__(self, filename, P, Q):
        
        self.magnitude, self.log_number_density = \
                              np.loadtxt(filename, unpack=True)
        self.P = P
        self.Q = Q

        self.__lf_interpolator = \
            RegularGridInterpolator((self.magnitude,), self.log_number_density,
                                    bounds_error=False, fill_value=None)

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__Phi_z01(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)
        
        return 10**log_lf

    def __Phi_z01(self, magnitude):
        # returns a spline fit to the LF at z=0.1 (using the cumulative LF)
        mags = np.arange(0, -25, -0.001)
        phi_cums = self.Phi_cumulative(mags, 0.1)
        phi = (phi_cums[:-1] - phi_cums[1:]) / 0.001
        tck = splrep((mags[1:]+0.0005)[::-1], np.log10(phi[::-1]))
        return splev(magnitude, tck)
        
    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude 
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # shift magnitudes to redshift z=0.1
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__lf_interpolator(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)
        
        return 10**log_lf
    
    
class LuminosityFunctionTarget(LuminosityFunction):
    """
    Target LF which transitions from SDSS at low redshifts to GAMA at
    high redshifts
    
    Args:
        lf_file: tabulated file of LF at z=0.1
        lf_param_file: file containing Schechter LF paramters at high z
    """
    
    def __init__(self, Phi_star=9.4e-3, M_star=-20.70, alpha=-1.23,
                 target_lf_file="target_lf.dat", P=1.8, Q=0.7):

        self.lf_sdss = LuminosityFunctionTabulated(target_lf_file, P, Q)
            
        self.lf_gama = LuminosityFunctionSchechter(Phi_star, M_star, alpha, P, Q)
        
    
    def transition(self, redshift):
        """
        Function which describes the transition between the SDSS LF
        at low z and the GAMA LF at high z
        """
        return 1. / (1. + np.exp(120*(redshift-0.15)))

    
    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        w = self.transition(redshift)
        
        lf_sdss = self.lf_sdss.Phi(magnitude, redshift)
        lf_gama = self.lf_gama.Phi(magnitude, redshift)

        return w*lf_sdss + (1-w)*lf_gama
        
    
    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude 
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        w = self.transition(redshift)

        lf_sdss = self.lf_sdss.Phi_cumulative(magnitude, redshift)
        lf_gama = self.lf_gama.Phi_cumulative(magnitude, redshift)
        
        return w*lf_sdss + (1-w)*lf_gama
        

def target_number_densities():
    """
    Get target number densities at z=0.2, for HOD fitting
    """
    lf = LuminosityFunctionTarget()
    mags = np.arange(-22, -16.9, 0.5)
    zs = np.ones(len(mags))*0.2
    n = lf.Phi_cumulative(mags, zs)
    np.savetxt("target_number_density.dat", np.array([mags, np.log10(n)]).transpose())
        
        
def test():
    """
    Example luminosity function plot
    """
    import matplotlib.pyplot as plt
    
    mags = np.arange(-16, -23.01, -0.01)
    lf = LuminosityFunctionTarget()
    
    # plot the differential luminosity function
    for z in np.arange(0,0.51,0.1):
        zs = np.ones(len(mags))*z
        plt.plot(mags, lf.Phi(mags, zs), label=r"$z=%.2f$" %z)
        #plt.plot(mags, lf.Phi_cumulative(mags, zs), label=r"$z=%.2f$" %z)

    plt.xlabel(r"$^{0.1}M_r - 5 \log h$")
    plt.ylabel(r"$\phi \ (h^3 \rm Mpc^{-3} mag^{-1})$")
    plt.xlim(-16,-23)
    plt.ylim(1e-6, 5e-2)
    plt.yscale("log")
    plt.legend(loc="lower left").draw_frame(False)

    plt.show()


def test_rescaling():
    """
    Example rescaled LF plots
    """
    import matplotlib.pyplot as plt

    cosmo_orig = CosmologyMXXL()
    cosmo_new = CosmologyUchuu()
    
    mags = np.arange(-16, -23.01, -0.01)
    lf = LuminosityFunctionTarget()

    zs = np.ones(len(mags)) * 0.2

    # differential LF
    plt.plot(mags, lf.Phi(mags, zs), label="Orig. cosmology (MXXL)")
    plt.plot(mags, lf.Phi_rescaled(mags, zs, cosmo_orig, cosmo_new), 
             label="New cosmo (Uchuu)")

    # or plot cumulative LF instead
    #plt.plot(mags, lf.Phi_cumulative(mags, zs), label="Orig. cosmology (MXXL)")
    #plt.plot(mags, lf.Phi_cumulative_rescaled(mags, zs, cosmo_orig,cosmo_new), 
    #         label="New cosmo (Uchuu)")
    
    plt.legend(loc="upper right").draw_frame(False)
    
    plt.xlabel(r"$^{0.1}M_r - 5 \log h$")
    plt.ylabel(r"$\phi \ (h^3 \rm Mpc^{-3} mag^{-1})$")
    plt.xlim(-16,-23)
    plt.ylim(1e-6, 5e-2)
    plt.yscale("log")
    plt.legend(loc="lower left").draw_frame(False)

    plt.show()


    # plot ratio
    plt.axvline(1, c="C0")
    plt.plot(mags, lf.Phi_rescaled(mags, zs, cosmo_orig, cosmo_new)/lf.Phi(mags, zs), c="C1")
    
    plt.xlabel(r"$^{0.1}M_r - 5 \log h$")
    plt.ylabel("Ratio")
    plt.xlim(-16,-23)
    #plt.ylim(1e-6, 5e-2)
    #plt.yscale("log")

    plt.show()


if __name__ == "__main__":
    
    # makes file of target number densities (used for HOD fits)
    #target_number_densities()
    
    # make test plot of LF at a few different redshifts
    #test()

    # make test plot of LF rescaled from MXXL to Uchuu cosmology
    test_rescaling()
