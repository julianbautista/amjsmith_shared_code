#! /usr/bin/env python
import numpy as np
from scipy.special import erfc, erf
from scipy.interpolate import RegularGridInterpolator
from hodpy import lookup


class Colour(object):
    """
    Class containing methods for randomly assigning galaxies a g-r
    colour from the parametrisation of the GAMA colour magnitude diagram
    in Smith et al. 2017. r-band absolute magnitudes are k-corrected
    to z=0.1 and use h=1. g-r colours are also k-corrected to z=0.1
    """

    def red_mean(self, magnitude, redshift):
        """
        Mean of the red sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """

        colour = 0.932 - 0.032 * (magnitude + 20)
        ind = redshift > 0.1
        colour[ind] -= 0.18 * (np.clip(redshift[ind], 0, 0.4)-0.1)

        return colour


    def red_rms(self, magnitude, redshift):
        """
        RMS of the red sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = 0.07 + 0.01 * (magnitude + 20)
        ind = redshift > 0.1
        colour[ind] += (0.05 + (redshift[ind]-0.1)*0.1) * (redshift[ind]-0.1)
        
        return colour


    def blue_mean(self, magnitude, redshift):
        """
        Mean of the blue sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour_bright = 0.62 - 0.11 * (magnitude + 20)
        colour_faint = 0.4 - 0.0286*(magnitude + 16)
        colour = np.log10(1e9**colour_bright + 1e9**colour_faint)/9
        ind = redshift > 0.1
        colour[ind] -= 0.25 * (np.clip(redshift[ind],0,0.4) - 0.1)
                                                          
        return colour


    def blue_rms(self, magnitude, redshift):

        """
        RMS of the blue sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = np.clip(0.12 + 0.02 * (magnitude + 20), 0, 0.15)
        ind = redshift > 0.1
        colour[ind] += 0.2*(redshift[ind]-0.1)

        return colour


    def satellite_mean(self, magnitude, redshift):
        """
        Mean satellite colour as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        
        colour = (0.86 - 0.065 * (magnitude + 20))
            
        ind = redshift > 0.1
        colour[ind] -= 0.18 * (redshift[ind]-0.1) 

        return colour


    def fraction_blue(self, magnitude, redshift):
        """
        Fraction of blue galaxies as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of blue galaxies
        """
        frac_blue = 0.2*magnitude + \
            np.clip(4.4 + (1.2 + 0.5*(redshift-0.1))*(redshift-0.1), 4.45, 10)
        frac_blue_skibba = 0.46 + 0.07*(magnitude + 20)

        frac_blue = np.maximum(frac_blue, frac_blue_skibba)

        return np.clip(frac_blue, 0, 1)


    def fraction_central(self, magnitude, redshift):
        """
        Fraction of central galaxies as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of central galaxies
        """
        # number of satellites divided by number of centrals
        nsat_ncen = 0.35 * (2 - erfc(0.6*(magnitude+20.5)))
        return 1 / (1 + nsat_ncen)


    def probability_red_satellite(self, magnitude, redshift):
        """
        Probability a satellite is red as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of probabilities
        """
        
        sat_mean  = self.satellite_mean(magnitude, redshift)
        blue_mean = self.blue_mean(magnitude, redshift)
        red_mean  = self.red_mean(magnitude, redshift)

        #p_red = np.clip(np.absolute(sat_mean-blue_mean) / \
        #                np.absolute(red_mean-blue_mean), 0, 1)

        p_red = np.clip((sat_mean-blue_mean) / \
                        (red_mean-blue_mean), 0, 1)

        f_blue = self.fraction_blue(magnitude, redshift)
        f_cen = self.fraction_central(magnitude, redshift)

        for i in range(2):
            idx = f_blue==i
            p_red[idx]=(1-i)

        return np.minimum(p_red, ((1-f_blue)/(1-f_cen)))


    def get_satellite_colour(self, magnitude, redshift):
        """
        Randomly assigns a satellite galaxy a g-r colour

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """

        num_galaxies = len(magnitude)

        # probability the satellite should be drawn from the red sequence
        prob_red = self.probability_red_satellite(magnitude, redshift)

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)
    
        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(magnitude[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(magnitude[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(magnitude[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(magnitude[is_blue], redshift[is_blue])

        # randomly select colour from Gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour


    def get_central_colour(self, magnitude, redshift):
        """
        Randomly assigns a central galaxy a g-r colour

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        num_galaxies = len(magnitude)

        # find probability the central should be drawn from the red sequence
        prob_red_sat  = self.probability_red_satellite(magnitude, redshift)
        prob_blue_sat = 1. - prob_red_sat

        frac_cent = self.fraction_central(magnitude, redshift)
        frac_blue = self.fraction_blue(magnitude, redshift)

        prob_blue = frac_blue/frac_cent - prob_blue_sat/frac_cent + \
                                                          prob_blue_sat
        prob_red = 1. - prob_blue

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)

        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(magnitude[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(magnitude[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(magnitude[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(magnitude[is_blue], redshift[is_blue])

        # randomly select colour from gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour



    def get_galaxy_colour(self, magnitude, redshift):
        """
        Randomly assigns a galaxy a g-r colour, from the distribution,
        treating centrals and satellites the same

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        num_galaxies = len(magnitude)

        prob_blue = self.fraction_blue(magnitude, redshift)
        prob_red = 1. - prob_blue

        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)

        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(magnitude[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(magnitude[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(magnitude[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(magnitude[is_blue], redshift[is_blue])

        # randomly select colour from gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour




class ColourNew(Colour):
    
    def __init__(self, frac_cen_func, lf, colour_fits=lookup.colour_fits):
        """
        Class containing methods for randomly assigning galaxies a g-r colour

        Args:
            frac_cen_function: function that
            lf: luminosity function
            colour_fits: location of file which contains fits to the GAMA 
                         colour-magnitude diagram
        """
        
        self.redshift_bin, self.redshift_median, \
                self.functions, self.parameters = self.read_fits(colour_fits)
        
        self.__mag_bins = np.arange(-23, -14, 0.01)
        
        self.__blue_mean_interpolator = self.__get_interpolator(0)
        self.__blue_rms_interpolator = self.__get_interpolator(1)
        self.__red_mean_interpolator = self.__get_interpolator(2)
        self.__red_rms_interpolator = self.__get_interpolator(3)
        self.__fraction_blue_interpolator = self.__get_interpolator(4)

        self.frac_cen_func=frac_cen_func
        self.lf = lf

        
    def read_fits(self, colour_fits):
        fits = np.load(colour_fits)
        Nbins = fits.shape[0] # number of redshift bins
        redshift_bin    = np.zeros(Nbins) # bin centres
        redshift_median = np.zeros(Nbins) # median redshift in bin
        functions       = np.zeros((5,Nbins),dtype="i")
        parameters      = [None]*Nbins

        for i in range(Nbins):
            redshift_bin[i] = fits[i,0,0]
            redshift_median[i] = fits[i,0,1]
            functions[:,i] = fits[i,1:,0]
            parameters[i] = fits[i,1:,1:]
        
        return redshift_bin, redshift_median, functions, parameters
            
        
    def broken(self, x, a, b, c, d):
        """
        Broken linear function with smooth transition
        """
        trans=20
        y1 = a*x + b
        y2 = c*x + d
        return np.log10(10**((y1)*trans) + 10**((y2)*trans)) / trans


    def broken_reverse(self, x, a, b, c, d):
        """
        Broken linear function with smooth transition
        """
        trans=20
        y1 = a*x + b
        y2 = c*x + d
        return 1-np.log10(10**((1-y1)*trans) + 10**((1-y2)*trans)) / trans
    
    
    def __get_interpolator(self, param_idx):
        
        redshifts = self.redshift_bin.copy()
        params = self.parameters.copy()
            
        array = np.zeros((len(self.__mag_bins), len(redshifts)))
        
        for i in range(len(redshifts)):
            if self.functions[param_idx][i] == 0:
                array[:,i] = self.broken(self.__mag_bins, *params[i][param_idx])
            else:
                array[:,i] = self.broken_reverse(self.__mag_bins, *params[i][param_idx])
                
        array = np.clip(array, -10, 10)
        
        func = RegularGridInterpolator((self.__mag_bins, redshifts), array,
                                       method='linear', bounds_error=False, fill_value=None)
        return func
        
        
    def blue_mean(self, magnitude, redshift):
        return self.__blue_mean_interpolator((magnitude, redshift))
    
    def blue_rms(self, magnitude, redshift):
        return np.clip(self.__blue_rms_interpolator((magnitude, redshift)), 0.02, 10)
    
    def red_mean(self, magnitude, redshift):
        return self.__red_mean_interpolator((magnitude, redshift))
    
    def red_rms(self, magnitude, redshift):
        return np.clip(self.__red_rms_interpolator((magnitude, redshift)), 0.02, 10)
        
    def fraction_blue(self, magnitude, redshift):
        frac_blue = np.clip(self.__fraction_blue_interpolator((magnitude, redshift)), 0, 1)
        
        # if at bright end blue_mean > red_mean, set all galaxies as being red
        b_m = self.blue_mean(magnitude, redshift)
        r_m = self.red_mean(magnitude, redshift)
        idx = np.logical_and(b_m > r_m, magnitude<-20)
        frac_blue[idx] = 0
        
        return frac_blue



    def fraction_central(self, magnitude, redshift):
        """
        Fraction of central galaxies as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of central galaxies
        """
        
        logn = np.log10(self.lf.Phi_cumulative(magnitude, redshift))
        fcen = self.frac_cen_func(logn)

        return np.clip(fcen, 0, 1)

            

    def satellite_mean(self, magnitude, redshift):
        """
        Mean satellite colour as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """

        blue_mean = self.blue_mean(magnitude, redshift)
        red_mean = self.red_mean(magnitude, redshift)
        
        frac = np.clip(-0.1*(magnitude+20.5) + 0.8, 0, 1)
        colour = (frac*red_mean + (1-frac)*blue_mean)

        return colour



