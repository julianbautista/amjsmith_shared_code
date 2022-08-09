#! /usr/bin/env python
import numpy as np
from hodpy.cosmology import CosmologyMXXL, CosmologyAbacus


def rescale_cosmology_abacus(cosmo, output_file=None, correlation_function="xi",
                             pimax=120, scale=8, power_spectrum="zel", z=0.2):
    """
        Returns the cosmology rescaling factors for the correlation
        function, for rescaling the MXXL cosmology to AbacusSummit

        Args:
            cosmo:     integer, AbacusSummit cosmology number
            output_file: file to save output, if provided
            correlation_function: "xi" or "wp"
            pimax:     maximum value of pi in integral, for wp only
            scale:     scale below which the scaling factor is set to 1
            power_spectrum: "lin", "nl" or "zel"
            z:         redshift
        """
    
    cosmo1 = CosmologyMXXL()
    cosmo2 = CosmologyAbacus(cosmo)

    dr = 0.05
    r_bins = 10**np.arange(-2+dr/2., 2.21-dr/2., dr)
    
    scaling_factor = cosmo1.get_xi_scaling_factor(cosmo2, r_bins, pimax=pimax,
                            correlation_function=correlation_function, scale=8, 
                            power_spectrum=power_spectrum, z=z)

    # save to file if filename provided
    if not output_file is None:
        np.savetxt(output_file, scaling_factor)
    
    return scaling_factor


if __name__ == "__main__":


    cosmo = 0
    
    correlation_function = "xi"
    pimax = 120 #Mpc/h, only needed if correlation_function="wp"
    scale = 8 #Mpc/h, scale where scaling factor normalised to be 1 (and set to 1 below this)
    z=0.2

    power_spectrum = "lin"
    
    output_file = "cosmology_rescaling_factor_%s_%s_%i.txt"%(correlation_function, power_spectrum, scale)
    
    rescale_cosmology_abacus(cosmo, output_file=output_file,
                correlation_function=correlation_function, pimax=pimax,
                scale=scale, power_spectrum=power_spectrum, z=z)
    
