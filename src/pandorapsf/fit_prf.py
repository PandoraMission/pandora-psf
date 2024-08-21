"""Recalculates the best fit parameters to the PRF maxima"""

import numpy as np

from pandorapsf.psf import PSF
from pandorapsf import PACKAGEDIR


_, A_fit, sigma_fit = PSF.from_name("visda").calc_prf_maxima()

np.savetxt(
    PACKAGEDIR + "/data/prf_gauss_params.csv",
    np.array([A_fit, sigma_fit]),
    delimiter=",",
    fmt="%f",
    header="amplitude, sigma",
)
