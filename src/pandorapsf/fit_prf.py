"""Recalculates the best fit parameters to the PRF maxima"""

import numpy as np

from .psf import PSF

_, A_fit, sigma_fit = PSF.from_name("visda").calc_prf_maxima()

np.savetxt(
    "data/prf_fit_params.csv",
    np.array([A_fit, sigma_fit]),
    delimiter=",",
    fmt="%f",
    header="amplitude, sigma",
)
