"""Class to deal with scenes?"""

from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse

from . import PACKAGEDIR
from .psf import PSF
from .utils import prep_for_add

__all__ = ["Scene"]


class Scene(object):
    def __init__(
        self,
        locations: npt.ArrayLike,
        psf: PSF = None,
        shape: Tuple = (2048, 2048),
        corner: Tuple = (-1024, -1024),
    ):
        self.locations = locations
        self.shape = shape
        self.corner = corner
        if psf is None:
            psf = PSF.from_name("visda")
        self.psf = psf
        self.rb, self.cb, self.prf = [], [], []
        self._get_X()

    def __repr__(self):
        return f"Scene Object [{self.psf.__repr__()}]"

    def __len__(self):
        return len(self.locations)

    def _get_X(self):
        self.X = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
        self.dX0 = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
        self.dX1 = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
        for idx, location in enumerate(self.locations):
            rb, cb, ar = prep_for_add(
                *self.psf.prf(
                    row=location[0],
                    column=location[1],
                ),
                shape=self.shape,
                corner=self.corner,
            )
            self.X[rb * self.shape[1] + cb, idx] = ar
            rb, cb, dar = prep_for_add(
                *self.psf.dprf(
                    row=location[0],
                    column=location[1],
                ),
                shape=self.shape,
                corner=self.corner,
            )
            self.dX0[rb * self.shape[1] + cb, idx] = dar[0]
            self.dX1[rb * self.shape[1] + cb, idx] = dar[1]
        self.X = self.X.tocsr()
        self.dX0 = self.dX0.tocsr()
        self.dX1 = self.dX1.tocsr()

    def model(self, flux: npt.ArrayLike) -> npt.ArrayLike:
        return self.X.dot(flux).reshape(self.shape)


class TraceScene(object):
    def __init__(
        self,
        locations: npt.ArrayLike,
        psf: PSF = None,
        shape: Tuple = (2048, 2048),
        corner: Tuple = (-1024, -1024),
    ):
        self.locations = locations
        self.shape = shape
        self.corner = corner
        if psf is None:
            psf = PSF.from_name("nirda")
        self.psf = psf
        if not "wavelength" in self.psf.dimension_names:
            raise ValueError(
                "Can only create a trace scene if `PSF` has a `'wavelength'` dimension"
            )
        if not hasattr(self.psf, "trace_dpixel"):
            raise ValueError("No trace parameters, you need to set them.")

        self.rb, self.cb, self.prf = [], [], []
        self._get_Xs()

    def __repr__(self):
        return f"TraceScene Object [{self.psf.__repr__()}]"

    def __len__(self):
        return len(self.locations)

    def _get_Xs(self):
        Xs = []
        for dpix, wav, sens in zip(
            self.psf.trace_dpixel, self.psf.trace_wavelength, self.psf.trace_sensitivity
        ):
            X = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
            for idx, location in enumerate(self.locations):
                rb, cb, ar = prep_for_add(
                    *self.psf.prf(
                        row=location[0] + dpix.value, column=location[1], wavelength=wav
                    ),
                    shape=self.shape,
                    corner=self.corner,
                )
                X[rb * self.shape[1] + cb, idx] = ar * sens.value
            Xs.append(X)
        self.X = sparse.hstack(Xs, format="csr")

    def model(self, spectra: npt.ArrayLike) -> npt.ArrayLike:
        """`spectra` must have shape nwav x ntargets"""
        if spectra.shape != (self.psf.trace_dpixel.shape[0], len(self)):
            raise ValueError("`spectra` must have shape (nwav, ntargets)")
        return self.X.dot(spectra.ravel()).reshape(self.shape)
