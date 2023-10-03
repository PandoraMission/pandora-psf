"""Class to deal with scenes?"""

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from tqdm import tqdm

from .psf import PSF
from .utils import prep_for_add

__all__ = ["Scene", "TraceScene"]


class Scene(object):
    def __init__(
        self,
        locations: npt.ArrayLike,
        psf: PSF = None,
        shape: Tuple = (2048, 2048),
        corner: Tuple = (-1024, -1024),
    ):
        if locations.shape[1] != 2:
            raise ValueError("`locations` must have shape (n, 2).")
        self.locations = locations
        self.shape = shape
        self.corner = corner
        if psf is None:
            psf = PSF.from_name("visda")
        self.psf = psf
        self.rb, self.cb, self.prf = [], [], []
        self.ntargets = len(self.locations)
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
        self.X = SparseWarp.from_coo(self.X.tocoo())
        self.dX0 = SparseWarp.from_coo(self.dX0.tocoo())
        self.dX1 = SparseWarp.from_coo(self.dX1.tocoo())

    def model(
        self, flux: npt.ArrayLike, jitter: Optional[npt.ArrayLike] = None
    ) -> npt.ArrayLike:
        """
        Parameters:
        -----------
        flux : npt.ArrayLike
            Array of flux values with shape (ntargets, ntimes)
        jitter : npt.ArrayLike
            Array of jitter values in row and column, has shape (2, ntimes)
        """
        if flux.ndim == 1:
            flux = flux[:, None]
        if flux.shape[0] != self.ntargets:
            raise ValueError("`flux` must be an array with shape (ntargets x ntimes).")
        nt = flux.shape[1]
        ar = self.X.dot(flux).T.reshape((nt, *self.shape))
        if jitter is not None:
            for tdx in np.arange(0, nt):
                ar[tdx] += (
                    (
                        self.dX0.multiply(jitter[0, tdx])
                        + self.dX1.multiply(jitter[1, tdx])
                    )
                    .dot(flux[:, tdx])
                    .reshape(self.shape)
                )
        return ar


class TraceScene(object):
    def __init__(
        self,
        locations: npt.ArrayLike,
        psf: PSF = None,
        shape: Tuple = (400, 80),
        corner: Tuple = (0, 0),
    ):
        if locations.shape[1] != 2:
            raise ValueError("`locations` must have shape (n, 2).")
        self.locations = locations
        self.shape = shape
        self.corner = corner
        if psf is None:
            psf = PSF.from_name("nirda")
        self.psf = psf
        if "wavelength" not in self.psf.dimension_names:
            raise ValueError(
                "Can only create a trace scene if `PSF` has a `'wavelength'` dimension"
            )
        if not hasattr(self.psf, "trace_pixel"):
            raise ValueError("No trace parameters, you need to set them.")
        self.ntargets = len(self.locations)
        self.rb, self.cb, self.prf = [], [], []
        self._get_Xs()

    def __repr__(self):
        return f"TraceScene Object [{self.psf.__repr__()}]"

    def __len__(self):
        return len(self.locations)

    def _get_Xs(self):
        Xs, dX0s, dX1s = [], [], []
        for pix, wav in zip(self.psf.trace_pixel, self.psf.trace_wavelength):
            X = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
            dX0 = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
            dX1 = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
            for idx, location in enumerate(self.locations):
                rb, cb, ar = prep_for_add(
                    *self.psf.prf(
                        row=location[0] + pix.value, column=location[1], wavelength=wav
                    ),
                    shape=self.shape,
                    corner=self.corner,
                )
                X[rb * self.shape[1] + cb, idx] = ar

                rb, cb, dar = prep_for_add(
                    *self.psf.dprf(
                        row=location[0] + pix.value, column=location[1], wavelength=wav
                    ),
                    shape=self.shape,
                    corner=self.corner,
                )
                dX0[rb * self.shape[1] + cb, idx] = dar[0]
                dX1[rb * self.shape[1] + cb, idx] = dar[1]
            Xs.append(X)
            dX0s.append(dX0)
            dX1s.append(dX1)
        self.X = SparseWarp.from_coo(sparse.hstack(Xs, format="coo"))
        self.dX0 = SparseWarp.from_coo(sparse.hstack(dX0s, format="coo"))
        self.dX1 = SparseWarp.from_coo(sparse.hstack(dX1s, format="coo"))

    def model(
        self,
        spectra: npt.ArrayLike,
        jitter: Optional[npt.ArrayLike] = None,
        quiet: bool = True,
    ) -> npt.ArrayLike:
        """`spectra` must have shape nwav x ntargets x ntime"""

        if spectra.ndim == 1:
            spectra = spectra[:, None, None]
        elif spectra.ndim == 2:
            spectra = spectra[:, :, None]
        elif spectra.ndim != 3:
            raise ValueError("Pass a 3D array for flux (nwav, ntargets, ntime).")
        if (spectra.shape[0] != self.psf.trace_pixel.shape[0]) | (
            spectra.shape[1] != len(self)
        ):
            raise ValueError("`spectra` must have shape (nwav, ntargets)")
        if jitter is not None:
            if spectra.ndim == 1:
                jitter = jitter[:, None]
            elif jitter.ndim != 2:
                raise ValueError("Pass 2D array for jitter (2, ntime).")

        nt = spectra.shape[2]
        ar = np.zeros((nt, *self.shape))
        for tdx in tqdm(range(nt), disable=quiet, desc="Time index"):
            ar[tdx, :, :] = self.X.dot(spectra[:, :, tdx].ravel()).T.reshape(
                *self.shape
            )
            if jitter is not None:
                ar[tdx] += (
                    (
                        self.dX0.multiply(jitter[0, tdx])
                        + self.dX1.multiply(jitter[1, tdx])
                    )
                    .dot(spectra[:, :, tdx].ravel())
                    .reshape(self.shape)
                )
        return ar


class SparseWarp(sparse.coo_matrix):
    """A special instance of a `coo_matrix` that can be translated in column and row."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_row = deepcopy(self.row)
        self.original_col = deepcopy(self.col)
        self.original_data = deepcopy(self.data)

    def translate(self, position: Tuple):
        """Translate the data in the array by `position` in (row, column)"""
        self.reset()
        # If translating to (0, 0) just return the original data
        if position == (0, 0):
            return self
        row, col = position
        # If you're translating more than one shape away just return 0s
        if (row > self.shape[0]) | (col > self.shape[1]):
            self.data *= 0
            return self
        # find where the data is within the array bounds
        new_row, new_col = self.row + row, self.col + col
        k = (
            (new_row >= 0)
            & (new_row < self.shape[0])
            & (new_col >= 0)
            & (new_col < self.shape[1])
        )

        self.data[~k] *= 0
        self.row[~k] *= row
        self.col[~k] *= col
        self.data[k] += deepcopy(self.original_data)
        self.row[k] += row
        self.col[k] += col
        return self

    def reset(self):
        """Reset any translation back to the original data"""
        self.row = deepcopy(self.original_row)
        self.col = deepcopy(self.original_col)
        self.data = deepcopy(self.original_data)

    @staticmethod
    def from_coo(coo):
        return SparseWarp((coo.data, (coo.row, coo.col)), shape=coo.shape, dtype=coo.dtype)