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

    # def _get_X(self):
    #     self.X = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
    #     self.dX0 = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
    #     self.dX1 = sparse.lil_matrix((np.prod(self.shape), len(self.locations)))
    #     for idx, location in enumerate(self.locations):
    #         rb, cb, ar = prep_for_add(
    #             *self.psf.prf(
    #                 row=location[0],
    #                 column=location[1],
    #             ),
    #             shape=self.shape,
    #             corner=self.corner,
    #         )
    #         self.X[rb * self.shape[1] + cb, idx] = ar
    #         rb, cb, dar = prep_for_add(
    #             *self.psf.dprf(
    #                 row=location[0],
    #                 column=location[1],
    #             ),
    #             shape=self.shape,
    #             corner=self.corner,
    #         )
    #         self.dX0[rb * self.shape[1] + cb, idx] = dar[0]
    #         self.dX1[rb * self.shape[1] + cb, idx] = dar[1]
    #     self.X = self.X.tocsr()
    #     self.dX0 = self.dX0.tocsr()
    #     self.dX1 = self.dX1.tocsr()

    def _get_X(self):
        row, col, data, grad0, grad1 = [], [], [], [], []
        for location in self.locations:
            r, c, ar, g0, g1 = self.psf.prf(
                        row=location[0],
                        column=location[1],
                        gradients=True
                    )
            row.append(r[:, None] * np.ones(c.shape[0], int))
            col.append(c[None, :] * np.ones(r.shape[0], int)[:, None])
            grad0.append(g0)
            grad1.append(g1)
            data.append(ar)
        data, row, col, grad0, grad1 = np.asarray(data), np.asarray(row), np.asarray(col), np.asarray(grad0), np.asarray(grad1)        
        self.X = SparseWarp3D(data.transpose([1, 2, 0]), row.transpose([1, 2, 0]) - self.corner[0], col.transpose([1, 2, 0]) - self.corner[1], imshape=self.shape)
        self.dX0 = SparseWarp3D(grad0.transpose([1, 2, 0]), row.transpose([1, 2, 0]) - self.corner[0], col.transpose([1, 2, 0]) - self.corner[1], imshape=self.shape)
        self.dX1 = SparseWarp3D(grad1.transpose([1, 2, 0]), row.transpose([1, 2, 0]) - self.corner[0], col.transpose([1, 2, 0]) - self.corner[1], imshape=self.shape)
        return


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
        if jitter is not None:
            if jitter.ndim == 1:
                jitter = jitter[:, None]
            if (jitter.shape[0] != 2) | (jitter.shape[1] != nt):
                raise ValueError("`jitter` must be an array with shape (2 x ntimes).")

        ar = self.X.dot(flux)
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
        self.X = sparse.hstack(Xs, format="csr")
        self.dX0 = sparse.hstack(dX0s, format="csr")
        self.dX1 = sparse.hstack(dX1s, format="csr")

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
            if jitter.ndim == 1:
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


class SparseWarp3D(sparse.coo_matrix):
    """Special class for working with stacks of sparse 3D images"""

    def __init__(self, data, row, col, imshape):
        if not np.all([row.ndim == 3, col.ndim == 3, data.ndim == 3]):
            raise ValueError("Pass a 3D array (nrow, ncol, nvecs)")
        self.nvecs = data.shape[-1]
        if not np.all(
            [
                row.shape[-1] == self.nvecs,
                col.shape[-1] == self.nvecs,
            ]
        ):
            raise ValueError("Must have the same 3rd dimension (nvecs).")
        self.subrow = row.astype(int)
        self.subcol = col.astype(int)
        self.subdepth = (
            np.arange(row.shape[-1], dtype=int)[None, None, :]
            * np.ones(row.shape, dtype=int)[:, :, None]
        )
        self.subdata = data
        self.imshape = imshape
        self.subshape = row.shape
        self.cooshape = (np.prod([*self.imshape[:2]]), self.nvecs)
        super().__init__(self.cooshape)
        self._set_data()

    def index(self, offset=(0, 0)):
        """Get the 2D positions of the data"""
        index0 = (np.vstack(self.subrow) + offset[0]) * self.imshape[0] + (
            np.vstack(self.subcol) + offset[1]
        )
        index1 = np.vstack(self.subdepth).ravel()
        return np.vstack([index0.ravel(), index1.ravel()])

    def _get_submask(self, offset=(0, 0)):
        # find where the data is within the array bounds
        kr = ((self.subrow[:, 0, 0] + offset[0]) < self.imshape[0]) & (
            (self.subrow[:, 0, 0] + offset[0]) >= 0
        )
        kc = ((self.subcol[0, :, 0] + offset[1]) < self.imshape[1]) & (
            (self.subcol[0, :, 0] + offset[1]) >= 0
        )
        return (kr[:, None, None] & kc[None, :, None]) * np.ones(self.subshape[-1], bool)

    def _set_data(self, offset=(0, 0)):
        # find where the data is within the array bounds
        k = np.vstack(
            self._get_submask(offset=offset)
        ).ravel()
        new_row, new_col = self.index(offset)
        self.row, self.col = new_row[k], new_col[k]
        self.data = np.vstack(deepcopy(self.subdata)).ravel()[k]

    def __repr__(self):
        return f"<{(*self.imshape, self.nvecs)} SparseWarp3D array of type {self.dtype}>"

    def dot(self, other):
        if other.ndim == 1:
            other = other[:, None]
        nt = other.shape[1]        
        return super().dot(other).reshape((nt, *self.imshape))

    def reset(self):
        """Reset any translation back to the original data"""
        self._set_data(offset=(0, 0))
        return

    def clear(self):
        """Clear data in the array"""
        self.data = np.asarray([])
        self.row = np.asarray([])
        self.col = np.asarray([])
        return

    def translate(self, position: Tuple):
        """Translate the data in the array by `position` in (row, column)"""
        self.reset()
        # If translating to (0, 0), do nothing
        if position == (0, 0):
            return
        self.clear()
        self._set_data(position)
        return
