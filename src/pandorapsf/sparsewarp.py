"""Magic classes for working with sparse data."""
from copy import deepcopy
from typing import Tuple

import numpy as np
from scipy import sparse

__all__ = ["SparseWarp3D", "ROISparseWarp3D"]


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
        self._kz = self.subdata != 0

        self.imshape = imshape
        self.subshape = row.shape
        self.cooshape = (np.prod([*self.imshape[:2]]), self.nvecs)
        self.coord = (0, 0)
        super().__init__(self.cooshape)
        index0 = (np.vstack(self.subrow)) * self.imshape[1] + (np.vstack(self.subcol))
        index1 = np.vstack(self.subdepth).ravel()
        self._index_no_offset = np.vstack([index0.ravel(), index1.ravel()])
        self._submask_no_offset = np.vstack(self._get_submask(offset=(0, 0))).ravel()
        self._subrow_v = deepcopy(np.vstack(self.subrow).ravel())
        self._subcol_v = deepcopy(np.vstack(self.subcol).ravel())
        self._subdata_v = deepcopy(np.vstack(deepcopy(self.subdata)).ravel())
        self._index1 = np.vstack(self.subdepth).ravel()

        self._set_data()

    def __add__(self, other):
        if isinstance(other, SparseWarp3D):
            data = deepcopy(self.subdata + other.subdata)
            if (
                (self.subcol != other.subcol)
                | (self.subrow != other.subrow)
                | (self.imshape != other.imshape)
                | (self.subshape != other.subshape)
            ):
                raise ValueError("Must have same base indicies.")
            return SparseWarp3D(
                data=data, row=self.subrow, col=self.subcol, imshape=self.imshape
            )
        else:
            return super(sparse.coo_matrix, self).__add__(other)

    def tocoo(self):
        return sparse.coo_matrix((self.data, (self.row, self.col)), shape=self.cooshape)

    def index(self, offset=(0, 0)):
        """Get the 2D positions of the data"""
        if offset == (0, 0):
            return self._index_no_offset
        index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
            self._subcol_v + offset[1]
        )
        #        index1 = np.vstack(self.subdepth).ravel()
        #        return np.vstack([index0.ravel(), index1.ravel()])
        # index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
        #     self._subcol_v * offset[1]
        # )
        return index0, self._index1
        # index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
        #     self._subcol_v * offset[1]
        # )
        # return index0, self._index1

    def _get_submask(self, offset=(0, 0)):
        # find where the data is within the array bounds
        kr = ((self.subrow + offset[0]) < self.imshape[0]) & (
            (self.subrow + offset[0]) >= 0
        )
        kc = ((self.subcol + offset[1]) < self.imshape[1]) & (
            (self.subcol + offset[1]) >= 0
        )
        return kr & kc & self._kz

    def _set_data(self, offset=(0, 0)):
        if offset == (0, 0):
            index0, index1 = self.index((0, 0))
            self.row, self.col = (
                index0[self._submask_no_offset],
                index1[self._submask_no_offset],
            )
            self.data = self._subdata_v[self._submask_no_offset]
        else:
            # find where the data is within the array bounds
            k = self._get_submask(offset=offset)
            k = np.vstack(k).ravel()
            new_row, new_col = self.index(offset=offset)
            self.row, self.col = new_row[k], new_col[k]
            self.data = self._subdata_v[k]
        self.coord = offset

    def __repr__(self):
        return (
            f"<{(*self.imshape, self.nvecs)} SparseWarp3D array of type {self.dtype}>"
        )

    def dot(self, other):
        if not isinstance(other, np.ndarray):
            raise NotImplementedError(
                f"dot products with type {type(other)} are not implemented."
            )
        if other.ndim == 1:
            other = other[:, None]
        nt = other.shape[1]
        return super().dot(other).reshape((*self.imshape, nt)).transpose([2, 0, 1])

    def reset(self):
        """Reset any translation back to the original data"""
        self._set_data(offset=(0, 0))
        self.coord = (0, 0)
        return

    def clear(self):
        """Clear data in the array"""
        self.data = np.asarray([])
        self.row = np.asarray([])
        self.col = np.asarray([])
        self.coord = (0, 0)
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

    def copy(self):
        return deepcopy(self)


class ROISparseWarp3D(SparseWarp3D):
    """Special version of a SparseWarp3D matrix which only populates and works with data within Regions of Interest."""

    def __init__(self, data, row, col, imshape, nROIs, ROI_size, ROI_corners):
        self.nROIs = nROIs
        self.ROI_size = ROI_size
        self.ROI_corners = ROI_corners
        self.get_ROI_mask = self.parse_ROIS(nROIs, ROI_size, ROI_corners)
        super().__init__(data=data, row=row, col=col, imshape=imshape)

    def parse_ROIS(self, nROIs: int, ROI_size: tuple, ROI_corners: list):
        if not len(ROI_corners) == nROIs:
            raise ValueError("Must pass corners for all ROIs.")
        if not np.all([isinstance(corner, tuple) for corner in ROI_corners]):
            raise ValueError("Pass corners as tuples.")

        def get_ROI_masks(row, column):
            mask = []
            for roi in range(nROIs):
                rmin, cmin = ROI_corners[roi]
                rmax, cmax = rmin + ROI_size[0], cmin + ROI_size[1]
                mask.append(
                    (row >= rmin) & (row < rmax) & (column >= cmin) & (column < cmax)
                )
            return np.asarray(mask)

        return get_ROI_masks

    def __repr__(self):
        return f"<{(*self.imshape, self.nvecs)} ROISparseWarp3D array of type {self.dtype}, {self.nROIs} Regions of Interest>"

    def _get_submask(self, offset=(0, 0)):
        # find where the data is within the array bounds
        kr = ((self.subrow + offset[0]) < self.imshape[0]) & (
            (self.subrow + offset[0]) >= 0
        )
        kc = ((self.subcol + offset[1]) < self.imshape[1]) & (
            (self.subcol + offset[1]) >= 0
        )
        kroi = self.get_ROI_mask(self.subrow + offset[0], self.subcol + offset[0]).any(
            axis=0
        )
        return kr & kc & kroi & self._kz

    def dot(self, other):
        if isinstance(other, np.ndarray):
            other = sparse.csr_matrix(other).T
        if not sparse.issparse(other):
            raise ValueError("Must pass a `sparse` array to dot.")
        if not other.shape[0] == self.nvecs:
            if other.shape[1] == self.nvecs:
                other = other.T
            else:
                raise ValueError(f"Must pass {(self.nvecs, 1)} shape object.")
        sparse_array = super().tocsr().dot(other)

        R, C = np.meshgrid(
            np.arange(0, self.ROI_size[0]),
            np.arange(0, self.ROI_size[1]),
            indexing="ij",
        )
        array = np.zeros((self.nROIs, other.shape[1], *self.ROI_size))
        for rdx, c in enumerate(self.ROI_corners):
            idx = (R.ravel() + c[0]) * self.imshape[1] + (C.ravel() + c[1])
            k = (idx >= 0) & (idx < self.shape[0])
            array[rdx, :, k.reshape(self.ROI_size)] = sparse_array[
                idx[k]
            ].toarray()  # ).reshape(self.ROI_size))
        return array
