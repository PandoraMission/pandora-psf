"""Class to deal with scenes?"""

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse
from tqdm import tqdm

from .psf import PSF
from .utils import downsample as downsample_array
from .utils import prep_for_add

__all__ = ["Scene", "TraceScene"]


class Scene(object):
    def __init__(
        self,
        locations: npt.ArrayLike,
        psf: PSF = None,
        shape: Tuple = (100, 100),
        corner: Tuple = (0, 0),
        scale: int = 1,
    ):
        if locations.shape[1] != 2:
            raise ValueError("`locations` must have shape (n, 2).")
        self.locations = locations
        #        self.shape = shape
        #        self.corner = corner
        if psf is None:
            psf = PSF.from_name("visda", scale=scale)
        self.psf = psf
        self.scale = self.psf.scale
        self.shape = tuple(np.asarray(shape) * self.scale)
        self.corner = tuple(np.asarray(corner) * self.scale)
        self.rb, self.cb, self.prf = [], [], []
        self.ntargets = len(self.locations)
        self._get_X()

    def _get_ar(self, flux, delta_pos=None, quiet=True):
        nt = flux.shape[1]
        ar = np.zeros((nt, *self.shape))
        if delta_pos is not None:
            jitterdec = (delta_pos - 0.5) % 1 - 0.5
            jitterint = np.round(delta_pos - jitterdec).astype(int)  # + 1
            unique_coordinates, unique_indices = np.unique(
                jitterint.T, axis=0, return_inverse=True
            )
            for index, coord in tqdm(
                enumerate(unique_coordinates),
                total=len(unique_coordinates),
                leave=True,
                position=0,
                desc="Modeling Pixel Positions",
                disable=quiet,
            ):
                self.X.translate(tuple(coord))
                self.dX0.translate(tuple(coord))
                self.dX1.translate(tuple(coord))
                tdxs = np.where(unique_indices == index)[0]
                ar[tdxs] = self.X.dot(flux[:, tdxs])
                ar[tdxs] += self.dX0.dot(
                    flux[:, tdxs] * -jitterdec[0, tdxs]
                ) + self.dX1.dot(flux[:, tdxs] * -jitterdec[1, tdxs])
        else:
            ar = self.X.dot(flux)
        return ar

    def __repr__(self):
        return f"Scene Object [{self.psf.__repr__()}]"

    def __len__(self):
        return len(self.locations)

    def _get_X(self):
        row, col, data, grad0, grad1 = [], [], [], [], []
        for location in self.locations * self.scale:
            r, c, ar, g0, g1 = self.psf.prf(
                row=location[0], column=location[1], gradients=True
            )
            row.append(r[:, None] * np.ones(c.shape[0], int))
            col.append(c[None, :] * np.ones(r.shape[0], int)[:, None])
            grad0.append(g0)
            grad1.append(g1)
            data.append(ar)
        data, row, col, grad0, grad1 = (
            np.asarray(data),
            np.asarray(row),
            np.asarray(col),
            np.asarray(grad0),
            np.asarray(grad1),
        )
        self.X = SparseWarp3D(
            data.transpose([1, 2, 0]),
            row.transpose([1, 2, 0]) - self.corner[0],
            col.transpose([1, 2, 0]) - self.corner[1],
            imshape=self.shape,
        )
        self.dX0 = SparseWarp3D(
            grad0.transpose([1, 2, 0]),
            row.transpose([1, 2, 0]) - self.corner[0],
            col.transpose([1, 2, 0]) - self.corner[1],
            imshape=self.shape,
        )
        self.dX1 = SparseWarp3D(
            grad1.transpose([1, 2, 0]),
            row.transpose([1, 2, 0]) - self.corner[0],
            col.transpose([1, 2, 0]) - self.corner[1],
            imshape=self.shape,
        )
        return

    def model(
        self,
        flux: npt.ArrayLike,
        delta_pos: Optional[npt.ArrayLike] = None,
        quiet: bool = False,
        downsample: bool = True,
    ) -> npt.ArrayLike:
        """
        Parameters:
        -----------
        flux : npt.ArrayLike
            Array of flux values with shape (ntargets, ntimes)
        delta_pos : npt.ArrayLike
            Array of jitter values in row and column, has shape (2, ntimes)
        """
        if flux.ndim == 1:
            flux = flux[:, None]
        if flux.shape[0] != self.ntargets:
            raise ValueError("`flux` must be an array with shape (ntargets x ntimes).")
        # nt = flux.shape[1]
        if delta_pos is not None:
            delta_pos = deepcopy(delta_pos) * float(self.scale)
        #     if delta_pos.ndim == 1:
        #         delta_pos = delta_pos[:, None]
        #     if (delta_pos.shape[0] != 2) | (delta_pos.shape[1] != nt):
        #         raise ValueError(
        #             "`delta_pos` must be an array with shape (2 x ntimes)."
        #         )

        # if delta_pos is not None:
        #     ar = np.zeros((nt, *self.shape))
        #     jitterdec = (delta_pos - 0.5) % 1 - 0.5
        #     jitterint = np.round(delta_pos - jitterdec).astype(int)  # + 1
        #     unique_coordinates, unique_indices = np.unique(
        #         jitterint.T, axis=0, return_inverse=True
        #     )
        #     for index, coord in enumerate(unique_coordinates):
        #         self.X.translate(tuple(coord))
        #         self.dX0.translate(tuple(coord))
        #         self.dX1.translate(tuple(coord))
        #         grad_ar = deepcopy(self.dX0)
        #         tdxs = np.where(unique_indices == index)[0]
        #         for tdx in tqdm(
        #             tdxs,
        #             desc=f"Time index {index + 1}/{len(unique_coordinates)}",
        #             leave=True,
        #             position=0,
        #             disable=quiet,
        #         ):
        #             grad_ar.data = (
        #                 deepcopy(self.dX0.data) * -jitterdec[0, tdx]
        #                 + deepcopy(self.dX1.data) * -jitterdec[1, tdx]
        #             )
        #             ar[tdx] += self.X.dot(flux[:, tdx].ravel())[0]
        #             ar[tdx] += grad_ar.dot(flux[:, tdx].ravel())[0]
        # else:
        #     ar = self.X.dot(flux)

        ar = self._get_ar(flux=flux, delta_pos=delta_pos, quiet=quiet)

        if self.scale == 1:
            return ar
        if downsample:
            return downsample_array(ar, self.scale)
        else:
            return ar


class TraceScene(Scene):
    def __init__(
        self,
        locations: npt.ArrayLike,
        psf: PSF = None,
        shape: Tuple = (400, 80),
        corner: Tuple = (0, 0),
        scale: int = 1,
        wav_bin: int = 4,
    ):
        if locations.shape[1] != 2:
            raise ValueError("`locations` must have shape (n, 2).")
        self.locations = locations
        if psf is None:
            psf = PSF.from_name("nirda", scale=scale)
        self.psf = psf
        self.scale = self.psf.scale
        self.shape = tuple(np.asarray(shape) * self.scale)
        self.corner = tuple(np.asarray(corner) * self.scale)
        if "wavelength" not in self.psf.dimension_names:
            raise ValueError(
                "Can only create a trace scene if `PSF` has a `'wavelength'` dimension"
            )
        if not hasattr(self.psf, "trace_pixel"):
            raise ValueError("No trace parameters, you need to set them.")
        self.ntargets = len(self.locations)
        self.rb, self.cb, self.prf = [], [], []
        self.wav_bin = wav_bin
        self._get_Xs(nbin=self.wav_bin)

    def __repr__(self):
        return f"TraceScene Object [{self.psf.__repr__()}]"

    def _get_Xs(self, nbin=4):
        def get_chunked_data(locations, pixels, waves):
            """Loops through pixel trave groups and gets each PRF element into a dictionary"""
            res = {}
            for pdx in range(len(pixels)):
                for location in locations:
                    key = f"({location[0]}, {location[1]}) {pdx}"
                    res[key] = {
                        pixels[pdx][idx].value: self.psf.prf(
                            row=location[0] + pixels[pdx][idx].value,
                            column=location[1],
                            wavelength=waves[pdx][idx],
                            gradients=True,
                        )
                        for idx in range(len(pixels[pdx]))
                    }
                    r = np.asarray(
                        [
                            [res[key][pix][0].min(), res[key][pix][0].max()]
                            for pix in pixels[pdx].value
                        ]
                    )
                    r = (np.min(r[:, 0]), np.max(r[:, 1]))
                    c = np.asarray(
                        [
                            [res[key][pix][1].min(), res[key][pix][1].max()]
                            for pix in pixels[pdx].value
                        ]
                    )
                    c = (np.min(c[:, 0]), np.max(c[:, 1]))
                    res[key]["r"] = r
                    res[key]["c"] = c
            return res

        def collapse(dic, shape):
            """For each location and pixel group, collapses the data into a single, small image"""
            rshape, cshape = shape
            corner = (dic["r"][0], dic["c"][0])
            ar, dar0, dar1 = np.zeros((3, rshape, cshape))
            for key, item in dic.items():
                if key in ["r", "c"]:
                    continue
                rb, cb, arb = prep_for_add(
                    row=item[0],
                    column=item[1],
                    prf=item[2],
                    shape=(rshape, cshape),
                    corner=corner,
                )
                _, _, dar0b = prep_for_add(
                    row=item[0],
                    column=item[1],
                    prf=item[3],
                    shape=(rshape, cshape),
                    corner=corner,
                )
                _, _, dar1b = prep_for_add(
                    row=item[0],
                    column=item[1],
                    prf=item[4],
                    shape=(rshape, cshape),
                    corner=corner,
                )
                ar[rb, cb] += arb
                dar0[rb, cb] += dar0b
                dar1[rb, cb] += dar1b

            R, C = np.mgrid[:rshape, :cshape]
            R += corner[0]
            C += corner[1]
            corr = np.sum(ar)

            return (
                R,
                C,
                ar / corr,
                (dar0 - dar0.mean()) / corr,
                (dar1 - dar1.mean()) / corr,
            )

        pixs, wavs = self.psf.trace_pixel * self.psf.scale, self.psf.trace_wavelength
        dwavs = np.gradient(wavs)
        pixs, wavs, dwavs = (
            np.array_split(pixs, len(pixs) / nbin),
            np.array_split(wavs, len(wavs) / nbin),
            np.array_split(dwavs, len(dwavs) / nbin),
        )

        bounds0 = np.asarray(
            [wav[0].value - dwav[0].value / 2 for wav, dwav in zip(wavs, dwavs)]
        )
        bounds1 = np.hstack(
            [*bounds0[1:], wavs[-1][-1].value + dwavs[-1][-1].value / 2]
        )
        self.bounds = np.asarray([bounds0, bounds1])
        self.nwav = self.bounds.shape[1]
        data, grad0, grad1 = np.zeros((3, len(pixs), len(self.locations), *self.shape))

        res = get_chunked_data(self.locations * self.psf.scale, pixs, wavs)

        r = np.asarray([item["r"] for _, item in res.items()])
        c = np.asarray([item["c"] for _, item in res.items()])
        shape = np.max(r[:, 1] - r[:, 0]), np.max(c[:, 1] - c[:, 0])

        R, C, data, grad0, grad1 = np.zeros((5, len(res), *shape))
        for idx, key in enumerate(res.keys()):
            R[idx], C[idx], data[idx], grad0[idx], grad1[idx] = collapse(
                res[key], shape
            )
        data, grad0, grad1 = data, grad0, grad1
        self.X = SparseWarp3D(
            data.transpose([1, 2, 0]),
            R.transpose([1, 2, 0]) - self.corner[0],
            C.transpose([1, 2, 0]) - self.corner[1],
            imshape=self.shape,
        )

        self.dX0 = SparseWarp3D(
            grad0.transpose([1, 2, 0]),
            R.transpose([1, 2, 0]) - self.corner[0],
            C.transpose([1, 2, 0]) - self.corner[1],
            imshape=self.shape,
        )

        self.dX1 = SparseWarp3D(
            grad1.transpose([1, 2, 0]),
            R.transpose([1, 2, 0]) - self.corner[0],
            C.transpose([1, 2, 0]) - self.corner[1],
            imshape=self.shape,
        )

    # def _get_Xs(self, nbin=4):
    #     pixs, wavs = self.psf.trace_pixel * self.psf.scale, self.psf.trace_wavelength
    #     mask = self.psf.trace_sensitivity/self.psf.trace_sensitivity.max() > 0.0001
    #     pixs, wavs = pixs[mask], wavs[mask]
    #     dwavs = np.gradient(wavs)
    #     pixs, wavs, dwavs = np.array_split(pixs, len(pixs)/nbin),
    #     np.array_split(wavs, len(wavs)/nbin), np.array_split(dwavs, len(dwavs)/nbin)

    #     bounds0 = np.asarray([wav[0].value - dwav[0].value/2 for wav, dwav in zip(wavs, dwavs)])
    #     bounds1 = np.hstack([*bounds0[1:], wavs[-1][-1].value + dwavs[-1][-1].value/2])
    #     self.bounds = np.asarray([bounds0, bounds1])
    #     self.nwav = self.bounds.shape[1]
    #     data, grad0, grad1 = np.zeros((3, len(pixs), len(self.locations), *self.shape))
    #     for pdx, pix, wav in tqdm(zip(range(len(pixs)), pixs, wavs), total=len(pixs)):
    #         for ldx, location in enumerate(self.locations * self.psf.scale):
    #             for idx in range(len(pix)):
    #                 r, c, ar, g0, g1 = self.psf.prf(
    #                     row=location[0] + pix.value[idx],
    #                     column=location[1],
    #                     wavelength=wav[idx],
    #                     gradients=True,
    #                 )
    #                 row, col, ar = prep_for_add(r, c, ar, shape=self.shape, corner=self.corner)
    #                 _, _, g0 = prep_for_add(r, c, g0, shape=self.shape, corner=self.corner)
    #                 _, _, g1 = prep_for_add(r, c, g1, shape=self.shape, corner=self.corner)
    #                 data[pdx, ldx, row, col] += ar
    #                 grad0[pdx, ldx, row, col] += g0
    #                 grad1[pdx, ldx, row, col] += g1
    #     data, grad0, grad1 = np.vstack(data), np.vstack(grad0), np.vstack(grad1)
    #     R, C = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing='ij')
    #     R, C = R[None, :, :] * np.ones(data.shape) - self.corner[0], C[None, :, :] * np.ones(data.shape) - self.corner[1]

    #     self.X = SparseWarp3D(
    #                 data.transpose([1, 2, 0]),
    #                 R.transpose([1, 2, 0]),
    #                 C.transpose([1, 2, 0]),
    #                 imshape=self.shape,
    #     )
    #     self.dX0 = SparseWarp3D(
    #                 grad0.transpose([1, 2, 0]),
    #                 R.transpose([1, 2, 0]),
    #                 C.transpose([1, 2, 0]),
    #         imshape=self.shape,
    #     )
    #     self.dX1 = SparseWarp3D(
    #                 grad1.transpose([1, 2, 0]),
    #                 R.transpose([1, 2, 0]),
    #                 C.transpose([1, 2, 0]),
    #         imshape=self.shape,
    #     )
    #     return
    # pixs, wavs = self.psf.trace_pixel * self.psf.scale, self.psf.trace_wavelength
    # mask = self.psf.trace_sensitivity/self.psf.trace_sensitivity.max() > 0.0001
    # pixs, wavs = pixs[mask], wavs[mask]
    # dwavs = np.gradient(wavs)
    # pixs, wavs, dwavs = np.array_split(pixs, len(pixs)/nbin), np.array_split(wavs, len(wavs)/nbin), np.array_split(dwavs, len(dwavs)/nbin)

    # bounds0 = np.asarray([wav[0].value - dwav[0].value/2 for wav, dwav in zip(wavs, dwavs)])
    # bounds1 = np.hstack([*bounds0[1:], wavs[-1][-1].value + dwavs[-1][-1].value/2])
    # self.bounds = np.asarray([bounds0, bounds1])
    # self.nwav = self.bounds.shape[1]
    # data, grad0, grad1 = np.zeros((3, len(pixs), len(self.locations), *self.shape))
    # rows, cols, datas, grad0s, grad1s = [], [], [], [], []
    # for pix, wav in tqdm(zip(pixs, wavs), total=len(pixs)):
    #     row, col, data, grad0, grad1 = [], [], [], [], []
    #     for location in self.locations * self.psf.scale:
    #         for idx in range(len(pix)):
    #             r, c, ar, g0, g1 = self.psf.prf(
    #                 row=location[0] + pix.value[idx],
    #                 column=location[1],
    #                 wavelength=wav[idx],
    #                 gradients=True,
    #             )

    #         row.append(r[:, None] * np.ones(c.shape[0], int))
    #         col.append(c[None, :] * np.ones(r.shape[0], int)[:, None])
    #         grad0.append(g0 - g0.mean())
    #         grad1.append(g1 - g1.mean())
    #         data.append(ar)
    #     data, row, col, grad0, grad1 = (
    #         np.asarray(data),
    #         np.asarray(row),
    #         np.asarray(col),
    #         np.asarray(grad0),
    #         np.asarray(grad1),
    #     )
    #     rows.append(row)
    #     cols.append(col)
    #     grad0s.append(grad0)
    #     grad1s.append(grad1)
    #     datas.append(data)
    # rows, cols, datas, grad0s, grad1s = (
    #     np.vstack(rows),
    #     np.vstack(cols),
    #     np.vstack(datas),
    #     np.vstack(grad0s),
    #     np.vstack(grad1s),
    # )
    # self.X = SparseWarp3D(
    #     datas.transpose([1, 2, 0]),
    #     rows.transpose([1, 2, 0]) - self.corner[0],
    #     cols.transpose([1, 2, 0]) - self.corner[1],
    #     imshape=self.shape,
    # )
    # self.dX0 = SparseWarp3D(
    #     grad0s.transpose([1, 2, 0]),
    #     rows.transpose([1, 2, 0]) - self.corner[0],
    #     cols.transpose([1, 2, 0]) - self.corner[1],
    #     imshape=self.shape,
    # )
    # self.dX1 = SparseWarp3D(
    #     grad1s.transpose([1, 2, 0]),
    #     rows.transpose([1, 2, 0]) - self.corner[0],
    #     cols.transpose([1, 2, 0]) - self.corner[1],
    #     imshape=self.shape,
    # )
    # return

    # def _get_Xs(self):
    #     rows, cols, datas, grad0s, grad1s = [], [], [], [], []
    #     for pix, wav in tqdm(
    #         zip(self.psf.trace_pixel * self.scale, self.psf.trace_wavelength),
    #         total=len(self.psf.trace_wavelength),
    #     ):
    #         row, col, data, grad0, grad1 = [], [], [], [], []
    #         for location in self.locations * self.scale:
    #             r, c, ar, g0, g1 = self.psf.prf(
    #                 row=location[0] + pix.value,
    #                 column=location[1],
    #                 wavelength=wav,
    #                 gradients=True,
    #             )
    #             row.append(r[:, None] * np.ones(c.shape[0], int))
    #             col.append(c[None, :] * np.ones(r.shape[0], int)[:, None])
    #             grad0.append(g0)
    #             grad1.append(g1)
    #             data.append(ar)
    #         data, row, col, grad0, grad1 = (
    #             np.asarray(data),
    #             np.asarray(row),
    #             np.asarray(col),
    #             np.asarray(grad0),
    #             np.asarray(grad1),
    #         )
    #         rows.append(row)
    #         cols.append(col)
    #         grad0s.append(grad0)
    #         grad1s.append(grad1)
    #         datas.append(data)
    #     rows, cols, datas, grad0s, grad1s = (
    #         np.vstack(rows),
    #         np.vstack(cols),
    #         np.vstack(datas),
    #         np.vstack(grad0s),
    #         np.vstack(grad1s),
    #     )
    #     self.X = SparseWarp3D(
    #         datas.transpose([1, 2, 0]),
    #         rows.transpose([1, 2, 0]) - self.corner[0],
    #         cols.transpose([1, 2, 0]) - self.corner[1],
    #         imshape=self.shape,
    #     )
    #     self.dX0 = SparseWarp3D(
    #         grad0s.transpose([1, 2, 0]),
    #         rows.transpose([1, 2, 0]) - self.corner[0],
    #         cols.transpose([1, 2, 0]) - self.corner[1],
    #         imshape=self.shape,
    #     )
    #     self.dX1 = SparseWarp3D(
    #         grad1s.transpose([1, 2, 0]),
    #         rows.transpose([1, 2, 0]) - self.corner[0],
    #         cols.transpose([1, 2, 0]) - self.corner[1],
    #         imshape=self.shape,
    #     )
    #     return

    def model(
        self,
        spectra: npt.ArrayLike,
        delta_pos: Optional[npt.ArrayLike] = None,
        quiet: bool = False,
        downsample: bool = True,
    ) -> npt.ArrayLike:
        """`spectra` must have shape nwav x ntargets x ntime"""
        if spectra.ndim == 1:
            spectra = spectra[:, None, None]
        elif spectra.ndim == 2:
            spectra = spectra[:, :, None]
        elif spectra.ndim != 3:
            raise ValueError("Pass a 3D array for flux (nwav, ntargets, ntime).")
        if (spectra.shape[0] != self.nwav) | (spectra.shape[1] != len(self)):
            raise ValueError("`spectra` must have shape (nwav, ntargets)")
        if delta_pos is not None:
            delta_pos = deepcopy(delta_pos) * float(self.scale)
            if delta_pos.ndim == 1:
                delta_pos = delta_pos[:, None]
            elif delta_pos.ndim != 2:
                raise ValueError("Pass 2D array for delta_pos (2, ntime).")

        ar = self._get_ar(flux=np.vstack(spectra), delta_pos=delta_pos, quiet=quiet)

        if self.scale == 1:
            return ar
        if downsample:
            return downsample_array(ar, self.scale)
        else:
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
