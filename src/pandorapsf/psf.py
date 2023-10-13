"""Defines the PSF class"""

# Standard library
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

# Third-party
import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits

from . import PACKAGEDIR
from .utils import bin_prf

__all__ = ["PSF"]


class PSF(object):
    """Class to use PSFs"""

    def __init__(
        self,
        X: npt.ArrayLike,
        psf_flux: npt.ArrayLike,
        dimension_names: List,
        dimension_units: List,
        pixel_size: u.Quantity,
        sub_pixel_size: u.Quantity,
        transpose: bool = False,
        freeze_dictionary: Dict = {},
        blur_value: Tuple = (0 * u.pixel, 0 * u.pixel),
        check_bounds: bool = True,
        extrapolate: bool = False,
        scale: int = 1,
    ):
        """
        PSF class for making PSFs, PRFs, and traces.

        Parameters:
        -----------
        X : np.NDArray
            Array of 1D vectors defining the value of each dimension for every element of `psf_flux`.
            Should have as many entries as `psf_flux` has dimensions.
        psf_flux: np.NDArray
            ND array of flux values
        dimension_names: list
            List of names for each of the N dimensions in `psf_flux`
        dimension_units: list
            List of `astropy.unit.Quantity`'s describing units of each dimension
        pixel_size: u.Quantity
            True detector pixel size in dimensions of length/pixel
        sub_pixel_size: u.Quantity
            PSF file pixel size in dimensions of length/pixel
        transpose: bool
            Whether to transpose the data in the column/row axis
        freeze_dictionary: dict
            Dictionary of dimensions to freeze
        blur_value: tuple
            Tuple of astropy quantities in pixels describing the amount of blur in each axis.
        extrapolate : bool
            Whether to allow the PSF to be evaluated outside of the bounds (i.e. will extrapolate)
        """
        self._psf_flux = psf_flux
        self.transpose = transpose
        if self.transpose:
            self._psf_flux = self._psf_flux.transpose(
                [1, 0, *np.arange(2, self._psf_flux.ndim)]
            )
        self.dimension_names = dimension_names
        self.dimension_units = dimension_units
        self.pixel_size = pixel_size
        self.sub_pixel_size = sub_pixel_size
        self.freeze_dictionary = freeze_dictionary
        self.blur_value = blur_value
        self.check_bounds = check_bounds
        self.extrapolate = extrapolate
        self.scale = scale

        for name, x in zip(dimension_names, X):
            setattr(self, name, x)
        self._validate()

    def _validate(self):
        """Method that finishes the set up of the PSF"""
        self.shape = self._psf_flux.shape[:2]
        self.ndims = len(self.dimension_names)
        # Set up 1D arrays
        if self.ndims == 1:
            setattr(
                self,
                self.dimension_names[0] + "1d",
                getattr(self, self.dimension_names[0]),
            )
            midpoint = getattr(self, self.dimension_names[0])
            midpoint = midpoint[len(midpoint) // 2]
            setattr(self, self.dimension_names[0] + "0d", midpoint)
        else:
            # Get 1D version of these grids
            dims = set(np.arange(self.ndims))
            for dim in np.arange(self.ndims):
                lp = getattr(self, self.dimension_names[dim]).transpose(
                    np.hstack([dim, list(dims - set([dim]))])
                )
                for d in range(self.ndims - 1):
                    lp = np.take(lp, 0, -1)
                s = np.argsort(lp.value)
                setattr(self, self.dimension_names[dim] + "1d", lp[s])
                # We have to do this to sort the axis, having them sorted will mean it's easier to interpolate later...
                reshape = np.hstack(
                    [np.hstack([dim, list(dims - set([dim]))]) + 2, 0, 1]
                )
                deshape = [
                    np.where(reshape == idx)[0][0] for idx in range(len(reshape))
                ]
                self._psf_flux = self._psf_flux.transpose(reshape)[s].transpose(deshape)
                midpoint = getattr(self, self.dimension_names[dim] + "1d")
                midpoint = midpoint[len(midpoint) // 2]
                setattr(self, self.dimension_names[dim] + "0d", midpoint)

        self.psf_column = (
            self.scale
            * (
                (np.arange(self.shape[1]) - self.shape[1] // 2)
                * u.pixel
                * self.sub_pixel_size
            )
            / self.pixel_size
        )
        self.psf_column -= np.median(np.diff(self.psf_column)) / 2
        self.psf_row = (
            self.scale
            * (
                (np.arange(self.shape[0]) - self.shape[0] // 2)
                * u.pixel
                * self.sub_pixel_size
            )
            / self.pixel_size
        )
        self.psf_row -= np.median(np.diff(self.psf_row)) / 2
        self.midpoint = tuple(
            [getattr(self, name + "0d") for name in self.dimension_names]
        )
        for dim, p in enumerate(self.dimension_names):
            lp = getattr(self, self.dimension_names[dim])
            setattr(self, p + "_bounds", [lp.min(), lp.max()])
        #        self._psf_flux_jitter = np.zeros_like(self._psf_flux)
        self._blur(self.blur_value)

    #       self.psf_flux = self._psf_flux_blur #+ self._psf_flux_jitter

    def _blur(self, blur_value: Tuple):
        """Blurs the PSF using a Gaussian Kernel. Will update `self._psf_flux_blur` and `self._psf_flux_blur_grad`.

        Parameters:
        -----------
        blur_value: tuple of astropy quantities, must be in units of pixels
        """
        xstd, ystd = blur_value
        xstd, ystd = u.Quantity(xstd, "pixel"), u.Quantity(ystd, "pixel")

        xstd = ((self.pixel_size * xstd) / self.sub_pixel_size).value
        ystd = ((self.pixel_size * ystd) / self.sub_pixel_size).value
        a = deepcopy(self._psf_flux)
        if (xstd == 0) & (ystd == 0):
            self._psf_flux_blur = deepcopy(self._psf_flux)
            self._psf_flux_blur_grad = (
                np.asarray(
                    np.gradient(
                        self._psf_flux_blur,
                        np.median(np.diff(self.psf_row)).value / self.scale,
                        axis=(0, 1),
                    )
                )
                * (self.sub_pixel_size / self.pixel_size).value
            )
            self.psf_flux = deepcopy(self._psf_flux)
            return
        s = a.shape
        a = a.reshape((s[0], s[1], np.prod(s[2:]).astype(int)))
        b = np.asarray(
            [
                convolve(
                    a[:, :, idx],
                    Gaussian2DKernel(
                        xstd,
                        ystd,
                    ),
                )
                for idx in range(a.shape[2])
            ]
        ).transpose([1, 2, 0])
        b = b.reshape(s)
        b /= b.sum(axis=(0, 1))[None, None]
        self._psf_flux_blur = b
        self._psf_flux_blur_grad = (
            np.asarray(
                np.gradient(
                    self._psf_flux_blur,
                    np.median(np.diff(self.psf_row)).value / self.scale,
                    axis=(0, 1),
                )
            )
            * (self.sub_pixel_size / self.pixel_size).value
        )
        self.psf_flux = self._psf_flux_blur  # + self._psf_flux_jitter
        return

    # def _jitter(self, row: npt.ArrayLike, column: npt.ArrayLike):
    #     """Jitters the PSF using gradients. Will update `self`

    #     Parameters:
    #     -----------
    #     row: npt.ArrayLike
    #         Row positions to jitter to. Will be downsampled
    #     column: npt.ArrayLike
    #         Column positions to jitter to. Will be downsampled
    #     """

    #     def grow(ar, ndims):
    #         for i in range(ndims):
    #             ar = ar[:, None]
    #         return ar

    #     def downsample(ar, npoints):
    #         nbin = np.ceil(ar.shape[0] / npoints).astype(int)
    #         a = np.asarray(
    #             [ar[idx * nbin : (idx + 1) * nbin] for idx in range(npoints)]
    #         )
    #         mean, weight = np.asarray([a1.mean() for a1 in a]), np.asarray(
    #             [len(a1) for a1 in a]
    #         ).astype(float)
    #         weight /= weight.max()
    #         return mean, weight

    #     if len(row) > 5:
    #         row, row_w = downsample(row, 5)
    #         column, column_w = downsample(column, 5)
    #     else:
    #         row_w = np.ones(len(row))
    #         column_w = np.ones(len(column))
    #     self._psf_flux_jitter *= 0
    #     if (row.sum() == 0) & (column.sum() == 0):
    #         self.psf_flux = self._psf_flux_blur + self._psf_flux_jitter
    #         return
    #     g1, g2 = self._psf_flux_blur_grad
    #     self._psf_flux_jitter = (
    #         g1 * grow(row * row_w, self.ndims + 2)
    #         + g2 * grow(column * column_w, self.ndims + 2)
    #     ).sum(axis=0)
    #     self.psf_flux = self._psf_flux_blur + self._psf_flux_jitter
    #     return

    def freeze_dimension(self, **kwargs):
        """Drop a dimension of the PSF model"""
        dnms = self.dimension_names.copy()
        duns = self.dimension_units.copy()
        PSF0 = self._psf_flux.copy()
        for key, point in kwargs.items():
            dim = np.where(np.in1d(dnms, key))[0][0]
            PSF0 = interpfunc(
                point.to(duns[dim]).value,
                getattr(self, dnms[dim] + "1d").value,
                reorder(PSF0, dim),
            )
            dnms.pop(dim)
            duns.pop(dim)

        X = {dnm: getattr(self, dnm) for dnm in dnms}
        dnms2 = self.dimension_names.copy()
        for key, point in kwargs.items():
            dim = np.where(np.in1d(dnms2, key))[0][0]
            for dnm in dnms:
                X[dnm] = X[dnm].transpose(
                    np.hstack([dim, list(set(np.arange(len(dnms2))) - set([dim]))])
                )[0]
            dnms2.pop(dim)
        psf2 = PSF(
            [X[dnm] for dnm in dnms],
            PSF0,
            dnms,
            duns,
            self.pixel_size,
            self.sub_pixel_size,
            freeze_dictionary=kwargs.copy(),
            blur_value=self.blur_value,
            extrapolate=self.extrapolate,
            scale=self.scale,
        )
        return psf2

    def integrate_dimension(self, name: str, func: Callable = None):
        """Integrates a dimension of the PSF

        This is how we would find the integrated shape on the visible side...in theory.
        """
        raise NotImplementedError

    def __repr__(self):
        freeze_dictionary = (
            f" (Frozen: {', '.join([f'{key}: {item}' for key, item in self.freeze_dictionary.items()])})"
            if len(self.freeze_dictionary) != 0
            else ""
        )
        return f"{self.ndims}D PSF Model [{', '.join(self.dimension_names)}]{freeze_dictionary}"

    @staticmethod
    def from_name(
        name: str,
        transpose: bool = False,
        scale: int = 1,
        blur_value: Tuple = (0.5 * u.pixel, 0.5 * u.pixel),
    ):
        """Open a PSF file based on the detector name"""
        if name.lower() in ["gauss", "gaussian", "test"]:
            p = PSF.from_file(
                f"{PACKAGEDIR}/data/pandora_gaussian_psf.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
            )
            return p

        if name.lower() in ["vis", "visda", "visible"]:
            p = PSF.from_file(
                f"{PACKAGEDIR}/data/pandora_vis_hr_20220506.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
            )
            p = p.freeze_dimension(wavelength=p.wavelength0d, temperature=-5 * u.deg_C)
            p.blur_value = blur_value
            p._blur(blur_value=p.blur_value)
            hdu = fits.open(f"{PACKAGEDIR}/data/visda-wav-solution.fits")
            for idx in np.arange(1, hdu[1].header["TFIELDS"] + 1):
                name, unit = hdu[1].header[f"TTYPE{idx}"], hdu[1].header[f"TUNIT{idx}"]
                setattr(p, f"trace_{name}", hdu[1].data[name] * u.Quantity(1, unit))
            p.trace_sensitivity *= hdu[1].header["SENSCORR"] * u.Quantity(
                1, hdu[1].header["CORRUNIT"]
            )
            return p

        elif name.lower() in ["nir", "nirda", "ir"]:
            p = PSF.from_file(
                f"{PACKAGEDIR}/data/pandora_nir_20220506.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
            )
            p = p.freeze_dimension(temperature=-5 * u.deg_C)
            p.blur_value = blur_value
            p._blur(blur_value=p.blur_value)
            hdu = fits.open(f"{PACKAGEDIR}/data/nirda-wav-solution.fits")
            for idx in np.arange(1, hdu[1].header["TFIELDS"] + 1):
                name, unit = hdu[1].header[f"TTYPE{idx}"], hdu[1].header[f"TUNIT{idx}"]
                setattr(p, f"trace_{name}", hdu[1].data[name] * u.Quantity(1, unit))
            p.trace_sensitivity *= hdu[1].header["SENSCORR"] * u.Quantity(
                1, hdu[1].header["CORRUNIT"]
            )
            return p
        else:
            raise ValueError(f"No such PSF as `{name}`")

    @staticmethod
    def from_file(
        filename: str = f"{PACKAGEDIR}/data/pandora_vis_20220506.fits",
        transpose: bool = False,
        blur_value: Tuple = (0 * u.pixel, 0 * u.pixel),
        extrapolate: bool = False,
        scale: int = 1,
    ):
        """Build a PSF class from an input fits file.

        Parameters:
        -----------
        filename: str
            Filename of PSF fits file. PSF cube must have a shape such that the first
            two dimensions are the x and y extent of the PSF. Defaults to visible PSF.
        transpose: bool
            Transpose the input file, i.e. rotate 90 degrees
        blur_value: tuple:
            A tuple describing the amount of motion blur in each axis. Optional, defaults to zero.

        Returns:
        --------
        psf : pandorapsf.PSF
            Returns a PSF object
        """
        hdu = fits.open(filename)
        pixel_size = hdu[0].header["PIXSIZE"] * u.micron / u.pix
        sub_pixel_size = hdu[0].header["SUBPIXSZ"] * u.micron / u.pix

        # LLNL's matlab files are COLUMN-major
        # This should make the array ROW-major
        replace = {"x": "column", "y": "row"}
        dimension_names = [
            replace[i.name.lower()] if i.name.lower() in replace else i.name.lower()
            for i in hdu[2:]
        ]

        if "row" in dimension_names:
            l = (
                np.where(np.asarray(dimension_names) == "row")[0][0],
                np.where(np.asarray(dimension_names) == "column")[0][0],
            )
            l = np.hstack(
                [l, list(set(list(np.arange(len(hdu) - 2))) - set(l))]
            ).astype(int)
        else:
            l = np.arange(len(hdu) - 2).astype(int)

        psf_flux = hdu[1].data.transpose(np.hstack([1, 0, *l + 2]))
        dimension_names = [dimension_names[l1] for l1 in l]
        dimension_units = [u.Unit(hdu[l1].header["UNIT"]) for l1 in l + 2]
        X = [hdu[l1].data.transpose(l) * u.Unit(hdu[l1].header["UNIT"]) for l1 in l + 2]

        return PSF(
            X,
            psf_flux,
            dimension_names,
            dimension_units,
            pixel_size,
            sub_pixel_size,
            transpose=transpose,
            blur_value=blur_value,
            extrapolate=extrapolate,
            scale=scale,
        )

    def _get_dim(self, dim: Union[int, str], dimension_names=None):
        """Return the numeric dimension of an input int or string"""
        if isinstance(dim, int):
            if (dim > self.ndims) | (dim < 0):
                raise ValueError(f"No dimension `{dim}`")
            l = dim
        elif isinstance(dim, str):
            if dimension_names is None:
                dimension_names = self.dimension_names
            l = np.where(np.asarray(dimension_names) == dim.lower())[0]
            if len(l) == 0:
                raise ValueError(f"No dimension `{dim}`")
            l = l[0]
        return l

    def _check_bounds(self, **kwargs):
        """Check a given point has the right shape, units and values in bounds"""
        cleaned = {}
        for key, value in kwargs.items():
            l = np.in1d(self.dimension_names, key)
            if not l.any():
                raise KeyError(f"`{key}` not in PSF dimensions.")
            dim = np.where(l)[0][0]
            value = u.Quantity(value, self.dimension_units[dim])
            bounds = getattr(self, self.dimension_names[dim] + "_bounds")
            if (value.value < bounds[0].value) | (value.value > bounds[1].value):
                if not self.extrapolate:
                    raise OutOfBoundsError(
                        f"Point ({value}) out of {self.dimension_names[dim]} bounds."
                    )
                else:
                    if value.value < bounds[0].value:
                        cleaned[key] = bounds[0]
                    elif value.value > bounds[1].value:
                        cleaned[key] = bounds[1]
            else:
                cleaned[key] = value
        return cleaned

    def psf(self, gradients=False, **kwargs):
        """Interpolate the PSF cube to a particular point

        Parameters
        ----------
        args: dict
            Dictionary of arguments to pass, set each dimension name to a value
        Returns
        -------
        ar : np.ndarray of shape self.shape
            The interpolated PSF
        """
        if self.check_bounds:
            kwargs = self._check_bounds(**kwargs)
        d = kwargs.copy()
        PSF0 = self.psf_flux
        if gradients:
            dPSF0 = self._psf_flux_blur_grad[0]
            dPSF1 = self._psf_flux_blur_grad[1]
        for key in self.dimension_names:
            value = d.pop(key, getattr(self, key + "0d"))
            PSF0 = interpfunc(
                value.value,
                getattr(self, key + "1d").value,
                PSF0,
            )
            if gradients:
                dPSF0 = interpfunc(
                    value.value,
                    getattr(self, key + "1d").value,
                    dPSF0,
                )
                dPSF1 = interpfunc(
                    value.value,
                    getattr(self, key + "1d").value,
                    dPSF1,
                )
        integral = PSF0.sum()
        if gradients:
            return (
                PSF0 / integral,
                (dPSF0 - dPSF0.mean()) / integral,
                (dPSF1 - dPSF1.mean()) / integral,
            )
        return PSF0 / integral

    def prf(self, row, column, gradients=False, **kwargs):
        """
        Bins the PSF down to the pixel scale.

        Parameters
        ----------
        args: dict
            Dictionary of arguments to pass, set each dimension name to a value

        Returns
        -------
        row: np.ndarray
            Array of integer row positions
        column: np.ndarray
            Array of integer column positions
        psf: np.ndarray
            2D array of PRF flux values with shape (nrows, ncolumns)
        """
        if self.check_bounds:
            test1 = np.in1d(list(kwargs.keys()), self.dimension_names)
            if not test1.all():
                raise ValueError(
                    f"Pass only dimension names from {self.dimension_names}"
                )
        row, column = u.Quantity(row, u.pixel), u.Quantity(column, u.pixel)
        if "row" in self.dimension_names:
            # if gradients:
            #     psf0, dpsf0, dpsf1 = self.psf(
            #         row=row, column=column, gradients=True, **kwargs
            #     )
            # else:
            psf0 = self.psf(row=row, column=column, **kwargs)
        else:
            # if gradients:
            #     psf0, dpsf0, dpsf1 = self.psf(gradients=True, **kwargs)
            # else:
            psf0 = self.psf(**kwargs)
        rb, cb, psfb = bin_prf(
            psf0,
            self.psf_row.value,
            self.psf_column.value,
            (row.value, column.value),
            normalize=False,
        )
        psfb /= np.sum(psfb)
        if gradients:
            # # _, _, dpsf0b = bin_prf(dpsf0, self.psf_row.value, self.psf_column.value, (row.value, column.value), normalize=False)
            # # _, _, dpsf1b = bin_prf(dpsf1, self.psf_row.value, self.psf_column.value, (row.value, column.value), normalize=False)
            # return rb, cb, psfb/integral, (dpsf0b - dpsf0b.mean())/integral, (dpsf1b - dpsf1b.mean())/integral
            dpsf0, dpsf1 = np.gradient(psfb)
            dpsf0 -= dpsf0.mean()
            dpsf1 -= dpsf1.mean()
            return rb, cb, psfb, dpsf0, dpsf1
        return rb, cb, psfb

    # def _bin_prf(self, psf0, row, column, normalize=True):
    #     mod = (self.psf_column.value + column.value) % 1
    #     cyc = ((self.psf_column.value + column.value) - mod).astype(int)
    #     colbin = np.unique(cyc)
    #     psf1 = np.asarray(
    #         [psf0[:, cyc == c].sum(axis=1) / (cyc == c).sum() for c in colbin]
    #     ).T
    #     mod = (self.psf_row.value + row.value) % 1
    #     cyc = ((self.psf_row.value + row.value) - mod).astype(int)
    #     rowbin = np.unique(cyc)
    #     psf2 = np.asarray(
    #         [psf1[cyc == c].sum(axis=0) / (cyc == c).sum() for c in rowbin]
    #     )
    #     if normalize:
    #         psf2 /= psf2.sum()
    #     return rowbin.astype(int), colbin.astype(int), psf2


def interpfunc(l, lp, PSF0):
    if l in lp:
        PSF1 = PSF0[:, :, np.where(lp == l)[0][0]]
    elif l < lp[0]:
        PSF1 = PSF0[:, :, 0]
    elif l > lp[-1]:
        PSF1 = PSF0[:, :, -1]
    else:
        # Find the two closest frames
        d = np.argsort(np.abs(lp - l))[:2]
        d = d[np.argsort(lp[d])]
        # Linearly interpolate
        slope = (PSF0[:, :, d[0]] - PSF0[:, :, d[1]]) / (lp[d[0]] - lp[d[1]])
        PSF1 = PSF0[:, :, d[1]] + (slope * (l - lp[d[1]]))
    return PSF1


def reorder(ar: np.ndarray, dim: int = 0):
    """Reorders a PSF array so that a different dimension is in the front, this helps when we interpolate."""
    if not isinstance(dim, (int, list, np.int_)):
        raise ValueError("Pass an `int` or a `list` of `int`s.")
    if not (ar.ndim - 2) > (np.max(dim)):
        raise ValueError(
            f"No dimension {[d + 2 for d in dim]} in array shape {ar.shape}"
        )
    if np.min(dim) < 0:
        raise ValueError("Can not reorder the first two dimensions.")
    if not hasattr(dim, "__iter__"):
        dim = [dim]
    cdim = [d + 2 for d in dim]
    l = set(np.arange(ar.ndim)[2:]) - set(cdim)
    return ar.transpose(np.hstack([0, 1, cdim, list(l)]).astype(int))


class OutOfBoundsError(Exception):
    """Exception raised if a point is out of bounds for this PSF"""

    pass
