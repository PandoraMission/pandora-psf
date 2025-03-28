"""Defines the PSF class"""

# Standard library
from typing import Dict, List, Union

# Third-party
import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandorasat as ps
from astropy.io import fits
from pandorasat.utils import get_phoenix_model

from . import DATADIR, PACKAGEDIR, logger
from .docstrings import add_docstring
from .plotting import plot_spatial, plot_spectral
from .utils import bin_prf, verify_nir_psf_files, verify_vis_psf_files

__all__ = ["PSF"]


class PSF(object):
    """Class to work with abstract PSFs"""

    @add_docstring(
        "X",
        "psf_flux",
        "dimension_names",
        "dimension_units",
        "pixel_size",
        "sub_pixel_size",
        "transpose",
        "freeze_dictionary",
        "check_bounds",
        "extrapolate",
        "scale",
        "bin",
    )
    def __init__(
        self,
        name: str,
        X: npt.NDArray,
        psf_flux: npt.NDArray,
        dimension_names: List,
        dimension_units: List,
        pixel_size: u.Quantity,
        sub_pixel_size: u.Quantity,
        transpose: bool = False,
        freeze_dictionary: Dict = {},
        check_bounds: bool = True,
        extrapolate: bool = False,
        scale: int = 1,
        bin: int = 1,
    ):
        """
        PSF class for making PSFs, PRFs, and traces.
        """
        if bin == 1:
            self.psf_flux = psf_flux
        elif bin >= 1:
            self.psf_flux = np.asarray(
                [
                    [psf_flux[idx::bin, jdx::bin] for idx in range(bin)]
                    for jdx in range(bin)
                ]
            ).mean(axis=(0, 1))
        else:
            raise ValueError("Can not parse `bin`")
        self.name = name
        self.bin = bin
        self.transpose = transpose
        if self.transpose:
            self.psf_flux = self.psf_flux.transpose(
                [1, 0, *np.arange(2, self.psf_flux.ndim)]
            )
        self.dimension_names = dimension_names
        self.dimension_units = dimension_units
        self.pixel_size = pixel_size
        self.sub_pixel_size = sub_pixel_size * bin
        self.freeze_dictionary = freeze_dictionary
        #        self.blur_value = blur_value
        self.check_bounds = check_bounds
        self.extrapolate = extrapolate
        self.scale = scale

        for name, x in zip(dimension_names, X):
            setattr(self, name, x)

        self._validate()
        self.psf_flux_grad = (
            np.asarray(
                np.gradient(
                    self.psf_flux,
                    np.median(np.diff(self.psf_row)).value / self.scale,
                    axis=(0, 1),
                )
            )
            # * (self.sub_pixel_size / self.pixel_size).value
        )

    def _validate(self):
        """Method that finishes the set up of the PSF"""
        self.shape = self.psf_flux.shape[:2]
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
                    np.where(reshape == idx)[0][0]
                    for idx in range(len(reshape))
                ]
                self.psf_flux = self.psf_flux.transpose(reshape)[s].transpose(
                    deshape
                )

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

    def freeze_dimension(self, **kwargs):
        """
        Drop a dimension of the PSF model. Dropped dimension must exist in self.dimension_names.

        Examples
        --------
        Freeze a PSF object `p` to have row of 10 pixels and wavelength of 1.3 microns
        >>> p.freeze_dimension(row=10*u.pixel, wavelength=1.3*u.micron)

        Returns
        -------
        new: pandorapsf.psf.PSF
            New PSF object
        """
        dnms = self.dimension_names.copy()
        duns = self.dimension_units.copy()
        PSF0 = self.psf_flux.copy()
        for key, point in kwargs.items():
            dim = np.where(np.in1d(dnms, key))[0][0]
            PSF0 = interpfunc(
                u.Quantity(point, duns[dim]).value,
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
                    np.hstack(
                        [dim, list(set(np.arange(len(dnms2))) - set([dim]))]
                    )
                )[0]
            dnms2.pop(dim)
        psf2 = PSF(
            self.name,
            [X[dnm] for dnm in dnms],
            PSF0,
            dnms,
            duns,
            self.pixel_size,
            self.sub_pixel_size,
            freeze_dictionary=kwargs.copy(),
            #            blur_value=self.blur_value,
            extrapolate=self.extrapolate,
            scale=self.scale,
        )
        return psf2

    @add_docstring("wavelength", "spectrum")
    def integrate_spectrum(
        self,
        wavelength: npt.NDArray,
        spectrum: npt.NDArray,
        wavelength_grid: npt.NDArray = None,
    ):
        """
        Create an integrated spectrum, integrated over wavelength bounds.

        Parameters
        ----------
        wavelength_grid: npt.NDArray
            Optional wavelength grid at which to integrate. If no grid is provided, will default to the
            `trace_wavelength` grid of this object.
        """
        if wavelength_grid is None:
            wavelength_grid = self.trace_wavelength.to(wavelength.unit)
        dwavs = np.gradient(wavelength_grid.to(wavelength.unit))
        bounds0, bounds1 = (
            (wavelength_grid.to(wavelength.unit) - dwavs / 2).value,
            (wavelength_grid.to(wavelength.unit) + dwavs / 2).value,
        )
        sens = np.interp(
            wavelength,
            self.trace_wavelength.to(wavelength.unit),
            self.trace_sensitivity,
        )

        if (
            not ((spectrum.unit * sens.unit) * wavelength.unit).to(
                u.electron / u.second
            )
            == 1
        ):
            raise ValueError(
                "Spectrum must be in units equivalent to erg / (Angstrom s cm2)."
            )

        # Calculate the weighted flux contribution in each wavelength bin
        integrated_flux = np.ones(len(wavelength_grid))
        for idx in range(bounds0.shape[0]):
            x = np.linspace(bounds0[idx], bounds1[idx], 100)
            integrated_flux[idx] = np.trapz(
                np.hstack(
                    [
                        0,
                        *np.interp(
                            x, wavelength.value, spectrum.value * sens.value
                        ),
                        0,
                    ]
                ),
                np.hstack([x[0] - 1e-10, *x, x[-1] + 1e-10]),
            )
        integrated_flux *= u.electron / u.second

        return integrated_flux

    @add_docstring("teff", "logg")
    def integrate_wavelength(self, teff=5500, logg=4.5, **kwargs):
        """Inte"""
        if "wavelength" not in self.dimension_names:
            raise ValueError("PSF has no wavelength dimension.")

        # calculate spectrum of star
        w, s = get_phoenix_model(teff, logg=logg, jmag=9)
        integrated_flux = self.integrate_spectrum(
            w, s, wavelength_grid=self.wavelength1d
        )

        weighted_flux = (integrated_flux / integrated_flux.sum()).value

        dnms = self.dimension_names.copy()
        duns = self.dimension_units.copy()
        PSF0 = self.psf_flux.copy()
        dim = np.where(np.in1d(dnms, "wavelength"))[0][0]

        reshape = np.ones(PSF0.ndim, dtype=int)
        reshape[dim + 2] = weighted_flux.shape[0]
        weighted_flux = weighted_flux.reshape(tuple(reshape.astype(int)))
        PSF0 = (PSF0 * weighted_flux).sum(axis=dim + 2)

        X = {
            dnm: getattr(self, dnm)
            for dnm in list(set(list(dnms)) - set(["wavelength"]))
        }

        for dnm in list(set(list(dnms)) - set(["wavelength"])):
            X[dnm] = X[dnm].transpose(
                np.hstack([dim, list(set(np.arange(len(dnms))) - set([dim]))])
            )[0]

        dnms.pop(dim)
        duns.pop(dim)

        psf2 = PSF(
            name=self.name,
            X=[X[dnm] for dnm in dnms],
            psf_flux=PSF0,
            dimension_names=dnms,
            dimension_units=duns,
            pixel_size=self.pixel_size,
            sub_pixel_size=self.sub_pixel_size,
            transpose=self.transpose,
            freeze_dictionary=kwargs.copy(),
            #            blur_value=self.blur_value,
            extrapolate=self.extrapolate,
            scale=self.scale,
        )
        return psf2

    def __repr__(self):
        freeze_dictionary = (
            f" (Frozen: {', '.join([f'{key}: {item:.3f}' for key, item in self.freeze_dictionary.items()])})"
            if len(self.freeze_dictionary) != 0
            else ""
        )
        return f"{self.ndims}D PSF Model [{', '.join(self.dimension_names)}]{freeze_dictionary}"

    @staticmethod
    @add_docstring("name", "transpose", "scale", "bin")
    def from_name(
        name: str,
        transpose: bool = False,
        scale: int = 1,
        bin: int = 1,
    ):
        """Open a PSF file based on the detector name. This will automatically freeze dimensions."""
        if name.lower() in ["gauss", "gaussian", "test"]:
            p = PSF.from_file(
                "gaussian",
                f"{PACKAGEDIR}/data/pandora_gaussian_psf.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
                bin=bin,
            )
            return p
        if name.lower() in ["vis", "visda", "visible"]:
            # Check that files are downloaded and current.
            verify_vis_psf_files()
            p = PSF.from_file(
                "visda",
                f"{DATADIR}/pandora_vis_psf.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
                bin=bin,
            ).integrate_wavelength(teff=5500, logg=4.5)
            return p
        elif name.lower() in [
            "vis_fallback",
            "visda_fallback",
            "visible_fallback",
        ]:
            logger.info("Loading low resolution fallback file for VISDA.")
            p = PSF.from_file(
                "visda",
                f"{PACKAGEDIR}/data/pandora_vis_psf_lowres.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
                bin=bin,
            )
            return p
        elif name.lower() in ["nir", "nirda", "ir"]:
            # Check that files are downloaded and current.
            verify_nir_psf_files()
            p = PSF.from_file(
                "nirda",
                f"{DATADIR}/pandora_nir_psf.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
                bin=bin,
            )
            return p
        elif name.lower() in ["nir_fallback", "nirda_fallback", "ir_fallback"]:
            logger.info("Loading low resolution fallback file for NIRDA.")
            p = PSF.from_file(
                "nirda",
                f"{PACKAGEDIR}/data/pandora_nir_psf_lowres.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
                bin=bin,
            )
            return p
        else:
            raise ValueError(f"No such PSF as `{name}`")

    def _add_trace_params(self):
        """Adds the expected trace parameters from pandorasat"""
        if self.name == "visda":
            detector = ps.VisibleDetector()
        elif self.name == "nirda":
            detector = ps.NIRDetector()
        else:
            raise ValueError(f"Can not parse name {self.name}.")
        self._trace_sensitivity, self._trace_pixel, self._trace_wavelength = (
            detector.trace_sensitivity,
            detector.trace_pixel,
            detector.trace_wavelength,
        )

    @property
    def trace_sensitivity(self):
        """The sensitivity of a spectrum trace at the `self.trace_wavelength` and `self.trace_pixel` positions."""
        if not hasattr(self, "_trace_sensitivity"):
            self._add_trace_params()
        return self._trace_sensitivity

    @property
    def trace_wavelength(self):
        """Wavelengths corresponding to `self.trace_sensitivity`"""
        if not hasattr(self, "_trace_wavelength"):
            self._add_trace_params()
        return self._trace_wavelength

    @property
    def trace_pixel(self):
        """Pixels corresponding to `self.trace_sensitivity`"""
        if not hasattr(self, "_trace_pixel"):
            self._add_trace_params()
        return self._trace_pixel

    @add_docstring("name", "transpose", "extrapolate", "scale", "bin")
    def from_file(
        name: str,
        file: str = f"{PACKAGEDIR}/data/pandora_vis_2024-05.fits",
        transpose: bool = False,
        extrapolate: bool = False,
        scale: int = 1,
        bin: int = 1,
    ):
        """Build a PSF class from an input fits file.

        Parameters:
        -----------
        file: str or fits.HDUList
            Filename of PSF fits file, or opened hdulist. PSF cube must have a shape such that the first
            two dimensions are the x and y extent of the PSF. Defaults to visible PSF.
        transpose: bool
            Transpose the input file, i.e. rotate 90 degrees

        Returns:
        --------
        psf : pandorapsf.PSF
            Returns a PSF object
        """
        if isinstance(file, str):
            hdu = fits.open(file)
        elif isinstance(file, fits.HDUList):
            hdu = file
        else:
            raise ValueError("Can not recognize filename.")
        pixel_size = hdu[0].header["PIXSIZE"] * u.micron / u.pix
        sub_pixel_size = hdu[0].header["SUBPIXSZ"] * u.micron / u.pix

        # LLNL's matlab files are COLUMN-major
        # This should make the array ROW-major
        replace = {"x": "column", "y": "row"}
        dimension_names = [
            (
                replace[i.name.lower()]
                if i.name.lower() in replace
                else i.name.lower()
            )
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

        # We expect the images to be in the first few dimensions
        psf_flux = hdu[1].data
        if (
            np.asarray(psf_flux.shape)[l] == np.asarray(hdu[2].data.shape)
        ).all():
            psf_flux = psf_flux.transpose(
                np.hstack([1 + l[-1] + 1, l[-1] + 1, *l])
            )
        else:
            psf_flux = psf_flux.transpose(np.hstack([1, 0, *l + 2]))
        dimension_names = [dimension_names[l1] for l1 in l]
        dimension_units = [u.Unit(hdu[l1].header["UNIT"]) for l1 in l + 2]
        X = [
            hdu[l1].data.transpose(l) * u.Unit(hdu[l1].header["UNIT"])
            for l1 in l + 2
        ]

        return PSF(
            name=name,
            X=X,
            psf_flux=psf_flux,
            dimension_names=dimension_names,
            dimension_units=dimension_units,
            pixel_size=pixel_size,
            sub_pixel_size=sub_pixel_size,
            transpose=transpose,
            #            blur_value=blur_value,
            extrapolate=extrapolate,
            scale=scale,
            bin=bin,
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
            if (value.value < bounds[0].value) | (
                value.value > bounds[1].value
            ):
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

    @add_docstring("gradients")
    def psf(self, gradients=False, **kwargs):
        """
        Interpolate the PSF cube to a particular point

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
            dPSF0 = self.psf_flux_grad[0]
            dPSF1 = self.psf_flux_grad[1]
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

    @add_docstring("row", "column", "gradients")
    def prf(self, row, column, gradients=False, **kwargs):
        """
        Bins the PSF down to the pixel scale.

        Returns
        -------
        row_array: npt.NDArray
            Array of integer row positions
        column_array: npt.NDArray
            Array of integer column positions
        prf: np.ndarray
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
            if gradients:
                psf0, dpsf0, dpsf1 = self.psf(
                    row=row, column=column, gradients=True, **kwargs
                )
            else:
                psf0 = self.psf(row=row, column=column, **kwargs)
        else:
            if gradients:
                psf0, dpsf0, dpsf1 = self.psf(gradients=True, **kwargs)
            else:
                psf0 = self.psf(**kwargs)
        rb, cb, psfb = bin_prf(
            psf0,
            self.psf_row.value,
            self.psf_column.value,
            (row.value, column.value),
            normalize=False,
        )
        integral = np.sum(psfb)
        if gradients:
            _, _, dpsf0b = bin_prf(
                dpsf0,
                self.psf_row.value,
                self.psf_column.value,
                (row.value, column.value),
                normalize=False,
            )
            _, _, dpsf1b = bin_prf(
                dpsf1,
                self.psf_row.value,
                self.psf_column.value,
                (row.value, column.value),
                normalize=False,
            )
            # return rb, cb, psfb/integral, (dpsf0b - dpsf0b.mean())/integral, (dpsf1b - dpsf1b.mean())/integral
            # dpsf0, dpsf1 = np.gradient(psfb)
            dpsf0b -= dpsf0b.mean()
            dpsf1b -= dpsf1b.mean()
            return (
                rb,
                cb,
                psfb / integral,
                dpsf0b / integral,
                dpsf1b / integral,
            )
        return rb, cb, psfb / integral

    def plot_spatial(self, n=3, image_type="PSF", **kwargs):
        plot_spatial(self, n=n, image_type=image_type, **kwargs)

    def plot_spectral(
        self, var="wavelength", n=5, npixels=20, image_type="psf", **kwargs
    ):
        plot_spectral(
            self,
            var=var,
            n=n,
            npixels=npixels,
            image_type=image_type,
            **kwargs,
        )


def interpfunc(l, lp, PSF0):
    """Interpolation function.
    Given a grid of points l and a desired point lp will interpolate n dimensional PSF0.
    Grid is always assumed to be the last dimension."""
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
