"""Defines the PRF class"""

# Third-party
import astropy.units as u
import numpy as np
import pandorasat as ps
from astropy.io import fits

from . import DATADIR
from .docstrings import add_docstring
from .plotting import plot_spatial
from .psf import PSF
from .utils import interp_prf, interpfunc, verify_vis_prf_files

__all__ = ["PRF"]


class PRF(PSF):
    def __repr__(self):
        freeze_dictionary = (
            f" (Frozen: {', '.join([f'{key}: {item:.3f}' for key, item in self.freeze_dictionary.items()])})"
            if len(self.freeze_dictionary) != 0
            else ""
        )
        return f"{self.ndims}D PRF Model [{', '.join(self.dimension_names)}]{freeze_dictionary}"

    def _add_trace_params(self):
        """Adds the expected trace parameters from pandorasat"""
        if self.name == "visda":
            raise NotImplementedError(
                "Can not add trace params to a visda PRF, it is not dispersed."
            )
        elif self.name == "nirda":
            detector = ps.NIRDetector()
            self._trace_pixel = np.arange(-150, 150, 0.25) * u.pixel
            hdulist = fits.open(detector.reference.spectrum_normalization_file)
            self._trace_wavelength = np.interp(
                self._trace_pixel.value,
                hdulist[1].data["pixel"],
                hdulist[1].data["wavelength"],
            ) * u.Unit(hdulist[1].header["TUNIT2"])
            self._trace_sensitivity = np.interp(
                self._trace_pixel.value,
                hdulist[1].data["pixel"],
                hdulist[1].data["Sensitivity per Pixel"],
            ) * u.Unit(hdulist[1].header["TUNIT3"])
            self._trace_sensitivity /= np.trapz(
                self._trace_sensitivity, self._trace_pixel
            )
        else:
            raise ValueError(f"Can not parse name {self.name}.")

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

    def psf(self, *args, **kwargs):
        raise NotImplementedError(
            "You can not create a PSF from a PRF object. Use the `prf` method instead."
        )

    @add_docstring("gradients")
    def _psf(self, gradients=False, **kwargs):
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
        if gradients:
            return (
                PSF0,
                (dPSF0 - dPSF0.mean()),
                (dPSF1 - dPSF1.mean()),
            )
        return PSF0

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
                psf0, dpsf0, dpsf1 = self._psf(
                    row=row, column=column, gradients=True, **kwargs
                )
            else:
                psf0 = self._psf(row=row, column=column, **kwargs)
        else:
            if gradients:
                psf0, dpsf0, dpsf1 = self._psf(gradients=True, **kwargs)
            else:
                psf0 = self._psf(**kwargs)
        rb, cb, psfb = interp_prf(
            psf0,
            self.psf_row.value,
            self.psf_column.value,
            (row.value, column.value),
            normalize=False,
        )
        # integral = np.sum(psfb)
        if gradients:
            _, _, dpsf0b = interp_prf(
                dpsf0,
                self.psf_row.value,
                self.psf_column.value,
                (row.value, column.value),
                normalize=False,
            )
            _, _, dpsf1b = interp_prf(
                dpsf1,
                self.psf_row.value,
                self.psf_column.value,
                (row.value, column.value),
                normalize=False,
            )
            return (
                rb,
                cb,
                psfb,
                dpsf0b,
                dpsf1b,
            )
        return rb, cb, psfb

        # return rb, cb, psfb/integral, (dpsf0b - dpsf0b.mean())/integral, (dpsf1b - dpsf1b.mean())/integral
        # dpsf0, dpsf1 = np.gradient(psfb)
        # dpsf0b -= dpsf0b.mean()
        # dpsf1b -= dpsf1b.mean()
        #     return (
        #         rb,
        #         cb,
        #         psfb / integral,
        #         dpsf0b / integral,
        #         dpsf1b / integral,
        #     )
        # return rb, cb, psfb / integral

    @staticmethod
    @add_docstring("name", "transpose", "scale", "bin")
    def from_name(
        name: str,
        transpose: bool = False,
        scale: int = 1,
        bin: int = 1,
    ):
        """Open a PRF file based on the detector name."""
        if name.lower() in ["vis", "visda", "visible"]:
            # Check that files are downloaded and current.
            verify_vis_prf_files()
            p = PRF.from_file(
                "visda",
                f"{DATADIR}/pandora_vis_prf.fits",
                transpose=transpose,
                extrapolate=True,
                scale=scale,
                bin=bin,
            )
            return p
        else:
            raise ValueError(f"No such PRF as `{name}`")

    def plot_spatial(self, n=3, **kwargs):
        plot_spatial(self, n=n, image_type="PRF", **kwargs)
