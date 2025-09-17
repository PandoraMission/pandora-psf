"""Defines the PRF class"""

# Third-party
import astropy.units as u
import numpy as np

from . import DATADIR
from .docstrings import add_docstring
from .plotting import plot_spatial
from .psf import PSF
from .utils import interp_prf, verify_vis_prf_files

__all__ = ["PRF"]


class PRF(PSF):
    def __repr__(self):
        freeze_dictionary = (
            f" (Frozen: {', '.join([f'{key}: {item:.3f}' for key, item in self.freeze_dictionary.items()])})"
            if len(self.freeze_dictionary) != 0
            else ""
        )
        return f"{self.ndims}D PRF Model [{', '.join(self.dimension_names)}]{freeze_dictionary}"

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
        rb, cb, psfb = interp_prf(
            psf0,
            self.psf_row.value,
            self.psf_column.value,
            (row.value, column.value),
            normalize=False,
        )
        integral = np.sum(psfb)
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
