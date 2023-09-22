"""Handy utilities for PSFs"""
from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.io import fits
from scipy.io import loadmat

from . import PACKAGEDIR


def make_PSF_fits_files(dir, nbin=2, suffix=""):
    d = loadmat(dir + "pandora_vis_20220506_hot_PSF_512.mat")

    PSF_hot = d["PSF"][:-1, :-1, :][128:384, 128:384]
    PSF_hot = np.asarray(
        [[PSF_hot[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
    ).mean(axis=(0, 1))
    PSF_hot = PSF_hot.reshape((*PSF_hot.shape[:2], 9, 9, 5))
    d = loadmat(dir + "pandora_vis_20220506_cold_PSF_512.mat")

    PSF_cold = d["PSF"][:-1, :-1, :][128:384, 128:384]
    PSF_cold = np.asarray(
        [[PSF_cold[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
    ).mean(axis=(0, 1))
    PSF_cold = PSF_cold.reshape((*PSF_cold.shape[:2], 9, 9, 5))

    sub_pixel_size = nbin * d["dx"][0][0] * u.micron / u.pix
    pixel_size = 6.5 * u.micron / u.pix
    x = d["x"][0].reshape((9, 9, 5)) * u.mm
    y = d["y"][0].reshape((9, 9, 5)) * u.mm
    x = (x / pixel_size).to(u.pixel)
    y = (y / pixel_size).to(u.pixel)
    wvl = d["wvl"][0].reshape((9, 9, 5)) * u.micron
    temp = np.asarray([-10.0, 30.0]) * u.deg_C
    PSF = np.asarray([PSF_cold, PSF_hot]).transpose([1, 2, 3, 4, 5, 0])
    PSF /= PSF.sum(axis=(0, 1))[None, None]

    x = x[:, :, :, None] * np.ones(len(temp))
    y = y[:, :, :, None] * np.ones(len(temp))
    wvl = wvl[:, :, :, None] * np.ones(len(temp))
    temp = temp[None, None, None, :] * np.ones(wvl.shape)

    hdr = fits.Header(
        [
            ("AUTHOR1", "LLNL"),
            ("AUTHOR2", "Christina Hedges"),
            ("ORIGIN1", "pandora_vis_20220506_cold_PSF_512.mat"),
            ("ORIGIN2", "pandora_vis_20220506_hot_PSF_512.mat"),
            ("CREATED", str(d["__header__"]).split("Created on: ")[-1][:-1]),
            ("DATE", datetime.now().isoformat()),
            (
                "PIXSIZE",
                pixel_size.value,
                f"PSF pixel size in {pixel_size.unit.to_string()}",
            ),
            (
                "SUBPIXSZ",
                sub_pixel_size.value,
                f"PSF sub pixel size in {sub_pixel_size.unit.to_string()}",
            ),
        ]
    )
    primaryhdu = fits.PrimaryHDU(header=hdr)
    hdu = fits.HDUList(
        [
            primaryhdu,
            fits.ImageHDU(PSF, name="PSF"),
            fits.ImageHDU(x.value, name="X"),
            fits.ImageHDU(y.value, name="Y"),
            fits.ImageHDU(wvl.value, name="WAVELENGTH"),
            fits.ImageHDU(temp.value, name="TEMPERATURE"),
        ]
    )
    hdu[2].header["UNIT"] = y.unit.to_string()
    hdu[3].header["UNIT"] = x.unit.to_string()
    hdu[4].header["UNIT"] = wvl.unit.to_string()
    hdu[5].header["UNIT"] = temp.unit.to_string()
    hdu.writeto(
        PACKAGEDIR + f"/data/pandora_vis{suffix}_20220506.fits",
        overwrite=True,
    )

    # NIR PSF with thermal info
    # -----------------------------------#

    d = loadmat(dir + "pandora_nir_20220506_thin_prism_hot_PSF_512.mat")
    PSF_hot = d["PSF"][:-1, :-1, :-1][128:384, 128:384]
    PSF_hot = np.asarray(
        [[PSF_hot[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
    ).mean(axis=(0, 1))

    d = loadmat(dir + "pandora_nir_20220506_thin_prism_cold_PSF_512.mat")

    PSF_cold = d["PSF"][:-1, :-1, :-1][128:384, 128:384]
    PSF_cold = np.asarray(
        [[PSF_cold[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
    ).mean(axis=(0, 1))

    sub_pixel_size = nbin * d["dx"][0][0] * u.micron / u.pix
    pixel_size = 18 * u.micron / u.pix

    # Bin down wavelength too, also too big.
    wbin = 10
    PSF_cold = np.asarray([PSF_cold[:, :, idx::wbin] for idx in range(wbin)]).mean(
        axis=(0)
    )
    PSF_hot = np.asarray([PSF_hot[:, :, idx::wbin] for idx in range(wbin)]).mean(
        axis=(0)
    )
    wvl = (
        np.asarray([d["wvl"][:900][idx::wbin] for idx in range(wbin)]).mean(axis=(0))[
            :, 0
        ]
        * u.micron
    )
    temp = np.asarray([-10.0, 30.0]) * u.deg_C
    wvl = wvl[:, None] * np.ones(len(temp))
    temp = temp[None, :] * np.ones(wvl.shape)

    PSF = np.asarray([PSF_cold, PSF_hot]).transpose([1, 2, 3, 0])

    PSF /= PSF.sum(axis=(0, 1))

    hdr = fits.Header(
        [
            ("AUTHOR1", "LLNL"),
            ("AUTHOR2", "Christina Hedges"),
            ("ORIGIN", "pandora_nir_20220506.mat"),
            ("CREATED", str(d["__header__"]).split("Created on: ")[-1][:-1]),
            ("DATE", datetime.now().isoformat()),
            (
                "PIXSIZE",
                pixel_size.value,
                f"PSF pixel size in {pixel_size.unit.to_string()}",
            ),
            (
                "SUBPIXSZ",
                sub_pixel_size.value,
                f"PSF sub pixel size in {sub_pixel_size.unit.to_string()}",
            ),
        ]
    )
    primaryhdu = fits.PrimaryHDU(header=hdr)
    hdu = fits.HDUList(
        [
            primaryhdu,
            fits.ImageHDU(PSF, name="PSF"),
            fits.ImageHDU(wvl.value, name="WAVELENGTH"),
            fits.ImageHDU(temp.value, name="TEMPERATURE"),
        ]
    )
    hdu[2].header["UNIT"] = wvl.unit.to_string()
    hdu[3].header["UNIT"] = temp.unit.to_string()
    hdu.writeto(
        PACKAGEDIR + f"/data/pandora_nir{suffix}_20220506.fits",
        overwrite=True,
    )


def prep_for_add(row, column, prf, shape=(100, 100), corner=(-50, -50)):
    Y, X = np.asarray(
        np.meshgrid(
            row - corner[0],
            column - corner[1],
            indexing="ij",
        )
    ).astype(int)
    k = (X >= 0) & (X < shape[1]) & (Y >= 0) & (Y < shape[0])
    if prf.ndim == 2:
        return Y[k], X[k], prf[k]
    elif prf.ndim == 3:
        return Y[k], X[k], prf[:, k]
    else:
        raise ValueError("can not parse prf for adding")
