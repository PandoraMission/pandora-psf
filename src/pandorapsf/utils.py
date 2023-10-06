"""Handy utilities for PSFs"""
from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.io import fits
from scipy.io import loadmat

from . import PACKAGEDIR


def make_vis_PSF_fits_files(dir, nbin=2, suffix=""):
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


def make_nir_PSF_fits_files(dir, nbin=2, suffix=""):
    # NIR PSF with thermal info
    # -----------------------------------#

    d = loadmat(dir + "pandora_nir_20220506_thin_prism_hot_PSF_512.mat")
    PSF_hot = d["PSF"][:-1, :-1, :-1]  # [128:384, 128:384]
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


def make_gaussian_psf():
    hdu1 = fits.open(f'{PACKAGEDIR}/data/pandora_vis_hr_20220506.fits')
    pixel_size = hdu1[0].header["PIXSIZE"] * u.micron / u.pix
    sub_pixel_size = hdu1[0].header["SUBPIXSZ"] * u.micron / u.pix
    shape = hdu1[1].data.shape[:2]
    psf_column = (
        (np.arange(shape[1]) - shape[1] // 2)
        * u.pixel
        * sub_pixel_size
    ) / pixel_size
    psf_row = (
        (np.arange(shape[0]) - shape[0] // 2)
        * u.pixel
        * sub_pixel_size
    ) / pixel_size

    def gauss(row, col, sigma):
        return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((col[None, :])**2/(2*sigma**2) + (row[:, None])**2/(2*sigma**2)))
        
    R = np.abs(hdu1[2].data[:, 0, 0, 0]) / 600 + 1
    C = np.abs(hdu1[3].data[0, :, 0, 0]) / 600 + 1

    ar = np.zeros((9, 9, 256, 256))
    for idx, r1 in enumerate(R):
        for jdx, c1 in enumerate(C):
            ar[idx, jdx] = gauss(psf_row.value, psf_column.value, np.hypot(r1, c1))
    ar = ar.transpose([2, 3, 0, 1])
    integral = np.trapz(np.trapz(ar, dx=np.median(np.diff(psf_row).value), axis=0), dx=np.median(np.diff(psf_column).value), axis=0)
    ar /= integral[None, None, :, :]
    hdr = fits.Header(
        [
            ("AUTHOR1", "Christina Hedges"),
            ("ORIGIN1", "Gaussian"),
            ("CREATED", datetime.now().isoformat()),
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
            fits.ImageHDU(ar, name="PSF"),
            fits.ImageHDU(hdu1[2].data[:, :, 0, 0], name="X"),
            fits.ImageHDU(hdu1[3].data[:, :, 0, 0], name="Y"),
        ]
    )
    hdu[2].header["UNIT"] = u.pixel.to_string()
    hdu[3].header["UNIT"] = u.pixel.to_string()
    hdu.writeto(
        PACKAGEDIR + f"/data/pandora_gaussian_psf.fits",
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


def bin_prf(psf0, psf_column, psf_row, location=(0, 0), normalize=True):
    mod = (psf_column + location[1]) % 1
    cyc = ((psf_column + location[1]) - mod).astype(int)
    colbin = np.unique(cyc)
    psf1 = np.asarray(
        [psf0[:, cyc == c].sum(axis=1) / (cyc == c).sum() for c in colbin]
    ).T
    
    mod = (psf_row + location[0]) % 1
    cyc = ((psf_row + location[0]) - mod).astype(int)
    rowbin = np.unique(cyc)
    psf2 = np.asarray(
        [psf1[cyc == c].sum(axis=0) / (cyc == c).sum() for c in rowbin]
    )
    if normalize:
        psf2 /= psf2.sum()
    return rowbin.astype(int), colbin.astype(int), psf2