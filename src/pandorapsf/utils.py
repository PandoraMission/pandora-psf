"""Handy utilities for PSFs"""
from datetime import datetime

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import c, h
from astropy.io import fits
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from . import PACKAGEDIR


def make_pixel_files():
    def photon_energy(wavelength):
        return ((h * c) / wavelength) * 1 / u.photon

    def sensitivity(wavelength):
        sed = 1 * u.erg / u.s / u.cm**2 / u.angstrom
        E = photon_energy(wavelength)
        telescope_area = np.pi * (mirror_diameter / 2) ** 2
        photon_flux_density = (telescope_area * sed * throughput(wavelength) / E).to(
            u.photon / u.second / u.angstrom
        ) * qe(wavelength)
        sensitivity = photon_flux_density / sed
        return sensitivity

    def qe(wavelength):
        wavelength = np.atleast_1d(wavelength)
        sw_coeffs = np.array([0.65830, -0.05668, 0.25580, -0.08350])
        sw_exponential = 100.0
        sw_wavecut_red = 1.62  # changed from 2.38 for Pandora
        sw_wavecut_blue = 0.85  # new for Pandora
        with np.errstate(invalid="ignore", over="ignore"):
            sw_qe = (
                sw_coeffs[0]
                + sw_coeffs[1] * wavelength.to(u.micron).value
                + sw_coeffs[2] * wavelength.to(u.micron).value ** 2
                + sw_coeffs[3] * wavelength.to(u.micron).value ** 3
            )

            sw_qe = np.where(
                wavelength.to(u.micron).value > sw_wavecut_red,
                sw_qe
                * np.exp(
                    (sw_wavecut_red - wavelength.to(u.micron).value) * sw_exponential
                ),
                sw_qe,
            )

            sw_qe = np.where(
                wavelength.to(u.micron).value < sw_wavecut_blue,
                sw_qe
                * np.exp(
                    -(sw_wavecut_blue - wavelength.to(u.micron).value)
                    * (sw_exponential / 1.5)
                ),
                sw_qe,
            )
        sw_qe[sw_qe < 1e-5] = 0
        return sw_qe * u.electron / u.photon

    def throughput(wavelength):
        df = pd.read_csv(f"{PACKAGEDIR}/data/dichroic-transmission.csv")
        throughput = np.interp(wavelength.to(u.angstrom).value, *np.asarray(df).T)
        return throughput**2 * 0.71

    mirror_diameter = 0.43 * u.m

    df = pd.read_csv(f"{PACKAGEDIR}/data/pixel_vs_wavelength.csv")
    pixel = np.round(np.arange(-400, 80, 0.25), 5) * u.pixel
    wav = np.polyval(np.polyfit(df.Pixel, df.Wavelength, 3), pixel.value) * u.micron

    sens = sensitivity(wav)
    mask = sens / sens.max() > 0.0001
    corr = np.trapz(sens, wav)
    hdu = fits.TableHDU.from_columns(
        [
            fits.Column(
                name="pixel",
                format="D",
                array=pixel.value[mask],
                unit=pixel.unit.to_string(),
            ),
            fits.Column(
                name="wavelength",
                format="D",
                array=wav.value[mask],
                unit=wav.unit.to_string(),
            ),
            fits.Column(
                name="sensitivity",
                format="D",
                array=(sens[mask] / corr),
                unit=(sens / corr).unit.to_string(),
            ),
        ]
    )
    hdu.header.append(
        fits.Card("SENSCORR", corr.value, "correction to apply to sensitivity")
    )
    hdu.header.append(
        fits.Card("CORRUNIT", corr.unit.to_string(), "units of correction")
    )
    hdu.writeto(f"{PACKAGEDIR}/data/nirda-wav-solution.fits", overwrite=True)

    def qe(wavelength):  # noqa: F811
        df = pd.read_csv(f"{PACKAGEDIR}/data/pandora_visible_qe.csv")
        wav, transmission = np.asarray(df.Wavelength) * u.angstrom, np.asarray(
            df.Transmission
        )
        return (
            np.interp(wavelength, wav, transmission, left=0, right=0)
            * u.electron
            / u.photon
        )

    def throughput(wavelength):  # noqa: F811
        df = pd.read_csv(f"{PACKAGEDIR}/data/dichroic-transmission.csv")
        throughput = 1 - np.interp(wavelength.to(u.angstrom).value, *np.asarray(df).T)
        return throughput * 0.752

    wav = np.arange(0.25, 1.3, 0.01) * u.micron
    pixel = np.zeros(len(wav)) * u.pixel
    sens = sensitivity(wav)
    corr = np.trapz(sens, wav)

    hdu = fits.TableHDU.from_columns(
        [
            fits.Column(
                name="pixel", format="D", array=pixel.value, unit=pixel.unit.to_string()
            ),
            fits.Column(
                name="wavelength",
                format="D",
                array=wav.value,
                unit=wav.unit.to_string(),
            ),
            fits.Column(
                name="sensitivity",
                format="D",
                array=(sens / corr),
                unit=(sens / corr).unit.to_string(),
            ),
        ]
    )
    hdu.header.append(
        fits.Card("SENSCORR", corr.value, "correction to apply to sensitivity")
    )
    hdu.header.append(
        fits.Card("CORRUNIT", corr.unit.to_string(), "units of correction")
    )
    hdu.writeto(f"{PACKAGEDIR}/data/visda-wav-solution.fits", overwrite=True)


def _open_LLNL_file(filename):
    """This opens the newest file format from LLNL. They might change it, so watch out."""
    d = loadmat(filename)
    subpixel_size = d["dx"][0][0] * u.micron / u.pix
    pixel_size = 6.5 * u.micron / u.pix

    wvl = np.unique(d["wvl"]) * u.micron
    nwav = wvl.shape[0]
    #    npoints = int(d["x"].shape[1] / nwav)

    TC, TR = np.meshgrid(
        np.unique(d["Tx"][0, ::nwav]), np.unique(d["Ty"][0, ::nwav]), indexing="xy"
    )
    sC, sR = np.meshgrid(
        np.arange(d["PSF"].shape[0]), np.arange(d["PSF"].shape[1]), indexing="xy"
    )
    mask = np.any(
        [
            (TR == row) & (TC == col)
            for col, row in zip(d["Tx"][0, ::nwav], d["Ty"][0, ::nwav])
        ],
        axis=0,
    )
    C, R = np.zeros((2, nwav, *TR.shape))

    for idx in range(nwav):
        C[idx, mask] = d["x"][0][idx::nwav]
        R[idx, mask] = d["y"][0][idx::nwav]
    row1d, column1d = R[:, C.shape[2] // 2].mean(axis=0), C[:, :, C.shape[2] // 2].mean(
        axis=0
    )

    PSF = np.zeros((*d["PSF"].shape[:2], *TR.shape, wvl.shape[0]))

    for wdx in np.arange(nwav):
        PSF[:, :, mask, wdx] = d["PSF"][:, :, wdx::nwav]
    # We need to sort of extrapolate the edges because we don't have those values.
    for count in range(2):
        extrap = np.zeros(PSF.shape)
        extrap[:, :, 1:, :, :] += PSF[:, :, :-1, :, :]
        extrap[:, :, :-1, :, :] += PSF[:, :, 1:, :, :]
        extrap[:, :, :, 1:, :] += PSF[:, :, :, :-1, :]
        extrap[:, :, :, :-1, :] += PSF[:, :, :, 1:, :]
        extrap /= 2
        PSF[:, :, ~mask, :] += extrap[:, :, ~mask, :]

    PSF /= PSF.sum(axis=(0, 1))[None, None]

    # Gaussian blur with std dev of 0.5 detector pixels
    blur_width = ((pixel_size * 0.5 * u.pix) / subpixel_size).value

    PSF = np.asarray(
        [
            [
                [
                    gaussian_filter(PSF[:, :, idx, jdx, kdx], blur_width)
                    for idx in range(PSF.shape[2])
                ]
                for jdx in range(PSF.shape[3])
            ]
            for kdx in range(PSF.shape[4])
        ]
    )

    PSF = PSF.transpose([4, 3, 2, 1, 0])
    PSF = PSF.transpose([1, 0, 3, 2, 4])
    row = row1d[:, None, None] * np.ones((*TR.shape, nwav)) * u.mm
    column = column1d[None, :, None] * np.ones((*TR.shape, nwav)) * u.mm
    wavelength = wvl[None, None, :] * np.ones((*TR.shape, nwav))
    subpixel_row = ((sR[:, 0] - sR.shape[0] / 2) * u.pixel * subpixel_size)[
        :, None
    ] * np.ones(d["PSF"].shape[:2])
    subpixel_column = ((sC[0] - sC.shape[0] / 2) * u.pixel * subpixel_size)[
        None
    ] * np.ones(d["PSF"].shape[:2])
    return (
        subpixel_size,
        subpixel_row,
        subpixel_column,
        row,
        column,
        wavelength,
        PSF,
        str(d["__header__"]).split("Created on: ")[-1][:-1],
    )


def make_PSF_fits_files(filename, suffix, pixel_size):
    (
        subpixel_size,
        subpixel_row,
        subpixel_column,
        row,
        column,
        wavelength,
        PSF,
        created_on,
    ) = _open_LLNL_file(filename)
    pixel_size = u.Quantity(pixel_size, u.micron / u.pixel)
    subpixel_row, subpixel_column, row, column = (
        (subpixel_row / pixel_size).to(u.pixel),
        (subpixel_column / pixel_size).to(u.pixel),
        (row / pixel_size).to(u.pixel),
        (column / pixel_size).to(u.pixel),
    )

    hdr = fits.Header(
        [
            ("AUTHOR1", "LLNL"),
            ("AUTHOR2", "Pandora DPC"),
            ("ORIGIN", filename.split("/")[-1]),
            ("SUFFIX", suffix),
            ("CREATED", created_on),
            ("DATE", datetime.now().isoformat()),
            (
                "PIXSIZE",
                pixel_size.value,
                f"PSF pixel size in {pixel_size.unit.to_string()}",
            ),
            (
                "SUBPIXSZ",
                subpixel_size.value,
                f"PSF sub pixel size in {subpixel_size.unit.to_string()}",
            ),
        ]
    )
    primaryhdu = fits.PrimaryHDU(header=hdr)
    hdu = fits.HDUList(
        [
            primaryhdu,
            fits.ImageHDU(PSF, name="PSF"),
            fits.ImageHDU(row.value, name="ROW"),
            fits.ImageHDU(column.value, name="COLUMN"),
            fits.ImageHDU(wavelength.value, name="WAVELENGTH"),
        ]
    )
    hdu[2].header["UNIT"] = row.unit.to_string()
    hdu[3].header["UNIT"] = column.unit.to_string()
    hdu[4].header["UNIT"] = wavelength.unit.to_string()
    hdu.writeto(
        PACKAGEDIR
        + f"/data/pandora_{suffix}_{datetime.strftime(datetime.now(), '%Y-%m')}.fits",
        overwrite=True,
    )


# def make_vis_PSF_fits_files(dir, nbin=1, suffix="_hr"):
#     d = loadmat(dir + "pandora_vis_20220506_hot_PSF_512.mat")

#     PSF_hot = d["PSF"][:-1, :-1, :]  # [128:384, 128:384]
#     PSF_hot = np.asarray(
#         [[PSF_hot[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
#     ).mean(axis=(0, 1))
#     PSF_hot = PSF_hot.reshape((*PSF_hot.shape[:2], 9, 9, 5))
#     d = loadmat(dir + "pandora_vis_20220506_cold_PSF_512.mat")

#     PSF_cold = d["PSF"][:-1, :-1, :]  # [128:384, 128:384]
#     PSF_cold = np.asarray(
#         [[PSF_cold[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
#     ).mean(axis=(0, 1))
#     PSF_cold = PSF_cold.reshape((*PSF_cold.shape[:2], 9, 9, 5))

#     sub_pixel_size = nbin * d["dx"][0][0] * u.micron / u.pix
#     pixel_size = 6.5 * u.micron / u.pix
#     x = d["x"][0].reshape((9, 9, 5)) * u.mm
#     y = d["y"][0].reshape((9, 9, 5)) * u.mm
#     x = (x / pixel_size).to(u.pixel)
#     y = (y / pixel_size).to(u.pixel)
#     wvl = d["wvl"][0].reshape((9, 9, 5)) * u.micron
#     temp = np.asarray([-10.0, 30.0]) * u.deg_C
#     PSF = np.asarray([PSF_cold, PSF_hot]).transpose([1, 2, 3, 4, 5, 0])
#     PSF /= PSF.sum(axis=(0, 1))[None, None]

#     PSF = np.asarray(
#         [
#             [
#                 [
#                     [
#                         gaussian_filter(PSF[:, :, idx, jdx, kdx, ldx], 6)
#                         for idx in range(PSF.shape[2])
#                     ]
#                     for jdx in range(PSF.shape[3])
#                 ]
#                 for kdx in range(PSF.shape[4])
#             ]
#             for ldx in range(PSF.shape[5])
#         ]
#     )
#     PSF = PSF.transpose([4, 5, 3, 2, 1, 0])
#     x = x[:, :, :, None] * np.ones(len(temp))
#     y = y[:, :, :, None] * np.ones(len(temp))
#     wvl = wvl[:, :, :, None] * np.ones(len(temp))
#     temp = temp[None, None, None, :] * np.ones(wvl.shape)

#     hdr = fits.Header(
#         [
#             ("AUTHOR1", "LLNL"),
#             ("AUTHOR2", "Christina Hedges"),
#             ("ORIGIN1", "pandora_vis_20220506_cold_PSF_512.mat"),
#             ("ORIGIN2", "pandora_vis_20220506_hot_PSF_512.mat"),
#             ("CREATED", str(d["__header__"]).split("Created on: ")[-1][:-1]),
#             ("DATE", datetime.now().isoformat()),
#             (
#                 "PIXSIZE",
#                 pixel_size.value,
#                 f"PSF pixel size in {pixel_size.unit.to_string()}",
#             ),
#             (
#                 "SUBPIXSZ",
#                 sub_pixel_size.value,
#                 f"PSF sub pixel size in {sub_pixel_size.unit.to_string()}",
#             ),
#         ]
#     )
#     primaryhdu = fits.PrimaryHDU(header=hdr)
#     hdu = fits.HDUList(
#         [
#             primaryhdu,
#             fits.ImageHDU(PSF, name="PSF"),
#             fits.ImageHDU(x.value, name="X"),
#             fits.ImageHDU(y.value, name="Y"),
#             fits.ImageHDU(wvl.value, name="WAVELENGTH"),
#             fits.ImageHDU(temp.value, name="TEMPERATURE"),
#         ]
#     )
#     hdu[2].header["UNIT"] = y.unit.to_string()
#     hdu[3].header["UNIT"] = x.unit.to_string()
#     hdu[4].header["UNIT"] = wvl.unit.to_string()
#     hdu[5].header["UNIT"] = temp.unit.to_string()
#     hdu.writeto(
#         PACKAGEDIR + f"/data/pandora_vis{suffix}_20220506.fits",
#         overwrite=True,
#     )


# def make_nir_PSF_fits_files(dir, nbin=1, suffix="_hr"):
#     # NIR PSF with thermal info
#     # -----------------------------------#

#     d = loadmat(dir + "pandora_nir_20220506_thin_prism_hot_PSF_512.mat")
#     PSF_hot = d["PSF"][:-1, :-1, :-1][128:384, 128:384]
#     PSF_hot = np.asarray(
#         [[PSF_hot[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
#     ).mean(axis=(0, 1))

#     d = loadmat(dir + "pandora_nir_20220506_thin_prism_cold_PSF_512.mat")

#     PSF_cold = d["PSF"][:-1, :-1, :-1][128:384, 128:384]
#     PSF_cold = np.asarray(
#         [[PSF_cold[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]
#     ).mean(axis=(0, 1))

#     sub_pixel_size = nbin * d["dx"][0][0] * u.micron / u.pix
#     pixel_size = 18 * u.micron / u.pix

#     # Bin down wavelength too, also too big.
#     wbin = 10
#     PSF_cold = np.asarray([PSF_cold[:, :, idx::wbin] for idx in range(wbin)]).mean(
#         axis=(0)
#     )
#     PSF_hot = np.asarray([PSF_hot[:, :, idx::wbin] for idx in range(wbin)]).mean(
#         axis=(0)
#     )
#     wvl = (
#         np.asarray([d["wvl"][:900][idx::wbin] for idx in range(wbin)]).mean(axis=(0))[
#             :, 0
#         ]
#         * u.micron
#     )
#     temp = np.asarray([-10.0, 30.0]) * u.deg_C
#     wvl = wvl[:, None] * np.ones(len(temp))
#     temp = temp[None, :] * np.ones(wvl.shape)

#     PSF = np.asarray([PSF_cold, PSF_hot]).transpose([1, 2, 3, 0])
#     PSF /= PSF.sum(axis=(0, 1))

#     PSF = np.asarray(
#         [
#             [gaussian_filter(PSF[:, :, idx, jdx], 3) for idx in range(PSF.shape[2])]
#             for jdx in range(PSF.shape[3])
#         ]
#     )
#     PSF = PSF.transpose([2, 3, 1, 0])

#     hdr = fits.Header(
#         [
#             ("AUTHOR1", "LLNL"),
#             ("AUTHOR2", "Christina Hedges"),
#             ("ORIGIN", "pandora_nir_20220506.mat"),
#             ("CREATED", str(d["__header__"]).split("Created on: ")[-1][:-1]),
#             ("DATE", datetime.now().isoformat()),
#             (
#                 "PIXSIZE",
#                 pixel_size.value,
#                 f"PSF pixel size in {pixel_size.unit.to_string()}",
#             ),
#             (
#                 "SUBPIXSZ",
#                 sub_pixel_size.value,
#                 f"PSF sub pixel size in {sub_pixel_size.unit.to_string()}",
#             ),
#         ]
#     )
#     primaryhdu = fits.PrimaryHDU(header=hdr)
#     hdu = fits.HDUList(
#         [
#             primaryhdu,
#             fits.ImageHDU(PSF, name="PSF"),
#             fits.ImageHDU(wvl.value, name="WAVELENGTH"),
#             fits.ImageHDU(temp.value, name="TEMPERATURE"),
#         ]
#     )
#     hdu[2].header["UNIT"] = wvl.unit.to_string()
#     hdu[3].header["UNIT"] = temp.unit.to_string()
#     hdu.writeto(
#         PACKAGEDIR + f"/data/pandora_nir{suffix}_20220506.fits",
#         overwrite=True,
#     )


# def make_gaussian_psf():
#     hdu1 = fits.open(f"{PACKAGEDIR}/data/pandora_vis_hr_20220506.fits")
#     pixel_size = hdu1[0].header["PIXSIZE"] * u.micron / u.pix
#     sub_pixel_size = hdu1[0].header["SUBPIXSZ"] * u.micron / u.pix
#     shape = hdu1[1].data.shape[:2]
#     psf_column = (
#         (np.arange(shape[1]) - shape[1] // 2) * u.pixel * sub_pixel_size
#     ) / pixel_size
#     psf_row = (
#         (np.arange(shape[0]) - shape[0] // 2) * u.pixel * sub_pixel_size
#     ) / pixel_size

#     def gauss(row, col, sigma):
#         return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
#             -(
#                 (col[None, :]) ** 2 / (2 * sigma**2)
#                 + (row[:, None]) ** 2 / (2 * sigma**2)
#             )
#         )

#     R = np.abs(hdu1[2].data[:, 0, 0, 0]) / 600 + 1
#     C = np.abs(hdu1[3].data[0, :, 0, 0]) / 600 + 1

#     ar = np.zeros((9, 9, 256, 256))
#     for idx, r1 in enumerate(R):
#         for jdx, c1 in enumerate(C):
#             ar[idx, jdx] = gauss(psf_row.value, psf_column.value, np.hypot(r1, c1))
#     ar = ar.transpose([2, 3, 0, 1])
#     integral = np.trapz(
#         np.trapz(ar, dx=np.median(np.diff(psf_row).value), axis=0),
#         dx=np.median(np.diff(psf_column).value),
#         axis=0,
#     )
#     ar /= integral[None, None, :, :]
#     hdr = fits.Header(
#         [
#             ("AUTHOR1", "Christina Hedges"),
#             ("ORIGIN1", "Gaussian"),
#             ("CREATED", datetime.now().isoformat()),
#             ("DATE", datetime.now().isoformat()),
#             (
#                 "PIXSIZE",
#                 pixel_size.value,
#                 f"PSF pixel size in {pixel_size.unit.to_string()}",
#             ),
#             (
#                 "SUBPIXSZ",
#                 sub_pixel_size.value,
#                 f"PSF sub pixel size in {sub_pixel_size.unit.to_string()}",
#             ),
#         ]
#     )
#     primaryhdu = fits.PrimaryHDU(header=hdr)
#     hdu = fits.HDUList(
#         [
#             primaryhdu,
#             fits.ImageHDU(ar, name="PSF"),
#             fits.ImageHDU(hdu1[2].data[:, :, 0, 0], name="X"),
#             fits.ImageHDU(hdu1[3].data[:, :, 0, 0], name="Y"),
#         ]
#     )
#     hdu[2].header["UNIT"] = u.pixel.to_string()
#     hdu[3].header["UNIT"] = u.pixel.to_string()
#     hdu.writeto(
#         f"{PACKAGEDIR}/data/pandora_gaussian_psf.fits",
#         overwrite=True,
#     )


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


# def bin_prf(psf0, psf_column, psf_row, location=(0, 0), normalize=True):
#     mod = (psf_column + location[1]) % 1
#     cyc = ((psf_column + location[1]) - mod).astype(int)
#     colbin = np.unique(cyc)
#     psf1 = np.asarray(
#         [psf0[:, cyc == c].sum(axis=1) / (cyc == c).sum() for c in colbin]
#     ).T

#     mod = (psf_row + location[0]) % 1
#     cyc = ((psf_row + location[0]) - mod).astype(int)
#     rowbin = np.unique(cyc)
#     psf2 = np.asarray([psf1[cyc == c].sum(axis=0) / (cyc == c).sum() for c in rowbin])
#     if normalize:
#         psf2 /= psf2.sum()
#     return rowbin.astype(int), colbin.astype(int), psf2


def bin_prf(psf0, psf_column, psf_row, location=(0, 0), normalize=True):
    df = pd.DataFrame(
        psf0, index=psf_row + location[0], columns=psf_column + location[1]
    )
    int_location = (
        int(location[0] - (location[0] % 1)),
        int(location[1] - (location[1] % 1)),
    )
    irange, crange = (
        np.floor(psf_row.min()).astype(int) - 1 + int_location[0],
        np.ceil(psf_row.max()).astype(int) + 2 + int_location[0],
    ), (
        np.floor(psf_column.min()).astype(int) - 1 + int_location[1],
        np.ceil(psf_column.max()).astype(int) + 2 + int_location[1],
    )
    df2 = (
        df.groupby(by=df.index - (df.index % 1))
        .sum()
        .T.groupby(by=df.columns - (df.columns % 1))
        .sum()
        .reindex(range(*crange), fill_value=0)
        .T.reindex(range(*irange), fill_value=0)
    )
    if normalize:
        df2 /= df2.sum().sum()
    #    import pdb;pdb.set_trace()
    return np.asarray(list(df2.index)), np.asarray(list(df2.columns)), df2.values


def downsample(ar, n=2):
    if not ar.ndim == 3:
        raise ValueError("Pass a 3D array")
    if not isinstance(n, int):
        raise ValueError("Can only downsample by an integer")

    # Calculate the dimensions for the result array
    new_shape = (ar.shape[0], *tuple(np.asarray(ar.shape[1:]) // n))

    # Reshape the input array into smaller blocks
    try:
        reshaped_array = ar.reshape(
            ar.shape[0],
            new_shape[1],
            ar.shape[1] // new_shape[1],
            new_shape[2],
            ar.shape[2] // new_shape[2],
        )
    except ValueError:
        raise ValueError(
            f"Can not reshape array shape {ar.shape} into array shape {new_shape}."
        )
    # Take the mean over the new dimensions
    return reshaped_array.sum(axis=(2, 4))
