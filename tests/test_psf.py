# Standard library
import os

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

# First-party/Local
from pandorapsf import DATADIR, DOCSDIR, PACKAGEDIR, PSF
from pandorapsf.plotting import plot_spatial, plot_spectral
from pandorapsf.psf import OutOfBoundsError
from pandorapsf.utils import prep_for_add


@pytest.mark.skip(reason="outdated functionality")
def test_gaussian():
    """Check that a gaussian PSF behaves as expected"""
    p = PSF.from_name("gaussian")
    psf, dx, dy = p.psf(
        row=p.row1d[0].value, column=p.column1d[0].value, gradients=True
    )
    R, C = np.meshgrid(
        np.arange(psf.shape[0]), np.arange(psf.shape[1]), indexing="ij"
    )
    original_shape = psf.shape
    binning_factor = 4

    def bin(ar):
        return np.mean(
            ar.reshape(
                (
                    original_shape[0] // binning_factor,
                    binning_factor,
                    original_shape[1] // binning_factor,
                    binning_factor,
                )
            ),
            axis=(1, 3),
        )

    # Binning the Array
    psfb, dxb, dyb, Rb, Cb = bin(psf), bin(dx), bin(dy), bin(R), bin(C)

    integral = np.sum(psfb)
    psfb, dxb, dyb = psfb / integral, dxb / integral, dyb / integral

    ar = psf + dx * 0.1 + dy * 0.2
    arb = psfb + dxb * 0.1 + dyb * 0.2

    assert np.isclose(
        (np.average(R, weights=psf) - np.average(R, weights=ar)),
        0.1,
        rtol=1e-4,
    )
    assert np.isclose(
        (np.average(C, weights=psf) - np.average(C, weights=ar)),
        0.2,
        rtol=1e-4,
    )

    assert np.isclose(
        (np.average(Rb, weights=psfb) - np.average(Rb, weights=arb)),
        0.1,
        rtol=1e-4,
    )
    assert np.isclose(
        (np.average(Cb, weights=psfb) - np.average(Cb, weights=arb)),
        0.2,
        rtol=1e-4,
    )

    rb, cb, prf, dx, dy = p.prf(row=0, column=0, gradients=True)

    R, C = np.meshgrid(
        np.arange(prf.shape[0]), np.arange(prf.shape[1]), indexing="ij"
    )

    ar = prf + dx * 0.1 + dy * 0.2

    assert np.isclose(
        (np.average(R, weights=prf) - np.average(R, weights=ar)),
        0.1,
        rtol=1e-4,
    )
    assert np.isclose(
        (np.average(C, weights=prf) - np.average(C, weights=ar)),
        0.2,
        rtol=1e-4,
    )


def test_vis_psf_init():
    p = PSF.from_file(
        file=f"{PACKAGEDIR}/data/pandora_vis_psf_lowres.fits", name="visda"
    )
    assert p.shape == (256, 256)
    assert p.ndims == 2


def test_nir_psf_init():
    p = PSF.from_file(
        file=f"{PACKAGEDIR}/data/pandora_nir_psf_lowres.fits", name="nirda"
    )
    assert p.shape == (256, 256)
    assert p.ndims == 1


def test_vis_psf():
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip(
            "Skipping this test on GitHub Actions, as this requires PSF file download."
        )
    p = PSF.from_file(file=f"{DATADIR}/pandora_vis_psf.fits", name="visda")

    # Should raise out of bounds
    with pytest.raises(OutOfBoundsError):
        p._check_bounds(wavelength=40 * u.micron)
    p._check_bounds(wavelength=0.5 * u.micron)
    p._check_bounds(wavelength=500 * u.nm)

    ar = p.psf(row=0, column=0, wavelength=0.5 * u.micron)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(
        wavelength=0.5 * u.micron,
        row=0,
        column=0,
    )
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(row=-600, column=600, wavelength=0.5 * u.micron)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(row=0, column=0)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    rb, cb, ar = p.prf(
        row=0 * u.pixel,
        column=0 * u.pixel,
        wavelength=0.5 * u.micron,
    )
    assert ar.ndim == 2
    assert rb.ndim == 1
    assert cb.ndim == 1
    assert np.isclose(ar.sum(), 1, rtol=1e-6)
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)

    fig = plot_spatial(p, n=3, image_type="PSF")
    fig.savefig(
        DOCSDIR + "images/test_vis_psf.png", dpi=150, bbox_inches="tight"
    )
    fig = plot_spatial(p, n=3, image_type="PRF")
    fig.savefig(
        DOCSDIR + "images/test_vis_prf.png", dpi=150, bbox_inches="tight"
    )

    fig = plot_spectral(p, var="wavelength", n=7, image_type="PSF")
    fig.savefig(
        DOCSDIR + "images/test_vis_psf_wav.png", dpi=150, bbox_inches="tight"
    )
    fig = plot_spectral(p, var="wavelength", n=7, image_type="PRF")
    fig.savefig(
        DOCSDIR + "images/test_vis_prf_wav.png", dpi=150, bbox_inches="tight"
    )

    # Can't check bounds of a dropped dimension
    p = p.freeze_dimension(wavelength=p.wavelength0d)
    with pytest.raises(KeyError):
        p._check_bounds(wavelength=10 * u.micron)

    rb, cb, ar = p.prf(row=0 * u.pixel, column=0 * u.pixel)
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)

    ar2 = np.zeros((100, 100))
    for r0, c0 in np.random.uniform(-50, 50, size=(10, 2)):
        rb, cb, ar = p.prf(row=r0 * u.pixel, column=c0 * u.pixel)
        row1, col1, prf1 = prep_for_add(
            rb, cb, ar, ar2.shape, corner=(-50, -50)
        )
        ar2[row1, col1] += prf1
    fig, ax = plt.subplots()
    ax.imshow(ar2, origin="lower")
    fig.savefig(DOCSDIR + "images/test.png", dpi=150, bbox_inches="tight")

    rb, cb, ar = p.prf(row=600, column=-600)
    assert np.isclose(rb.mean(), 600, atol=1)
    assert np.isclose(cb.mean(), -600, atol=1)

    fig = plot_spatial(p, n=3, image_type="PSF")
    plt.close("all")

    p = p.freeze_dimension(row=0 * u.pixel, column=0 * u.pixel)
    ar2 = np.zeros((100, 100))
    for r0, c0 in np.random.uniform(-50, 50, size=(10, 2)):
        rb, cb, ar = p.prf(row=r0 * u.pixel, column=c0 * u.pixel)
        row1, col1, prf1 = prep_for_add(
            rb, cb, ar, ar2.shape, corner=(-50, -50)
        )
        ar2[row1, col1] += prf1
    fig, ax = plt.subplots()
    ax.imshow(ar2, origin="lower")
    fig.savefig(DOCSDIR + "images/test.png", dpi=150, bbox_inches="tight")

    p = PSF.from_name("visda")
    assert p.psf().shape == p.shape
    assert np.asarray(p.psf(gradients=True)[1:]).shape == (2, *p.shape)
    assert np.asarray(p.prf(row=0, column=0, gradients=True)[3:5]).ndim == 3


def test_nir_psf():
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip(
            "Skipping this test on GitHub Actions, as this requires PSF file download."
        )
    p = PSF.from_file(file=f"{DATADIR}/pandora_nir_psf.fits", name="nirda")
    # Should raise out of bounds
    with pytest.raises(OutOfBoundsError):
        p._check_bounds(wavelength=40 * u.micron)
    p._check_bounds(wavelength=1 * u.micron)
    p._check_bounds(wavelength=1000 * u.nm)

    ar = p.psf(wavelength=1 * u.micron)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(wavelength=1 * u.micron)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    rb, cb, ar = p.prf(
        row=0 * u.pixel,
        column=0 * u.pixel,
        wavelength=1 * u.micron,
    )
    assert ar.ndim == 2
    assert rb.ndim == 1
    assert cb.ndim == 1
    assert np.isclose(ar.sum(), 1, rtol=1e-6)
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)

    fig = plot_spectral(p, var="wavelength", n=7, image_type="PSF")
    fig.savefig(
        DOCSDIR + "images/test_nir_psf.png", dpi=150, bbox_inches="tight"
    )
    fig = plot_spectral(p, var="wavelength", n=7, image_type="PRF")
    fig.savefig(
        DOCSDIR + "images/test_nir_prf.png", dpi=150, bbox_inches="tight"
    )

    rb, cb, ar = p.prf(
        row=0 * u.pixel, column=0 * u.pixel, wavelength=1 * u.micron
    )
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)
    rb, cb, ar = p.prf(row=200, column=-200, wavelength=1.0 * u.micron)
    assert np.isclose(rb.mean(), 200, atol=1)
    assert np.isclose(cb.mean(), -200, atol=1)

    p = PSF.from_name("nirda")
    assert hasattr(p, "trace_wavelength")
    assert hasattr(p, "trace_sensitivity")


# Needs to test the `extrapolate` funcionality
