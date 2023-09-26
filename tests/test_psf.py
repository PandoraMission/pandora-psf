# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

# First-party/Local
from pandorapsf import PACKAGEDIR, PSF, TESTDIR, __version__
from pandorapsf.plotting import plot_spatial, plot_spectral
from pandorapsf.psf import OutOfBoundsError
from pandorapsf.utils import prep_for_add


def test_version():
    assert __version__ == "0.1.0"


def test_vis_psf_init():
    p = PSF.from_file(
        filename=f"{PACKAGEDIR}/data/pandora_vis_20220506.fits",
        blur_value=(0.25 * u.pixel, 0.25 * u.pixel),
    )
    assert p.shape == (128, 128)
    assert p.ndims == 4

    # jittering does something...
    #    p._jitter(np.arange(0, 0.5, 0.1), np.arange(0, 0.5, 0.1))
    #    assert p._psf_flux_jitter.sum() != 0

    # freeze some dimensions
    p = p.freeze_dimension(wavelength=p.wavelength0d, temperature=p.temperature0d)
    assert p.shape == (128, 128)
    assert p.ndims == 2
    assert (
        p.__repr__()
        == "2D PSF Model [row, column] (Frozen: wavelength: 0.54 micron, temperature: 30.0 deg_C)"
    )


def test_nir_psf_init():
    p = PSF.from_file(
        filename=f"{PACKAGEDIR}/data/pandora_nir_20220506.fits",
        blur_value=(0.25 * u.pixel, 0.25 * u.pixel),
    )
    assert p.shape == (128, 128)
    assert p.ndims == 2

    # jittering does something...
    #    p._jitter(np.arange(0, 0.5, 0.1), np.arange(0, 0.5, 0.1))
    #    assert p._psf_flux_jitter.sum() != 0

    # freeze some dimensions
    p = p.freeze_dimension(temperature=p.temperature0d)
    assert p.shape == (128, 128)
    assert p.ndims == 1
    assert p.__repr__() == "1D PSF Model [wavelength] (Frozen: temperature: 30.0 deg_C)"


def test_vis_psf():
    p = PSF.from_file(
        filename=f"{PACKAGEDIR}/data/pandora_vis_20220506.fits",
        blur_value=(0.25 * u.pixel, 0.25 * u.pixel),
    )
    # Should raise out of bounds
    with pytest.raises(OutOfBoundsError):
        p._check_bounds(temperature=-40 * u.deg_C)
    p._check_bounds(temperature=10 * u.deg_C)
    p._check_bounds(temperature=10 * u.deg_C, wavelength=0.5 * u.micron)
    p._check_bounds(temperature=10 * u.deg_C, wavelength=500 * u.nm)

    ar = p.psf(row=0, column=0, wavelength=0.5 * u.micron, temperature=10 * u.deg_C)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(
        wavelength=0.5 * u.micron,
        temperature=10 * u.deg_C,
        row=0,
        column=0,
    )
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(
        row=-600, column=600, wavelength=0.5 * u.micron, temperature=10 * u.deg_C
    )
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(row=0, column=0)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    rb, cb, ar = p.prf(
        row=0 * u.pixel,
        column=0 * u.pixel,
        wavelength=0.5 * u.micron,
        temperature=10 * u.deg_C,
    )
    assert ar.ndim == 2
    assert rb.ndim == 1
    assert cb.ndim == 1
    assert np.isclose(ar.sum(), 1, rtol=1e-6)
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)
    # with pytest.raises(ValueError):
    #     rb, cb, ar = p.prf(
    #         row=0 * u.pixel, column=0 * u.pixel, temperature=10 * u.deg_C
    #     )
    fig = plot_spatial(p, n=3, image_type="PSF")
    fig.savefig(TESTDIR + "output/test_vis_psf.png", dpi=150, bbox_inches="tight")
    fig = plot_spatial(p, n=3, image_type="PRF")
    fig.savefig(TESTDIR + "output/test_vis_prf.png", dpi=150, bbox_inches="tight")

    fig = plot_spectral(p, var="wavelength", n=7, image_type="PSF")
    fig.savefig(TESTDIR + "output/test_vis_psf_wav.png", dpi=150, bbox_inches="tight")
    fig = plot_spectral(p, var="wavelength", n=7, image_type="PRF")
    fig.savefig(TESTDIR + "output/test_vis_prf_wav.png", dpi=150, bbox_inches="tight")

    fig = plot_spectral(p, var="temperature", n=7, image_type="PSF")
    fig.savefig(TESTDIR + "output/test_vis_psf_temp.png", dpi=150, bbox_inches="tight")
    fig = plot_spectral(p, var="temperature", n=7, image_type="PRF")
    fig.savefig(TESTDIR + "output/test_vis_prf_temp.png", dpi=150, bbox_inches="tight")

    # Can't check bounds of a dropped dimension
    p = p.freeze_dimension(wavelength=p.wavelength0d, temperature=p.temperature0d)
    with pytest.raises(KeyError):
        p._check_bounds(temperature=10 * u.deg_C)

    rb, cb, ar = p.prf(row=0 * u.pixel, column=0 * u.pixel)
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)

    ar2 = np.zeros((100, 100))
    for r0, c0 in np.random.uniform(-50, 50, size=(10, 2)):
        rb, cb, ar = p.prf(row=r0 * u.pixel, column=c0 * u.pixel)
        row1, col1, prf1 = prep_for_add(rb, cb, ar, ar2.shape, corner=(-50, -50))
        ar2[row1, col1] += prf1
    fig, ax = plt.subplots()
    ax.imshow(ar2, origin="lower")
    fig.savefig(TESTDIR + "output/test.png", dpi=150, bbox_inches="tight")

    with pytest.raises(ValueError):
        rb, cb, ar = p.prf(
            row=0 * u.pixel, column=0 * u.pixel, temperature=10 * u.deg_C
        )
    rb, cb, ar = p.prf(row=600, column=-600)
    assert np.isclose(rb.mean(), 600, atol=1)
    assert np.isclose(cb.mean(), -600, atol=1)

    fig = plot_spatial(p, n=3, image_type="PSF")
    plt.close("all")

    p = p.freeze_dimension(row=0 * u.pixel, column=0 * u.pixel)
    ar2 = np.zeros((100, 100))
    for r0, c0 in np.random.uniform(-50, 50, size=(10, 2)):
        rb, cb, ar = p.prf(row=r0 * u.pixel, column=c0 * u.pixel)
        row1, col1, prf1 = prep_for_add(rb, cb, ar, ar2.shape, corner=(-50, -50))
        ar2[row1, col1] += prf1
    fig, ax = plt.subplots()
    ax.imshow(ar2, origin="lower")
    fig.savefig(TESTDIR + "output/test.png", dpi=150, bbox_inches="tight")

    p = PSF.from_name("visda")
    assert p.psf().shape == p.shape
    assert p.dpsf().shape == (2, *p.shape)
    assert p.dprf(row=0, column=0)[2].ndim == 3


def test_nir_psf():
    p = PSF.from_file(
        filename=f"{PACKAGEDIR}/data/pandora_nir_20220506.fits",
        blur_value=(0.25 * u.pixel, 0.25 * u.pixel),
    )
    # Should raise out of bounds
    with pytest.raises(OutOfBoundsError):
        p._check_bounds(temperature=-40 * u.deg_C)
    p._check_bounds(temperature=10 * u.deg_C)
    p._check_bounds(temperature=10 * u.deg_C, wavelength=1 * u.micron)
    p._check_bounds(temperature=10 * u.deg_C, wavelength=1000 * u.nm)

    ar = p.psf(wavelength=1 * u.micron, temperature=10 * u.deg_C)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    ar = p.psf(wavelength=1 * u.micron)
    assert ar.shape == p.shape
    assert np.isclose(ar.sum(), 1, rtol=1e-6)

    rb, cb, ar = p.prf(
        row=0 * u.pixel,
        column=0 * u.pixel,
        wavelength=1 * u.micron,
        temperature=10 * u.deg_C,
    )
    assert ar.ndim == 2
    assert rb.ndim == 1
    assert cb.ndim == 1
    assert np.isclose(ar.sum(), 1, rtol=1e-6)
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)
    # with pytest.raises(ValueError):
    #     rb, cb, ar = p.prf(
    #         row=0 * u.pixel, column=0 * u.pixel, temperature=10 * u.deg_C
    #     )
    fig = plot_spectral(p, var="wavelength", n=7, image_type="PSF")
    fig.savefig(TESTDIR + "output/test_nir_psf.png", dpi=150, bbox_inches="tight")
    fig = plot_spectral(p, var="wavelength", n=7, image_type="PRF")
    fig.savefig(TESTDIR + "output/test_nir_prf.png", dpi=150, bbox_inches="tight")

    fig = plot_spectral(p, var="temperature", n=5, image_type="PRF")
    fig.savefig(TESTDIR + "output/test_nir_psf_temp.png", dpi=150, bbox_inches="tight")

    # Can't check bounds of a dropped dimension
    p = p.freeze_dimension(temperature=p.temperature0d)
    with pytest.raises(KeyError):
        p._check_bounds(temperature=10 * u.deg_C)

    rb, cb, ar = p.prf(row=0 * u.pixel, column=0 * u.pixel, wavelength=1 * u.micron)
    assert np.isclose(rb.mean(), 0, atol=1)
    assert np.isclose(cb.mean(), 0, atol=1)
    with pytest.raises(ValueError):
        rb, cb, ar = p.prf(
            row=0 * u.pixel, column=0 * u.pixel, temperature=10 * u.deg_C
        )
    rb, cb, ar = p.prf(row=600, column=-600, wavelength=1.0 * u.micron)
    assert np.isclose(rb.mean(), 600, atol=1)
    assert np.isclose(cb.mean(), -600, atol=1)

    p = PSF.from_name("nirda")
    assert hasattr(p, "trace_wavelength")
    assert hasattr(p, "trace_sensitivity")
