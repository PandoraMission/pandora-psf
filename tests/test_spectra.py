"""Tests to check spectra look reasonable"""

# Third-party
import astropy.units as u
import numpy as np
import pandorasat as ps

# First-party/Local
import pandorapsf as pp


def test_vega():
    nirda = ps.NIRDetector()
    p = pp.PSF.from_name("nirda_fallback", transpose=True)
    ts = pp.TraceScene(
        np.asarray([[300, 40]]), psf=p, wavelength=p.trace_wavelength[:-1]
    )

    wav, sed = ps.phoenix.load_vega()
    wav = wav.to(u.micron)
    k = (wav.value > 0.6) & (wav.value < 2)
    sens = nirda.sensitivity(wav)

    # The total flux through the sensitivity curve
    a = np.trapz(sens[k] * sed[k], wav[k]).to(u.electron / u.s)

    f = p.integrate_spectrum(wav[k], sed[k], ts.wavelength)
    k = np.isfinite(f)

    # The total flux through the sensitivity curve after integrating to a lower resolution
    b = np.trapz(f[k], ts.wavelength[k]).to(u.electron / u.s)

    # Flux should be conserved
    assert np.isclose(a, b, rtol=1e-3)

    # ar = ts.model(f * np.gradient(ts.wavelength), quiet=True)[0]
    ar = ts.model(f, quiet=True)[0]

    # Total flux per pixel
    spectrum_per_pix = (ar.sum(axis=1)).to(u.electron / u.second / u.pixel)

    # Flux should be conserved
    pix = (np.arange(0, 400, 1) - 300) * u.pixel
    c = np.trapz(spectrum_per_pix, pix)
    assert np.isclose(a, c, rtol=1e-3)

    # convert to function of wavelength
    w = np.interp(pix.value, ts.pixel.value, ts.wavelength.value) * u.micron
    dw = np.gradient(w, pix)

    spectrum_per_wav = spectrum_per_pix / dw

    # Flux should be conserved
    d = np.trapz(spectrum_per_wav[dw != 0], w[dw != 0]).to(u.electron / u.s)
    assert np.isclose(a, d, rtol=1e-3)

    # When divided by the original sensitivity this should have the correct units.
    assert (spectrum_per_wav / nirda.sensitivity(w)).unit.is_equivalent(
        sed.unit
    )

    # The average measured spectrum at 1 micron should be close to the input spectrum
    a = np.interp(
        1,
        w.to(u.micron).value,
        (spectrum_per_wav / nirda.sensitivity(w))
        .to(u.erg / (u.AA * u.s * u.cm**2))
        .value,
    )

    b = np.interp(1, wav.to(u.micron).value, sed.value)

    assert np.isclose(a, b, rtol=1e-3)
