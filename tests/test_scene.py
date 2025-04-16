# Standard library
import os
from copy import deepcopy

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytest

# First-party/Local
from pandorapsf import DOCSDIR, PSF, ROIScene, Scene, TraceScene
from pandorapsf.utils import downsample, prep_for_add

if os.getenv("GITHUB_ACTIONS") == "true":
    # github actions uses the fallback data so we don't download from zenodo
    pn = PSF.from_name("nirda_fallback")
    pv = PSF.from_name("visda_fallback")
else:
    pn = PSF.from_name("nirda")
    pv = PSF.from_name("visda")


def test_simple_vis_scene():
    row, column = np.meshgrid(
        np.arange(-1024, 1024, 100), np.arange(-1024, 1024, 100), indexing="ij"
    )
    locations = np.vstack([row.ravel(), column.ravel()]).T
    s = Scene(
        locations=locations, shape=(2048, 2048), corner=(-1024, -1024), psf=pv
    )
    assert (s.X.dot(np.ones(s.X.shape[-1])).sum(axis=0) <= 1.0 + 1e10).all()
    img = s.model(np.ones(s.X.shape[1]), quiet=True)
    assert img.ndim == 3
    if os.getenv("GITHUB_ACTIONS") != "true":
        fig, ax = plt.subplots()
        ax.imshow(np.log10(img[0]), origin="lower")
        ax.set(
            title="Simple Visible Scene Test", xlabel="Pixels", ylabel="Pixels"
        )
        fig.savefig(
            DOCSDIR + "images/test_vis_scene.png", dpi=150, bbox_inches="tight"
        )


def test_vis_grad_scene():
    row, column = np.meshgrid(
        np.arange(0, 100, 10), np.arange(0, 100, 10), indexing="ij"
    )
    locations = np.vstack([row.ravel(), column.ravel()]).T
    s = Scene(locations=locations, shape=(100, 100), corner=(0, 0), psf=pv)
    img0 = s.dX0.dot(np.ones(s.dX0.shape[1])).reshape((100, 100))
    img1 = s.dX1.dot(np.ones(s.dX1.shape[1])).reshape((100, 100))
    img = img0 + img1
    if os.getenv("GITHUB_ACTIONS") != "true":
        fig, ax = plt.subplots()
        ax.imshow(img, origin="lower")
        ax.set(
            title="Simple Visible Grad Test", xlabel="Pixels", ylabel="Pixels"
        )
        fig.savefig(
            DOCSDIR + "images/test_vis_grad_scene.png",
            dpi=150,
            bbox_inches="tight",
        )


def test_simple_IR_scene():
    row, column = np.meshgrid(
        np.arange(-200, 200, 20), np.arange(-20, 20, 8), indexing="ij"
    )
    locations = np.vstack([row.ravel(), column.ravel()]).T

    s = Scene(locations=locations, psf=pn, shape=(400, 80), corner=(-200, -40))
    img = s.model(np.ones(s.X.shape[1]), quiet=True)
    assert img.ndim == 3
    if os.getenv("GITHUB_ACTIONS") != "true":
        fig, ax = plt.subplots()
        ax.imshow(np.log10(img[0]), origin="lower")
        ax.set(title="Simple IR Scene Test", xlabel="Pixels", ylabel="Pixels")
        fig.savefig(
            DOCSDIR + "images/test_nir_scene.png", dpi=150, bbox_inches="tight"
        )


def test_roiscene():
    ntimes = 20
    roi_size = (30, 30)
    corners = [(0, 0), (0, 70), (30, 30)]
    locations = np.vstack(
        [
            np.vstack(
                [
                    np.random.uniform(10, roi_size[0] - 10, size=3),
                    np.random.uniform(10, roi_size[1] - 10, size=3),
                ]
            ).T
            + np.asarray(corner)
            for corner in corners
        ]
    )
    true_fluxes = (
        10 ** np.random.uniform(1, 3, size=(locations.shape[0]))[:, None]
        * np.ones(ntimes)[None, :]
    )
    s = ROIScene(
        locations=locations,
        shape=(100, 100),
        corner=(0, 0),
        psf=pv,
        nROIs=len(corners),
        ROI_size=roi_size,
        ROI_corners=corners,
    )
    roi_data = s.model(true_fluxes)
    assert roi_data.shape == (len(corners), ntimes, *roi_size)
    assert np.abs(roi_data[1, 1, :, :] - roi_data[1, 0, :, :]).sum() == 0

    true_fluxes = (
        10 ** np.random.uniform(1, 3, size=(locations.shape[0]))[:, None]
        * np.ones(ntimes)[None, :]
    )
    shot_noise = np.random.normal(
        0, true_fluxes[:, 0] ** 0.5, size=(ntimes, s.ntargets)
    ).T
    true_fluxes += shot_noise
    true_shift = np.random.normal(0, 0.1, size=(2, ntimes))

    roi_data = s.model(true_fluxes, true_shift)
    assert roi_data.shape == (len(corners), ntimes, *roi_size)
    assert np.abs(roi_data[1, 1, :, :] - roi_data[1, 0, :, :]).sum() != 0


@pytest.mark.skip(
    reason="this is very slow and needs API tweaking for a speed up."
)
# Image shapes are wrong here need to fix
def test_trace_scene():
    locations = np.vstack(
        [np.asarray([250])[:, None], np.asarray([40])[:, None]]
    ).T
    s = TraceScene(locations=locations, psf=pn, shape=(400, 80), corner=(0, 0))

    spectra = np.ones(s.nwav)
    assert (s.X.dot(np.ones(s.X.shape[-1])).sum(axis=0) < 4).all()
    #        img = s.model(spectra, quiet=True)
    #        assert img.ndim == 3
    #        assert img.shape == (1, 400, 80)
    img = s.model(spectra[:, None], quiet=True)
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)
    img = s.model(spectra[:, None, None], quiet=True)
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)

    locations = np.vstack([np.asarray([250, 300]), np.asarray([20, 60])]).T
    s = TraceScene(locations=locations, psf=pn, shape=(400, 80), corner=(0, 0))
    spectra = np.ones(s.nwav)[:, None] * np.ones(2)
    spectra[:, 1] *= 0.1

    img = s.model(spectra, quiet=True)
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)

    img = s.model(spectra[:, :, None], quiet=True)
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)

    assert img[0, :, :40].sum() > 5 * img[0, :, 40:].sum()

    fig, ax = plt.subplots()
    ax.imshow(img[0], origin="lower")
    ax.set(title="IR Trace Test", xlabel="Pixels", ylabel="Pixels")
    fig.savefig(
        DOCSDIR + "images/test_nir_trace.png", dpi=150, bbox_inches="tight"
    )

    fig, ax = plt.subplots()
    img0 = s.dX0.dot(spectra.ravel())
    img1 = s.dX1.dot(spectra.ravel())
    img = img0 + img1
    ax.imshow(img[0], origin="lower")
    ax.set(title="IR Trace Test", xlabel="Pixels", ylabel="Pixels")
    fig.savefig(
        DOCSDIR + "images/test_nir_trace_grad.png",
        dpi=150,
        bbox_inches="tight",
    )


@pytest.mark.skip(reason="outdated functionality for now")
def test_scale():
    for detector in ["Gaussian", "VISDA", "NIRDA"]:
        for scale in [1, 2]:
            p = PSF.from_name(detector, scale=scale)
            s = Scene(
                locations=np.asarray([[0, 0]]),
                shape=(50, 50),
                psf=p,
                corner=(-25, -25),
            )
            R, C = np.arange(-0.5, 0.5, 0.02), np.arange(-0.5, 0.5, 0.02)
            truth = np.zeros(
                (R.shape[0], C.shape[0], 50 * p.scale, 50 * p.scale)
            )
            estimate = np.zeros((R.shape[0], C.shape[0], 50, 50))
            for (
                idx,
                r,
            ) in enumerate(R):
                for (
                    jdx,
                    c,
                ) in enumerate(C):
                    rb, cb, ar = prep_for_add(
                        *p.prf(row=r * p.scale, column=c * p.scale),
                        shape=(50 * p.scale, 50 * p.scale),
                        corner=(-25 * p.scale, -25 * p.scale),
                    )
                    truth[idx, jdx, rb, cb] = ar
                    estimate[idx, jdx, :, :] = s.model(
                        np.ones((1)),
                        np.asarray([[r, c]]).T,
                        downsample=True,
                        quiet=True,
                    )[0]
            truth = np.asarray(
                [
                    downsample(truth[idx, :, :, :], p.scale)
                    for idx in range(truth.shape[0])
                ]
            )
            fig, ax = plt.subplots()
            im = ax.pcolormesh(
                R,
                C,
                np.abs((truth - estimate)).sum(axis=(2, 3)).T,
                vmin=0,
                vmax=0.01,
            )
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Residuals")
            plt.gca().set_aspect(1)
            ax.set(
                title=f"{detector}, Scale:{scale}",
                xlabel="Column Sub Pixel Position",
                ylabel="Row Sub Pixel Position",
            )
            fig.savefig(
                DOCSDIR + f"images/test_{detector}_scale_{scale}.png",
                dpi=150,
                bbox_inches="tight",
            )


@pytest.mark.skip(reason="outdated functionality for now")
def test_centered():
    for scale in [1, 2]:
        for name in ["gaussian", "visda", "nirda"]:
            p = PSF.from_name(name, scale=scale)
            locations = np.vstack([np.asarray([0]), np.asarray([0])]).T
            s = Scene(locations, psf=p, shape=(50, 70), corner=(-25, -35))
            delta_pos = np.random.normal(0, 3, size=(2, 100))
            delta_pos[1] *= 2
            flux = np.ones(delta_pos.shape[1])[None, :]

            ar = s.model(flux, delta_pos, quiet=True)

            R, C = np.meshgrid(np.arange(50), np.arange(70), indexing="ij")
            rmid = np.asarray(
                [np.average(R, weights=ar[tdx]) for tdx in range(ar.shape[0])]
            )
            cmid = np.asarray(
                [np.average(C, weights=ar[tdx]) for tdx in range(ar.shape[0])]
            )
            dr = rmid - R.mean() - delta_pos[0]
            dc = cmid - C.mean() - delta_pos[1]
            assert np.allclose(dr - np.median(dr), 0, atol=0.05)
            assert np.allclose(dc - np.median(dc), 0, atol=0.05)


def test_aligned():
    p = PSF.from_name("visda_fallback")
    c, r = deepcopy(p.psf_column.value), deepcopy(p.psf_row.value)
    C, R = np.meshgrid(c, r)
    ar = p.psf(row=0, column=0)
    x1, y1 = np.average(C, weights=ar), np.average(R, weights=ar)

    dR, dC = np.mgrid[-0.5:0.5:11j, -0.5:0.5:11j]

    shape = (100, 101)
    corner = (-50, -50)
    R, C = np.mgrid[: shape[0], : shape[1]]
    R -= 50
    C -= 50
    x, y = np.zeros((2, np.prod(dC.shape)))
    for idx, dc, dr in zip(range(np.prod(dC.shape)), dC.ravel(), dR.ravel()):
        s = Scene(
            locations=np.asarray([dr, dc])[None, :],
            shape=shape,
            corner=corner,
            psf=p,
        )
        ar = s.model(np.ones(1))[0]
        x[idx], y[idx] = (
            np.average(C + 0.5, weights=ar),
            np.average(R + 0.5, weights=ar),
        )
    x, y = x.reshape(dC.shape), y.reshape(dR.shape)
    assert np.allclose(dC + x1, x, atol=0.05)
    assert np.allclose(dR + y1, y, atol=0.05)

    x, y = np.zeros((2, np.prod(dC.shape)))
    s = Scene(
        locations=np.asarray([0, 0])[None, :],
        shape=shape,
        corner=corner,
        psf=p,
    )
    for idx, dc, dr in zip(range(np.prod(dC.shape)), dC.ravel(), dR.ravel()):
        delta_pos = (np.asarray([dr, dc])[None, :] * np.ones(2)[:, None]).T
        ar = s.model(np.ones((1, 2)), delta_pos=delta_pos)[0]
        x[idx], y[idx] = (
            np.average(C + 0.5, weights=ar),
            np.average(R + 0.5, weights=ar),
        )
    x, y = x.reshape(dC.shape), y.reshape(dR.shape)
    assert np.allclose(dC + x1, x, atol=0.05)
    assert np.allclose(dR + y1, y, atol=0.05)
