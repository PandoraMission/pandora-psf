# Third-party
import matplotlib.pyplot as plt
import numpy as np

# import pytest
from scipy import sparse

# First-party/Local
from pandorapsf import PSF, TESTDIR, Scene, TraceScene
from pandorapsf.scene import SparseWarp3D


def test_centered():
    for name in ['gaussian', 'visda', 'nirda']:
        p = pandorapsf.PSF.from_name(name)
        locations = np.vstack([np.asarray([0]), np.asarray([0])]).T
        s = Scene(locations, psf=p, shape=(50, 70), corner=(-25, -35))
        delta_pos = np.random.normal(0, 3, size=(2, 100))
        delta_pos[1] *= 2
        flux = np.ones(delta_pos.shape[1])[None, :]
        
        ar = s.model(flux, delta_pos)
        
        R, C = np.meshgrid(np.arange(50), np.arange(70), indexing='ij')
        rmid = np.asarray([np.average(R, weights=ar[tdx]) for tdx in range(ar.shape[0])])
        cmid = np.asarray([np.average(C, weights=ar[tdx]) for tdx in range(ar.shape[0])])
        dr = rmid - R.mean() - delta_pos[0]
        dc = cmid - C.mean() - delta_pos[1]
        np.allclose(dr - np.mean(dr), 0, atol=0.01)
        np.allclose(dc - np.mean(dc), 0, atol=0.01)

def test_simple_vis_scene():
    row, column = np.meshgrid(
        np.arange(-1024, 1024, 100), np.arange(-1024, 1024, 100), indexing="ij"
    )
    locations = np.vstack([row.ravel(), column.ravel()]).T
    s = Scene(locations=locations, shape=(2048, 2048), corner=(-1024, -1024))
    assert (s.X.sum(axis=0) <= 1.0 + 1e10).all()
    img = s.model(np.ones(s.X.shape[1]))
    assert img.ndim == 3
    fig, ax = plt.subplots()
    ax.imshow(np.log10(img[0]), origin="lower")
    ax.set(title="Simple Visible Scene Test", xlabel="Pixels", ylabel="Pixels")
    fig.savefig(TESTDIR + "output/test_vis_scene.png", dpi=150, bbox_inches="tight")


def test_vis_grad_scene():
    row, column = np.meshgrid(
        np.arange(0, 100, 10), np.arange(0, 100, 10), indexing="ij"
    )
    locations = np.vstack([row.ravel(), column.ravel()]).T
    s = Scene(locations=locations, shape=(100, 100), corner=(0, 0))
    img0 = s.dX0.dot(np.ones(s.dX0.shape[1])).reshape((100, 100))
    img1 = s.dX1.dot(np.ones(s.dX1.shape[1])).reshape((100, 100))
    img = img0 + img1
    fig, ax = plt.subplots()
    ax.imshow(img, origin="lower")
    ax.set(title="Simple Visible Grad Test", xlabel="Pixels", ylabel="Pixels")
    fig.savefig(
        TESTDIR + "output/test_vis_grad_scene.png",
        dpi=150,
        bbox_inches="tight",
    )


def test_simple_IR_scene():
    row, column = np.meshgrid(
        np.arange(-200, 200, 20), np.arange(-20, 20, 8), indexing="ij"
    )
    locations = np.vstack([row.ravel(), column.ravel()]).T
    p = PSF.from_name("nirda")
    s = Scene(locations=locations, psf=p, shape=(400, 80), corner=(-200, -40))
    img = s.model(np.ones(s.X.shape[1]))
    assert img.ndim == 3
    fig, ax = plt.subplots()
    ax.imshow(np.log10(img[0]), origin="lower")
    ax.set(title="Simple IR Scene Test", xlabel="Pixels", ylabel="Pixels")
    fig.savefig(TESTDIR + "output/test_nir_scene.png", dpi=150, bbox_inches="tight")


# Image shapes are wrong here need to fix
def test_trace_scene():
    p = PSF.from_name("nirda")
    locations = np.vstack([np.asarray([250])[:, None], np.asarray([40])[:, None]]).T
    s = TraceScene(locations=locations, psf=p, shape=(400, 80), corner=(0, 0))

    spectra = np.ones(p.trace_pixel.shape[0])
    assert (s.X.sum(axis=0) < 4).all()
    img = s.model(spectra)
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)
    img = s.model(spectra[:, None])
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)
    img = s.model(spectra[:, None, None])
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)

    locations = np.vstack([np.asarray([250, 300]), np.asarray([20, 60])]).T
    s = TraceScene(locations=locations, psf=p, shape=(400, 80), corner=(0, 0))
    spectra = np.ones(s.psf.trace_pixel.shape[0])[:, None] * np.ones(2)
    spectra[:, 1] *= 0.1

    img = s.model(spectra)
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)

    img = s.model(spectra[:, :, None])
    assert img.ndim == 3
    assert img.shape == (1, 400, 80)

    assert img[0, :, :40].sum() > 5 * img[0, :, 40:].sum()
    fig, ax = plt.subplots()
    ax.imshow(img[0], origin="lower")
    ax.set(title="IR Trace Test", xlabel="Pixels", ylabel="Pixels")
    fig.savefig(TESTDIR + "output/test_nir_trace.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots()
    img0 = s.dX0.dot(spectra.ravel())
    img1 = s.dX1.dot(spectra.ravel())
    img = img0 + img1
    ax.imshow(img[0], origin="lower")
    ax.set(title="IR Trace Test", xlabel="Pixels", ylabel="Pixels")
    fig.savefig(
        TESTDIR + "output/test_nir_trace_grad.png", dpi=150, bbox_inches="tight"
    )


def test_sparsewarp():
    R, C = np.meshgrid(
        np.arange(20, 25).astype(int), np.arange(10, 16).astype(int), indexing="ij"
    )
    R = R[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    C = C[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    data = np.ones_like(R).astype(float)

    sw = SparseWarp3D(data, R, C, (50, 50))
    assert sw.imshape == (50, 50)
    assert sw.shape == sw.cooshape == (2500, 10)
    assert sw.subshape == R.shape
    assert isinstance(sw, sparse.coo_matrix)
    assert len(sw.data) == 300
    assert sw.data.sum() == 300
    assert sw.dtype == float

    # Move data out of frame
    sw = SparseWarp3D(data, R + 50, C, (50, 50))
    assert len(sw.data) == 0
    # translate back into frame
    sw.translate((-50, 0))
    assert len(sw.data) == 300
    # reset it
    sw.reset()
    assert len(sw.data) == 0

    sw = SparseWarp3D(data, R + np.arange(10), C + np.arange(10), (50, 50))
    sw.translate((-1, 1))
    assert len(sw.data) == 300

    assert sw.dot(np.ones(10)).shape == (1, 50, 50)
    assert isinstance(sw.dot(np.ones(10)), np.ndarray)
    assert sw.dot(np.ones(10)).sum() == 300
