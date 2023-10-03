# Third-party
import matplotlib.pyplot as plt
import numpy as np

# import pytest
from scipy import sparse

# First-party/Local
from pandorapsf import PSF, TESTDIR, Scene  # , TraceScene
from pandorapsf.scene import SparseWarp


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
# def test_trace_scene():
#     locations = np.vstack([np.asarray([250])[:, None], np.asarray([40])[:, None]]).T
#     p = PSF.from_name("nirda")
#     spectra = np.ones(p.trace_pixel.shape[0])[:, None]
#     s = TraceScene(locations=locations, psf=p, shape=(400, 80), corner=(0, 0))
#     assert (s.X.sum(axis=0) < 4).all()
#     img = s.X.dot(spectra.ravel()).reshape(s.shape)

#     locations = np.vstack([np.asarray([250, 300]), np.asarray([40, 60])]).T
#     spectra = np.ones(s.psf.trace_pixel.shape[0])[:, None] * np.ones(2)
#     spectra[:, 1] *= 0.1
#     p = PSF.from_name("nirda")
#     s = TraceScene(locations=locations, psf=p, shape=(400, 80), corner=(0, 0))

#     img = s.model(spectra)
#     fig, ax = plt.subplots()
#     ax.imshow(img[0], origin="lower")
#     ax.set(title="IR Trace Test", xlabel="Pixels", ylabel="Pixels")
#     fig.savefig(TESTDIR + "output/test_nir_trace.png", dpi=150, bbox_inches="tight")

#     img = s.model(spectra)
#     fig, ax = plt.subplots()
#     img0 = s.dX0.dot(spectra.ravel()).reshape(s.shape)
#     img1 = s.dX1.dot(spectra.ravel()).reshape(s.shape)
#     img = img0 + img1
#     ax.imshow(img[0], origin="lower")
#     ax.set(title="IR Trace Test", xlabel="Pixels", ylabel="Pixels")
#     fig.savefig(
#         TESTDIR + "output/test_nir_trace_grad.png", dpi=150, bbox_inches="tight"
#     )

#     with pytest.raises(ValueError):
#         img = s.model(spectra[:, 0][:, None])

# test trace (without sensitivity) dot ones results in roughly ones


def test_sparsewarp():
    R, C = np.meshgrid(np.arange(20, 25), np.arange(20, 25), indexing="ij")
    data = np.ones((5, 5))
    sw = SparseWarp((data.ravel(), (R.ravel(), C.ravel())), shape=(50, 50))
    assert isinstance(sw, sparse.coo_matrix)
    assert sparse.issparse(sw)
    assert hasattr(sw, "translate")
    assert sw.shape == (50, 50)
    assert sw.data.sum() != 0
    sw.translate((1, 1))
    assert sw.data.sum() != 0
    sw.translate((100, 1))
    assert sw.data.sum() == 0
