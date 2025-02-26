# Third-party
import numpy as np

# import pytest
from scipy import sparse
from sparse3d import ROISparse3D, Sparse3D


def test_sparse():
    R, C = np.meshgrid(
        np.arange(20, 25).astype(int),
        np.arange(10, 16).astype(int),
        indexing="ij",
    )
    R = R[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    C = C[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    data = np.ones_like(R).astype(float)

    sw = Sparse3D(data, R, C, (50, 50))
    assert sw.imshape == (50, 50)
    assert sw.shape == sw.cooshape == (2500, 10)
    assert sw.subshape == R.shape
    assert isinstance(sw, sparse.coo_matrix)
    assert len(sw.data) == 300
    assert sw.data.sum() == 300
    assert sw.dtype == float

    # Move data out of frame
    sw = Sparse3D(data, R + 50, C, (50, 50))
    assert len(sw.data) == 0
    # translate back into frame
    sw.translate((-50, 0))
    assert len(sw.data) == 300
    # reset it
    sw.reset()
    assert len(sw.data) == 0

    sw = Sparse3D(data, R + np.arange(10), C + np.arange(10), (50, 50))
    sw.translate((-1, 1))
    assert len(sw.data) == 300

    assert sw.dot(np.ones(10)).shape == (50, 50)
    assert isinstance(sw.dot(np.ones(10)), np.ndarray)
    assert sw.dot(np.ones(10)).sum() == 300


def test_roiSparse():
    R, C = np.mgrid[:20, :20]
    R, C = (
        R + np.arange(2, 48, 5)[:, None, None],
        C + np.arange(2, 48, 5)[:, None, None],
    )
    data = np.random.normal(0, 1, size=R.shape) ** 0

    sw = ROISparse3D(
        data,
        R,
        C,
        imshape=(50, 50),
        nROIs=3,
        ROI_size=(10, 10),
        ROI_corners=[(0, 0), (10, 40), (40, 41)],
    )
    assert sw.imshape == (50, 50)
    assert sw.ROI_size == (10, 10)
    assert sw.shape == sw.cooshape == (2500, 20)
    assert sw.subshape == R.shape
    assert isinstance(sw, sparse.coo_matrix)
    #    assert len(sw.data) == 370
    #    assert sw.data.sum() == 370
    assert sw.dtype == float

    assert sw.dot(np.ones(20)).shape == (3, 1, 10, 10)
    assert isinstance(sw.dot(np.ones(20)), np.ndarray)
    assert np.prod(sw.tocsr().dot(np.ones(20)).shape) == np.prod(sw.imshape)

    # translate the data away everything should be zero:
    sw.translate((-150, -150))
    assert sw.dot(np.ones(20)).shape == (3, 1, 10, 10)
    assert isinstance(sw.dot(np.ones(20)), np.ndarray)
    assert sw.dot(np.ones(20)).sum() == 0
    assert sw.tocsr().dot(np.ones(20)).sum() == 0

    sw.reset()
    assert sw.dot(np.ones(20)).shape == (3, 1, 10, 10)
    assert isinstance(sw.dot(np.ones(20)), np.ndarray)
    assert sw.dot(np.ones(20)).sum() != 0
    assert sw.tocsr().dot(np.ones(20)).sum() != 0
