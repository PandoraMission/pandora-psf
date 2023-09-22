"""Functions to plot up PSFs"""

import matplotlib.pyplot as plt
import numpy as np


def plot_spatial(
    psf,
    n=3,
    image_type="PSF",
):
    """Plot a PSF that has some spatial dimesions"""
    if "row" not in psf.dimension_names:
        raise ValueError("Not a spatial PSF")
    if "column" not in psf.dimension_names:
        raise ValueError("Not a spatial PSF")

    if not (n % 2) == 1:
        n += 1

    kwargs = {}
    for dnm in psf.dimension_names:
        if (dnm == "row") | (dnm == "column"):
            continue
        kwargs[dnm] = getattr(psf, dnm + "0d")

    fig, ax = plt.subplots(n, n, figsize=(n * 2, n * 2))
    for x1, y1 in (
        np.asarray(((np.mgrid[:n, :n] - n // 2) * (600 / (n // 2))))
        .reshape((2, n**2))
        .T
    ):
        jdx = int(x1 // (600 / (n // 2)) + n // 2)
        idx = int(-y1 // (600 / (n // 2)) + n // 2)
        kwargs["row"] = x1
        kwargs["column"] = y1
        if image_type.lower() == "psf":
            y, x, f = (
                psf.psf_row.value,
                psf.psf_column.value,
                psf.psf(**kwargs),
            )
            ax[idx, jdx].set(xticklabels=[], yticklabels=[])
        elif image_type.lower() == "prf":
            y, x, f = psf.prf(**kwargs)
            ax[idx, jdx].set(xticklabels=[], yticklabels=[])
        else:
            raise ValueError("No such image type. Choose from `'PSF'`.")
        ax[idx, jdx].pcolormesh(
            x,
            y,
            f,
            vmin=0,
            vmax=[0.05 if image_type.lower() == "prf" else 0.001][0],
        )
    ax[n // 2, 0].set(ylabel="Y Pixel")
    ax[n - 1, n // 2].set(xlabel="X Pixel")
    ax[0, n // 2].set(title=image_type.upper())
    return fig


def plot_spectral(
    psf,
    var="wavelength",
    n=5,
    npixels=20,
    image_type="psf",
):
    wavs = np.linspace(
        getattr(psf, var + "1d").min(), getattr(psf, var + "1d").max(), n
    )
    m = npixels // 2
    fig, ax = plt.subplots(
        1,
        n,
        figsize=(n * 3, 3),
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    kwargs = {}
    for dnm in psf.dimension_names:
        if dnm == var:
            continue
        kwargs[dnm] = getattr(psf, dnm + "0d")

    for ndx in np.arange(n):
        kwargs[var] = wavs[ndx]
        if image_type.lower() == "psf":
            y, x, f = (
                psf.psf_row.value,
                psf.psf_column.value,
                psf.psf(**kwargs),
            )
        elif image_type.lower() == "prf":
            y, x, f = psf.prf(
                row=kwargs.pop("row", 0), column=kwargs.pop("column", 0), **kwargs
            )
        im = ax[ndx].pcolormesh(
            x,
            y,
            f,
            vmin=0,
            vmax=[0.1 if image_type.lower() == "prf" else 0.01][0],
        )
        ax[ndx].set(
            xlim=(-m, m),
            ylim=(-m, m),
            xticks=np.arange(-(m - 1), m, 2),
            yticks=np.arange(-(m - 1), m, 2),
            title=f"{wavs[ndx]:0.2} $\mu$m",
            xlabel="Pixels",
        )
    #            ax[ndx].grid(True, ls="-", color="white", lw=0.5, alpha=0.5)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax[0].set(ylabel="Pixels")
    # for idx in range(n):
    #     ax[idx, 0].set(ylabel="Pixels")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Flux Value")
    # fig.suptitle(
    #     f"{image_type.upper()} Across {var.capitalize()}", fontsize=15
    # )
    return fig
