"""Functions to plot up PSFs"""

# Third-party
import matplotlib.pyplot as plt
import numpy as np


def _get_v(kwargs, image_type):
    if image_type.lower() == "psf":
        vmin, vmax = kwargs.pop("vmin", 0), kwargs.pop("vmax", 0.001)
    elif image_type.lower() == "prf":
        vmin, vmax = kwargs.pop("vmin", 0), kwargs.pop("vmax", 0.05)
    elif image_type.lower() == "dprf":
        vmin, vmax = kwargs.pop("vmin", -0.05), kwargs.pop("vmax", 0.05)
    elif image_type.lower() == "dpsf":
        vmin, vmax = kwargs.pop("vmin", -0.001), kwargs.pop("vmax", 0.001)
    return vmin, vmax


def plot_spatial(psf, n=3, image_type="PSF", **kwargs):
    """Plot a PSF that has some spatial dimesions"""
    if "row" not in psf.dimension_names:
        raise ValueError("Not a spatial PSF")
    if "column" not in psf.dimension_names:
        raise ValueError("Not a spatial PSF")

    if not (n % 2) == 1:
        n += 1

    locdict = {}
    for dnm in psf.dimension_names:
        if (dnm == "row") | (dnm == "column"):
            continue
        locdict[dnm] = getattr(psf, dnm + "0d")
    vmin, vmax = _get_v(kwargs, image_type)
    cmap = kwargs.pop("cmap", "viridis")
    fig, ax = plt.subplots(n, n, figsize=(n * 2, n * 2))
    for x1, y1 in (
        np.asarray(((np.mgrid[:n, :n] - n // 2) * (600 / (n // 2))))
        .reshape((2, n**2))
        .T
    ):
        jdx = int(x1 // (600 / (n // 2)) + n // 2)
        idx = int(-y1 // (600 / (n // 2)) + n // 2)
        locdict["row"] = x1
        locdict["column"] = y1
        if image_type.lower() == "psf":
            y, x, f = (
                psf.psf_row.value,
                psf.psf_column.value,
                psf.psf(**locdict),
            )
            ax[idx, jdx].set(xticklabels=[], yticklabels=[])
        elif image_type.lower() == "dpsf":
            y, x, df = (
                psf.psf_row.value,
                psf.psf_column.value,
                psf.dpsf(**locdict),
            )
            f = df[0]
            ax[idx, jdx].set(xticklabels=[], yticklabels=[])
        elif image_type.lower() == "prf":
            y, x, f = psf.prf(**locdict)
            ax[idx, jdx].set(xticklabels=[], yticklabels=[])
        elif image_type.lower() == "dprf":
            y, x, df = psf.dprf(**locdict)
            f = df[0]
            ax[idx, jdx].set(xticklabels=[], yticklabels=[])
        else:
            raise ValueError("No such image type. Choose from `'PSF'`.")
        ax[idx, jdx].pcolormesh(
            x,
            y,
            f,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax[idx, jdx].set(xticks=[], yticks=[])
        if idx == n - 1:
            ax[idx, jdx].set(xlabel=f"{x1.astype(int)} Pix")
        if jdx == 0:
            ax[idx, jdx].set(ylabel=f"{y1.astype(int)} Pix")
    # ax[n // 2, 0].set(ylabel="Y Pixel")
    # ax[n - 1, n // 2].set(xlabel="X Pixel")
    ax[0, n // 2].set(title=image_type.upper())
    plt.subplots_adjust(hspace=0.051, wspace=0.051)
    return fig


def plot_spectral(
    psf, var="wavelength", n=5, npixels=20, image_type="psf", **kwargs
):

    vmin, vmax = _get_v(kwargs, image_type)
    cmap = kwargs.pop("cmap", "viridis")

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
                row=kwargs.pop("row", 0),
                column=kwargs.pop("column", 0),
                **kwargs,
            )
        im = ax[ndx].pcolormesh(
            x,
            y,
            f,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
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
