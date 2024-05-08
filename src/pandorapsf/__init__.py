__version__ = "0.2.3"
# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"
PANDORASTYLE = "{}/data/pandora.mplstyle".format(PACKAGEDIR)

# Standard library
import logging  # noqa: E402
import shutil  # noqa: E402

# Third-party
from rich.logging import RichHandler  # noqa: E402

logging.basicConfig()
log = logging.getLogger("pandora-psf")
log.addHandler(RichHandler(markup=True))
log.setLevel("INFO")

if not os.path.isfile(f"{PACKAGEDIR}/data/pandora_vis_20220506.fits"):
    from astropy.utils.data import download_file  # noqa: E402

    # Download vis PSF
    log.warning("No PSF file found. Downloading 2GB VIS PSF file.")
    p = download_file(
        "https://zenodo.org/records/11153153/files/pandora_vis_2024-05.fits?download=1",
        pkgname="pandora-psf",
    )
    shutil.move(p, f"{PACKAGEDIR}/data/pandora_vis_20220506.fits")
    log.warning(f"VIS PSF downloaded to {PACKAGEDIR}/data/pandora_vis_20220506.fits.")

if not os.path.isfile(f"{PACKAGEDIR}/data/pandora_nir_20220506.fits"):
    from astropy.utils.data import download_file  # noqa: E402

    # Download nir PSF
    log.warning("No PSF file found. Downloading 2GB NIR PSF")
    p = download_file(
        "https://zenodo.org/records/11153153/files/pandora_nir_2024-05.fits?download=1",
        pkgname="pandora-psf",
    )
    shutil.move(p, f"{PACKAGEDIR}/data/pandora_nir_20220506.fits")
    log.warning(f"NIR PSF downloaded to {PACKAGEDIR}/data/pandora_nir_20220506.fits.")

from .psf import PSF  # noqa: F401, E402
from .scene import ROIScene, Scene, TraceScene  # noqa: F401, E402
from .sparsewarp import ROISparseWarp3D, SparseWarp3D  # noqa: F401, E402
