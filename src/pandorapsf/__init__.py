# Standard library
import configparser  # noqa: E402
import os  # noqa
from importlib.metadata import PackageNotFoundError, version  # noqa

from appdirs import user_config_dir, user_data_dir  # noqa: E402
from astropy.io import fits


def get_version():
    try:
        return version("pandorapsf")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"
PANDORASTYLE = "{}/data/pandora.mplstyle".format(PACKAGEDIR)


CONFIGDIR = user_config_dir("pandorapsf")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")


def reset_config():
    config = configparser.ConfigParser()
    config["SETTINGS"] = {
        "storage_dir": user_data_dir("pandorapsf"),
        "log_level": "INFO",
    }

    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


def load_config() -> configparser.ConfigParser:
    """
    Loads the configuration file, creating it with defaults if it doesn't exist.

    Returns
    -------
    configparser.ConfigParser
        The loaded configuration.
    """

    config = configparser.ConfigParser()

    if not os.path.exists(CONFIGPATH):
        # Create default configuration
        reset_config()
    config.read(CONFIGPATH)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    """
    Saves the configuration to the file.

    Parameters
    ----------
    config : configparser.ConfigParser
        The configuration to save.
    app_name : str
        Name of the application.
    """
    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


config = load_config()

STORAGEDIR = config["SETTINGS"]["storage_dir"]
os.makedirs(STORAGEDIR, exist_ok=True)

from pandorasat import get_logger  # noqa: E402

logger = get_logger("pandorapsf")
logger.setLevel(config["SETTINGS"]["log_level"])


# Standard library
import shutil  # noqa: E402

if not os.path.isfile(f"{STORAGEDIR}/pandora_vis_psf.fits"):
    from astropy.utils.data import download_file  # noqa: E402

    # Download vis PSF
    logger.warning("No PSF file found. Downloading 200MB VIS PSF file.")
    p = download_file(
        "https://zenodo.org/records/11228523/files/pandora_vis_2024-05.fits?download=1",
        pkgname="pandora-psf",
    )
    shutil.move(p, f"{STORAGEDIR}/pandora_vis_psf.fits")
    logger.warning(f"VIS PSF downloaded to {STORAGEDIR}/pandora_vis_psf.fits.")

if not os.path.isfile(f"{STORAGEDIR}/pandora_nir_psf.fits"):
    from astropy.utils.data import download_file  # noqa: E402

    # Download nir PSF
    logger.warning("No PSF file found. Downloading 200MB NIR PSF")
    p = download_file(
        "https://zenodo.org/records/11153153/files/pandora_nir_2024-05.fits?download=1",
        pkgname="pandora-psf",
    )
    shutil.move(p, f"{STORAGEDIR}/pandora_nir_psf.fits")
    logger.warning(f"NIR PSF downloaded to {STORAGEDIR}/pandora_nir_psf.fits.")

hdulist = fits.open(STORAGEDIR + "/pandora_vis_psf.fits")
hdulist.verify("exception")
if not (hdulist[0].header["DATE"] == "2024-05-14T11:38:14.755119"):
    raise ValueError("Out of date visible PRF file.")

hdulist = fits.open(STORAGEDIR + "/pandora_nir_psf.fits")
hdulist.verify("exception")
if not (hdulist[0].header["DATE"] == "2024-05-08T15:02:58.461202"):
    raise ValueError("Out of date NIR PRF file.")


from .psf import PSF  # noqa: F401, E402
from .scene import ROIScene, Scene, TraceScene  # noqa: F401, E402
