# Standard library
import configparser  # noqa: E402
import os  # noqa
from importlib.metadata import PackageNotFoundError, version  # noqa

# Third-party
import numpy as np
import pandas as pd
from appdirs import user_config_dir, user_data_dir  # noqa: E402


def get_version():
    try:
        return version("pandorapsf")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"
DOCSDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/docs/"
PANDORASTYLE = "{}/data/pandora.mplstyle".format(PACKAGEDIR)

# Third-party
from pandorasat import get_logger  # noqa: E402

logger = get_logger("pandorapsf")

CONFIGDIR = user_config_dir("pandorapsf")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")


def reset_config():
    config = configparser.ConfigParser()
    config["SETTINGS"] = {
        "data_dir": user_data_dir("pandorapsf"),
        "log_level": "WARNING",
        "vis_psf_download_location": "https://zenodo.org/records/15101982/files/pandora_vis_psf.fits?download=1",
        "nir_psf_download_location": "https://zenodo.org/records/15101982/files/pandora_nir_psf.fits?download=1",
        "vis_psf_creation_date": "2025-03-28T10:30:27.329441",
        "nir_psf_creation_date": "2025-03-28T10:06:08.770326",
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

for key in ["data_dir", "log_level"]:
    if key not in config["SETTINGS"]:
        logger.error(
            f"`{key}` missing from the `gaiaoffline` config file. Your configuration is being reset."
        )
        reset_config()
        config = load_config()


def display_config() -> pd.DataFrame:
    dfs = []
    for section in config.sections():
        df = pd.DataFrame(
            np.asarray(
                [(key, value) for key, value in dict(config[section]).items()]
            )
        )
        df["section"] = section
        df.columns = ["key", "value", "section"]
        df = df.set_index(["section", "key"])
        dfs.append(df)
    return pd.concat(dfs)


DATADIR = config["SETTINGS"]["data_dir"]
os.makedirs(DATADIR, exist_ok=True)
logger.setLevel(config["SETTINGS"]["log_level"])

from .psf import PSF  # noqa: F401, E402
from .scene import ROIScene, Scene, TraceScene  # noqa: F401, E402
