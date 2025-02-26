# Standard library
import configparser  # noqa: E402
import os  # noqa
from importlib.metadata import PackageNotFoundError, version  # noqa

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
PANDORASTYLE = "{}/data/pandora.mplstyle".format(PACKAGEDIR)


CONFIGDIR = user_config_dir("pandorapsf")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")


def reset_config():
    config = configparser.ConfigParser()
    config["SETTINGS"] = {
        "storage_dir": user_data_dir("pandorapsf"),
        "log_level": "INFO",
        "vis_psf_download_location": "https://zenodo.org/records/11228523/files/pandora_vis_2024-05.fits?download=1",
        "nir_psf_download_location": "https://zenodo.org/records/11153153/files/pandora_nir_2024-05.fits?download=1",
        "vis_psf_creation_date": "2024-05-14T11:38:14.755119",
        "nir_psf_creation_date": "2024-05-08T15:02:58.461202",
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


def display_config() -> pd.DataFrame:
    dfs = []
    for section in config.sections():
        df = pd.DataFrame(
            np.asarray([(key, value) for key, value in dict(config[section]).items()])
        )
        df["section"] = section
        df.columns = ["key", "value", "section"]
        df = df.set_index(["section", "key"])
        dfs.append(df)
    return pd.concat(dfs)


STORAGEDIR = config["SETTINGS"]["storage_dir"]
os.makedirs(STORAGEDIR, exist_ok=True)

from pandorasat import get_logger  # noqa: E402

logger = get_logger("pandorapsf")
logger.setLevel(config["SETTINGS"]["log_level"])

from .psf import PSF  # noqa: F401, E402
from .scene import ROIScene, Scene, TraceScene  # noqa: F401, E402
