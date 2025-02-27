<a href="https://github.com/pandoramission/pandora-psf/actions/workflows/black.yml"><img src="https://github.com/pandoramission/pandora-psf/workflows/black/badge.svg" alt="black status"/></a> <a href="https://github.com/pandoramission/pandora-psf/actions/workflows/flake8.yml"><img src="https://github.com/pandoramission/pandora-psf/workflows/flake8/badge.svg" alt="flake8 status"/></a> [![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://pandoramission.github.io/pandora-psf/)
[![PyPI - Version](https://img.shields.io/pypi/v/pandorapsf)](https://pypi.org/project/pandorapsf/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pandorapsf)](https://pypi.org/project/pandorapsf/)

# pandorapsf

This is a standalone tool for working with the NASA Pandora Mission PSF. This can be used to model the expected distribution of starlight on the detector.

## Installation

To install you can use

```sh
pip install pandorapsf --upgrade
```

You should update your package often, as we frequently put out new versions with updated Current Best Estimates, and some limited new functionality. Check your version number using

```python
import pandorapsf as ppsf
ppsf.__version__
```

## Configuration

`pandorapsf` has a configuration system. This determines where `pandorapsf` will expect your data to be stored.

Users can find where the configuration file is stored using

  ```python
  from pandorapsf import CONFIGDIR
  print(CONFIGDIR)
  ```

You can display your current configuration using

  ```python
  from pandorapsf import display_config
  display_config()
  ```

You can access particular configuration parameters using

You can display your current configuration, for example use the following to find the `data_dir` parameter.

  ```python
  from pandorapsf import config
  config["SETTINGS"]["data_dir"]
  ```

You can update the configuration either by updating the `config.ini` file in your `CONFIRDIR` or you can update them in your environment using `save_config`, e.g.

  ```python
  from pandorapsf import save_config
  config["SETTINGS"]["log_level"] = "INFO"
  save_config(config)
  ```

If you want to reset back to defaults you can use

  ```python
  from pandorapsf import reset_config
  reset_config()
  ```

### Configuration Parameters

Below is a table of all the configuration parameters in `pandorapsf` and their current defaults

|                 (section, key)                          | Description                                                                                                                                              |
|:------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| ('SETTINGS', 'data_dir')                  | Where data will be stored for the package. This includes ~0.5Gb of PSF files which will be downloaded.                                                   |
| ('SETTINGS', 'log_level')                 | Default level for the logger. Change this to make the tool more or less verbose by default.                                                              |
| ('SETTINGS', 'vis_psf_download_location') | Where the visible channel PSF file is located online for download.                                                                                       |
| ('SETTINGS', 'nir_psf_download_location') | Where the NIR channel PSF file is located online for download.                                                                                           |
| ('SETTINGS', 'vis_psf_creation_date')     | This is the string provided in the "CREATION" keyword of the PSF fits file header. This will be used to verify that the PSF file is the correct version. |
| ('SETTINGS', 'nir_psf_creation_date')     | This is the string provided in the "CREATION" keyword of the PSF fits file header. This will be used to verify that the PSF file is the correct version. |

### Obtaining PSF files

When you install this tool you will additionally need to download the PSF grid online. This grid will be updated periodically as we improve our estimates of Pandora's performance. `pandorapsf` will download the files for you and put them in your configured data directory.

As we release new versions of `pandorapsf`, the default configuration will change to point to new PSF files that contain our new best estimates. Keep your `pandorapsf` version up to date to ensure you have the most recent changes.

## Dependencies

This tool depends on [`sparse3d`](https://github.com/christinahedges/sparse3d) which implements the sparse matrix maths needed to quickly model scenes.
