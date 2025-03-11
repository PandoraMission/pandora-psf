# PSF Class

This class implements tools needed to work with the PSF. Below is an example of the Pandora PSF in pixel space. This class enables users to work with the PSF files and

- Evaluate it at any grid point (e.g. position, temperature, wavelength)
- Evaluate the PRF (i.e. the PSF integrated on a pixel grid)
- Find the gradient of the PSF or PRF

![Visible PSF](images/test_vis_psf.png)

::: pandorapsf.psf.PSF
    selection:
      docstring_style: numpy
      members:
        - psf
