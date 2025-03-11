# Standard library
import functools

# Third-party
import astropy.units as u
import numpy.typing as npt

PARAMETERS_DOCSTRINGS = {
    "name": (str, "Name of detector"),
    "psf_flux": (npt.NDArray, "ND array of flux values"),
    "wavelength": (
        npt.NDArray,
        "Array of wavelength values. If no wavelength unit is present, assumed to be in microns.",
    ),
    "spectrum": (
        npt.NDArray,
        "Array of spectrum intensity values. Should have units of erg / (Angstrom s cm2).",
    ),
    "gradients": (
        bool,
        "Whether to return gradients. If True, will return an additional 2 arrays that contain the gradients in each axis.",
    ),
    "row": (float, "Row position on the detector"),
    "column": (float, "Column position on the detector"),
    "X": (
        npt.NDArray,
        (
            "Array of 1D vectors defining the value of each dimension for every element of `psf_flux`. "
            "Should have as many entries as `psf_flux` has dimensions."
        ),
    ),
    "transpose": (
        bool,
        "Whether to transpose the input data in the column/row axis.",
    ),
    "scale": (
        float,
        "How much to scale the PSF grid. Scale of 2 makes the PSF 2x broader. Default is 1.",
    ),
    "dimension_names": (
        list,
        "List of names for each of the N dimensions in `psf_flux`",
    ),
    "dimension_units": (
        list,
        "List of `astropy.unit.Quantity`'s describing units of each dimension",
    ),
    "pixel_size": (
        u.Quantity,
        "True detector pixel size in dimensions of length/pixel",
    ),
    "sub_pixel_size": (
        u.Quantity,
        "PSF file pixel size in dimensions of length/pixel",
    ),
    "freeze_dictionary": (dict, "Dictionary of dimensions to freeze"),
    "blur_value": (
        tuple,
        "Tuple of astropy quantities in pixels describing the amount of blur in each axis.",
    ),
    "extrapolate": (
        bool,
        "Whether to allow the PSF to be evaluated outside of the bounds (i.e. will extrapolate)",
    ),
    "check_bounds": (
        bool,
        (
            "Whether to check if the inputs are within the bounds of the PSF model and have the right units. "
            "This check causes a small slowdown."
        ),
    ),
    "bin": (
        int,
        (
            "Optional amount to bin the input PSF file by. "
            "Binning the PSF file will result in faster computation, but less accurate modeling. "
            "Default is 1.",
        ),
    ),
    "teff": (float, ("Effective temperature of a star in K.")),
    "logg": (float, ("Surface gravity of a star. log(g)")),
    "locations": (
        npt.NDArray,
        ("Set of locations in row and column for every source. "),
    ),
    "psf": (str, "A PSF object to use within the scene."),
}


# Decorator to add common parameters to docstring
def add_docstring(*param_names):
    def decorator(func):
        param_docstring = ""
        if func.__doc__:
            # Determine the current indentation level
            lines = func.__doc__.splitlines()
            if len(lines[0]) == 0:
                indent = len(lines[1]) - len(lines[1].lstrip())
            else:
                indent = len(lines[0]) - len(lines[0].lstrip())
        else:
            indent = 0
        indent_str = " " * indent
        for name in param_names:
            if name in PARAMETERS_DOCSTRINGS:
                dtype, desc = PARAMETERS_DOCSTRINGS[name]
                if isinstance(dtype, tuple):
                    dtype_str = " or ".join(
                        [t._name if hasattr(t, "_name") else t.__name__][0]
                        for t in dtype
                        if t is not None
                    )
                    dtype_str += " or None" if None in dtype else ""
                else:
                    dtype_str = dtype.__name__
                param_docstring += f"{indent_str}{name}: {dtype_str}\n{indent_str}    {desc}\n"
        existing_docstring = func.__doc__ or ""
        if "Parameters" in existing_docstring:
            func.__doc__ = (
                existing_docstring.split("---\n")[0]
                + "---\n"
                + param_docstring
                + "---\n".join(existing_docstring.split("---\n")[1:])
            )
        else:
            func.__doc__ = (
                existing_docstring
                + f"\n\n{indent_str}Parameters\n{indent_str}----------\n"
                + param_docstring
            )
        return func

    return decorator


# Decorator to inherit docstring from base class
def inherit_docstring(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    if func.__doc__ is None:
        for base in func.__qualname__.split(".")[0].__bases__:
            base_func = getattr(base, func.__name__, None)
            if base_func and base_func.__doc__:
                func.__doc__ = base_func.__doc__
                break
    return wrapper
