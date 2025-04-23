"""MRI_utils"""

from .kspace import (
    forward,
    backward,
    zeropad,
    extract_data_acs,
)

from .bids import (
    get_function_session,
    get_attr_files,
)

from .plot import (
    save_quadrants_ipe,
)

__all__ = [
    "forward",
    "backward",
    "zeropad",
    "extract_data_acs",
    "get_function_session",
    "get_attr_files",
    "save_quadrants_ipe",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
