"""Retrieve files from a bids-structured directory module."""

from ._bids import get_function_session, get_attr_files


__all__ = [
    "get_function_session",
    "get_attr_files",
]
