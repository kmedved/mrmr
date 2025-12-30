"""Centralized optional dependency handling."""

from importlib import import_module
from importlib.util import find_spec

HAS_NUMBA = HAS_CATBOOST = HAS_CATEGORY_ENCODERS = HAS_POLARS = False
njit = None
catboost = None
ce = None
polars = None

# numba
if find_spec("numba") is not None:
    try:
        njit = import_module("numba").njit
        HAS_NUMBA = True
    except Exception:
        pass

if not HAS_NUMBA:
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper

# catboost
if find_spec("catboost") is not None:
    try:
        catboost = import_module("catboost")
        HAS_CATBOOST = True
    except Exception:
        pass

# category_encoders
if find_spec("category_encoders") is not None:
    try:
        ce = import_module("category_encoders")
        HAS_CATEGORY_ENCODERS = True
    except Exception:
        pass

# polars
if find_spec("polars") is not None:
    try:
        polars = import_module("polars")
        HAS_POLARS = True
    except Exception:
        pass
