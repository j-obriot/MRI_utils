import importlib

__all__ = [
    "bids",
    "dataset",
    "fmri",
    "kspace",
    "nifti",
    "plot",
]

# lazy loader
def __getattr__(name):
    try:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    except:
        raise AttributeError(f"module {__name__} has no attribute {name}")
