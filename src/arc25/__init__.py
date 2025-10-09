import importlib.metadata

try:
    from ._version import __version__
except ImportError:
    try:
        __version__ = importlib.metadata.version("yde-arc25")
    except Exception:
        __version__ = "?"
