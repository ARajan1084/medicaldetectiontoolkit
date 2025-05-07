"""
Re‑export symbols from the compiled nms extension.

This file is intentionally small:
* It does **not** depend on the legacy `torch.utils.ffi`.
* Every symbol defined in `_nms` that isn’t private (`_foo`) is pushed into the
  package namespace so you can write, e.g.:

    from _ext.nms import nms_cpu, nms_gpu, ...

If you prefer to expose only specific names, replace the `__all__` definition
with an explicit list.
"""

from . import _nms as _C  # compiled C++/CUDA module produced by setup.py

# All public attributes (skip those that start with an underscore)
__all__ = [name for name in dir(_C) if not name.startswith("_")]

# Inject those symbols into the current module’s globals()
globals().update({name: getattr(_C, name) for name in __all__})

# Optional: tidy up
del _C