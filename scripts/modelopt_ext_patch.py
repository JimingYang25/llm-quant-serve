"""Make ModelOpt's CUDA extensions buildable on this machine.

IMPORT THIS BEFORE MODELOPT. Two independent defects, both diagnosed empirically:

  A. torch's compiler-resolution check crashes with a logging TypeError when the
     C++ compiler is named anything other than `g++` (it formats a warning with the
     wrong number of arguments). With CXX unset, torch falls back to the name `c++`
     and the build dies before compiling a single file. Fix: point CXX at a real
     g++.

  B. The failing translation units are `.cu` files, so adding `-std=c++20` to
     `extra_cflags` does NOT reach them — nvcc forwards the standard through
     `extra_cuda_cflags`. Without C++20, torch 2.9.1's own headers
     (ATen/core/List_inl.h:201, a `decltype(...)::difference_type` in a
     static_cast) fail with "need 'typename' ... dependent scope".
     Fix: add `-std=c++20` to BOTH flag lists.

ModelOpt calls torch's loader internally and passes no flags of its own, so the
only place to intervene is to wrap the loader before ModelOpt imports it.

Verified outcome (2026-09-14): modelopt_cuda_ext, _fp8 and _mx all build and load;
before the patch, none of them did, and every quantization number was produced by
the silent CPU fallback.
"""
from __future__ import annotations

import os

os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("MAX_JOBS", "4")

BOTH_FLAG_LISTS = (("extra_cflags", "-std=c++20"), ("extra_cuda_cflags", "-std=c++20"))


def _patched_load(name, sources, *args, **kwargs):
    for key, flag in BOTH_FLAG_LISTS:
        existing = list(kwargs.get(key) or [])
        if flag not in existing:
            existing.append(flag)
        kwargs[key] = existing
    return _ORIGINAL_LOAD(name, sources, *args, **kwargs)


import torch.utils.cpp_extension as _torch_ext  # noqa: E402

_ORIGINAL_LOAD = _torch_ext.load
_torch_ext.load = _patched_load

try:  # modelopt may have already imported `load` into its own namespace
    import modelopt.torch.utils.cpp_extension as _mo_ext  # noqa: E402

    _mo_ext.load = _patched_load
except Exception:  # modelopt not imported yet: patching torch is enough
    pass


def extensions_status() -> dict:
    """Report which ModelOpt CUDA extensions are usable (builds them on first call)."""
    status = {}
    try:
        from modelopt.torch.quantization import extensions as moext
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    for label, fn in (("int8", moext.get_cuda_ext),
                      ("fp8", moext.get_cuda_ext_fp8),
                      ("mx", moext.get_cuda_ext_mx)):
        try:
            status[label] = fn(raise_if_failed=False) is not None
        except Exception:
            status[label] = False
    return status


if __name__ == "__main__":
    print(extensions_status())
