"""
Pytest configuration for AURA test suite.

Engineering Review Findings & Native Teardown Fix:
- Root cause of exit code 139 (SIGSEGV): Pytest invoked via system Python
  (/Library/Frameworks/Python.framework/...) imported 'torchvision' from the system
  site-packages. That torchvision extension was compiled against an older Torch ABI (2.12.0),
  conflicting with the project venv's PyTorch (2.14.1). During process exit, libc's
  __cxa_finalize_ranges invoked at::native::mps::MetalShaderLibrary::~MetalShaderLibrary()
  via torchvision's vision::ops::mps::lib, crashing on a corrupted pointer (0x6e6f69).
- Safe Alternative Solution: Blocking unisolated external 'torchvision' prevents the ABI
  mismatch completely. Python and PyTorch clean up naturally with exit code 0.
- Optional fallback: AURA_FORCE_EXIT_WORKAROUND=1 allows re-enabling os._exit() if ever needed.
"""

import logging
import os
import platform
import sys
import pytest

# Prevent external/system torchvision from polluting venv's torch runtime
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None

_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = exitstatus


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """
    Optional guarded fallback. Only invoked if explicitly requested via
    AURA_FORCE_EXIT_WORKAROUND=1 on macOS Apple Silicon when torch is loaded.
    By default, clean process termination is preserved.
    """
    if os.environ.get("AURA_FORCE_EXIT_WORKAROUND") == "1":
        if sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64") and "torch" in sys.modules:
            try:
                logging.shutdown()
            except Exception:
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(_exit_status)
