"""Resolve duplicate-OpenMP conflicts by import order, not by suppression.

The conflict
------------
Several environments used by this project ship more than one copy of the
Intel OpenMP runtime ``libiomp5md.dll``: conda's MKL build (which
NumPy/SciPy link against, under ``Library/bin``) and Torch's bundled copy
(under ``site-packages/torch/lib``). Whichever library initializes OpenMP
second aborts the process::

    OMP: Error #15: Initializing libiomp5md.dll, but found
    libiomp5md.dll already initialized.

Measured on the project's GPU environment (torch 2.5.1+cu121): importing
``torch`` before ``numpy`` aborts at startup, while importing ``numpy``
first is clean and leaves CUDA fully available. On the CPU environment the
same collision surfaced later and more confusingly, as
``numpy.corrcoef`` aborting the interpreter mid-test-session.

Why not ``KMP_DUPLICATE_LIB_OK``
--------------------------------
Setting ``KMP_DUPLICATE_LIB_OK=TRUE`` tells OpenMP to continue with two
runtimes loaded. Intel documents that variable as "unsafe, unsupported,
undocumented" and warns it "may cause crashes or silently produce incorrect
results". Silently incorrect numerical results are not an acceptable risk
for a benchmark whose output becomes paper evidence, and the experiment
protocol requires ``duplicate_openmp_override`` to stay unset so that runs
are auditable.

Several modules previously called
``os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`` at import time.
That had two effects beyond their own execution: it enabled the override
for the whole process, including any measurement taken afterwards, and --
because module import happens during pytest collection -- it leaked into
unrelated tests. Use :func:`ensure_single_openmp_runtime` instead.
"""

from __future__ import annotations

import importlib
import os
from typing import Dict, List

__all__ = [
    "OPENMP_OVERRIDE_ENV",
    "PROCESS_START_OVERRIDE",
    "SKLEARN_OVERRIDE_VALUE",
    "assert_no_external_openmp_override",
    "describe_openmp_override",
    "ensure_single_openmp_runtime",
]

OPENMP_OVERRIDE_ENV = "KMP_DUPLICATE_LIB_OK"

PROCESS_START_OVERRIDE = os.environ.get(OPENMP_OVERRIDE_ENV)
"""Value of the override when this module was first imported.

Captured at import time so that a value inherited from the shell can be told
apart from one a library sets later. Import this module early -- before
``sklearn`` -- for the distinction to be meaningful.
"""

SKLEARN_OVERRIDE_VALUE = "True"
"""The value scikit-learn writes.

scikit-learn's Windows ``__init__`` executes
``os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")`` as a documented
workaround for an intel-openmp 2019.5 issue. Any process importing sklearn --
which includes most of this project's entry points -- therefore has the
override set regardless of the caller's intent. It cannot be prevented, only
recorded.

Note the consequence for existing artifacts: the ``duplicate_openmp_override:
null`` recorded in ``dinov2_runtime_probe.json`` reflects a probe that never
imported sklearn, not a guarantee about runs that do.
"""

_MKL_LINKED_MODULES = ("numpy", "scipy.linalg", "sklearn")


def ensure_single_openmp_runtime() -> List[str]:
    """Import the MKL-linked stack so it claims OpenMP before Torch loads.

    Call this at the top of any entry point that will import Torch, *before*
    that import. Idempotent and cheap after the first call, since Python
    caches modules.

    Returns:
        The module names successfully imported, for logging. A module that is
        absent or broken is skipped rather than raised: the caller's own
        imports will fail with a clearer error if they genuinely need it.
    """
    loaded: List[str] = []
    for name in _MKL_LINKED_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            continue
        loaded.append(name)
    return loaded


def describe_openmp_override() -> Dict[str, object]:
    """Report the override's value and, as far as possible, its origin.

    Distinguishes three cases, because they carry different weight:

    ``unset``
        No override anywhere. The strongest position.
    ``library_default``
        Absent at process start and now equal to
        :data:`SKLEARN_OVERRIDE_VALUE`, i.e. scikit-learn's documented
        workaround. Unavoidable while sklearn is a dependency; must be
        disclosed rather than presented as ``null``.
    ``external``
        Present already at process start, so it came from the shell or a
        wrapper. This is a deliberate suppression by whoever launched the run
        and invalidates the measurement.

    Returns:
        A JSON-compatible record for the run's environment freeze.
    """
    current = os.environ.get(OPENMP_OVERRIDE_ENV)
    if current is None:
        origin = "unset"
    elif PROCESS_START_OVERRIDE is not None:
        origin = "external"
    elif current == SKLEARN_OVERRIDE_VALUE:
        origin = "library_default"
    else:
        origin = "unknown"
    return {
        "duplicate_openmp_override": current,
        "duplicate_openmp_override_origin": origin,
        "duplicate_openmp_override_at_process_start": PROCESS_START_OVERRIDE,
        "note": (
            "scikit-learn sets KMP_DUPLICATE_LIB_OK='True' on Windows as a "
            "documented workaround, so `library_default` is expected in any run "
            "that imports sklearn and is disclosed rather than suppressed. "
            "`external` means the launcher set it and the run must not be "
            "reported."
        ),
    }


def assert_no_external_openmp_override() -> None:
    """Fail if the override was supplied by the launcher rather than a library.

    Intended for measurement entry points. A shell-level
    ``KMP_DUPLICATE_LIB_OK`` is an undisclosed suppression of a real conflict,
    which the experiment protocol forbids. scikit-learn's own default is
    tolerated because it cannot be prevented, but it is recorded in the
    artifacts by :func:`describe_openmp_override`.

    Raises:
        RuntimeError: the override was already set when the process started.
    """
    if PROCESS_START_OVERRIDE is not None:
        raise RuntimeError(
            f"{OPENMP_OVERRIDE_ENV}={PROCESS_START_OVERRIDE!r} was set before this "
            "process started. This override lets two OpenMP runtimes coexist, "
            "which Intel documents as a cause of crashes or silently incorrect "
            "results, so measurements taken under it must not be reported. Unset "
            "it and rely on import ordering (ensure_single_openmp_runtime)."
        )
