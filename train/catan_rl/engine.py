"""The compiled engine module, `ce`.

Imports catan_engine from the Python path, or else from build/python/bind (where
`cmake --build --preset python` puts it), so scripts run without setting PYTHONPATH.
"""

import sys
from pathlib import Path

try:
    import catan_engine as ce
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build" / "python" / "bind"))
    try:
        import catan_engine as ce
    except ImportError as e:
        raise ImportError(
            "catan_engine is not built: run `cmake --preset python && "
            "cmake --build --preset python` from the repository root") from e

__all__ = ["ce"]
