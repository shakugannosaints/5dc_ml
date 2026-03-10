# alphazero/ - Semimove-level AlphaZero for 5D Chess (Very Small - Open variant)

import sys
import os

# Ensure the compiled engine module (engine.pyd) is importable.
# Prefer build_py_ml because it is the actively maintained Python 3.12 build.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _candidate in reversed(("build_py_ml", "build_py", "build")):
    _p = os.path.join(_root, _candidate)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Verify engine can be imported
try:
    import engine as _engine  # noqa: F401
except ImportError as e:
    raise ImportError(
        f"Cannot import compiled 'engine' module. "
        f"Build it first with: cmake -DPYMODULE=on .. && cmake --build .\n"
        f"Searched in: {[os.path.join(_root, d) for d in ('build_py', 'build_py_ml', 'build')]}"
    ) from e
