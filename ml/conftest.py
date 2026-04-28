"""Make the ml/ package importable when running pytest from this directory.

Without this, `from pipelines.grading.centering import ...` would fail because
pytest's rootdir-based sys.path setup doesn't include the package root."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
