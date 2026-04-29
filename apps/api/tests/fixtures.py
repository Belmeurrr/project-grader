"""Shim that re-exports ml/tests/fixtures.py as `tests.fixtures`.

The API test package (`apps/api/tests/`) and the ML test package
(`ml/tests/`) both want to live as `tests.*`. Python imports whichever
one is found first on sys.path, so `from tests.fixtures import ...` from
inside an API test resolves to `apps/api/tests/fixtures.py` (this file)
rather than reaching across into ml/. This shim is the bridge: it loads
ml/tests/fixtures.py via importlib and re-exports its public surface,
so callers keep writing `from tests.fixtures import card_in_scene`.

Why not delete `apps/api/tests/__init__.py` instead? That would let
ml/tests/ win the namespace, but it changes pytest's collection mode for
the whole API suite. A re-export shim is a smaller, safer change."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ML_FIXTURES = Path(__file__).resolve().parents[3] / "ml" / "tests" / "fixtures.py"
_spec = importlib.util.spec_from_file_location("_ml_test_fixtures", _ML_FIXTURES)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_ml_test_fixtures", _mod)
_spec.loader.exec_module(_mod)

# Re-export. Add new fixtures here when ml/ adds them and an apps/api test
# wants them — keeping the surface explicit avoids a `from x import *` that
# would silently pull in private helpers.
synth_card = _mod.synth_card
synth_card_with_pattern = _mod.synth_card_with_pattern
card_in_scene = _mod.card_in_scene
blurry = _mod.blurry
with_glare = _mod.with_glare
canonical_clean = _mod.canonical_clean
canonical_with_edge_defect = _mod.canonical_with_edge_defect
encode_jpeg = _mod.encode_jpeg
synth_halftone_card = _mod.synth_halftone_card
synth_continuous_tone_card = _mod.synth_continuous_tone_card
