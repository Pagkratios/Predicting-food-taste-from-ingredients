"""Compatibility shim for the shared repo-level HS export."""

from __future__ import annotations

import importlib.util
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SHARED_FILE = os.path.join(REPO_ROOT, "data", "hs_predictions.py")

_SPEC = importlib.util.spec_from_file_location("shared_hs_predictions", SHARED_FILE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

hs_predictions = _MODULE.hs_predictions
