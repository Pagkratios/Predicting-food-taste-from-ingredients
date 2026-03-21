"""Compatibility shim for the shared repo-level prediction store."""

from __future__ import annotations

import importlib.util
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SHARED_FILE = os.path.join(REPO_ROOT, "data", "data_predictions.py")

_SPEC = importlib.util.spec_from_file_location("shared_data_predictions", SHARED_FILE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

pred_data = _MODULE.pred_data
