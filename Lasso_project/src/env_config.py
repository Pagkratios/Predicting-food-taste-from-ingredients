#!/usr/bin/env python3

import os
from functools import lru_cache


SRC_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_RAW_RECIPES_PATH = os.path.join(PROJECT_ROOT, "data", "raw_recipes.py")


def _strip_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


@lru_cache(maxsize=1)
def _load_dotenv():
    values = {}
    if not os.path.isfile(ENV_PATH):
        return values

    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_quotes(value.strip())
            if key:
                values[key] = value
    return values


def _get_env_value(name):
    value = os.environ.get(name)
    if value:
        return value.strip()
    return _load_dotenv().get(name)


def _resolve_repo_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(REPO_ROOT, path_value)


def get_raw_recipes_path():
    configured = _get_env_value("RAW_RECIPES_PATH")
    if configured:
        return os.path.abspath(_resolve_repo_path(configured))
    return os.path.abspath(DEFAULT_RAW_RECIPES_PATH)
