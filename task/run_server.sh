#!/bin/sh
# Always use the project venv so Flask/dependencies resolve (avoids ModuleNotFoundError).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" app.py
