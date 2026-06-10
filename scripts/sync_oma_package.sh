#!/usr/bin/env bash
# Sync the vendored coast_content_oma package from the development repo.
# Run this after editing anything in ~/Desktop/coast-content-oma so the
# deployed copy (this repo) picks up the changes.
set -euo pipefail

SRC="$(dirname "$0")/../../coast-content-oma/src/coast_content_oma"
DST="$(dirname "$0")/../coast_content_oma"

if [ ! -d "$SRC" ]; then
  echo "Source package not found: $SRC" >&2
  exit 1
fi

rm -rf "$DST"
cp -R "$SRC" "$DST"
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Synced coast_content_oma → $(cd "$DST" && pwd)"
