#!/usr/bin/env python3
"""One-off: build shared Content OMA + RAG for all premade courses locally.

Run from coast-local-oma before starting the server so every student gets
instant access without waiting for OMA ingestion:

    cd coast-local-oma
    python3 scripts/build_curated_oma.py

Re-run with --force to rebuild after changing lecture PDFs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env when run from CLI
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        import os
        os.environ.setdefault(key.strip(), val.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build shared Content OMA for premade courses")
    parser.add_argument("--force", action="store_true", help="Rebuild even if already indexed")
    parser.add_argument("--folder", help="Only build this premade folder name")
    args = parser.parse_args()

    from curated_config import (
        CURATED_FOLDER_NAMES,
        bootstrap_all_curated_content,
        bootstrap_curated_content,
        is_curated_content_ready,
    )

    if args.folder:
        if args.folder not in CURATED_FOLDER_NAMES:
            print(f"Unknown premade folder: {args.folder}")
            print(f"Available: {', '.join(sorted(CURATED_FOLDER_NAMES))}")
            return 1
        result = bootstrap_curated_content(args.folder, force=args.force)
        print(result)
        return 0 if result.get("content_ready") else 1

    results = bootstrap_all_curated_content(force=args.force)
    ok = all(r.get("content_ready") for r in results.values())
    for folder, result in results.items():
        ready = is_curated_content_ready(folder)
        print(f"  {folder}: {'ready' if ready else 'NOT READY'} — {result}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
