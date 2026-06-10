"""Merge near-duplicate Content OMA concepts and remap student evidence.

Run after long courses accumulate fragmented concept nodes
("eigenvalue" vs "eigenvalue decomposition" as separate nodes).

Usage:
  RAG_PROVIDER=oma python3 scripts/dedupe_concepts.py --user-id 13 --folder ML --dry-run
  RAG_PROVIDER=oma python3 scripts/dedupe_concepts.py --user-id 13 --folder ML
  RAG_PROVIDER=oma python3 scripts/dedupe_concepts.py --user-id 13 --folder ML --threshold 0.93
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

import oma_provider  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not oma_provider.is_oma_enabled():
        print("FAIL: RAG_PROVIDER is not 'oma'")
        return 1

    result = oma_provider.dedupe_folder_concepts(
        args.user_id, args.folder,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.dry_run and result.get("merged"):
        print("\n(dry run — re-run without --dry-run to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
