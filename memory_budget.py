"""How much memory the server and its helper processes may use, so heavy work (reading an
upload, indexing a file) waits for room instead of pushing the machine over its limit.

COAST_MEMORY_LIMIT_MB is the machine's memory (2048 on Render's Standard plan). Unset, as
locally, nothing waits."""
from __future__ import annotations

import os

# Measured on an 82-page lecture with 97 figures (2026-10-07): reading the upload peaked at
# 160 MB in its own process, indexing at +116 MB in the server. Estimates round those up.
READ_UPLOAD_MB = 220
INDEX_FILE_MB = 180
INDEX_REMOTE_MB = 40  # indexed in a container: the server holds the pages' text and stores results
_SHARE = 0.85  # leave room for chat, the database and the unexpected


def used_mb() -> float:
    import psutil
    me = psutil.Process()
    total = me.memory_info().rss
    for child in me.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass  # a helper that just finished
    return total / 1e6


def index_file_mb() -> float:
    from coast_content_oma import remote
    return INDEX_REMOTE_MB if remote.enabled() else INDEX_FILE_MB


def has_room(need_mb: float) -> bool:
    limit = os.getenv("COAST_MEMORY_LIMIT_MB")
    if not limit:
        return True
    return used_mb() + need_mb <= float(limit) * _SHARE
