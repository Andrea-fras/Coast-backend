"""Request hardening shared by the API: rate limits, the admin check and security headers.

Rate limits live in memory (production runs one gunicorn worker, see render.yaml), so a
restart forgets them; they exist to make guessing codes and passwords impractical and to
keep anonymous callers from running up AI costs, not to meter usage.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

ADMIN_EMAILS = {
    "andreaf.fraschetti@gmail.com",
    "rio.mauss@gmail.com",
}


def is_admin(user) -> bool:
    """A listed email whose ownership was proven (Google sign-in or an emailed code).

    The email alone is not enough: before email verification was enforced, anyone could
    register an unclaimed address and would have inherited admin rights with it.
    """
    return bool(user) and (user.email or "").lower() in ADMIN_EMAILS and bool(getattr(user, "email_verified", False))


class RateLimiter:
    """Sliding-window counters keyed by any string (an IP, an email, a user id)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _window(self, key: str, window: float, now: float) -> deque:
        hits = self._hits[key]
        while hits and hits[0] <= now - window:
            hits.popleft()
        return hits

    def check(self, key: str, limit: int, window: float, message: str | None = None) -> None:
        """Count one hit for `key`; raise 429 once more than `limit` hits land in `window` seconds."""
        now = time.monotonic()
        with self._lock:
            hits = self._window(key, window, now)
            if len(hits) >= limit:
                retry = max(1, int(window - (now - hits[0])))
                raise HTTPException(429, message or "Too many attempts. Please wait a few minutes and try again.",
                                    headers={"Retry-After": str(retry)})
            hits.append(now)

    def blocked(self, key: str, limit: int, window: float) -> bool:
        """True when `key` already has `limit` hits in the window (does not count a hit)."""
        with self._lock:
            return len(self._window(key, window, time.monotonic())) >= limit

    def record(self, key: str) -> None:
        with self._lock:
            self._hits[key].append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


limiter = RateLimiter()

MINUTE = 60
HOUR = 60 * MINUTE


def client_ip(request: Request) -> str:
    """The caller's address. On Render the app sits behind its proxy, which appends the real
    client to X-Forwarded-For; locally the socket address is the client."""
    if os.getenv("RENDER"):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# The API only returns JSON and files, so it can refuse to be framed, sniffed or used as a page.
API_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; sandbox",
}


async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in API_HEADERS.items():
        response.headers.setdefault(name, value)
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
    return response


# ── uploaded files ─────────────────────────────────────────────────────────
_MAGIC = {
    ".pdf": (b"%PDF-",),
    ".pptx": (b"PK\x03\x04",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".bmp": (b"BM",),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".webp": (b"RIFF",),
}
PPTX_MAX_UNPACKED = 600 * 1024 * 1024  # a real deck is rarely a tenth of this
PPTX_MAX_ENTRIES = 20000


def check_upload(path, ext: str) -> None:
    """Refuse a file whose bytes don't match its extension, and PowerPoint files that would
    unpack into something enormous (a "zip bomb"), before any parser touches them."""
    ext = ext.lower()
    with open(path, "rb") as fh:
        head = fh.read(1024)
    signatures = _MAGIC.get(ext)
    ok = signatures is None or (
        # PDFs may carry a few junk bytes before the header; the spec allows it within 1 KB.
        any(sig in head for sig in signatures) if ext == ".pdf" else any(head.startswith(sig) for sig in signatures)
    )
    if ok and ext == ".webp":
        ok = head[8:12] == b"WEBP"
    if not ok:
        raise HTTPException(400, f"This file isn't a real {ext.lstrip('.').upper()} file. Try exporting it again.")
    if ext == ".pptx":
        import zipfile
        try:
            with zipfile.ZipFile(path) as zf:
                entries = zf.infolist()
        except zipfile.BadZipFile:
            raise HTTPException(400, "This PowerPoint file is damaged. Try saving a new copy.")
        if len(entries) > PPTX_MAX_ENTRIES or sum(e.file_size for e in entries) > PPTX_MAX_UNPACKED:
            raise HTTPException(400, "This PowerPoint file is too large to open. Export it as a PDF instead.")
