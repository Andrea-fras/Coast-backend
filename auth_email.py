"""Email validation and verification codes for signup."""

from __future__ import annotations

import os
import random
import re
import string
from datetime import datetime, timedelta, timezone

import requests

# Common throwaway domains — not exhaustive, blocks obvious fakes.
DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com", "throwaway.email",
    "yopmail.com", "sharklasers.com", "trashmail.com", "10minutemail.com",
    "fakeinbox.com", "getnada.com", "maildrop.cc", "dispostable.com",
    "temp-mail.org", "emailondeck.com", "mintemail.com", "mytemp.email",
}

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_email_address(email: str, *, strict: bool | None = None) -> tuple[bool, str]:
    """Return (ok, error_message). Checks format, disposable list, and MX record."""
    email = normalize_email(email)
    if not email or not EMAIL_RE.match(email):
        return False, "Enter a valid email address."
    domain = email.split("@", 1)[1]
    if domain in DISPOSABLE_DOMAINS:
        return False, "Disposable email addresses aren't allowed. Use your real inbox or Google sign-in."
    if strict is None:
        strict = bool(os.getenv("RENDER"))
    if not strict:
        return True, ""
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, "MX")
        if not answers:
            return False, "That email domain doesn't look reachable. Check for typos."
    except Exception:
        return False, "That email domain doesn't look reachable. Check for typos."
    return True, ""


def generate_code() -> str:
    return "".join(random.choices(string.digits, k=6))


def send_verification_email(email: str, code: str) -> tuple[bool, str]:
    """Send code via Resend. Returns (sent, detail)."""
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    from_addr = os.environ.get("RESEND_FROM", "Coast <onboarding@resend.dev>").strip()

    if not api_key:
        if os.environ.get("AUTH_DEV_EXPOSE_CODES", "").lower() in ("1", "true", "yes"):
            return False, f"dev_code:{code}"
        return False, "Email verification is not configured. Please sign in with Google."

    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": from_addr,
                "to": [email],
                "subject": "Your Coast verification code",
                "html": (
                    f"<div style='font-family:sans-serif;max-width:420px'>"
                    f"<h2>Welcome to Coast</h2>"
                    f"<p>Your verification code is:</p>"
                    f"<p style='font-size:28px;font-weight:bold;letter-spacing:4px'>{code}</p>"
                    f"<p style='color:#666'>This code expires in 15 minutes.</p>"
                    f"</div>"
                ),
            },
            timeout=15,
        )
        if r.status_code in (200, 201):
            return True, "sent"
        return False, f"Could not send email ({r.status_code})"
    except Exception as e:
        return False, str(e)[:120]
