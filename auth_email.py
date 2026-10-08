"""Email validation and verification codes for signup."""

from __future__ import annotations

import os
import secrets
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


def email_verification_required() -> bool:
    """When true, signup must complete verify-email before register."""
    return os.getenv("AUTH_EMAIL_VERIFICATION", "").lower() in ("1", "true", "yes")


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
        strict = email_verification_required()
    if not strict:
        return True, ""
    # Only a domain that doesn't exist, or takes no mail, is refused. When the lookup itself fails
    # (our resolver timing out), the address is allowed: the emailed code still has to arrive.
    try:
        import dns.resolver
    except ImportError:
        return True, ""  # no resolver here (a local setup): the emailed code is the check
    try:
        answers = dns.resolver.resolve(domain, "MX", lifetime=4)
        if not answers:
            return False, "That email domain doesn't look reachable. Check for typos."
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return False, "That email domain doesn't look reachable. Check for typos."
    except Exception:
        pass
    return True, ""


def generate_code() -> str:
    return "".join(secrets.choice(string.digits) for _ in range(6))


EMAILS = {
    "verify": ("Your Coast verification code", "Welcome to Coast",
               "Enter this code to finish creating your account:",
               "If you didn't try to sign up for Coast, you can ignore this email."),
    "reset": ("Reset your Coast password", "Reset your password",
              "Enter this code in Coast to choose a new password:",
              "If you didn't ask to reset your password, you can ignore this email; your password stays the same."),
}


def send_code_email(email: str, code: str, purpose: str = "verify") -> tuple[bool, str]:
    """Send a six-digit code via Resend. Returns (sent, detail)."""
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    from_addr = os.environ.get("RESEND_FROM", "Coast <onboarding@resend.dev>").strip()
    subject, heading, lead, footer = EMAILS[purpose]

    if not api_key:
        if os.environ.get("AUTH_DEV_EXPOSE_CODES", "").lower() in ("1", "true", "yes"):
            return False, f"dev_code:{code}"
        return False, "Email is not configured. Please sign in with Google."

    html = (
        "<div style='font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:440px;margin:0 auto;"
        "padding:24px;color:#1a1a1a'>"
        f"<p style='font-size:20px;font-weight:800;margin:0 0 16px'>{heading}</p>"
        f"<p style='margin:0 0 12px'>{lead}</p>"
        f"<p style='font-size:32px;font-weight:800;letter-spacing:6px;margin:8px 0 16px'>{code}</p>"
        "<p style='color:#666;margin:0 0 4px'>The code expires in 15 minutes.</p>"
        f"<p style='color:#666;margin:0'>{footer}</p>"
        "<p style='color:#999;font-size:12px;margin:24px 0 0'>Coast · coast.academy</p>"
        "</div>"
    )
    text = f"{heading}\n\n{lead}\n\n{code}\n\nThe code expires in 15 minutes.\n{footer}\n\nCoast · coast.academy"
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"from": from_addr, "to": [email], "subject": subject, "html": html, "text": text},
            timeout=15,
        )
        if r.status_code in (200, 201):
            return True, "sent"
        return False, f"Could not send email ({r.status_code}: {r.text[:160]})"
    except Exception as e:
        return False, str(e)[:120]


def send_verification_email(email: str, code: str) -> tuple[bool, str]:
    return send_code_email(email, code, "verify")
