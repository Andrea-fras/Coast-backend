"""Briefly stop optional embedding requests after a non-retryable credit error."""
import threading
import time
_lock=threading.Lock()
_retry_at=0.0

def available():
    with _lock: return time.monotonic() >= _retry_at

def note_error(error):
    global _retry_at
    message=str(error).lower()
    if not any(marker in message for marker in ('insufficient_quota','credit_balance_exhausted','no credits remaining')):
        return False
    with _lock: _retry_at=time.monotonic()+300
    return True
