"""Process-wide admission control shared by tutoring, evaluation and ingestion.

Configure per-process limits below the provider project's actual quotas. This
bounds bursts; it does not increase provider quotas or coordinate multiple hosts.
"""
from collections import deque
from contextlib import contextmanager
import logging
import os
import threading
import time

log = logging.getLogger(__name__)

def is_credit_error(error):
    return any(marker in str(error).lower() for marker in (
        'prepayment credits are depleted','credit_balance_exhausted',
        'insufficient_quota','no credits remaining'))


class ProviderUnavailable(RuntimeError):
    """The provider was recently out of credits; callers fall back without waiting on it."""


# A provider that answers "out of credits" keeps answering that, so skip it for a
# while instead of paying a failed round trip (and retries) on every request.
CREDIT_COOLDOWN_SEC = float(os.getenv('COAST_CREDIT_COOLDOWN_SEC', '300'))
_down_until = {}


def _check_available(provider):
    until = _down_until.get(provider, 0)
    if until > time.monotonic():
        raise ProviderUnavailable(f'{provider}: no credits remaining (skipped until the cooldown ends)')


def _note_failure(provider, error):
    if not is_credit_error(error):
        return
    if is_available(provider):
        log.warning('provider=%s out of credits; skipping it for %ds', provider, CREDIT_COOLDOWN_SEC)
    _down_until[provider] = time.monotonic() + CREDIT_COOLDOWN_SEC


def is_available(provider):
    return _down_until.get(provider, 0) <= time.monotonic()


class Capacity:
    def __init__(self, concurrent=6, background=4, max_waiters=64, rpm=0):
        self.concurrent = max(1, concurrent)
        self.background = min(self.concurrent, max(1, background))
        self.max_waiters = max_waiters
        self.rpm = max(0, rpm)
        self.condition = threading.Condition()
        self.active = self.background_active = self.waiters = self.interactive_waiters = 0
        self.starts = deque()

    @contextmanager
    def slot(self, priority='background', timeout=45):
        interactive = priority == 'interactive'
        started = time.monotonic()
        with self.condition:
            if self.waiters >= self.max_waiters:
                raise TimeoutError('AI request queue is full; retry shortly')
            self.waiters += 1
            self.interactive_waiters += int(interactive)
            try:
                while True:
                    now = time.monotonic()
                    while self.starts and self.starts[0] <= now - 60:
                        self.starts.popleft()
                    available = self.active < self.concurrent and (not self.rpm or len(self.starts) < self.rpm)
                    if not interactive:
                        available = available and self.background_active < self.background and self.interactive_waiters == 0
                    if available:
                        self.active += 1
                        self.background_active += int(not interactive)
                        self.starts.append(now)
                        break
                    remaining = timeout - (now - started)
                    if remaining <= 0:
                        raise TimeoutError('AI request queue wait exceeded; retry shortly')
                    delay = max(0.01, self.starts[0] + 60 - now) if self.rpm and len(self.starts) >= self.rpm else remaining
                    self.condition.wait(min(delay, remaining))
            finally:
                self.waiters -= 1
                self.interactive_waiters -= int(interactive)
                self.condition.notify_all()
        try:
            yield time.monotonic() - started
        finally:
            with self.condition:
                self.active -= 1
                self.background_active -= int(not interactive)
                self.condition.notify_all()

_gates = {}
_lock = threading.Lock()

def gate(provider):
    with _lock:
        if provider not in _gates:
            prefix = 'COAST_' + provider.upper()
            concurrent = int(os.getenv(prefix + '_CONCURRENCY', '6'))
            _gates[provider] = Capacity(concurrent, int(os.getenv(prefix + '_BACKGROUND_CONCURRENCY', str(max(1,concurrent-2)))),
                int(os.getenv('COAST_AI_MAX_WAITERS','64')), int(os.getenv(prefix + '_RPM','0')))
        return _gates[provider]

def _record(provider, usage, started, ok):
    try:
        import ai_usage
        ai_usage.record(provider, usage, latency_ms=(time.monotonic() - started) * 1000, ok=ok)
    except Exception:
        log.exception('ai_usage: could not record a %s call', provider)


def _usage(provider, response):
    try:
        import ai_usage
        return ai_usage.read_usage(provider, response)
    except Exception:
        return None


def call(provider, operation, *, priority='background'):
    _check_available(provider)
    started = time.monotonic()
    with gate(provider).slot(priority) as waited:
        call_started = time.monotonic()
        try:
            response = operation()
        except Exception as e:
            _note_failure(provider, e)
            _record(provider, None, call_started, ok=False)
            raise
        finally:
            log.info('ai_call provider=%s priority=%s wait_ms=%d elapsed_ms=%d',provider,priority,waited*1000,(time.monotonic()-started)*1000)
        _record(provider, _usage(provider, response), call_started, ok=True)
        return response

def stream(provider, operation, *, priority='interactive'):
    _check_available(provider)
    with gate(provider).slot(priority):
        import ai_usage
        meter = ai_usage.StreamMeter(provider)
        started = time.monotonic()
        ok = True
        try:
            response = operation()
        except Exception as e:
            _note_failure(provider, e)
            _record(provider, None, started, ok=False)
            raise
        try:
            for chunk in response:
                meter.see(chunk)
                # OpenAI's final usage-only chunk carries no text; callers never see it.
                if getattr(chunk, 'choices', None) == [] and getattr(chunk, 'usage', None) is not None:
                    continue
                yield chunk
        except GeneratorExit:
            raise  # the student left mid-answer; the tokens so far are still recorded
        except Exception as e:
            ok = False
            _note_failure(provider, e)
            raise
        finally:
            _record(provider, meter.usage, started, ok=ok)
            close = getattr(response, 'close', None)
            if callable(close): close()
