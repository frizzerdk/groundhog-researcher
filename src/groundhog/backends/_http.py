"""Shared HTTP helper for backends — urlopen with periodic warnings."""

import sys
import threading
import urllib.request


def _urlopen_with_warnings(req, label="LLM", warn_interval=30, timeout=None):
    """urlopen that prints warnings if the request takes too long.

    Prints "still waiting..." every warn_interval seconds so the user
    knows it's not frozen. ``timeout`` (seconds) caps a single request so a
    stalled backend surfaces as a socket timeout instead of hanging the
    optimizer forever; None keeps the old unbounded behavior.
    """
    warnings_printed = 0

    def _warn():
        nonlocal warnings_printed
        warnings_printed += 1
        elapsed = warnings_printed * warn_interval
        print(f"\n  [{label}] Still waiting... ({elapsed}s)", file=sys.stderr, flush=True)
        # Schedule next warning
        timer = threading.Timer(warn_interval, _warn)
        timer.daemon = True
        timer.start()
        return timer

    timer = threading.Timer(warn_interval, _warn)
    timer.daemon = True
    timer.start()

    try:
        return urllib.request.urlopen(req, timeout=timeout)
    finally:
        timer.cancel()
