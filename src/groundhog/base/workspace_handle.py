"""WorkspaceHandle — the toolkit's pointer at the attempt currently in flight.

``toolkit.ws`` (alias ``toolkit.workspace``) is a stable handle created once by
``assemble_toolkit``. Whoever holds the live attempt points it —
``set_attempt()`` in the strategy when a workspace is born, or in the CLI when
a command targets an attempt — and build-time tools that closed over
``toolkit.ws`` read the in-flight attempt's files at invoke time.

Design contract (2026-07 sessions):
- ``set_attempt()/clear()`` is the primitive; ``attempt()`` is the RAII
  bracket whose ``finally`` makes the clear unforgettable.
- Reads HARD-FAIL when nothing is set (``WorkspaceNotSetError``) — never
  ``None``, never a stale attempt.
- Plain instance attribute, deliberately not a ContextVar: agent tools are
  invoked over HTTP on a server thread, where a ContextVar set on the
  strategy thread would be invisible. Sequential contract — one current
  attempt per toolkit; parallel attempts get per-worker toolkits.
- Read-pointer only: the handle never commits or aborts. The workspace
  lifecycle stays owned by the strategy / CLI. Convention: ``commit()`` is
  the LAST statement inside the bracket (the folder is renamed at commit).
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional


class WorkspaceNotSetError(RuntimeError):
    """Nothing is in flight — set_attempt() (or the attempt() bracket) first."""


class WorkspaceStateError(RuntimeError):
    """Double-set without clear, unresolvable target, or misuse of a view."""


class ReadOnlyWorkspaceView:
    """A committed attempt exposed through the workspace ``.path`` contract.

    Lets one tool serve both worlds: during a run ``toolkit.ws`` points at the
    live workspace; from the CLI it can point at any committed attempt, whose
    folder is ensured on disk via ``history.materialize``. Mutating lifecycle
    calls fail loudly.
    """

    def __init__(self, attempt, history):
        self._attempt = attempt
        self._history = history

    @property
    def attempt(self):
        return self._attempt

    @property
    def display_id(self) -> str:
        return getattr(self._attempt, "name", "") or str(self._attempt.id)

    @property
    def path(self) -> Path:
        return Path(self._history.materialize(self._attempt))

    def heartbeat(self):
        return None

    def commit(self, *a, **k):
        raise WorkspaceStateError(
            f"attempt {self._attempt.id!r} is already committed — this is a "
            f"read-only view, not a live workspace"
        )

    def abort(self, *a, **k):
        raise WorkspaceStateError(
            f"attempt {self._attempt.id!r} is already committed — this is a "
            f"read-only view, not a live workspace"
        )


class WorkspaceHandle:
    """One current attempt per toolkit; see module docstring."""

    def __init__(self, history):
        self._history = history
        self._current = None  # live Workspace or ReadOnlyWorkspaceView

    # --- the primitive -----------------------------------------------------

    def set_attempt(self, target):
        """Point the handle at ``target`` and return the workspace-shaped
        object. Accepts a live Workspace, an in-progress workspace id, a
        committed Attempt, or a committed attempt id."""
        ws = self._resolve(target)
        if self._current is not None and self._current is not ws:
            raise WorkspaceStateError(
                "an attempt is already in flight; clear() it first "
                "(one current attempt per toolkit)"
            )
        self._current = ws
        hb = getattr(ws, "heartbeat", None)
        if callable(hb):
            hb()  # liveness refresh — reap must never kill what's current
        return ws

    def clear(self):
        self._current = None

    # --- the bracket ---------------------------------------------------------

    @contextmanager
    def attempt(self, target):
        """RAII bracket: set on entry, ALWAYS clear on exit (return, continue,
        or exception). Asserts nothing was leaked in flight on entry."""
        if self._current is not None:
            raise WorkspaceStateError(
                "nested or leaked attempt: the handle is already set — "
                "the previous bracket did not clear"
            )
        ws = self.set_attempt(target)
        try:
            yield ws
        finally:
            self.clear()

    # --- reads (hard-fail) -----------------------------------------------------

    @property
    def current(self):
        if self._current is None:
            raise WorkspaceNotSetError(
                "no attempt in flight: call toolkit.ws.set_attempt(...) or "
                "wrap the work in `with toolkit.ws.attempt(...):`"
            )
        return self._current

    @property
    def path(self) -> Path:
        return Path(self.current.path)

    def is_set(self) -> bool:
        """Non-raising probe for tools that want graceful degrade."""
        return self._current is not None

    def heartbeat(self):
        """Refresh the current workspace's liveness marker (no-op if unset)."""
        if self._current is not None:
            hb = getattr(self._current, "heartbeat", None)
            if callable(hb):
                hb()

    # --- namespace conveniences ---------------------------------------------

    def get_prior(self, toolkit) -> Optional[object]:
        """Selection through the workspace namespace — delegates to the
        toolkit's standing ``get_prior`` (fulfils the old
        ``toolkit.workspace.get_prior`` plan)."""
        return toolkit.get_prior(toolkit)

    # --- resolution ----------------------------------------------------------

    def _resolve(self, target):
        from groundhog.base.attempt_history import Attempt, Workspace

        if isinstance(target, (Workspace, ReadOnlyWorkspaceView)):
            return target
        if isinstance(target, Attempt):
            return ReadOnlyWorkspaceView(target, self._history)
        if isinstance(target, str):
            # In-progress workspace id first (live re-bind) …
            try:
                return self._history.resume(target)
            except (KeyError, NotImplementedError):
                pass
            # … then a committed attempt id (read-only, materialized on demand).
            attempt = self._history.get(target)
            if attempt is not None:
                return ReadOnlyWorkspaceView(attempt, self._history)
        raise WorkspaceStateError(
            f"cannot resolve attempt {target!r}: not a live workspace, "
            f"in-progress workspace id, or committed attempt id"
        )
