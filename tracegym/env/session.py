"""The concrete environment session.

Ties the three pieces together: a :class:`~tracegym.env.store.Store`
materialized from the schema and seeded from the task, a
:class:`~tracegym.env.tools.ToolRuntime` that dispatches actions, and an
:class:`~tracegym.env.ledger.EffectLedger` that catches everything irreversible.

Determinism is the contract. Same schema + same task + same actions => same
snapshot, same digest, same effects. No clock, no randomness, no network. That
is what makes the fidelity gate in stage 6 mean anything.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping

from tracegym.contracts import (
    Effect,
    EnvManifest,
    JsonObject,
    JsonValue,
    Observation,
    StateSchema,
    TaskCase,
    ToolClass,
)

from .interface import BaseEnvSession, SessionClosedError
from .ledger import EffectLedger
from .store import Store
from .tools import Rule, ToolRuntime

__all__ = ["TraceGymSession", "build_session"]


class TraceGymSession(BaseEnvSession):
    """One live environment for exactly one :class:`TaskCase`.

    Parameters
    ----------
    schema:
        Inferred state schema (stage 3).
    task:
        The task, whose ``pre_state`` seeds the store (stage 5).
    tool_classes:
        ``tool -> ToolClass`` (stage 2). Tools absent from this map are not in
        the action space and raise when called.
    rules:
        Per-tool overrides for the inferred read/write/external rules. This is
        the escape hatch for real customer tools whose argument names do not
        line up with their columns.
    external_stubs:
        ``tool -> response`` for EXTERNAL tools, so the stub can match the
        acknowledgement the real tool returned. It is still only an
        acknowledgement: nothing is performed.
    db_path:
        Where SQLite lives. ``None`` means in-memory (default); pass a path to
        keep the store for inspection. A temp file we create is deleted at close.
    """

    def __init__(
        self,
        schema: StateSchema,
        task: TaskCase,
        tool_classes: Mapping[str, ToolClass],
        *,
        rules: Mapping[str, Rule] | None = None,
        external_stubs: Mapping[str, JsonValue] | None = None,
        db_path: str | None = None,
        env_id: str | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.task = task
        self.tool_classes = dict(tool_classes)
        self._rule_overrides = dict(rules or {})
        self._external_stubs = dict(external_stubs or {})
        self._db_path = db_path
        self._own_tempfile = False
        self._env_id = env_id
        self._step = 0
        self.ledger = EffectLedger()
        self.store = Store(schema, path=db_path or ":memory:")
        self.runtime: ToolRuntime | None = None
        self._manifest: EnvManifest | None = None

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> None:
        if self._db_path == "":  # explicit request for a real file we manage
            fd, path = tempfile.mkstemp(prefix="tracegym-env-", suffix=".sqlite")
            os.close(fd)
            self._db_path = path
            self._own_tempfile = True
            self.store = Store(self.schema, path=path)
        self.store.open()
        self.store.seed(self.task)
        self.runtime = ToolRuntime(
            self.schema,
            self.tool_classes,
            self.store,
            self.ledger,
            rules=self._rule_overrides,
            external_stubs=self._external_stubs,
        )
        self._manifest = self._build_manifest()

    def _close(self) -> None:
        # Deterministic teardown, in a fixed order: freeze the ledger so no
        # late effect can be appended, drop the connection, remove any temp
        # file we created. Idempotent via BaseEnvSession.
        self.ledger.freeze()
        self.store.close()
        if self._own_tempfile and self._db_path and os.path.exists(self._db_path):
            os.unlink(self._db_path)

    # -- identity ----------------------------------------------------------

    def _build_manifest(self) -> EnvManifest:
        assert self.runtime is not None
        schema_digest = self.store.schema_digest()
        env_id = self._env_id or f"env-{self.task.task_id}-{schema_digest[:12]}"
        static = tuple(sorted(e.name for e in self.schema.entities if e.static_snapshot))
        return EnvManifest(
            env_id=env_id,
            task_id=self.task.task_id,
            schema_digest=schema_digest,
            tool_classes=dict(self.tool_classes),
            static_entities=static,
            unsupported_tools=self.runtime.unsupported_tools,
        )

    def manifest(self) -> EnvManifest:
        if self._manifest is None:
            if self._closed:
                raise SessionClosedError("session was closed before it was opened")
            self.open()
        assert self._manifest is not None
        return self._manifest

    def unsupported_reason(self, tool: str) -> str | None:
        """Why a tool is unsupported, for operator-facing reports."""
        self._require_live()
        assert self.runtime is not None
        return self.runtime.reason(tool)

    # -- the four methods --------------------------------------------------

    def execute(self, tool: str, arguments: JsonObject) -> Observation:
        """Run one action. The only sanctioned way to change this world."""
        self._require_live()
        assert self.runtime is not None
        step = self._step
        self._step += 1
        return self.runtime.execute(tool, dict(arguments or {}), step)

    def snapshot(self) -> dict[str, list[dict]]:
        self._require_live()
        return self.store.snapshot()

    def digest(self) -> str:
        self._require_live()
        return self.store.digest()

    def effects(self) -> tuple[Effect, ...]:
        return self.ledger.all()

    @property
    def step_count(self) -> int:
        return self._step


def build_session(
    schema: StateSchema,
    task: TaskCase,
    tool_classes: Mapping[str, ToolClass],
    **kwargs,
) -> TraceGymSession:
    """Materialize and open an environment in one call."""
    session = TraceGymSession(schema, task, tool_classes, **kwargs)
    session.open()
    return session
