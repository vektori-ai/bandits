"""Reconstruct a task's starting state from its own trace (PLAN.md Step 10).

The rule, stated once and enforced everywhere below:

    The trace tells you what the world looked like at the start, *because the
    agent read it*. Every read BEFORE the first write is evidence of the
    starting state. Every read AFTER a write is not - don't let it leak
    backward.

Why the leak matters. In ``ep-refund-ok`` the agent reads order 7741 at step 2
(``status == "delivered"``), refunds it at step 4, then reads it again at step 5
(``status == "refunded"``). If the step-5 observation lands in the pre-state,
the task starts in the state the agent was supposed to *produce*. Every rollout
then scores as correct without doing anything, and the reward signal is dead.

The refinement this module implements: **the flag is per entity and per row, not
global.** A write to ``orders/7741`` says nothing about the ``products`` table,
and nothing about ``orders/7750`` either. Blanking every read after the first
write throws away most of the evidence in a long trace, which produces sparse,
under-seeded environments - a different failure with the same smell. So we track
a dirty set of ``(entity, row_key)`` pairs, and only reads that touch a dirty row
(or a dirty-in-full entity) are rejected.

When a write's target row cannot be identified - no primary key in its arguments
or response - we mark the *whole entity* dirty from that step onward. Being
conservative there costs seed rows; being permissive there corrupts the task.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from tracegym.contracts import (
    CallStatus,
    EntityRows,
    EntitySchema,
    InvocationPoint,
    JsonObject,
    JsonValue,
    StateSchema,
    Trace,
)

__all__ = [
    "PreState",
    "RowRef",
    "attribute_object",
    "entity_field_names",
    "iter_objects",
    "key_of",
    "reconstruct_final_state",
    "reconstruct_pre_state",
    "solvability_warnings",
    "write_tools_of",
]


# --------------------------------------------------------------------------
# row attribution: turning a JSON response into (entity, primary key, row)
# --------------------------------------------------------------------------

#: An object must match this many of an entity's declared fields (the primary
#: key included) before we are willing to call it a row of that entity. Two is
#: the minimum that rules out coincidental id-only payloads such as
#: ``{"error": "not_found", "order_id": 9999}``.
MIN_FIELD_OVERLAP = 2


@dataclass(frozen=True)
class RowRef:
    """A row identified inside a tool response."""

    entity: str
    key: JsonValue
    row: JsonObject
    step: int
    tool: str

    @property
    def key_id(self) -> str:
        return json.dumps(self.key, sort_keys=True, default=str)


def entity_field_names(entity: EntitySchema) -> set[str]:
    return {f.name for f in entity.fields}


def iter_objects(value: JsonValue) -> Iterator[JsonObject]:
    """Yield every dict inside a JSON value, outermost first.

    Responses are not always a bare row: they wrap rows in envelopes
    (``{"order": {...}}``) or lists (``{"results": [{...}, {...}]}``). We look at
    all of them and let attribution decide which are rows.
    """
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from iter_objects(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_objects(v)


def attribute_object(obj: JsonObject, schema: StateSchema) -> tuple[str, JsonValue] | None:
    """Decide which entity an object is a row of, and what its key is.

    Returns ``(entity_name, primary_key_value)`` or None when the object is not
    recognisably a row. Requires the entity's primary key to be present and at
    least :data:`MIN_FIELD_OVERLAP` declared fields to match. Ties are broken by
    entity name so the result is deterministic.

    Catches: rows in envelopes, rows in lists, write-response rows.
    Misses: entities with no inferred primary key (they are skipped entirely -
    we cannot address their rows, so we cannot assert on them either), and rows
    whose response uses field names the schema never saw.
    """
    if not isinstance(obj, dict):
        return None
    if "error" in obj:
        # An error payload is not a row even when it echoes an id.
        return None
    best: tuple[int, str, JsonValue] | None = None
    for entity in schema.entities:
        pk = entity.primary_key
        if not pk or pk not in obj:
            continue
        names = entity_field_names(entity)
        overlap = len(names & set(obj)) if names else len(obj)
        if overlap < MIN_FIELD_OVERLAP:
            continue
        cand = (overlap, entity.name, obj[pk])
        if best is None or (-cand[0], cand[1]) < (-best[0], best[1]):
            best = cand
    if best is None:
        return None
    return best[1], best[2]


def _project(entity: EntitySchema, obj: JsonObject) -> JsonObject:
    """Keep only fields the schema knows about.

    A write response often carries fields the entity does not have
    (``refund_amount_cents`` on ``refund_order``). Letting those through would
    make the verifier assert on columns the rebuilt store never materializes,
    and every rollout would fail on a field that does not exist.
    """
    names = entity_field_names(entity)
    if not names:
        return dict(obj)
    return {k: v for k, v in obj.items() if k in names}


def rows_in(inv: InvocationPoint, schema: StateSchema) -> list[RowRef]:
    """Every row observable in one invocation's response."""
    out: list[RowRef] = []
    seen: set[tuple[str, str]] = set()
    for obj in iter_objects(inv.response):
        hit = attribute_object(obj, schema)
        if hit is None:
            continue
        name, key = hit
        entity = schema.entity(name)
        if entity is None:  # pragma: no cover - attribute_object only names real entities
            continue
        ref = RowRef(entity=name, key=key, row=_project(entity, obj), step=inv.step, tool=inv.tool)
        ident = (ref.entity, ref.key_id)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(ref)
    return out


def key_of(entity: EntitySchema, obj: JsonObject) -> JsonValue | None:
    pk = entity.primary_key
    if pk and isinstance(obj, dict) and pk in obj:
        return obj[pk]
    return None


# --------------------------------------------------------------------------
# tool classes
# --------------------------------------------------------------------------


def write_tools_of(schema: StateSchema) -> set[str]:
    """Write-class tools according to the schema's ``written_by`` sets."""
    return {t for e in schema.entities for t in e.written_by}


def read_tools_of(schema: StateSchema) -> set[str]:
    return {t for e in schema.entities for t in e.read_by}


# --------------------------------------------------------------------------
# the reconstruction
# --------------------------------------------------------------------------


@dataclass
class PreState:
    """Reconstructed starting state plus the reasoning that produced it."""

    trace_id: str
    rows: dict[str, dict[str, JsonObject]] = field(default_factory=dict)
    """entity -> key_id -> row. Insertion ordered; first observation wins."""

    keys: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    """entity -> key_id -> the raw primary key value."""

    first_write_step: int | None = None
    blocked: list[JsonObject] = field(default_factory=list)
    """Observations rejected as post-state. The audit trail for the leak rule."""

    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    """Solvability warnings. A non-empty list means: do not train on this task
    until a human has looked, because every rollout may fail identically."""

    def add(self, ref: RowRef) -> bool:
        bucket = self.rows.setdefault(ref.entity, {})
        if ref.key_id in bucket:
            # First observation wins. A later pre-write read of the same row can
            # only add fields, never overwrite - the earlier value is closer to
            # the true start of the episode.
            for k, v in ref.row.items():
                bucket[ref.key_id].setdefault(k, v)
            return False
        bucket[ref.key_id] = dict(ref.row)
        self.keys.setdefault(ref.entity, {})[ref.key_id] = ref.key
        return True

    def has_value(self, value: Any) -> bool:
        """True when some pre-state row carries this value in any field."""
        target = str(value)
        for bucket in self.rows.values():
            for row in bucket.values():
                for v in row.values():
                    if str(v) == target:
                        return True
        return False

    def to_entity_rows(self) -> tuple[EntityRows, ...]:
        return tuple(
            EntityRows(entity=name, rows=tuple(bucket.values()))
            for name, bucket in self.rows.items()
            if bucket
        )


class _DirtySet:
    """Per-entity, per-row write tracking."""

    def __init__(self) -> None:
        self.rows: dict[str, set[str]] = {}
        self.entities: set[str] = set()

    def mark_row(self, entity: str, key_id: str) -> None:
        self.rows.setdefault(entity, set()).add(key_id)

    def mark_entity(self, entity: str) -> None:
        self.entities.add(entity)

    def is_dirty(self, entity: str, key_id: str) -> bool:
        return entity in self.entities or key_id in self.rows.get(entity, set())


def _write_targets(
    inv: InvocationPoint, schema: StateSchema
) -> tuple[list[tuple[str, str, JsonValue]], list[str]]:
    """What a write invocation touched.

    Returns (specific rows, whole entities). We look at the arguments first -
    ``refund_order(order_id=7741)`` names its target even when the call errored -
    then at the response. An entity that ``written_by`` claims this tool writes,
    but for which we cannot pin a row, is dirtied in full.
    """
    rows: list[tuple[str, str, JsonValue]] = []
    whole: list[str] = []
    for entity in schema.entities:
        if inv.tool not in entity.written_by:
            continue
        found: list[JsonValue] = []
        k = key_of(entity, inv.arguments)
        if k is not None:
            found.append(k)
        for obj in iter_objects(inv.response):
            k2 = key_of(entity, obj) if isinstance(obj, dict) else None
            if k2 is not None and k2 not in found:
                found.append(k2)
        if found:
            for k3 in found:
                rows.append((entity.name, json.dumps(k3, sort_keys=True, default=str), k3))
        else:
            whole.append(entity.name)
    return rows, whole


_ID_TOKEN = re.compile(r"\b(?:[A-Z0-9]+(?:[-_][A-Z0-9]+)+|\d{2,})\b")
_STOP_TOKENS = {"00"}


def solvability_warnings(instruction: str, pre: PreState) -> list[str]:
    """Flag identifiers in the instruction that have no pre-state row.

    PLAN.md Step 10 is emphatic about why this matters: if the task says "refund
    order 7741" and order 7741 does not exist in the starting state, every
    rollout fails identically. At the pass@k gate that is indistinguishable from
    "beyond the model's ability" - a false positive of exactly the kind this
    whole system exists to avoid. So we surface it as a warning on the task
    rather than shipping a silently impossible environment.

    Catches: ids the agent only learned about *after* a write (so the pre-state
    lost them), ids mentioned in the instruction but never read at all.
    Misses: identifiers written in prose ("my last order"), ids that coincide
    with an unrelated field value, and semantic preconditions - an order that
    exists but is not refundable still passes this check.
    """
    out: list[str] = []
    for token in dict.fromkeys(_ID_TOKEN.findall(instruction or "")):
        if token in _STOP_TOKENS:
            continue
        if pre.has_value(token):
            continue
        if token.isdigit() and pre.has_value(int(token)):
            continue
        out.append(f"instruction references '{token}' but no pre-state row carries that value")
    return out


def reconstruct_pre_state(
    trace: Trace,
    schema: StateSchema,
    *,
    write_tools: Iterable[str] | None = None,
    include_error_responses: bool = False,
) -> PreState:
    """Reconstruct the starting state for one trace.

    ``write_tools`` defaults to the union of every entity's ``written_by``. Pass
    it explicitly when the schema is partial - a write tool we fail to recognise
    is the one way this function can leak post-state.

    Error responses are not evidence by default: ``{"error": "not_found",
    "order_id": 9999}`` describes a row that is *absent*, and seeding it would
    invert the task.
    """
    writes = set(write_tools) if write_tools is not None else write_tools_of(schema)
    pre = PreState(trace_id=trace.trace_id)
    dirty = _DirtySet()

    for inv in sorted(trace.invocations, key=lambda i: i.step):
        is_write = inv.tool in writes
        if not is_write:
            if inv.status is CallStatus.ERROR and not include_error_responses:
                continue
            for ref in rows_in(inv, schema):
                if dirty.is_dirty(ref.entity, ref.key_id):
                    pre.blocked.append(
                        {
                            "step": inv.step,
                            "tool": inv.tool,
                            "entity": ref.entity,
                            "row_key": ref.key,
                            "reason": "read after a write to this row; post-state, not pre-state",
                        }
                    )
                    continue
                pre.add(ref)
            continue

        # a write: everything it touched is post-state from here on
        if pre.first_write_step is None:
            pre.first_write_step = inv.step
        rows, whole = _write_targets(inv, schema)
        for entity_name, key_id, key in rows:
            dirty.mark_row(entity_name, key_id)
            pre.notes.append(
                f"step {inv.step}: {inv.tool} wrote {entity_name}[{key!r}]; "
                f"later reads of that row are post-state"
            )
        for entity_name in whole:
            dirty.mark_entity(entity_name)
            pre.notes.append(
                f"step {inv.step}: {inv.tool} wrote {entity_name} but its target row could not "
                f"be identified; the whole entity is treated as post-state from here"
            )
        if not rows and not whole:
            pre.notes.append(
                f"step {inv.step}: {inv.tool} is a write tool no entity claims; "
                f"no rows dirtied - check the schema's written_by sets"
            )

    pre.warnings = solvability_warnings(trace.instruction or "", pre)
    return pre


def reconstruct_final_state(
    trace: Trace,
    schema: StateSchema,
    *,
    write_tools: Iterable[str] | None = None,
) -> dict[str, dict[str, JsonObject]]:
    """Reconstruct the state at the END of the trace: last observation wins.

    Both reads and successful write responses count here, because by the end of
    the episode there is no ordering left to violate. Write responses are
    projected onto the entity's declared fields first (see :func:`_project`).
    Errored calls are ignored - they changed nothing and returned no row.
    """
    _ = write_tools  # accepted for symmetry; final state does not depend on ordering
    state: dict[str, dict[str, JsonObject]] = {}
    for inv in sorted(trace.invocations, key=lambda i: i.step):
        if inv.status is CallStatus.ERROR:
            continue
        for ref in rows_in(inv, schema):
            bucket = state.setdefault(ref.entity, {})
            if ref.key_id in bucket:
                bucket[ref.key_id] = {**bucket[ref.key_id], **ref.row}
            else:
                bucket[ref.key_id] = dict(ref.row)
    return state
