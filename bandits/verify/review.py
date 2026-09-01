"""Persist the explicit owner decision that promotes one calibrated verifier.

Promotion is not a formality. A measurement record existing is not the same as
that measurement saying anything, so the evidence is assessed before an owner is
allowed to sign it off, and an owner overriding that assessment produces a
visibly different artifact rather than the same one.
"""

from __future__ import annotations

import hashlib

from pydantic import model_validator

from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract
from bandits.verify.models import VerifierDraft, VerifierSpec, VerifierStatus
from bandits.verify.validate import Validation, accept, calibrate


class PromotionBlocker(Contract):
    """One reason the evidence does not support promoting a verifier."""

    code: str
    detail: str


def assess_promotion(validation: Validation, verifier_id: str) -> tuple[PromotionBlocker, ...]:
    """What stands between this verifier's measurements and an ordinary review.

    Deliberately short. Every criterion here rules out a verifier that would look
    calibrated while having established nothing, and each one has to stay
    satisfiable on a corpus of realistic size — a bar no real family can clear
    just moves every promotion onto the override path and tells a reader less
    than no bar at all.
    """
    blockers: list[PromotionBlocker] = []
    held_out = validation.held_out(verifier_id)

    if held_out is None or held_out.labeled == 0:
        blockers.append(
            PromotionBlocker(
                code="no_held_out_measurement",
                detail=(
                    "no held-out run carries a label, so the only agreement measured is "
                    "on the traces the check was drafted from"
                ),
            )
        )
    elif held_out.scored == 0:
        blockers.append(
            PromotionBlocker(
                code="no_scorable_held_out",
                detail=(
                    f"all {held_out.labeled} labeled held-out run(s) were unscorable; "
                    "the check has never been observed deciding anything"
                ),
            )
        )

    if not validation.success_labels or not validation.failure_labels:
        blockers.append(
            PromotionBlocker(
                code="single_verdict_labels",
                detail=(
                    f"labels record {validation.success_labels} success(es) and "
                    f"{validation.failure_labels} failure(s); a check measured against one "
                    "verdict has agreed with nothing it could have disagreed with"
                ),
            )
        )

    for result in validation.gameability:
        if result.verifier_id == verifier_id and result.passed:
            blockers.append(
                PromotionBlocker(
                    code="gameable",
                    detail=(
                        f"a constructed run passed this verifier using {result.forged_facts} "
                        f"forged fact(s): {result.hypothesis}"
                    ),
                )
            )
    return tuple(blockers)


class ReviewedVerifier(Contract):
    schema_version: int = 1
    source_draft_id: str
    validation_id: str
    human_acceptance_id: str
    spec: VerifierSpec

    success_threshold: float
    """The threshold this verifier was measured at, frozen at acceptance.

    Carried rather than re-defaulted downstream: a run scoring 0.6 is a failure
    to a validation run at 0.8 and a success to an exporter that assumed 0.5, and
    nothing in the artifact would have recorded the disagreement.
    """

    accepted_risks: tuple[PromotionBlocker, ...] = ()
    """What the owner promoted this over. Empty for an ordinary review."""

    @model_validator(mode="after")
    def is_actually_reviewed(self) -> ReviewedVerifier:
        if self.spec.status not in {VerifierStatus.REVIEWED, VerifierStatus.RISK_ACCEPTED}:
            raise ValueError("a reviewed-verifier artifact must contain an accepted spec")
        if self.spec.validation_artifact_id != self.validation_id:
            raise ValueError("reviewed verifier and validation id disagree")
        if self.spec.human_acceptance_id != self.human_acceptance_id:
            raise ValueError("reviewed verifier and acceptance id disagree")
        if not 0 <= self.success_threshold <= 1:
            raise ValueError("success threshold must be in [0, 1]")
        risk_accepted = self.spec.status is VerifierStatus.RISK_ACCEPTED
        if risk_accepted and not self.accepted_risks:
            raise ValueError("a risk-accepted verifier must name what was accepted")
        if not risk_accepted and self.accepted_risks:
            raise ValueError("an ordinary review cannot carry accepted risks")
        return self


def review_verifier(
    draft: VerifierDraft,
    draft_id: str,
    validation: Validation,
    validation_id: str,
    verifier_id: str,
    acceptance_id: str,
    *,
    accept_risks: bool = False,
) -> ReviewedVerifier:
    """Promote one measured spec after a named human acceptance event.

    Refuses outright when the evidence does not support promotion. ``accept_risks``
    is the only way past that, and it does not produce the same artifact: the
    blockers are recorded on the result and the status says the decision went
    against them.
    """
    if not acceptance_id.strip():
        raise ValueError("human acceptance id must not be empty")
    if validation.source_draft_id != draft_id:
        raise ValueError("validation was produced from another verifier draft")
    if validation.family_id != draft.family_id:
        raise ValueError("validation and verifier draft belong to different families")
    spec = next((item for item in draft.verifiers if item.verifier_id == verifier_id), None)
    if spec is None:
        raise ValueError(f"unknown verifier id: {verifier_id!r}")
    measured = {item.verifier_id for item in validation.agreements}
    if verifier_id not in measured:
        raise ValueError("the selected verifier has no validation measurements")

    blockers = assess_promotion(validation, verifier_id)
    if blockers and not accept_risks:
        detail = "; ".join(f"{item.code}: {item.detail}" for item in blockers)
        raise ValueError(f"the evidence does not support promoting {verifier_id!r} — {detail}")

    reviewed = accept(calibrate(spec, validation_id), acceptance_id, over_risk=bool(blockers))
    return ReviewedVerifier(
        source_draft_id=draft_id,
        validation_id=validation_id,
        human_acceptance_id=acceptance_id,
        spec=reviewed,
        success_threshold=validation.success_threshold,
        accepted_risks=blockers,
    )


def compute_reviewed_verifier_id(reviewed: ReviewedVerifier) -> str:
    digest = hashlib.sha256(reviewed.model_dump_json().encode()).hexdigest()
    return f"reviewed-verifier-{digest[:16]}"


def save_reviewed_verifier(reviewed: ReviewedVerifier, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_reviewed_verifier_id(reviewed),
        kind="reviewed_verifier",
        parent_artifact_id=reviewed.validation_id,
        payload=reviewed.model_dump_json().encode(),
        summary={"verifiers": 1, "accepted_risks": len(reviewed.accepted_risks)},
    )


def load_reviewed_verifier(reviewed_id: str, store: DerivedStore) -> ReviewedVerifier:
    return ReviewedVerifier.model_validate_json(store.read_payload(reviewed_id))
