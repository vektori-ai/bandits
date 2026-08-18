"""Verifier synthesis, grading and anti-cheat. The reward function lives here."""

from tracegym.verify.anticheat import (
    AntiCheatReport,
    Finding,
    RolloutAction,
    RolloutRecord,
    check_rollout,
    enforce,
)
from tracegym.verify.run import RewardMode, UnreviewedVerifierError, evaluate
from tracegym.verify.synthesize import UnlabeledTraceError, synthesize_verifier

__all__ = [
    "AntiCheatReport",
    "Finding",
    "RewardMode",
    "RolloutAction",
    "RolloutRecord",
    "UnlabeledTraceError",
    "UnreviewedVerifierError",
    "check_rollout",
    "enforce",
    "evaluate",
    "synthesize_verifier",
]
