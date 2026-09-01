# Research source record: Bandits vs. Experiential

Audience: Bandits maintainers. Date: 2026-08-29. Scope: code-visible architecture, capability,
maturity, and positioning. Excludes company/team/traction and unverified hosted behavior.

Canonical user-facing synthesis: `docs/BANDITS_VS_EXPERIENTIAL.md`.

## Claim-to-source ledger

- Bandits product flow, verifier states, evidence hierarchy and RL requirements — `docs/RL_READINESS.md`, local working tree, accessed 2026-08-29.
- Bandits eval/SFT eligibility and portable schema — `docs/EXPORTS.md` and `bandits/export/`, local working tree, accessed 2026-08-29.
- Bandits CLI workflow — `bandits/cli.py`, local working tree, accessed 2026-08-29.
- Bandits source set — `bandits/ingest/__init__.py`, local working tree, accessed 2026-08-29.
- Experiential product and quickstart — Experiential README, Experiential Labs, revision `9f08e1a`, https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/README.md, accessed 2026-08-29.
- Experiential supported/excluded release claims — Release scope, Experiential Labs, revision `9f08e1a`, https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/docs/release-scope.md, accessed 2026-08-29.
- Experiential command behavior — CLI usage, Experiential Labs, revision `9f08e1a`, https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/docs/usage.md, accessed 2026-08-29.
- Experiential source set and ingest guarantees — Ingest reference and `exp/simulation/ingest/sources.py`, Experiential Labs, revision `9f08e1a`, https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/docs/reference/ingest.md, accessed 2026-08-29.
- Experiential gateway/runtime claims — Gateway architecture, Experiential Labs, revision `9f08e1a`, https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/docs/reference/gateway-architecture.md, accessed 2026-08-29.
- Experiential public Python API — `exp/__init__.py`, Experiential Labs, revision `9f08e1a`, https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/exp/__init__.py, accessed 2026-08-29.
- Experiential evidence-cited rubric proposals and review — `exp/common/judging/proposal.py`,
  `review.py`, and `rubric.py`, Experiential Labs, revision `9f08e1a`, accessed 2026-08-29.
- Experiential judge calibration and human-review guarantees — `exp/common/judging/calibration.py`,
  `calibration_contracts.py`, `risk_acceptance.py`, and `exp/optimize/router/judging/`, Experiential
  Labs, revision `9f08e1a`, accessed 2026-08-29.
- Experiential production/teacher acceptance and SFT exclusions —
  `exp/optimize/model/sft/contracts.py`, `sources.py`, `builder.py`, and their tests, Experiential
  Labs, revision `9f08e1a`, accessed 2026-08-29.

## Verification record

- Experiential cloned at tag v0.7.5 / commit `9f08e1aac7011c4be310f0fd2787366f1aec399d`.
- Bandits base commit `756841440f2fb2819efc62b7cf0d348acf3f1ca7`; dirty working tree explicitly included.
- Bandits: 6,411 non-test Python LOC, 3,608 test LOC, 20 test files, 211 tests passed via `uv run pytest -q`.
- Experiential: 133,188 non-test Python LOC, 114,290 test LOC, 309 Python test files. Suite not executed locally.
- The initial synthesis understated overlap. A second implementation-level pass established direct
  mappings among outcomes, task mining, rubric proposals, human approval, judge calibration,
  acceptance evidence, exclusions and SFT construction. The corrected report no longer claims
  Experiential lacks an equivalent evidence-to-evaluator workflow.

## Research stop reason

All requested comparison slots have primary repository evidence. The consequential overlap claims
were checked against implementation and first-party docs, major negative claims were bounded, and
another broad search was unlikely to change the product-level conclusion.
