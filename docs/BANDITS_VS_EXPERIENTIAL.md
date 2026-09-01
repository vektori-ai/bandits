# Bandits vs. Experiential

**Audience:** Bandits maintainers  
**Compared:** Bandits working tree on 2026-08-29 (base commit `7568414`) and Experiential
v0.7.5 at [`9f08e1a`](https://github.com/experientiallabs/experiential/tree/9f08e1aac7011c4be310f0fd2787366f1aec399d)  
**Bottom line:** these projects overlap substantially in the trace-to-task-to-evaluator-to-training
data pipeline. Bandits is a focused implementation of that pipeline centered on executable task
verifiers and portable exports. Experiential contains a more mature version of much of the same
pipeline—outcomes, leakage-safe task mining, evidence-cited rubric proposals, human rubric review,
judge calibration, acceptance rules, exclusions, and SFT datasets—inside a much broader inference
gateway, simulator, router optimizer, and training runtime.

## Executive answer

Experiential is broader than Bandits and, in the shared offline-learning area, is also farther ahead
on several important guarantees. Its primary product includes a live model-serving path—an
OpenAI/Anthropic-compatible gateway, identity and spend control, provider routing, traffic-derived
simulation, router fitting, and managed SFT—but its codebase also implements a serious evidence and
dataset-construction subsystem. Its repository is
roughly 133k non-test Python lines plus a native Rust data plane, versus Bandits' roughly 6.4k
non-test Python lines. Experiential v0.7.5 has 309 Python test files; Bandits has 20 test files and
its current suite passes locally (211 tests).

The shared seam can be phrased as:

> Given historical, heterogeneous agent traces, what task was being attempted, what evidence says
> it succeeded, what executable or rubric-based check could score it, how well does that check agree
> with history, and which rows are defensible enough to export?

Experiential answers that question mainly with trace outcomes, evidence-grounded rubric dimensions,
calibrated model judges, production/teacher acceptance rules, and recursively verified artifact
graphs. It then continues into a second question that Bandits does not address:

> Given production traffic, model candidates, a simulation/world model, a calibrated judge, and a
> finite budget, which model should serve each request—and can selected runtime interactions become
> a bounded SFT run?

The projects are therefore **directly overlapping in their offline evidence/evaluation/data layer**
and complementary only outside that overlap. Bandits can still differentiate through executable
stateful verifiers, explicit verifier-input missingness, compact portable eval rows, and a
vendor-neutral evaluator handoff. Those are narrower distinctions than the earlier claim that
Experiential lacked an equivalent workflow.

## Capability map

| Area | Bandits (“us”) | Experiential (“them”) | Practical difference |
|---|---|---|---|
| Product center | Local CLI and immutable artifact chain from traces to tasks, evidence, verifier drafts, validation, review, eval/SFT exports | The same broad offline chain via outcomes, tasks, rubrics, calibrated judges, acceptance evidence and SFT, plus gateway/simulation/routing/training | Experiential subsumes much of Bandits' layer and continues much farther downstream |
| Trace sources | `otlp`, `chat-json`, `claude-code` | `braintrust`, `chat-json`, `langfuse`, `langsmith`, `mastra`, `otel-genai`, `otlp`, `phoenix`, `posthog`; authorized PostHog pull in Python | Experiential has much wider vendor coverage; Bandits uniquely has a named Claude Code adapter |
| Normalization posture | Preserves missingness, source digest, issues, redaction state, trace/span lineage | Preserves source-byte digest and source IDs; malformed rows remain issues; deterministic W3C-like IDs | Philosophically similar, both fail closed; Experiential is more mature and more interoperable |
| Privacy | Configurable ingest-time redaction (`secrets-only-v1`) and redaction lineage | Gateway keeps prompts/responses/tool arguments out of accounting SQLite; normalized project snapshots still contain trace content | Bandits explicitly transforms trace content; Experiential minimizes content in the serving ledger |
| Task discovery | Extracts candidate tasks/outcome evidence, clusters families, selects medoids and fit/held-out splits; users can merge/split | Deduplicates/mines representative tasks and constructs fit/held-out simulation inputs | Strong overlap; Experiential connects the result to a full simulator and router optimizer |
| Outcome model | Typed evidence with source/strength; verifier returns `pass`, `fail`, or `unknown` | Trace outcome is `success`, `failure`, `abandoned`, or `unknown`; calibrated judges score rubric dimensions; missing fit evidence is retained | Both preserve uncertainty; Bandits attaches it to verifier execution while Experiential carries it through traces, calibration and evaluation matrices |
| Success-check authoring | Drafts verifier specifications, interviews users for missing facts, executes drafts, validates on history, then separately records owner acceptance | Evidence-cited rubric proposals require successful and failed rollouts; humans can accept, edit, order, replace and finalize dimensions; default task-success axis is 0/1 | Direct overlap. Bandits additionally targets executable/state checks; Experiential's implemented authoring target is a rubric/judge |
| Validation | Checks verifier behavior against labelled/evidenced traces and records measured status; acceptance is separate | Human labels, judge-vs-human trace review, grouped out-of-fold calibration, sparse-evidence refusal/risk acceptance, held-out exclusion and immutable audits | Experiential's judge validation is currently more statistically and operationally developed; Bandits validates a broader verifier type |
| Eval export | Portable JSONL containing instruction, complete reviewed grader, and corpus/task/family/trace/verifier/validation lineage | Immutable rollout, judgment, calibration, execution-plan and held-out-report artifacts bound inside a project graph | Both retain deep lineage; Bandits' distinctive choice is a compact external eval-row interchange |
| SFT | Exports successful high-quality chat/tool trajectories; quarantines incomplete, erroring, repetitive, overlong, duplicate, or unverifiable rows | Admits recursively verified production, teacher and runtime sources; binds acceptance rules/evidence; records typed exclusions; prevents cross-split fingerprints; can dispatch/resume bounded Tinker training | Strong direct overlap. Experiential has the deeper artifact/training lifecycle; Bandits currently applies some additional demonstration-quality heuristics and emits portable chat JSONL |
| Preference/RL data | Product plan describes preference and RL-candidate outputs, but current CLI implements only eval and SFT | No general preference/DPO export found; no policy-gradient/RLVR trainer; router optimization is not model-policy RL | Neither currently delivers general RL from arbitrary traces |
| Environment | Historical replay/verifier execution only; readiness doc correctly says reset/action/observation machinery is missing for RL | Local/Harbor environment interfaces, sandbox execution, immutable-trace RAG world model, text/sandbox simulation | Experiential is materially ahead on executable/simulated environments |
| Online serving | None | Native Rust data plane; Chat Completions, Responses, Anthropic Messages; streaming, fallback, replay, guardrails | Entirely Experiential territory |
| Routing | None | Candidate selection, bounded evaluation, fitted immutable router, exact-model pools and provider fallback | Entirely Experiential territory |
| Governance/cost | Artifact lineage and reviewer acceptance | Auth identities, keys, grants, quotas, micro-USD reservation/settlement, consent gates and spend ledgers | Experiential is production-grade operational governance; Bandits is evidence governance |
| Public surface | One Python package but effectively CLI-first; package root exports only `__version__` | CLI plus a broad lazy-loaded public Python API and OpenAI client-compatible serving API | Experiential is much more embeddable and deployable |
| Packaging/maturity | v0.1.0, Python ≥3.11, 3 runtime dependencies, no repository license file found | v0.7.5, Python ≥3.11, many provider/runtime dependencies, Apache-2.0 license, wheel and release certification | Bandits is simpler to audit/install but not yet externally consumable as open source without a license |

## Where the designs genuinely overlap

### 1. Ingestion and evidence preservation

Both reject magical format detection in favor of an explicitly named source and both retain
normalization failures rather than silently skipping them. Bandits additionally applies a named
redaction ruleset before storage. Experiential supports three times as many declared source types
and has deeper vendor-specific canonicalization, model/provider identity handling, environment
capture, remote PostHog ingestion, and an explicit 100–1000 valid-trace build boundary. See
[Experiential ingest documentation](https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/docs/reference/ingest.md)
and Bandits' `bandits/ingest/` implementation.

### 2. Task mining

Both normalize traces, derive task descriptors, deduplicate similar work, and preserve fit/held-out
partitions. Bandits exposes family inspection and human merge/split commands, which makes taxonomy
correction a first-class user operation. Experiential uses the mined task set as the substrate for
RAG-backed world simulation and router evaluation. Its public build boundary is documented in
[the package API](https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/exp/__init__.py)
and its mining implementation lives under
[`exp/simulation/mining`](https://github.com/experientiallabs/experiential/tree/9f08e1aac7011c4be310f0fd2787366f1aec399d/exp/simulation/mining).

### 3. Rubrics, judges, and human calibration

This is direct, deep overlap. Experiential's `RubricProposal` requires both successful and failed
rollout evidence, span-level citations, source fit lineages, and explicit exclusion of router-held-
out lineages. `RubricReview` supports accepting, editing, reordering, replacing and finalizing cards
into an immutable `Rubric` whose status is `provisional` or `human_approved`. Its default rubric is
literally a binary “Task success” axis.

The judging system then preserves model/prompt/rubric/rollout identity, collects human labels,
reviews judge decisions axis by axis, constructs monotonic score maps, and uses grouped out-of-fold
calibration. Sparse per-dimension evidence remains insufficient unless a separate risk-acceptance
artifact authorizes proceeding. Router-held-out labels can be reported but cannot train the
calibration. This is more developed than Bandits' current validation implementation.

The remaining architectural difference is the evaluator target. Experiential's concrete path
turns the accepted rubric into a calibrated LM judge for scoring rollouts. Bandits' verifier model
also intends to cover deterministic output checks, programmatic state predicates, commands,
API/database queries and composites. Experiential's `build` command intentionally stops before
proposal/judging; that is a phase boundary, not absence of the later machinery.

### 4. SFT

Experiential is much farther downstream. It can freeze a runtime journal, construct and verify an
SFT artifact graph, render training data, estimate a full schedule, require cost authorization,
dispatch Tinker cross-entropy training, resume it, and register the trained alias. Its release notes
carefully state that no paid real Tinker training or trained-vs-base quality comparison was run for
v0.7.5 ([release scope](https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/docs/release-scope.md)).

Bandits' narrower advantage is portability and its current demonstration-selection policy: it reconstructs tool
calls across source formats, demands verifier success, rejects missing results/errors/repetition/
outliers/duplicates, emits explicit warnings, and writes every excluded trace to an unresolved
file. Experiential is at least as strict on provenance: production data needs a trusted successful
outcome or immutable human approval; teacher data binds rollout, task, judgment, calibration and
acceptance rule; calibrated scores are recomputed; accepted sources are recursively verified; and
typed exclusions plus leakage-safe partitions are persisted. Bandits does not train a model.

## What Experiential has that Bandits does not

1. **A real inference product.** A native Rust server supports OpenAI Chat Completions and
   Responses plus Anthropic Messages, streaming and non-streaming paths, tool calls, cancellation,
   refusal, idempotent replay, and provider normalization. The architecture is first-party
   documented [here](https://github.com/experientiallabs/experiential/blob/9f08e1aac7011c4be310f0fd2787366f1aec399d/docs/reference/gateway-architecture.md).
2. **Provider and tenant authority.** Provider connections, model catalogs, aliases, exact-model
   pools, identities, virtual keys, grants, revocation, monthly quotas, attempt accounting, and
   guardrails are built into the runtime.
3. **Closed-loop candidate evaluation.** It can use trace-derived tasks with a world model or
   sandbox, run multiple candidate models, judge them, fit a selection policy, evaluate held-out
   behavior, and activate the frozen router.
4. **Environment machinery.** Local process and Harbor interfaces, ledgers, cleanup rules, sandbox
   recordings, simulation clocks, and RAG-based transition retrieval provide an actual rollout
   substrate.
5. **Cost and consent controls.** Conservative estimates, command budgets, confirmation rules,
   spend ledgers, dispatch reservations, and provider-attempt attribution exist throughout.
6. **Operational hardening.** SQLite WAL/migrations, immutable catalogs, crash reconciliation,
   concurrency controls, replay, capability negotiation, release evidence, native-extension
   certification, and broad SDK/provider fixtures dwarf Bandits' current operational surface.
7. **Distribution and product UX.** Guided setup, hosted login/platform integration, provider
   selectors, an interactive gateway UI, public Python API, published package, telemetry controls,
   documentation, and Apache-2.0 licensing.

## What remains meaningfully different in Bandits

1. **Executable verifier type.** Bandits asks for the smallest defensible success check and models
   required inputs, preconditions, checks, blind spots and missing-input behavior. Its draft can be
   executed locally and may represent state predicates or commands rather than an LM score.
   Experiential's analogous mature path is rubric proposal/review plus calibrated LM judging;
   its production acceptance rules consume already-recorded trusted outcome names rather than
   synthesizing a new deterministic checker from traces.
2. **Verifier-specific `unknown`.** Experiential absolutely preserves unknown trace outcomes,
   missing calibration evidence, failed/unjudged evaluation cells and unknown operational facts.
   Bandits' narrower distinction is requiring each newly executed verifier itself to return
   `pass`, `fail`, or `unknown` based on its declared input availability.
3. **Explicit evidence-strength ordering.** Bandits separates executable state evidence,
   structured results, external evaluators, user feedback, human labels, model judgments, and
   agent self-report. Experiential distinguishes production outcomes, human approval, teacher
   judgment/calibration and runtime evidence structurally, but does not expose the same single
   cross-source strength ladder.
4. **Portable graders in exported eval rows.** Each eval row carries the complete reviewed grader
   and full derivation lineage rather than being bound to a particular simulation/router project.
5. **Different SFT heuristics and export shape.** Bandits gates repeated actions, recovery/error
   paths, family-relative length and terminal-message completeness, then emits chat-completions
   JSONL plus an unresolved sidecar. Experiential also records typed exclusions and deduplicates
   leakage groups, so explicit quarantine alone is not a differentiator.
6. **Human-editable task taxonomy.** CLI-level family merge/split makes a proposed clustering easy
   to correct without reinterpreting the entire project pipeline.
7. **Low-dependency local core.** Pydantic, Typer, and Rich are the only runtime dependencies. This
   makes Bandits easier to isolate as a small verifier/data compiler, although its Python API must
   be made intentionally public first.

## Important corrections to possible positioning

- **Do not claim Bandits is more “RL-ready.”** Today Bandits has no resettable environment, online
  rollout executor, policy optimizer, or reward-training loop. Experiential also is not a general
  RL/RLVR framework, but it has far more of the environment and evaluation substrate.
- **Do not call Experiential merely a router.** At v0.7.5 it is also a gateway, authority and spend
  ledger, trace simulator/world model, judge-calibration system, artifact graph, and SFT runner.
- **Do not claim Experiential lacks the trace-to-evaluator-to-SFT loop.** It implements outcomes,
  task mining, evidence-cited rubric proposal, human review, judge calibration, acceptance rules,
  exclusions, leakage-safe SFT construction and training orchestration. The bounded distinction is
  executable/stateful checker synthesis and portable grader export—not evaluation in general.
- **Do not sell raw source-count parity.** Experiential supports nine named formats to Bandits'
  three. Bandits should win on evidence semantics and verifier/data quality, or add adapters.
- **Do not treat file/line count as product quality.** It indicates scope and maturity burden. The
  direct evidence is feature behavior and tests, not size alone.

## Recommended product boundary

Position Bandits as **a portable executable-verifier and learning-data compiler**, while being
explicit that Experiential already covers the broader evidence/judge/SFT workflow:

```text
trace stores / gateways / agent logs
                 │
                 ▼
 Bandits: normalize → mine → gather evidence → draft/validate/review verifier
                 │
                 ├── portable evals + graders
                 ├── curated SFT trajectories
                 └── unresolved evidence + RL-environment requirements
                                      │
                                      ▼
       Experiential or another runner: calibrate judges / simulate / route / train / serve
```

That boundary is credible only if Bandits makes executable, externally runnable graders genuinely
better than Experiential's rubric/judge path. Lineage, eligibility, human review and SFT exclusions
cannot by themselves carry the positioning because Experiential implements them extensively.
Trying to match its gateway, provider adapters, runtime, router, budget authority, and training
orchestration would turn a focused project into a much larger infrastructure program.

## Roadmap implications

### Defend the differentiator

- Formalize a stable verifier protocol and plugin boundary for deterministic state checks, command
  checks, API/database probes, judges, and composites.
- Add mutation, invariance, adversarial/gaming, and held-out calibration tests promised by
  `RL_READINESS.md`; current validation is narrower than the product thesis.
- Make verifier review auditable with reviewer identity, supersession/revocation, rationale, and
  versioned acceptance policy.
- Export a small standalone evaluator runner or SDK so a Bandits eval row can score a fresh attempt
  outside the CLI.

### Close the most damaging gaps

- Add high-demand adapters in this order: OpenInference/Phoenix, Langfuse, LangSmith, Braintrust,
  then PostHog. Reuse a common vendor-observation layer rather than one-off mappings.
- Publish a supported Python API instead of exposing only `__version__` at package root.
- Add a license before presenting the repository as reusable open source.
- Add CI, a compatibility matrix, fixture provenance, and reproducible coverage/release evidence.
- Implement preference export only when comparable starting state and defensible preference are
  actually present; keep RL-candidate output a readiness manifest until reset/step/observation and
  terminal scoring are executable.

### Integrate rather than duplicate

- Define a lossless mapping between Bandits `VerifierDraft`/`ReviewedVerifier` and Experiential
  `Rubric`/`JudgeCalibration`/acceptance evidence, documenting which deterministic checks cannot be
  represented by a rubric judge.
- Accept Experiential normalized trace snapshots or project bundles as a Bandits ingest source.
- Return reviewed Bandits verifier IDs and grader payloads as external evaluation assets that an
  Experiential optimization run can bind immutably.
- Let Experiential own provider calls, cost preflight, candidate rollout, and serving; let Bandits
  own verifier evidence and dataset admission.

## Confidence and limitations

Confidence is high on repository-visible capabilities and current command surfaces. The comparison
used the complete local Bandits working tree and the exact tagged Experiential revision above,
including source, tests, docs, package metadata, and release scope. Bandits' passing test count is
from a local run on 2026-08-29. Experiential's test *inventory* was counted, but its full suite was
not executed locally because the repository includes a compiled native extension and optional/live
provider lanes; no claim is made that all upstream tests pass here. Hosted Platform behavior,
commercial traction, performance, and undocumented private services were intentionally excluded.

The local Bandits worktree contains uncommitted export/review changes, so this report reflects what
is in the IDE now, not solely commit `7568414`.
