# Bandits: Traces to Verifiers and Learning Data

Status: focused product and implementation plan  
Scope: domain-agnostic trace analysis, draft verifiers, and learning-data export

## 1. What Bandits does

People already have traces from agents doing real work. Those traces contain
tasks, actions, observations, mistakes, and sometimes evidence of success, but
they are not directly usable for evaluation or training.

Bandits turns them into concrete learning assets:

```text
agent traces
    -> candidate tasks
    -> evidence of success and failure
    -> draft verifier
    -> verifier validation against past traces
    -> eval and training-data exports
```

The one-sentence product definition is:

> Bandits reads agent traces, drafts a way to check whether the agent succeeded,
> and turns the verified runs into reusable evaluation and training data.

Bandits is not limited to coding agents. A trace may describe customer support,
browser automation, API workflows, research, data analysis, coding, or another
tool-using process. Domain-specific checks differ, but the workflow is the same.

## 2. Why someone uses it

A team has hundreds or thousands of traces. They want to improve their agent,
but first they must answer:

- What repeatable tasks are represented?
- Which runs actually succeeded?
- How could success be checked automatically?
- Which traces are safe and useful for training?
- Which training method can each trace support?

Today, teams usually inspect traces manually, label examples, write evaluators,
and reshape the data by hand. General evaluation platforms can store traces and
promote selected runs into datasets. Bandits should focus on the harder step in
the middle: proposing and testing the success check.

The minimum useful result is not `RL-ready: 72%`. It is:

```text
Task family: refund an eligible order

Observed task:
  Refund order 7741

Draft verifier:
  Query the order after the run.
  Pass when status == "refunded" and refunded_amount == charged_amount.

Historical validation:
  Agrees with 18 known successful runs.
  Rejects 7 known failed runs.
  Cannot score 4 runs because the final order state was not captured.

Exports:
  18 SFT examples
  5 success/failure preference pairs
  25 evaluation cases
  RL environment candidate blocked: reset procedure is unknown
```

## 3. The core product loop

### Step 1: ingest without losing evidence

Normalize source-specific logs while preserving:

- the user task and relevant context;
- ordered model and tool actions;
- tool arguments and results;
- errors, exit codes, and timeouts;
- final output;
- externally recorded scores or feedback;
- run, agent, model, and environment metadata;
- source location and digest;
- explicit markers for truncation, redaction, and importer limitations.

Missing information must remain missing. Bandits must never convert an absent
tool result into an empty successful result.

### Step 2: extract candidate tasks

For each trace, produce a task record:

```text
task input
starting context or state reference
available actions/tools
observed trajectory
final output or state
existing outcome evidence
```

Then group sufficiently similar tasks into task families. Task grouping helps
Bandits find repeated checks and comparable success/failure examples, but each
grouping remains a proposal that users can split or merge.

### Step 3: collect outcome evidence

Outcome evidence can come from:

- a structured final state;
- a test or command result;
- an API or database response;
- an existing evaluator score;
- user acceptance or correction;
- a human label;
- an agent's own claim;
- a model-based judgement.

These sources are not equally trustworthy. Bandits records the source and
strength of each claim instead of reducing everything immediately to pass/fail.

Suggested evidence ordering:

```text
executable state check
  > structured external result
  > existing trusted evaluator
  > explicit user feedback
  > reviewed human label
  > model judgement
  > agent self-report
```

### Step 4: draft a verifier

Bandits proposes the smallest check that appears to represent task success.

Verifier types may include:

- exact or schema-constrained output checks;
- programmatic state predicates;
- test commands;
- API or database queries;
- before/after state comparisons;
- rule-based trajectory checks;
- rubric-based model judges;
- combinations of deterministic checks and judges;
- human review when automation is not defensible.

The draft must contain its inputs, preconditions, checks, pass/fail/unknown
behavior, supporting evidence, blind spots, possible reward-hacking paths, and
required external systems.

Example interface:

```python
class VerificationResult(BaseModel):
    outcome: Literal["pass", "fail", "unknown"]
    score: float | None
    details: dict[str, object]
    evidence: tuple[str, ...]


def verify(
    task: Task,
    initial_state: State | None,
    final_state: State | None,
    trajectory: tuple[Event, ...],
) -> VerificationResult:
    ...
```

`unknown` is essential. Missing data should not be treated as failure, and a
verifier should fail closed when its required evidence is unavailable.

### Step 5: test the verifier against history

A generated verifier is a hypothesis. Bandits should test it before recommending
it.

Validation includes:

- agreement with existing strong outcome evidence;
- disagreement analysis;
- known-success and known-failure examples;
- mutation tests that alter a relevant result;
- invariance tests that alter irrelevant details;
- missing-input behavior;
- attempts to satisfy the check without completing the intended task;
- separate calibration and validation trace sets where volume permits.

The verifier receives a status:

```text
suggested       plausible idea, not executed
executable      can run, but is not calibrated
calibrated      tested against historical examples
reviewed        accepted by a human owner
rejected        contradicted or easily gamed
```

Bandits should never call a verifier trusted solely because generated code runs.

### Step 6: materialize the supported assets

Different evidence supports different outputs.

#### Evaluation case

Requires a task and some means of judging a new attempt. This may use a reviewed
automatic verifier or human rubric.

#### Supervised fine-tuning example

Requires a successful, high-quality trajectory. Output is the task paired with
the selected action/message sequence in a target training format.

#### Preference example

Requires comparable attempts at the same task or starting condition, plus a
defensible preference. Output contains chosen and rejected trajectories with the
reason for the preference.

#### RL/RLVR candidate

Requires more than historical data:

- a task that can be instantiated repeatedly;
- an environment that can reset;
- actions that can execute;
- observations returned after new actions;
- a verifier that can score new terminal states;
- acceptable safety and cost boundaries.

Bandits may output the environment interface and missing components, but traces
alone cannot always provide the transition system.

## 4. What Bandits returns

For a corpus, Bandits produces a project directory:

```text
bandits-output/
  summary.md
  task_families.json
  evidence.jsonl
  verifiers/
    refund_order/
      verifier_spec.yaml
      verifier.py          optional draft
      test_verifier.py
      validation.json
  datasets/
    eval.jsonl
    sft.jsonl
    preference.jsonl
    rl_candidates.jsonl
  unresolved.jsonl
```

Every output row links back to its source trace and records whether values were
observed directly, deterministically derived, inferred by a model, or supplied
or reviewed by a human.

## 5. End-to-end example

This example uses a support agent because it exercises the same machinery as
coding or browser traces without making the product coding-specific.

### 5.1 Source trace

```json
{
  "trace_id": "run-104",
  "task": "Refund order 7741",
  "spans": [
    {
      "kind": "tool",
      "name": "lookup_order",
      "arguments": {"order_id": "7741"},
      "output": {
        "status": "paid",
        "charged_amount": 48.0,
        "refundable": true
      }
    },
    {
      "kind": "tool",
      "name": "refund_order",
      "arguments": {"order_id": "7741", "amount": 48.0},
      "output": {"status": "refunded", "refunded_amount": 48.0}
    }
  ]
}
```

Bandits preserves this as evidence. It does not immediately decide that the run
succeeded just because the last tool used the word `refunded`.

### 5.2 Extracted task candidate

```json
{
  "task_id": "task-run-104",
  "trace_id": "run-104",
  "instruction": "Refund order 7741",
  "initial_state": {
    "order_id": "7741",
    "status": "paid",
    "charged_amount": 48.0,
    "refundable": true
  },
  "observed_actions": [
    {"tool": "lookup_order", "arguments": {"order_id": "7741"}},
    {
      "tool": "refund_order",
      "arguments": {"order_id": "7741", "amount": 48.0}
    }
  ],
  "observed_final_state": {
    "status": "refunded",
    "refunded_amount": 48.0
  },
  "outcome": "unknown",
  "outcome_evidence": ["ev-run-104-refund-result"]
}
```

The task extraction makes the input, relevant state, actions, and result
explicit. It remains reviewable independently of verifier generation.

### 5.3 Verifier specification

Across similar traces, Bandits observes that successful refunds end with a
refunded order and that the refunded amount matches the charged amount:

```yaml
schema_version: 1
verifier_id: refund-order-v1
task_family: refund_order
status: suggested

inputs:
  required:
    - initial_state.charged_amount
    - final_state.status
    - final_state.refunded_amount

checks:
  - final_state.status == "refunded"
  - final_state.refunded_amount == initial_state.charged_amount

unknown_when:
  - initial or final order state is missing
  - amounts use currencies that cannot be compared

blind_spots:
  - trace does not prove that the refund reached the payment processor
  - a duplicate refund may still satisfy the final-state predicate

possible_gaming:
  - changing a cached order object without changing the authoritative record

evidence:
  supporting: [ev-run-104-refund-result, ev-run-118-refund-result]
  contradicting: [ev-run-121-payment-pending]
```

The specification is the review boundary. A user can correct it before Bandits
generates or runs code.

### 5.4 Generated verifier draft

For a structured-state predicate, Bandits can use a constrained template:

```python
from decimal import Decimal, InvalidOperation


def verify(task, initial_state, final_state, trajectory):
    if initial_state is None or final_state is None:
        return {
            "outcome": "unknown",
            "score": None,
            "details": {"reason": "initial or final order state missing"},
        }

    try:
        charged = Decimal(str(initial_state["charged_amount"]))
        refunded = Decimal(str(final_state["refunded_amount"]))
        status = final_state["status"]
    except (KeyError, InvalidOperation):
        return {
            "outcome": "unknown",
            "score": None,
            "details": {"reason": "required refund fields are unreadable"},
        }

    passed = status == "refunded" and refunded == charged
    return {
        "outcome": "pass" if passed else "fail",
        "score": 1.0 if passed else 0.0,
        "details": {
            "status": status,
            "charged_amount": str(charged),
            "refunded_amount": str(refunded),
        },
    }
```

In the implementation this uses typed Bandits contracts and runs in a restricted
verifier runner. Its manifest links generated source to the verifier and
evidence IDs.

### 5.5 Generated verifier tests

Bandits creates tests from observed examples and mutations:

```python
def test_full_refund_passes():
    result = verify(
        task={"order_id": "7741"},
        initial_state={"charged_amount": 48.0},
        final_state={"status": "refunded", "refunded_amount": 48.0},
        trajectory=(),
    )
    assert result["outcome"] == "pass"


def test_partial_refund_fails():
    result = verify(
        task={"order_id": "7741"},
        initial_state={"charged_amount": 48.0},
        final_state={"status": "refunded", "refunded_amount": 20.0},
        trajectory=(),
    )
    assert result["outcome"] == "fail"


def test_missing_final_state_is_unknown():
    result = verify(
        task={"order_id": "7741"},
        initial_state={"charged_amount": 48.0},
        final_state=None,
        trajectory=(),
    )
    assert result["outcome"] == "unknown"
```

Passing these tests means the code matches its specification. It does not prove
the specification represents the business's definition of success.

### 5.6 Historical validation

Bandits runs the draft over traces whose outcomes have stronger evidence:

```json
{
  "verifier_id": "refund-order-v1",
  "status": "calibrated",
  "runs": 29,
  "agreement": {"pass": 18, "fail": 7},
  "disagreement": 1,
  "unknown": 3,
  "counterexamples": [
    {
      "trace_id": "run-121",
      "verifier": "pass",
      "known_outcome": "failure",
      "reason": "payment processor remained pending"
    }
  ],
  "recommendation": "Add authoritative payment status to the verifier"
}
```

The counterexample matters more than the aggregate agreement rate: it shows how
the draft would reward the wrong behavior.

### 5.7 Learning-data exports

After review, one trace may produce several assets.

Evaluation case:

```json
{
  "id": "eval-refund-7741",
  "input": {"instruction": "Refund order 7741", "order_id": "7741"},
  "verifier_id": "refund-order-v2",
  "source_trace_id": "run-104"
}
```

SFT example:

```json
{
  "messages": [
    {"role": "user", "content": "Refund order 7741"},
    {
      "role": "assistant",
      "tool_calls": [
        {"name": "lookup_order", "arguments": {"order_id": "7741"}}
      ]
    },
    {"role": "tool", "name": "lookup_order", "content": "{...}"}
  ],
  "provenance": {"trace_id": "run-104", "verifier_id": "refund-order-v2"}
}
```

Preference example, only when both runs began from comparable states:

```json
{
  "prompt": {"instruction": "Refund order 7741"},
  "chosen_trace_id": "run-104",
  "rejected_trace_id": "run-109",
  "preference_evidence": {
    "verifier_id": "refund-order-v2",
    "chosen_outcome": "pass",
    "rejected_outcome": "fail"
  }
}
```

RL candidate handoff:

```yaml
task_family: refund_order
reset:
  status: unresolved
  need: safe way to restore an eligible paid order
actions:
  status: observed
  tools: [lookup_order, refund_order]
observations:
  status: observed
verifier:
  status: reviewed
  id: refund-order-v2
environment_status: blocked
```

Bandits exports SFT and eval data now, but does not pretend that a resettable
refund environment exists.

## 6. Domain independence

Bandits should have a domain-independent core and small domain extensions.

The core understands tasks, actions, observations, state references, outcomes,
verifier contracts, evidence provenance, and dataset transformations.

Domain extensions understand possible checks:

```text
coding       tests, build output, repository diff
browser      URL, DOM state, downloaded artifact
support      ticket state, policy compliance, user resolution
commerce     order/payment/inventory state
data         query result, schema, row-level invariants
research     citation validity, coverage rubric, factual checks
API agent    response schema and downstream resource state
```

An extension supplies evidence extractors and verifier templates. It must not
hard-code one company's task semantics into the canonical trace model. Unknown
domains can still use generic structured checks, model rubrics, or human review.

## 7. Human involvement

The realistic workflow is assisted automation, not zero-click truth discovery.

Bandits should ask humans to review the highest-leverage decisions:

- Is this a coherent task family?
- Does the verifier represent the actual business intention?
- Are these successful trajectories worth imitating?
- Is this preference caused by agent quality rather than environment differences?
- Can the underlying system be reset safely?

A good product makes these decisions fast and evidence-rich. It does not hide
them behind a confidence score.

## 8. CLI workflow

```bash
# Normalize source traces.
bandits ingest traces.jsonl --source otlp

# Discover tasks, outcomes, and possible checks.
bandits analyze corpus-abc

# Inspect and edit a proposed task family.
bandits show family-refund-order

# Draft and test a verifier.
bandits draft-verifier family-refund-order
bandits validate-verifier family-refund-order

# Export only assets supported by the evidence.
bandits export corpus-abc --format eval
bandits export corpus-abc --format sft
bandits export corpus-abc --format preference
bandits export corpus-abc --format rl-candidates
```

The first implementation does not need every command. They define the intended
workflow and keep ingestion, inference, validation, and export separate.

## 9. Core data contracts

The existing immutable `TraceCorpus` remains source evidence. Derived artifacts
live separately so changing an analysis policy does not change the corpus.

Conceptual additions:

```python
class TaskCandidate:
    task_id: str
    trace_id: str
    instruction: str
    initial_state_ref: str | None
    trajectory_span_ids: tuple[str, ...]
    final_state_ref: str | None
    outcome_evidence_ids: tuple[str, ...]


class Evidence:
    evidence_id: str
    provenance: Literal["observed", "derived", "model", "human"]
    strength: Literal["strong", "moderate", "weak"]
    claim: str
    trace_id: str
    span_id: str | None
    value: object


class TaskFamily:
    family_id: str
    name: str
    task_ids: tuple[str, ...]
    common_action_schema: dict[str, object]
    proposed_by: Literal["rule", "model", "human"]
    review_status: str


class VerifierDraft:
    verifier_id: str
    family_id: str
    spec: dict[str, object]
    code_path: str | None
    evidence_ids: tuple[str, ...]
    blind_spots: tuple[str, ...]
    status: str


class LearningAsset:
    kind: Literal["eval", "sft", "preference", "rl_candidate"]
    source_trace_ids: tuple[str, ...]
    verifier_id: str | None
    payload: dict[str, object]
    warnings: tuple[str, ...]
```

## 10. Technical architecture

```text
bandits/
  ingest/              source adapters
  traces.py            immutable normalized evidence
  store.py             content-addressed artifacts
  analyze/
    tasks.py            task extraction
    outcomes.py         outcome evidence extraction
    families.py         task grouping
    models.py           derived artifact contracts
  verify/
    candidates.py       verifier strategy selection
    draft.py            specification and optional code generation
    validate.py         historical and mutation validation
    templates/          domain extensions
  export/
    eval.py
    sft.py
    preference.py
    rl_candidate.py
```

Deterministic extraction runs first. Model assistance is appropriate for task
grouping, semantic outcome interpretation, and verifier drafting, but all
model-produced claims are labelled and linked to the prompt and model version.

### 10.1 Mapping from the current repository

The existing code is the ingestion foundation, not disposable prototype work:

| Current component | Keep | Next responsibility |
|---|---|---|
| `bandits/traces.py` | Yes | Preserve normalized source evidence and add only source-level metadata |
| `bandits/ingest/` | Yes | Declare importer capabilities and retain tool results, errors, and outcome signals |
| `bandits/store.py` | Yes | Store derived analysis artifacts beside immutable corpora |
| `bandits/cli.py` | Yes | Add `analyze`, then verifier and export commands incrementally |

Do not put inferred outcomes or verifier decisions directly on `Trace`. The same
trace may be reassessed with a newer policy or corrected by a reviewer without
changing the evidence artifact.

### 10.2 First internal API

The first vertical slice can remain deliberately small:

```python
from bandits.analyze import analyze_corpus

corpus = store.read("corpus-abc")
analysis = analyze_corpus(corpus)

for task in analysis.tasks:
    print(task.instruction)
    print(task.outcome_evidence)
    print(task.verifier_candidates)
```

Conceptually:

```python
def analyze_corpus(corpus: TraceCorpus) -> CorpusAnalysis:
    tasks = tuple(extract_task(trace) for trace in corpus.traces)
    evidence = extract_outcome_evidence(corpus)
    candidates = propose_deterministic_verifiers(tasks, evidence)
    return CorpusAnalysis(
        corpus_digest=compute_corpus_digest(corpus),
        tasks=tasks,
        evidence=evidence,
        verifier_candidates=candidates,
    )
```

The first version should propose only verifiers supported by deterministic
patterns. Model-assisted drafting comes after this data flow is stable.

### 10.3 Artifact lifecycle

```text
corpus artifact (immutable source evidence)
    |
    +-- analysis artifact v1
          |
          +-- verifier draft v1
          |     |
          |     +-- validation run v1
          |     +-- human review v1
          |
          +-- dataset export v1
```

Every child records its parent ID. Updating a verifier produces a new artifact;
it does not rewrite the earlier analysis or corpus.

## 11. Build plan

### Milestone 1: one trace becomes one task record

Add the source fidelity needed to extract the instruction, actions,
observations, final output, outcome evidence, and importer limitations. Use
several domains in fixtures so the core does not become coding-specific.

Proof: one trace emits an auditable `TaskCandidate` without losing or inventing
evidence.

### Milestone 2: outcome evidence and verifier specifications

Detect exit codes, tests, structured field predicates, exact expected outputs,
existing scores, user feedback, and before/after state differences. Emit typed
verifier specifications, not code, first.

Proof: domain owners understand the proposed check and its weakness.

### Milestone 3: task families and comparable attempts

Group repeated tasks conservatively, support human split/merge corrections, and
identify successful, failed, and unknown attempts under comparable conditions.

Proof: real corpora yield meaningful families without mixing different tasks.

### Milestone 4: draft verifier code

Generate code only from typed specifications. Begin with constrained templates:
JSON/schema predicates, command runners, state-diff predicates, composed checks,
and rubric-judge adapters. Run drafts in isolated tests with timeouts.

Proof: drafts pass contract tests and return `unknown` when evidence is missing.

### Milestone 5: verifier validation

Run drafts across historical outcomes, holdouts, and mutations. Report
agreements, disagreements, unknowns, and concrete counterexamples. Require human
acceptance before marking a verifier reviewed.

Proof: Bandits surfaces plausible false positives rather than only an accuracy
number.

### Milestone 6: eval and SFT export

Export reviewed tasks with verifier references as evals and reviewed successful
trajectories as SFT examples. Support a few explicit target schemas.

Proof: exports validate against their schemas and retain source provenance.

### Milestone 7: preference export

Create chosen/rejected pairs only when attempts share a comparable task and
starting condition. Attach evidence explaining the preference.

Proof: no pair depends solely on agent self-report or incomparable states.

### Milestone 8: RL candidate handoff

When reset, execution, observation, and verifier evidence exist, emit task rows,
reset requirements, action/observation schemas, the reviewed verifier adapter,
missing environment components, and example trajectories.

Proof: an environment engineer can decide what remains without rereading the
source traces. This is a handoff package, not a claim that arbitrary environments
can be reconstructed from logs.

## 12. What not to build first

- A dashboard.
- General trace search competing with observability platforms.
- A universal environment generator.
- A learned world model.
- RL training infrastructure.
- Automatic dense reward shaping.
- Dozens of ingestion formats.
- A verifier marketplace.

None of these proves the central value.

## 13. First product test

Before implementing the full plan, take three real corpora from different
domains and manually produce:

1. five task candidates;
2. one task family;
3. one verifier specification;
4. one executable draft where possible;
5. a validation table against past runs;
6. eval, SFT, preference, and RL-candidate eligibility for each trace.

The key question is:

> Does the draft verifier save meaningful human work, and would a domain owner
> trust it after reviewing the evidence and counterexamples?

If the answer is no, more architecture will not rescue the product. If the
answer is yes across more than one domain, Bandits has a credible core.

## 14. Product promise and boundary

The promise:

> Give Bandits your agent traces. It will recover candidate tasks, draft and test
> ways to verify success, and export the runs that are usable for evaluation or
> training.

The boundary:

> Bandits proposes and validates from available evidence. A reviewed verifier or
> real environment remains the authority for correctness.
