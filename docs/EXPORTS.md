# Eval and SFT exports

Bandits exports only through an explicitly accepted verifier. Validation alone
caps a verifier at `calibrated`; `review-verifier` records the separate owner
decision that promotes one measured verifier.

A validation artifact existing is not evidence that the validation established
anything, so promotion is assessed before an owner can sign it off. Review is
refused outright unless:

- a held-out run carries a label, so agreement is measured somewhere other than
  the traces the check was drafted from;
- at least one held-out labeled run was scorable, so the check has been observed
  deciding something;
- the labels contain both a success and a failure, so agreement was reachable in
  both directions; and
- no constructed run passed the verifier without performing the task.

`--accept-risks` is the only way past a refusal, and it does not produce the same
artifact. The verifier lands on `risk_accepted` rather than `reviewed`, the
blockers are recorded on it, and their codes travel onto every export manifest
that verifier authorizes.

```bash
bandits review-verifier <verifier-draft-id> \
  --validation <validation-id> \
  --verifier <verifier-id> \
  --acceptance-id <ticket-or-review-id>
```

The returned `reviewed-verifier-*` id gates both export formats:

```bash
bandits export <task-set-id> --format eval \
  --verifier <reviewed-verifier-id> --output eval.jsonl

bandits export <task-set-id> --format sft \
  --verifier <reviewed-verifier-id> --output sft.jsonl
```

## Splits

Demonstrations and evaluation cases are drawn from opposite sides of the family
split: `--format sft` reads `fit` and `--format eval` reads `held_out`. A task
trained on and then scored measures nothing, and the split is already grouped by
lineage, so the two exports are disjoint by construction rather than by
convention. `--split fit|held_out|all` overrides it; `all` is recorded as a
manifest warning, because whatever it exports overlaps whatever the other format
drew from either side.

An empty partition produces an empty export and says so in the manifest warnings
rather than resembling a clean run with nothing to report.

## What the manifest records

Every export carries the policy it was produced under, so a row's admission can
be explained and reproduced later: the partition and how many traces it offered,
the success threshold, the step-count bound, the authorizing verifier's status,
and any risk codes it was promoted over.

The success threshold is frozen when the verifier is accepted and is not a
parameter of export. A caller re-choosing it could admit as a demonstration a run
the validation being cited recorded as a failure.

Each command writes the requested file and a sibling `<stem>.unresolved.jsonl`.
Excluded rows are never silently dropped; every quarantine row contains its
trace and family ids and one or more reasons.

## Portable eval JSONL

Each row contains an `instruction`, the complete reviewed verifier under
`grader`, and corpus/task-set/family/trace/verifier/validation lineage. The
schema is deliberately a lossless Bandits interchange contract. A harness-
specific adapter can reduce it later without having to reconstruct provenance.

Eligibility requires a task in the selected family, a prompt composed only of
`at_start` evidence, and a reviewed verifier from the same artifact lineage.

## SFT JSONL

Each row contains ordered `messages` in chat-completions shape, generating
model/scaffold metadata, any `warnings` about the demonstration, and the same
complete lineage.

The action the agent chose is a training target, so it is announced on an
`assistant` message as a `tool_calls` entry, and the result comes back on a
`tool` message carrying the matching `tool_call_id`. Recording the arguments on
the tool message instead would place them in the model's context rather than in
its output, and the example would teach the model to answer without ever calling
anything.

```json
{"role": "assistant", "content": "working on it",
 "tool_calls": [{"id": "call-refund-1-s1", "type": "function",
                 "function": {"name": "lookup_order",
                              "arguments": "{\"order_id\": \"7001\"}"}}]}
{"role": "tool", "name": "lookup_order", "tool_call_id": "call-refund-1-s1",
 "content": "{\"charged_amount\": 48.0, \"status\": \"paid\"}"}
```

Sources disagree about where a call is recorded: OTLP puts the arguments on the
tool span, while the chat-JSON and Claude Code adapters put them on the model
span that emitted the call and leave the tool span holding only the result. Both
are read, so no adapter silently exports trajectories with no actions in them.

Calls are never batched into one assistant turn, even when the source hangs them
off a shared parent span. A shared parent records only that both calls happened
inside that span; presenting a lookup and the action that depended on its result
as one parallel batch would teach an agent to commit before it has read.

Every stored artifact keeps the full message model. The emitted JSONL carries
only the keys each message actually uses, because a trainer reading
`tool_calls: []` on a user message either rejects the row or treats it as a turn
that called nothing.

### Eligibility

All eval prompt gates, plus:

- the reviewed verifier scores the trajectory at or above `0.5`;
- every required verifier input is present (`unknown` is quarantined);
- every announced call has a recorded result — a missing one is never
  serialized as an empty observation;
- span count is no more than 1.5 times the family median;
- no span records an error or recovery path;
- no exact tool action is repeated, measured on the reconstructed calls rather
  than on the tool spans, since two of the three source shapes leave a tool
  span's arguments empty and make every call look alike;
- at least one assistant target and generating model are recorded; and
- the normalized trajectory is not a near duplicate already selected.

These are conservative demonstration-quality gates. Verifier success proves an
outcome hypothesis; it does not by itself prove that the route to that outcome
is worth imitating.

### Warnings

An episode ending on a tool result is exported with a warning rather than
quarantined. Most exporters never record the agent's closing turn, so treating
its absence as disqualifying would reduce a real corpus to nothing, and the
actions leading to the verified outcome are faithful regardless. The warning
travels on the row so whoever trains on it can see what is missing.
