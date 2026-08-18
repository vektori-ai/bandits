# Traces → RL Environments

**The plan, in steps.** No code yet. Two parts: prove it on a benchmark, then point it at a
real production agent.

---

## The idea in one page

A model that fails at agentic tasks is usually missing a few specific skills, not "bad at
everything." If you can name those skills, build a small environment that trains each one, and
grade it with real code instead of an LLM's opinion, you get a much better model for much less
compute.

Traces are how you find out which skills are missing.

**The one thing to keep straight:** traces are not an environment. They are *recordings* of one.
You have to rebuild the environment from the recordings. There are three ways, and picking the
right one decides everything else:

| Approach | How it works | Verdict |
|---|---|---|
| **Replay** | Play back the recorded response for each recorded action | Breaks the moment the agent does anything new. No RL value. |
| **LLM simulator** | An LLM makes up the response to any action | Works everywhere, but the dynamics are invented and there's no ground truth left — so you're forced into an LLM judge for reward. |
| **Rebuild the state** | Figure out the database behind the responses, build a real one, reimplement the tools on top | Deterministic, works off-path, reward is a real state check. **This is what we do.** |

> The second option is the trap. It's where every trace-to-eval pipeline currently ends up,
> including WMO. Once reward is an LLM judge, the pass@k routing gate stops meaning anything —
> and that gate is the product.

**Why traces beat generic synthetic environments:** the TRACE paper (Stanford) tested this
directly. Generic synthetic envs (AWM) scored 38.4 on τ²-Bench. Envs targeted at the model's
own measured failures scored 48.2. Same compute. Targeting wins.

---

## The key concept: invocation points

Everything below depends on one thing, so it gets a name.

An **invocation point** is a single tool call recorded in a trace:

```
tool name · arguments · response · status · position in sequence
```

That one record is what turns a transcript into an environment. Each field buys a specific piece:

| Field | What it gives you |
|---|---|
| Tool name + schema | The **action space** |
| Argument values | Which parameters are actually used (usually ~5 of 38), and the value patterns for generating filler data |
| Response body | The **response shape** — and, through repeated IDs, the evidence for the database schema |
| Error responses | The **failure modes** the environment must reproduce. An env that can't return `not_found` trains an agent that has never seen adversity. |
| Position in sequence | Reads *before* the first write = the starting state. A write followed by a read on the same ID = the write-tool/read-tool relation. |
| State change across a write | What the **verifier checks** |

Take invocation points away and every row above goes blank. You're left with instruction text and
a final answer — which is exactly the input WMO kept, and exactly why WMO ends at an LLM judge.
That isn't a design preference on their part. It's forced by the data they retained.

> **The chain:** no invocation points → no action space → no state → no verifier → judge.

So this is the **first thing to check on any customer's telemetry**, before promising anything.
It shows up again in Step 6 (getting them), Step 7 (reading them), and Step 9 (creating the ones
the traces are missing).

---

## Two kinds of environment

We will build both. They do different jobs and it's important not to mix them up.

**Type A — Skill environment.** Small, isolated, trains one missing skill. Doesn't copy the
customer's world, just their tool interface. Tasks are generated from random seeds, so you get
unlimited instances. This is what TRACE builds.

**Type B — Mirror environment.** Rebuilds the customer's actual world from their traces. Fewer
tasks, but it looks like their real system. This is what proves to a customer that we understand
their agent, and it's what eval runs on.

Type A gives you training volume. Type B gives you credibility and honest evaluation.

---

# PART 1 — Prove it on a benchmark

Do this first. On a benchmark, every task comes with a pass/fail label already attached. That
removes the single hardest problem (see Step 8) and lets us prove the machinery works before
adding it back.

**Start with τ²-Bench.** 164 tasks (50 airline + 114 retail). Tool use, a database, and a
simulated user — the exact shape of a production customer-service agent. Both AWM and TRACE
published numbers on it, so we have something to beat. SWE-bench Verified comes second, since
our repo mining already lives there.

### Step 1 — Roll out and collect

Run the base model on the benchmark. Save every trajectory with its pass/fail label. This is our
trace corpus, with labels included for free.

### Step 2 — Find the missing skills

Split trajectories into passed and failed. Then:

1. **Discover** — read across all trajectories and write a fixed list of candidate skills, each
   with a name and description. Do this once so the names stay stable.
2. **Label** — for every trajectory and every skill, mark one of: `NOT NEEDED`, `PRESENT`,
   `MISSING`.
3. **Score** — for each skill, compute how much more often it's missing in failures than in
   successes. TRACE's thresholds: keep a skill only if that gap is ≥ 20 points **and** it
   explains ≥ 10% of all failures.
4. **Repeat** the labeling a few times and keep only skills that survive every run.

The gap is what matters, not the raw failure count. A skill that's missing in failures *and*
successes equally isn't a deficit — it's noise or an ambiguous task.

**Known weak point:** this labeling is done by an LLM. Reward stays hard, but *choosing what to
train* is judge-driven. That's a deliberate tradeoff, not something we should inherit without
noticing. Flag it, measure how stable it is across runs, revisit later.

### Step 3 — Build one environment per skill

For each surviving skill, generate an environment that:

- **Keeps the target's interface.** Same tool schemas, same protocol, same output format. The
  agent shouldn't be able to tell it's somewhere else.
- **Makes the skill mandatory.** Every generated task must be unsolvable without it.
- **Generates from a seed.** Seed → user profile, database records, task parameters, *and the
  correct answer*. Every seed is a different scenario, so the model can't memorize.
- **Grades with code.** Compare the final database state against the known-correct state. Hash
  check. Plus verify the agent told the user the right thing.

### Step 4 — Train

One LoRA adapter per skill, GRPO, base model frozen. TRACE used 2–4 adapters at ~5% of
parameters each, 40 iterations, 4–8 A100s.

### Step 5 — Combine and measure

Train a small gate that picks which adapter to use per task (TRACE's MoE gate is ~490k
parameters — cheap). Then measure back on the original benchmark.

**Targets to beat:** base 32.9, GRPO-on-target 37.8, AWM 38.4, GEPA 39.6, TRACE 48.2.

If we can't reproduce roughly this, stop and fix it before touching real traces.

---

# PART 2 — Point it at a real production agent

Everything above assumed labeled tasks handed to us. Production gives us neither labels nor a
ready-made environment. Here's what changes.

### Step 6 — Get two things, not one

People conflate these. They're different.

**The tool list** — the complete set of things the agent *can* do. From the MCP `list_tools`
call, an OpenAPI spec, or the function schemas in the request body. Complete, but abstract.

**The traces** — what was actually called, with what values, returning what. Real, but partial.

Traces alone → an environment with holes exactly where production is thin, and production is
thin on failure paths, which is where the training signal lives. Tool list alone → AWM: a clean
invented world nobody uses.

**Watch out:** most stacks log LLM calls, not tool calls. You can usually recover tool calls
from the assistant message's `tool_calls` blocks plus the matching `tool`-role replies. If you
can't recover them at all — stop. You have text, and text only supports the LLM-simulator path.

### Step 7 — Figure out the database behind the tools

This is the hard technical step. You have a stream of `(tool, arguments, response)`. You need
the state that makes them all consistent.

**The trick is that IDs repeat.** `create_order` returns `order_id: 7741`. Later
`get_order(7741)` returns a body. Later `list_orders(customer=88)` returns an array containing
7741. That repetition tells you:

- There's an `orders` table.
- Its columns are the union of every field any response ever showed for it.
- `create_order` writes to it; `get_order` and `list_orders` read from it.
- Foreign keys, wherever an ID shows up as a field in one place and a primary key in another.

Then sort every tool into one of three buckets, because each is handled differently:

| Bucket | Example | What we do |
|---|---|---|
| **Read** | `get_order` | Query the rebuilt database |
| **Write** | `refund_order` | Change the rebuilt database — *this is what reward checks* |
| **External** | `send_email`, `charge_card` | Fake the response, but write the attempt to an **effect log** so reward can check "an email was sent" without sending one |

The effect log isn't optional. For a lot of production agents, success lives entirely in effects,
not in stored data.

**Where this fails:** a table that's only ever read and never cross-referenced can't be properly
modeled. Make it a fixed snapshot and say so in the manifest. Don't invent structure.

### Step 8 — Get outcome labels (the hardest problem)

A trace tells you what the agent *did*. It does not tell you whether that was *right*.

> **"Do what production did" is not a reward function.** It teaches the model to copy the old
> system's mistakes, and it looks like it's working, because agreement is highest on the easy
> traces.

You need a label from outside the trace. Best to worst:

1. **A real downstream signal** — ticket closed and not reopened in 7 days, refund not reversed,
   no human escalation, payment settled, thumbs-up. This usually already exists in the customer's
   own systems and **nobody has ever joined it to their traces.** Ask for this first, every time.
2. **Human review of a sample** — expensive, but this is the part the LangChain eval-engineering
   framing gets right: human feedback isn't a nice-to-have bolted on, it's *where reward comes
   from* when signal 1 doesn't exist.
3. **Throw the trace away.** Not every trace becomes a task. Low yield is a correct outcome, not
   a bug. Our repo mining runs ~10%; expect similar.

### Step 9 — Fill in the gaps by calling the tools directly

Traces are a biased sample. Where they're silent about a tool — never called, or only ever with
default arguments — call it yourself to learn its response shape, its errors, and its validation
limits. Valid call, missing field, wrong type, nonexistent ID, boundary values.

**Rules, non-negotiable:**

- **Staging only.** Never production.
- **Written per-tool authorization**, listed in a manifest.
- **Sort tools into read/write/external *before* probing**, not after. You do not find out a tool
  was `charge_card` by charging a card.
- Record every probe as provenance.

Design the pipeline to work **without** this step. It improves coverage; it must not be
load-bearing, because plenty of security reviews will say no.

### Step 10 — Build the starting state per task

Neither reference handles this, because neither has to. AWM writes the task first and the
database second, so they always match. We get the task from a trace and have to build a database
that fits it.

If the task says *"refund order 7741"*, then order 7741 has to exist, belong to the right
customer, and be refundable. Otherwise every rollout fails identically — which at the pass@k
gate looks exactly like "beyond the model's ability." That would manufacture the very result the
gate exists to test for.

**So the starting state is per-task, not one shared fixture.** And the rule is clean:

> The trace tells you what the world looked like at the start, **because the agent read it.**
> Every read *before* the first write is evidence of the starting state. Every read *after* a
> write is not — don't let it leak backward.

Around that, generate filler rows that match the inferred schema and the observed value
patterns, so search and list tools aren't trivially easy.

**Copy the shapes, never the values.** Real values drag PII into an artifact we intend to keep,
share and train on — and they let the model memorize instead of learn.

### Step 11 — Write the verifier

Reward is an assertion over the final state and the effect log. Never a judge.

```
orders[7741].status == 'refunded'
  AND orders[7741].refund_amount == 4200
  AND effects.emails_to(customer_88) == 1
  AND inventory unchanged
```

Note that last line. Include the things that must **stay the same**, not just what must change,
or the agent learns that collateral damage is fine as long as the target field is right.

Generate it once from the state diff of a trace with a positive outcome label, then **freeze it
and have a person read it** before it grades anything. A generated reward function nobody has
read is an unexamined reward function.

**Anti-cheat**, which ports straight over from our existing `check-env --reward-hack`:

- Writing the database directly instead of calling the tool → must fail.
- Reading the verifier, the effect log, or the answer from inside the episode → must fail.
- Network access from inside the episode → must fail.
- The honest control and the cheating control must both still score as expected.

A rebuilt environment is simpler than reality, so it has exploits reality doesn't. Assume they're
there and go looking.

### Step 12 — The fidelity gate (the accept/reject test)

Before any Type B environment is allowed near training: **replay its own source trace against
it.** Feed the recorded actions in order and compare each response to the recorded one.

- Exact match on IDs, statuses, and structure.
- Allow differences in timestamps, generated IDs, unordered collections, free text.
- **Report per tool, not as one number.** One tool at 40% divergence tells you exactly what to
  fix. An average tells you nothing.

An environment that can't reproduce its own source trace is not a model of anything. Reject it
and go back to Step 7.

Neither reference does this and neither can — AWM has no trace to replay, WMO's text engine
can't execute an action at all.

**This is also the best thing we can show a customer.** "Your rebuilt environment reproduces 94%
of your recorded production trajectories, here's the per-tool breakdown" is a claim their
engineers can check themselves. Far stronger than any eval score, because it's a statement about
*their* system, not about a model.

### Step 13 — Hand off to what we already have

Emit as a Harbor task. Nothing downstream changes:

```
environment + task + frozen verifier
   → pass@k sweep
   → routing decision (RL / OPD / neither)
   → OPD loop or GRPO branch
   → non-regression check
```

This is the discipline that keeps the whole thing from becoming a side quest. The trace pipeline
is **a new source of tasks feeding an existing gate**, not a new product with its own new way of
measuring success.

It also closes a real gap: repo mining needs a linked GitHub issue, which is why yield sits near
10%. Traces need no linked issue — the user's request *is* the task, and the outcome label *is*
the grade.

---

## The dependency chain

```
Tool list ──────┐
                ├─→ [7] Rebuild database ──→ [10] Per-task state ──┐
Traces ─────────┤        ↑                                          │
                │        └── [9] Probing (optional)                 ▼
Outcome labels ─┴─→ [8] Task + label ────→ [11] Verifier ──→ [12] FIDELITY GATE
                                                                    │
                                                                    ▼
                                                      [13] pass@k → routing → training
```

**Critical path:** recoverable tool calls → rebuilt database → fidelity gate. If any one of those
three fails, there's no product. Everything else is refinement.

**Start chasing outcome labels at the same time as traces**, not after — they come from a
completely different system and a completely different person at the customer.

---

## What will go wrong

**Traces only cover the happy path.** The failures we want to train on are underrepresented.
*Fix, and this is the whole payoff of rebuilding instead of replaying:* once the environment
actually runs, we're not limited to what happened. Make the order non-refundable, make the
customer unverified, make the API rate-limit — generate situations that never occurred in
production but are entirely reachable in the real system. A replay environment can never do this.
This is where task volume actually comes from.

**Long trajectories.** Real agent runs are long, and OPD's dense signal may not survive the
horizon — thunlp's own caveat, which we already carry. Trace-derived tasks will be *longer* than
mined repo tasks, so this gets worse, not better. Note that TRACE ran at 32k context and 50 steps,
so their results don't tell us anything about this.

**Exploits.** A rebuilt world is simpler than the real one and admits strategies the real one
doesn't. Step 11's guards are necessary but not sufficient.

**PII.** Traces are the most sensitive thing a company owns — prompts, customer names, order
contents, internal responses. Shapes-not-values, scrub before storing, everything stays local.

**Underdetermined schemas.** Some tables can't be inferred properly. Degrade to fixed snapshots
and label them honestly instead of inventing structure.

---

## What to reuse vs. what to build

| Piece | Take it from | Why |
|---|---|---|
| Trace loading | **WMO** `simulation/ingest/` | 11 vendor adapters, no format guessing, source digests, proper call-ID pairing. Good and boring — don't rebuild. |
| Task deduplication | **WMO** `simulation/mining/` | Hashing embedder, grouping, fit/held-out split, no paid API calls. |
| Database + env + verifier generation | **AWM** `core/{db,sample,spec,env,verifier}.py` | The schema → sample data → interface → generated env → state-diff verifier chain. |
| Skill discovery + targeted envs | **TRACE** | Contrastive labeling, the 20pt/10% thresholds, per-skill LoRA + MoE gate. |
| Task emission, anti-cheat, pass@k, routing | **vektori-trace** | Already built. Unchanged. |

**What's actually new — and it's a narrow band:**

1. Sorting tools into read / write / external from traces.
2. Inferring the database schema from repeated IDs.
3. Building the starting state per task from the trace's own reads.
4. Writing state-diff + effect-log verifiers from labeled traces.
5. **The fidelity gate.**

Number 5 has no prior art anywhere, is the accept/reject test for everything else, and is the
one number a customer can verify without trusting us.

---

## The thesis, in one paragraph

> Text traces give you tasks and nothing else — which is why every trace-to-eval pipeline today
> ends in an LLM judge. **Tool calls give you the action space, the response shapes, and the
> state changes**, which together are enough to *rebuild* the world instead of simulating it.
> Rebuild it, prove the rebuild by replaying the traces it came from, and reward becomes a fact
> instead of an opinion. Every company running a production agent already has the raw material
> and is dumping it into a logging bucket.
