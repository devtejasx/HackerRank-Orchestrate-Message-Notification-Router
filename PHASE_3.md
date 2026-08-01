# Phase 3 — Personalisation: Routing Signals

Turns Phase 2 output plus repository context into **ten independent,
normalised, explained routing signals**, personalised to the receiving user.

**Scope:** signals only. Phase 3 makes no routing decision. No `notify`,
`digest` or `mute`, no `output.csv`, no evidence retrieval, no LLMs, OCR, ASR
or embeddings. Those are Phase 4 and beyond.

The brief's framing: *"the routing decision must be personalized to the
receiving user"*. So the same message reaching two different people should
accumulate different evidence, and Phase 4 should be able to act on that
difference without re-deriving anything.

---

## Quick start

```python
from src.pipeline import MessagePipeline

pipeline = MessagePipeline.load()          # personalisation on by default
analysis = pipeline.analyse_all()[0]

signals = analysis.routing
signals.risk_modifier.score                # 0.94
signals.risk_modifier.confidence           # 0.94
signals.risk_modifier.reason               # "Classified as scam; ..."
signals.reasons                            # ordered across all ten signals
```

```bash
python main.py --message msg_091   # full chain for one message
python main.py --all               # whole-dataset signal summary
python main.py --no-personalize    # Phase 2 behaviour, unchanged
python -m pytest                   # 519 tests
```

---

## The ten signals

| Signal | Polarity | Asks |
|---|---|---|
| `sender_priority` | boost | Does this individual matter to this user? |
| `business_priority` | boost | Does this brand matter to this user? |
| `group_priority` | boost | Does this group matter to this user? |
| `relationship_strength` | boost | How strong is the tie, whoever sent it? |
| `historical_importance` | boost | Did comparable messages matter before? |
| `engagement_modifier` | boost | How receptive is this user generally? |
| `fatigue_modifier` | **suppress** | How overloaded is this user already? |
| `risk_modifier` | **suppress** | How unsafe does this look? |
| `trust_modifier` | boost | How much standing does the sender have? |
| `urgency_modifier` | boost | How time-critical is it? |

Each exposes `score` (0–1, 0.5 = neutral), `confidence` (0–1) and `reasons`.

`signed_strength` resolves polarity and scales by confidence into `[-1, 1]`, so
Phase 4 can combine signals directly without remembering which way each points.

---

## Layout

```text
src/personalization/
├── normalization.py        # Contribution, blend, saturating, decay, one_sided
├── signal_models.py        # RoutingSignal, RoutingSignals, SignalPolarity
├── interaction_stats.py    # scoped reaction rates, memoised
├── base.py                 # SignalContext + SignalCalculator contract
├── sender_priority.py      ┐
├── business_priority.py    │
├── group_priority.py       │  ten independent calculators,
├── relationship.py         │  one per signal
├── historical_importance.py│
├── engagement.py           │
├── fatigue.py              │
├── trust.py                │
├── risk.py                 │
├── urgency.py              ┘
└── engine.py               # ← PersonalizationEngine
```

---

## Design decisions

### No arbitrary weights, two ways

The brief forbids inventing weights. Two mechanisms keep that honest.

**Every raw quantity is mapped into `[0, 1]` by a named curve with one
parameter that states a claim.** `saturating(count, half_point=5)` says *"five
prior messages is where familiarity is half-formed"* — arguable, inspectable,
wrong-able. "Multiply by 0.17" is none of those.

**Scores are weighted means of those normalised values.** A weighted mean of
numbers in `[0, 1]` is provably in `[0, 1]`, so nothing is clamped, weights
express only relative importance, and an unavailable input is simply omitted
while the rest re-normalise themselves.

### Reasons cannot drift from scores

The same `Contribution` objects that produce the score produce the explanation.
A contribution carries `high_reason` and `low_reason`; whichever fires depends
on the value that went into the blend. It is structurally impossible for a
signal to report a reason that contradicts its own score.

### One-sided signals

Most signals are genuinely two-sided: a sender the user ignores is *real
evidence* for holding a message back, not merely an absence of evidence for
sending it.

Two are not. Early output showed a clean message producing a **+0.65 priority
boost** from `risk_modifier` and a non-urgent message a **−0.94 suppression**
from `urgency_modifier` — purely because their scores sat below neutral.
Neither is an argument for anything: "shows no sign of being a scam" is not a
reason to interrupt someone, and "not urgent", true of most messages, is not a
reason to bury one. `SignalCalculator.one_sided` rescales those onto
`[0.5, 1]` so absence lands on neutral.

### Reuse over recomputation

- `risk` reads only the Phase 2 verdict. It converts per-category rule scores
  into **shares of total evidence** — "scam accounts for 78% of everything the
  classifier found" is interpretable and already normalised, where a raw 5.8 is
  not. Its confidence *is* the classifier's confidence; inventing a second one
  would duplicate and drift.
- `engagement` reads Phase 2's `features.history` for base rates, adding only
  the trend Phase 2 does not compute.
- `interaction_stats` supplies the one thing Phase 2 genuinely lacks: rates
  scoped to **one** sender, business or group. One implementation shared by
  three calculators, memoised per `(scope, id, user)`.

### Trust is not priority

A verified bank the user never opens is highly trustworthy and low priority; a
chatty friend is the reverse. Keeping them separate preserves a distinction
Phase 4 needs.

---

## Self-evaluation

Phase 3 makes no decision, so "accuracy" is the wrong measure. The right
question is whether the signals **discriminate** — a signal that looks the same
for `notify` and `mute` is dead weight.

Measured against the ground-truth `action` on the 30 labelled sample rows:

| Signal | notify | digest | mute | spread | ordered? |
|---|---:|---:|---:|---:|:--:|
| `historical_importance` | +0.309 | +0.041 | −0.204 | **0.512** | yes |
| `sender_priority` | +0.251 | +0.036 | −0.138 | **0.390** | yes |
| `risk_modifier` | +0.000 | −0.022 | −0.373 | **0.373** | yes |
| `engagement_modifier` | +0.248 | +0.077 | −0.085 | **0.333** | yes |
| `relationship_strength` | +0.113 | −0.086 | −0.175 | 0.288 | yes |
| `group_priority` | +0.259 | +0.029 | −0.012 | 0.271 | yes |
| `trust_modifier` | +0.380 | +0.220 | +0.176 | 0.204 | yes |
| `urgency_modifier` | +0.195 | +0.011 | +0.005 | 0.189 | yes |
| `business_priority` | +0.076 | +0.033 | −0.057 | 0.133 | yes |
| `fatigue_modifier` | +0.141 | +0.174 | +0.091 | 0.083 | **no** |

**Nine of ten signals order monotonically `notify > digest > mute`** without
ever having seen an action label. That is the strongest available evidence that
Phase 4 can combine them into a decision.

### Weaknesses found and fixed

**Fatigue boosted 102 of 110 messages.** "Outside quiet hours" was contributing
`0.0` at full weight, dragging the signal below neutral for the ~90% of
messages arriving at a normal hour — reading nearly every user as unfatigued
regardless of actual load. Quiet hours now contributes only when the message is
*inside* the window. Mean moved 0.33 → 0.42.

**Relationship strength penalised every business message** for a near-zero
reply rate. People do not chat with brands; that is the expected shape of a
healthy commercial relationship. Reciprocity is now omitted for business
counterparties.

### Weaknesses found and deliberately *not* "fixed"

**`fatigue_modifier` does not discriminate between messages** — spread 0.083,
and the only signal that breaks the ordering. This is inherent, not a defect.
Measured across the dataset, fatigue varies **4× more across users
(sd 0.062) than within a single user (sd 0.015)**. It is a per-*user* property:
its job is to shift *when* a given person should be interrupted, not to rank
their messages against each other. Phase 4 should read it as a per-user
threshold shift, not as a per-message score. Making it discriminate between
messages would require inventing a message-level dependency it does not have.

**`relationship_strength` averages 0.37, mostly below neutral.** Checked
against the data rather than tuned away: the median counterparty pair has only
**4 prior messages**, and **46 of 77** sender pairs have a *zero* reply rate.
Most ties in this dataset really are weak, so a low mean is honest. Moving the
half-points to recentre the distribution would be overfitting to make a table
look tidy.

**`business_priority` averages 0.135 confidence overall** — but 0.548 *when it
applies*. It is zero for the 83 non-business messages by design. Low overall
confidence here means "rarely applicable", not "weak".

### Unavoidable limitations

- **Notification-load data predates every message.**
  `daily_notification_summary` covers 2026-07-04 to 07-17; all 110 incoming
  messages fall in 07-18 to 07-31. The rolling window therefore ends at the
  last *recorded* day. Treating absent days as zero load would be far worse —
  every user would read as completely unfatigued.
- **Sparse per-pair history.** Six of 110 messages have no prior history with
  their counterparty at all, and the median pair has four. Confidence values
  report this honestly rather than hiding it.
- **No message-level fatigue.** As above; inherent to the data model.

---

## Testing

```bash
python -m pytest        # 519 passed
```

| File | Covers |
|---|---|
| `test_normalization.py` | curves, blending, one-sided rescaling, signal records |
| `test_calculators.py` | all ten independently, plus a universal contract |
| `test_personalization_engine.py` | scoped stats, engine validation, pipeline wiring |

Two regressions are pinned: messages outside quiet hours must not read as
unfatigued, and a clean message must never produce a priority boost.

A dataset-wide test asserts **every signal varies** across the 110 messages — a
constant signal would carry no information — and that **no routing action ever
appears** in Phase 3 output.

---

## For Phase 4

```python
from src.pipeline import MessagePipeline

for analysis in MessagePipeline.load().analyse_all():
    signals = analysis.routing

    # Every signal already resolved for direction and scaled by confidence.
    total = sum(s.signed_strength for s in signals.all_signals)

    # Or weigh them yourself; nothing is pre-combined.
    signals.risk_modifier.signed_strength      # < 0 for anything risky
    signals.fatigue_modifier                   # per-user threshold shift
    signals.reasons                            # become the routing reasons

    # ... notify / digest / mute decision goes here, in Phase 4
```

`signed_strength` is offered as a convenience, not a decision: it resolves
polarity and confidence but applies no weighting between signals. Choosing
those weights *is* Phase 4's job.

Backwards compatible throughout: `MessagePipeline(repo, personalize=False)`
reproduces Phase 2 behaviour exactly, and `MessageAnalysis.routing` is `None`
in that case.
