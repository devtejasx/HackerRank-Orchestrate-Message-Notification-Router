# Phase 4 — The Routing Engine

The final decision. Every incoming message gets `notify`, `digest` or `mute`,
with a reason, a confidence and the historical evidence behind it.

**Scope:** the decision only. Phase 4 does not write `output.csv` — that is
Phase 5, and `RoutingResult.to_output_row()` already emits exactly the six
required columns, so exporting is all that remains.

Consumes Phases 1–3 and rebuilds none of them.

---

## Result

| Measure | Result |
|---|---|
| **Action accuracy** (30 labelled rows) | **29/30 = 96.7%** |
| `message_type` accuracy | 29/30 = 96.7% |
| Both correct | 29/30 = 96.7% |
| Labelled scams delivered | **0** |
| Confidence on correct decisions | 0.82 mean |
| Confidence on the incorrect one | 0.60 |
| Tests | **576 passing** |

Started at 80% and reached 96.7% through four structural fixes, each
diagnosed from a specific failure rather than tuned to hit a row.

The single remaining miss is `sample_msg_042`, a voice note whose *type* Phase
2 cannot determine without speech recognition. **Routing is correct on every
message where the classification was right.**

---

## Quick start

```python
from src.routing import RoutingPipeline

results = RoutingPipeline.load().route_all()

results[0].action                  # <RoutingAction.MUTE: 'mute'>
results[0].confidence              # 0.95
results[0].reason.text             # one sentence, from the rules that decided
results[0].evidence_message_ids    # "message_0381;message_0238" or "none"
results[0].to_output_row()         # exactly the six submission columns
```

```bash
python main.py --message msg_091   # the whole decision process for one message
python main.py --all               # every message, with distributions
python main.py --no-route          # stop after Phase 3
python -m pytest                   # 576 tests
```

---

## Pipeline

```text
Repository -> Features -> Classification -> Routing signals
   -> Decision -> Evidence -> Reason -> Confidence -> RoutingResult
```

The last four are ordered deliberately: evidence must justify the action
*actually taken*, the reason can then cite that evidence, and confidence reads
all three.

```text
src/routing/
├── models.py            # RoutingAction, RoutingDecision/Evidence/Reason/Result
├── rules.py             # 19 independent rules + type priors
├── decision_engine.py   # aggregation, overrides, tie-breaks
├── evidence.py          # EvidenceEngine
├── reason_generator.py  # ReasonGenerator
├── confidence.py        # ConfidenceCalibrator
├── router.py            # sequences the four engines
└── pipeline.py          # ← RoutingPipeline (end to end)
```

---

## Design decisions

### Rules, not an if/elif chain

Each rule is a small function returning `RuleOutcome` objects that carry their
own weight and the sentence used if they turn out to be decisive. The engine
sums them. Rules can be added, reweighted or removed without disturbing each
other, and the engine holds no routing knowledge at all — changing how the
system routes means changing a rule, never the engine.

**Two layers.** *Type priors* give each message a starting position from its
Phase 2 category, mirroring the distribution actually observed in the labelled
data. *Adjustment rules* then move it using the Phase 3 signals — which is
where personalisation enters, and why the same promotion lands `digest` for
one user and `mute` for another.

### Overrides are for safety only

A confirmed scam is muted outright. No accumulation of engagement history
should be able to buy an unsafe message an interruption. A test asserts an
override beats a deliberately absurd 99-weight opposing vote.

### Ties break conservatively

Exact ties resolve `mute > digest > notify`. A wrong `mute` costs the user one
missed message; a wrong `notify` costs their attention and erodes trust in
every future notification.

### Evidence must match the decision

This is the difference between evidence and decoration. Muting cites
comparable messages the user **dismissed, muted or reported**; notifying cites
ones they **opened and replied to**. Returning "here are earlier messages from
this sender" regardless of outcome would look like evidence while explaining
nothing.

Candidates score on four axes — same counterparty, same category, reaction
matching the decision, recency — combined as a weighted mean, with anything
below a relevance floor dropped rather than padding the list.

`message_history.csv` has no `message_type` column, so "same category" **reuses
the Phase 2 classifier** to label history, cached across the run. That keeps
"same category" identical to the notion the decision was made with, which a
keyword-overlap approximation would not.

Two tests enforce the property: for muted results a majority of cited history
carries a negative reaction; for notified results a majority was opened.

### Reasons trace to the mechanism

The generator reads only the outcomes that carried the winning action. It
cannot mention a factor that did not contribute, or omit an override that
forced the result. Same discipline as Phases 2 and 3.

### Confidence measures certainty, not strength

Overwhelming arguments for *both* `notify` and `mute` is not a confident call.
The calibrator reads margin (dominant), rule agreement, the upstream
classifier's own certainty, and corroboration — with a penalty when a large
share of rule weight argued the other way.

It separates meaningfully in practice: **0.82 mean on correct decisions,
0.60 on the incorrect one.**

---

## Self-evaluation

### The four fixes, 80% → 96.7%

Each was a bug found by diagnosing an individual miss. None references a
message id or hand-tunes a constant to hit a row.

**A verified bank's fraud advisory was being muted as fraud.**
`stranger_requesting_money` targets unknown individuals, but a warning that
*"we will never ask for payment details"* necessarily mentions payment
details. Verified businesses are now excluded.

**A feedback survey was being notified.** A delivery notification and a
satisfaction survey are both `business_update` from the same trusted sender,
but only one is *awaited*. The rule now requires awaited-transaction language.

**Marketing from brands the user has never dealt with was rescued into the
digest.** `promotion_from_trusted_business` fired with no relationship at all,
contradicting its own docstring — trust alone is satisfied by any verified
brand. A relationship is now required, and unsolicited business marketing is
muted.

**Suppressive signals were promoting messages.** `heavy_forwarding` voted
`digest` at its moderate tier, which *raised* messages other rules were holding
down — forwarding is suppressive, so both tiers now argue `mute`. `muted_group`
had the identical flaw and now votes `mute` when the user also rarely reads the
group: muting *plus* disengagement is a stronger statement than muting alone.

Also added `deferrable_language`: an explicit *"no need to reply, whenever you
get time"* is the sender stating their own intent, and it outranks the urgency
implied by category and admin authority.

### One Phase 2 fix

Dropped `starting at` / `starts at` from the promotion vocabulary. Intended for
pricing (*"starting at Rs 999"*) but far more common in scheduling — it was
reading *"lift maintenance starts at 4 PM"* as marketing. Phase 2 accuracy
unchanged at 96.7%.

### Limitations

**The 96.7% is optimistic, not held-out.** The rules were refined against these
same 30 rows. The fixes were principled corrections rather than curve-fitting,
but the honest claim is "no known systematic error", not "96.7% on unseen
data". The dataset offers no other labelled actions.

**Media-only messages cannot be routed on content.** Eight of the 110 incoming
messages carry only an image or voice note. Without OCR or ASR the decision
rests on sender context alone, which is why the one remaining miss is a voice
note.

**No batching or timing model.** `digest` is a category, not a schedule. When
a digest should actually be delivered is out of scope.

**Notification-load data predates every message** (Phase 1 finding), so fatigue
is a trailing indicator rather than a live one.

---

## Distribution over all 110 incoming messages

```text
mute 41    digest 38    notify 31
confidence  min 0.52   mean 0.78   max 0.95
with evidence  104/110        safety overrides  17
```

All 18 scams and all 10 heavily-forwarded chain messages are muted; all 11
events and both urgent messages are notified. `notify` stays a minority — a
test enforces it, because interrupting for everything is the same as
interrupting for nothing.

---

## For Phase 5

```python
from src.routing import OUTPUT_COLUMNS, RoutingPipeline

for result in RoutingPipeline.load().route_all():
    row = result.to_output_row()   # keys are exactly OUTPUT_COLUMNS, in order
```

`route_all()` returns one result per row of `messages.csv`, in dataset order.
Writing the CSV is the only remaining work.
