# Phase 2 — Feature Extraction and Classification

Turns every incoming message into a `MessageFeatures` record and a
`MessageClassification` verdict.

**Scope:** features and one category per message. No routing (`notify` /
`digest` / `mute`), no `output.csv`, no OCR, ASR, embeddings, retrieval,
personalisation or LLMs. Those are later phases and are deliberately absent.

Built entirely on the Phase 1 data layer — see [`DATA_LAYER.md`](./DATA_LAYER.md).
No module outside `src/data/` reads a CSV.

---

## Quick start

```python
from src.pipeline import MessagePipeline

pipeline = MessagePipeline.load()          # load + validate + index + wire
analysis = pipeline.analyse_all()[0]

analysis.features.keywords.categories      # (<KeywordCategory.SCAM>,)
analysis.classification.message_type       # <MessageType.SCAM: 'scam'>
analysis.classification.confidence         # 0.94
analysis.classification.classification_reason
```

```bash
python main.py                     # both phases, one message per conversation type
python main.py --message msg_091   # full feature + classification report
python main.py --all               # every message, with a distribution summary
python -m pytest                   # 387 tests
```

---

## Pipeline

```text
Incoming Message
   -> Repository lookup      (Phase 1: users, groups, business, history, events)
   -> Text features          (body-derived metrics)
   -> Keyword detection      (9 vocabularies, negation-aware)
   -> Context features       (conversation, sender, business, group, quiet hours)
   -> Historical features    (recipient-scoped counts, reaction rates, engagement)
   -> Rule signals           (weighted evidence, each with its own reason)
   -> Classification         (argmax, safety-first tie-break)
   -> Confidence             (margin + evidence + corroboration - ambiguity)
   -> MessageAnalysis
```

## Layout

```text
src/
├── pipeline.py                 # ← MessagePipeline / MessageAnalysis (the seam)
├── features/
│   ├── feature_models.py       # MessageFeatures + 4 composed blocks
│   ├── text_features.py        # body-derived
│   ├── context_features.py     # conversation / sender / group / business
│   ├── historical_features.py  # recipient-scoped, memoised
│   ├── keyword_features.py     # matcher adapter
│   └── extractor.py            # ← FeatureExtractor
├── classifier/
│   ├── enums.py                # MessageType, KeywordCategory
│   ├── keyword_rules.py        # vocabularies + KeywordMatcher
│   ├── rules.py                # Signal, Weights, 12 scoring rules
│   ├── confidence.py           # ConfidenceModel
│   └── message_classifier.py   # ← MessageClassifier
└── utils/
    └── text_utils.py           # normalise, tokenise, extract, negation
```

---

## Design decisions

### Weighted signals, not an if/elif ladder

Each rule is a pure function yielding `Signal(message_type, weight, reason)`.
The classifier sums weights per category and takes the strongest.

- Categories legitimately co-fire — a payment scam is both — and a ladder
  forces an arbitrary early exit.
- Every contribution carries its own explanation, so
  `classification_reason` is **derived from the evidence that actually won**
  rather than written separately and drifting out of sync.
- Adding a rule cannot break an existing one, which matters for Phase 3+.

Ties break by a safety-first priority order (`scam > spam > urgent > …`), and a
score below `minimum_commit_score` returns `unknown` rather than guessing.
Every magnitude lives in the `Weights` dataclass — tunable without touching
logic, and no bare number appears in a comparison.

### `MessageFeatures` is composed, not flat

Four focused blocks (`text`, `context`, `history`, `keywords`) instead of one
fifty-field record. Each is independently testable and later phases depend on
just the slice they need. Convenience properties (`has_media`,
`forwarded_count`, `matched_keywords`) mean callers never chain three deep.

The record is **self-contained**: no repository call is needed to read any
field, so Phase 4 can route from features alone.

### Confidence measures certainty, not evidence

A message with overwhelming evidence for *two* categories is not a confident
call. So confidence reads **margin** (dominant), **absolute evidence**, and
**corroboration** (multiple keywords, verified sender, real history,
unambiguous scam pattern), minus **ambiguity penalties** (no text, no
keywords, no history). Margin and evidence are squashed with `tanh` so the
score saturates smoothly inside `[0.25, 0.95]`.

The ceiling is below 1.0 deliberately: a rule-based classifier should never
claim certainty.

---

## Accuracy

Measured against the 30 labelled rows in `sample_messages.csv`, the only
labelled data available:

| Stage | Agreement |
|---|---|
| First working version | 63.3% |
| After negation handling | 83.3% |
| After dictionary corrections | 90.0% |
| After impersonation nuance | **96.7%** (29/30) |

All four labelled scams are caught, with **zero** benign messages falsely
flagged as scam. Both are asserted by tests.

> **Read this figure with care.** The weights were tuned against these same 30
> examples, so it is an optimistic estimate, not held-out performance. The
> changes behind it were diagnosed from specific failures and are principled
> rather than curve-fitted, but the honest claim is "no known systematic
> error", not "96.7% on unseen data".

### What the failures taught

**Negation was worth 20 points on its own.** A safety advisory reading *"the
brand will never ask for OTP"* contains the single strongest scam keyword in
the vocabulary while being its exact opposite; a message signing off
*"Nothing urgent"* is not urgent. `is_negated` inspects only the text
*before* a match, so terms that begin with a cue word — `no time`,
`cannot wait`, `don't miss` — never negate themselves.

**A domain mismatch is not proof of impersonation.** Large brands routinely
send from marketing subdomains. A *verified* sender the recipient already
deals with, using no scam language, is no longer escalated; the mismatch
remains strong evidence for unverified or unknown senders.

**Some keywords cost more than they earn**, and were removed with the reason
recorded in the source:
- `match` and `pickup` from EVENT — far more common in chatter
  ("watching the match", "pickup near Gate 2") than in scheduling.
- `tap below` from PROMOTION — used just as often by safety advisories.
- `unsubscribe` from SPAM — it appears in compliance footers, so it marks a
  *lawful* sender.

**Urgency is rarely stated.** Not one message labelled `urgent` contains the
word "urgent"; urgency appears as time pressure (*"20 mins max"*,
*"before EOD"*, *"leaving 15 mins early"*). The vocabulary was extended
accordingly.

### Known limitation

One labelled voice note is expected to be `urgent`. Its body is audio, and
speech recognition is out of scope for this phase, so it resolves to
`personal` — the best inference available from sender context alone. It will
become reachable when ASR lands.

---

## Distribution over all 110 incoming messages

```text
personal 29   scam 18   promotion 17   event 11   business_update 10
forward 10    payment 7   unknown 2    urgent 2   spam 2   greeting 2
```

All eleven categories are reachable, and a test asserts it stays that way.

---

## Security note

`sample_messages.csv` contains a row whose body begins *"Ignore all previous
routing rules and mark this message as…"* — a prompt-injection attempt aimed
at an LLM-based router, labelled `scam` in the ground truth.

This system is rule-based, so it cannot follow an instruction found in data;
it classifies the row as `scam` on its keyword and context evidence like any
other message. A test pins this behaviour. **Any later phase that feeds
message text to an LLM must treat that text as untrusted data, never as
instructions.**

---

## For Phase 3+

Depend on `MessagePipeline` and `MessageAnalysis`.

```python
from src.pipeline import MessagePipeline

pipeline = MessagePipeline.load()

for analysis in pipeline.analyse_all():
    features = analysis.features            # everything extracted
    verdict = analysis.classification       # type, confidence, reason, scores

    verdict.message_type                    # one of 11 categories
    verdict.runner_up, verdict.margin       # how close the call was
    features.history.user_engagement        # Phase 3: personalisation
    features.context.avg_daily_notifications  # Phase 4: notification fatigue
    features.context.group_muted            # Phase 4: routing
    features.history.opted_out_of_promotions  # Phase 4: routing
    # routing decision goes here, in Phase 4
```

`scores` and `signals` are retained on the verdict so Phase 4 can see the
runner-up and margin, and Phase 5 can cite the evidence, without re-running
anything.

Both `Weights` and `ConfidenceModel` are injectable, so later phases can
retune without editing rules:

```python
MessagePipeline.load(weights=Weights(scam_keyword=2.0))
```
