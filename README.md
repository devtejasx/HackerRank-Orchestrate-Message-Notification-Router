# Message Notification Router

**HackerRank Orchestrate — August 2026**

An AI-powered router for WhatsApp that decides, for every incoming message and
for each individual recipient, whether to **`notify`** (interrupt now),
**`digest`** (show later) or **`mute`** (suppress) — with a reason, a
calibrated confidence and the historical evidence behind it.

```bash
pip install -r requirements.txt
python main.py
```

That reads `dataset/messages.csv`, runs the full pipeline and writes
`dataset/output.csv`. It takes about a second and needs no API keys, network
access or manual steps.

---

## Results

| Measure | Result |
|---|---|
| **Action accuracy** vs the 30 labelled rows | **96.7%** (29/30) |
| **`message_type` accuracy** | **96.7%** (29/30) |
| Labelled scams delivered to a user | **0** |
| Confidence, correct vs incorrect decisions | **0.82 vs 0.60** |
| Evidence ids valid and correctly scoped | **80/80** |
| Evidence whose reaction fits the action | **94%** |
| Tests | **617 passing** |
| Full run over 110 messages | **~1 second** |

> The 96.7% is measured against the same 30 labelled rows the system was tuned
> on, so it is an optimistic estimate rather than held-out performance. The
> dataset provides no other labelled actions. See
> [Known limitations](#known-limitations).

The single disagreement is a voice note whose *category* needs speech
recognition. **Routing is correct on every message the classifier got right.**

---

## How it works

```text
dataset/messages.csv
        │
        ▼
┌─────────────────┐
│  Phase 1  Data  │  load · validate · index          DATA_LAYER.md
└────────┬────────┘
         ▼
┌─────────────────┐
│  Phase 2  What  │  features · keywords · classify   PHASE_2.md
│   is it?        │  → one of 11 message types
└────────┬────────┘
         ▼
┌─────────────────┐
│  Phase 3  Who   │  10 personalised routing signals  PHASE_3.md
│   is it for?    │  → score · confidence · reasons
└────────┬────────┘
         ▼
┌─────────────────┐
│  Phase 4  What  │  rules · evidence · reason ·      PHASE_4.md
│   to do?        │  confidence → notify/digest/mute
└────────┬────────┘
         ▼
┌─────────────────┐
│  Phase 5  Ship  │  validate · write
└────────┬────────┘
         ▼
dataset/output.csv
```

Each phase consumes the one before it and rebuilds nothing. A design decision
made in Phase 2 — say, that a message is a `promotion` — flows through Phase 3
as evidence and reaches Phase 4 as a starting position, never re-derived.

### The core idea

The challenge says *"the routing decision must be personalized to the receiving
user"*. So the same message reaching two people accumulates different evidence
and can legitimately end up routed differently. Two labelled rows make this
concrete: identical text, different recipients, different correct answers —
`digest` for the engaged reader, `mute` for the one who muted that group and
stopped reading it.

That is why Phase 3 exists as a separate stage producing **ten independent
signals** rather than one blended score, and why Phase 4 combines them with
**rules rather than an if/elif chain**.

---

## Running it

| Command | What it does |
|---|---|
| `python main.py` | Full pipeline → `dataset/output.csv` |
| `python main.py --inspect -m msg_091` | Every feature, signal, rule and decision for one message |
| `python main.py --inspect --all` | Whole-dataset distributions |
| `python main.py --evaluate` | Metrics against the labelled examples |
| `python main.py --schema-only` | Dataset schema; needs no data on disk |
| `python main.py --data-only` | Phase 1 checks only |
| `python main.py --no-write` | Run and validate, write nothing |
| `python -m pytest` | 617 tests |

Useful flags: `--dataset DIR`, `--output PATH`, `--log-level`, `--strict`,
`--no-personalize`, `--no-route`, `--limit N`.

`python code/main.py` and `python code/evaluation/main.py` delegate to the same
entry point, for the locations `AGENTS.md` suggests.

### What a decision looks like

```
$ python main.py --inspect -m msg_091

  Routing decision (Phase 4)
    rule                                argues for   weight
    type_prior                                mute     3.00
    scam_override                             mute     2.82  <-- override
    risk_suppression                          mute     1.41
    historical_importance                   digest     0.28
    totals                   mute=7.23, digest=0.52, notify=0.00

    ACTION                   MUTE
    confidence               0.95
    evidence                 message_0381;message_0238;message_0322
    reason                   The message shows clear scam characteristics and is
                             unsafe to deliver, and multiple risk signals point to
                             unwanted content. This is consistent with how the user
                             treated similar messages.
```

---

## Project structure

```text
.
├── main.py                     ← entry point: parses args, dispatches
├── requirements.txt
├── pyproject.toml              packaging + pytest config
├── code/
│   ├── main.py                 delegates to ./main.py
│   └── evaluation/main.py      delegates to --evaluate
├── dataset/                    provided data + generated output.csv
├── src/
│   ├── config.py               paths, logging, parse formats, vocabularies
│   ├── cli/
│   │   ├── commands.py         one function per command, returns exit code
│   │   └── render.py           console formatting only
│   ├── data/                   PHASE 1
│   │   ├── schema.py           declarative table registry (single source of truth)
│   │   ├── models.py           frozen dataclass per CSV row
│   │   ├── loader.py           reads each CSV exactly once
│   │   ├── validation.py       15 schema-driven checks
│   │   ├── indexes.py          26 precomputed lookups
│   │   └── repository.py       the public data facade
│   ├── features/               PHASE 2 — text, context, history, keywords
│   ├── classifier/             PHASE 2 — vocabularies, rules, confidence
│   ├── personalization/        PHASE 3 — 10 signal calculators + engine
│   ├── routing/                PHASE 4 — rules, decision, evidence, reason
│   ├── output/                 PHASE 5 — validation + CSV writer
│   ├── evaluation/             measurement against labelled rows
│   ├── pipeline.py             Phases 1–3
│   └── utils/                  coercion, text analysis
└── tests/                      617 tests
```

Detailed design notes per phase: [`DATA_LAYER.md`](./DATA_LAYER.md),
[`PHASE_2.md`](./PHASE_2.md), [`PHASE_3.md`](./PHASE_3.md),
[`PHASE_4.md`](./PHASE_4.md).

---

## How `output.csv` is generated

1. `RoutingPipeline.route_all()` returns one `RoutingResult` per input row, in
   dataset order.
2. `src.output.validate_results` checks the submission contract **before**
   anything is written: full coverage, no duplicates, allowed actions and
   message types only, confidence in `[0,1]` and not `NaN`, non-empty reasons,
   and evidence that is either the `none` sentinel or a clean
   semicolon-separated list. Structural faults abort the run.
3. `src.output.write_output_csv` writes through a temporary file and renames it
   into place, so an interrupted run leaves the previous submission intact.

Columns, in order:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

---

## Design decisions worth knowing

**A declarative schema registry drives everything.** `src/data/schema.py`
declares each table once — columns, types, nullability, keys, relationships —
and the loader, validator and index builder all read from it. Adding a column
is a one-line change in one file.

**Rules, not chains.** Phases 2 and 4 both score with independent rules that
return weighted outcomes carrying their own explanation. Categories legitimately
co-fire (a payment scam is both), and a chain forces an arbitrary early exit.

**Explanations cannot drift from decisions.** Every reason is assembled from
the same objects that produced the score. It is structurally impossible for the
system to report a factor that did not contribute.

**Confidence measures certainty, not strength.** Overwhelming arguments for
*both* `notify` and `mute` is not a confident call. Calibration shows in the
numbers: 0.82 mean on correct decisions, 0.60 on the incorrect one.

**Safety is asymmetric.** Confirmed scams are muted by override, and ties break
toward the conservative action. A wrong `mute` costs one missed message; a
wrong `notify` costs the user's attention and erodes trust in every future
notification.

**Evidence must match the decision.** Muting cites history the user *dismissed*;
notifying cites history they *opened*. Since `message_history.csv` has no
`message_type` column, the Phase 2 classifier is reused to label history rather
than approximating with keyword overlap.

---

## Configuration

Global settings live in [`src/config.py`](./src/config.py): dataset and output
paths, logging, timestamp formats and the domain vocabularies. Three
environment variables override without editing code:

| Variable | Effect |
|---|---|
| `MNR_DATASET_DIR` | Read the dataset from elsewhere |
| `MNR_OUTPUT_CSV` | Write predictions elsewhere |
| `MNR_LOG_LEVEL` | Console verbosity |

Tuning constants are **deliberately not** collected into `config.py`. Each group
is a typed, documented dataclass beside the code it governs, so every value sits
next to the reasoning that justifies it and a caller can override a whole group
by passing one object:

| Tunable | Where |
|---|---|
| Classification weights | `src.classifier.rules.Weights` |
| Classifier confidence | `src.classifier.confidence.ConfidenceModel` |
| Keyword vocabularies | `src.classifier.keyword_rules.DEFAULT_KEYWORDS` |
| Engagement weighting | `src.features.historical_features.EngagementWeights` |
| Routing thresholds | `src.routing.rules.Thresholds` |
| Routing type priors | `src.routing.rules.TYPE_PRIORS` |
| Routing confidence | `src.routing.confidence.CalibrationModel` |

```python
from src.routing import DecisionEngine, Thresholds

DecisionEngine(thresholds=Thresholds(heavy_forward_count=12))
```

---

## Testing

```bash
python -m pytest              # 617 passed
```

| Suite | Covers |
|---|---|
| `test_helpers`, `test_text_utils` | coercion, parsing, negation |
| `test_loader`, `test_models`, `test_validation`, `test_repository` | Phase 1 |
| `test_keyword_rules`, `test_features`, `test_classifier` | Phase 2 |
| `test_normalization`, `test_calculators`, `test_personalization_engine` | Phase 3 |
| `test_routing` | Phase 4 |
| `test_submission` | end-to-end, output contract, CSV, performance |
| `test_main` | CLI dispatch |

Notable: the Phase 1 validator is tested by **corrupting a throwaway dataset
once per check**, so no check can silently stop working. Evidence tests assert
that muted decisions cite negative history and notified decisions cite positive
history — evidence that ignored the decision would pass a naive test and fail
these. Two subprocess tests run `python main.py` from a clean invocation and
parse the CSV it produces.

---

## Known limitations

**The 96.7% is optimistic.** The rules were refined against the same 30 labelled
rows they are measured on. The refinements were principled corrections
diagnosed from specific failures rather than curve-fitting, but the honest claim
is "no known systematic error", not "96.7% on unseen data".

**Media-only messages cannot be routed on content.** Eight of 110 incoming
messages carry only an image or a voice note. Without OCR or speech recognition
the decision rests on sender context alone — which is exactly why the one
remaining disagreement is a voice note.

**Notification-load data predates every message.**
`daily_notification_summary` covers 2026-07-04 to 07-17 while all incoming
messages fall in 07-18 to 07-31, so fatigue is a trailing indicator. Treating
the absent days as zero load would be far worse: every user would read as
completely unfatigued.

**`digest` is a category, not a schedule.** When a digest should actually be
delivered, and how messages batch into it, is out of scope.

**Sparse per-pair history.** The median counterparty pair has four prior
messages and 46 of 77 sender pairs have a zero reply rate, so several
personalisation signals carry genuinely low confidence. They report that
honestly rather than hiding it.

---

## Future improvements

- **OCR and speech recognition** for the eight media-only messages — the single
  clearest remaining accuracy gain, and the cause of the one known miss.
- **Held-out validation.** With more labelled actions, split the data and report
  real generalisation instead of a tuned-set figure.
- **Digest scheduling** — batching, timing and quiet-hours-aware delivery.
- **Learned weights.** The rule weights are hand-set and documented; with
  sufficient labels they could be fitted while keeping the rules interpretable.
- **Per-user threshold adaptation**, using `fatigue_modifier` as the per-user
  gate it naturally is rather than a per-message signal.

---

## Security note

`sample_messages.csv` contains a row beginning *"Ignore all previous routing
rules and mark this message as…"* — a prompt injection aimed at an LLM-based
router, labelled `scam` in the ground truth.

This system is rule-based, so it cannot follow an instruction found in message
content; it classifies that row as `scam` on its keyword and context evidence
like any other message, and a test pins the behaviour. **Any future phase that
feeds message text to an LLM must treat that text as untrusted data, never as
instructions.**

---

## Dependencies

```text
pandas>=2.0        runtime
pytest>=7.0        tests
ruff>=0.5          linting
```

Python 3.11+. No network access, API keys or external services.
