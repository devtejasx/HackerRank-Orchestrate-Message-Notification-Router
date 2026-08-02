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

That reads `dataset/messages.csv`, runs the full pipeline and writes the
submission to **`output.csv`** — both at the repository root and at
`dataset/output.csv`, filling the template shipped there. It takes about a
second and needs no API keys, network access or manual steps.

---

## Results

| Measure | Result |
|---|---|
| **Action accuracy** vs the 30 labelled rows | **96.7%** (29/30) |
| **`message_type` accuracy** | **96.7%** (29/30) |
| Labelled scams delivered to a user | **0** |
| Confidence, correct vs incorrect decisions | **0.81 vs 0.50** (gap **+0.31**) |
| Evidence ids valid and correctly scoped | **80/80** |
| Evidence whose reaction fits the action | **94%** |
| Dataset corruptions survived with a complete, valid submission | **40/40** |
| Tests | **684 passing** |
| Full run over 110 messages | **~1 second** (1.9 ms/message, linear to 5,500) |

> The 96.7% is measured against the same 30 labelled rows the system was tuned
> on, so it is an optimistic estimate rather than held-out performance. The
> dataset provides no other labelled actions. See
> [Known limitations](#known-limitations).

The single disagreement is a voice note whose *category* needs speech
recognition. **Routing is correct on every message the classifier got right.**

---

## Against the brief

Every requirement in the challenge specification, and where it is met:

| Requirement | Where |
|---|---|
| `notify` / `digest` / `mute` per message | `src/routing/decision_engine.py` |
| Personalised to the **receiving user** | `src/personalization/` — 10 signals, `SignalContext` is always recipient-scoped |
| Handles text, image and voice messages | `src/features/`, plus `src/media/` for attachment resolution |
| Uses user, group, business, history, image, voice and interaction data | All 12 tables loaded and indexed; `src/data/schema.py` |
| Suppresses low-value, repetitive, unwanted, suspicious and unsafe content | `src/routing/rules.py` — promotion, forwarding, muted-group, risk and scam rules |
| One row per `message_id`, exact columns | `src/output/validation.py` enforces it before writing |
| `none` when no evidence applies | `NO_EVIDENCE` sentinel, `src/routing/models.py` |
| Runnable from the terminal | `python main.py` |
| Reads only from `dataset/`, no organizer-only files | `src.config.DATASET_DIR` is the single data root |
| No hardcoded labels | No message id maps to an outcome anywhere. At runtime the `sample_messages.csv` labels are read **only** by `src/evaluation/`; they informed the rule weights during development, which the comments beside those weights say openly |
| Deterministic | Pinned by `tests/test_robustness.py::TestDeterminism` |
| Secrets from environment only | No credentials of any kind; the run needs no network |

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
output.csv  +  dataset/output.csv
```

Alongside Phase 2 sits the **multimodal seam** (`src/media/`): attachments are
resolved to files on disk and offered to an OCR / speech-to-text provider whose
recovered text rejoins the message body. No model is installed, so it recovers
nothing today — but the wiring, the feature block and the tests already exist.
See [Multimodal](#multimodal-where-ocr-and-whisper-plug-in).

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
| `python main.py` | Full pipeline → `output.csv` and `dataset/output.csv` |
| `python main.py --inspect -m msg_091` | Every feature, signal, rule and decision for one message |
| `python main.py --inspect --all` | Whole-dataset distributions |
| `python main.py --evaluate` | Metrics against the labelled examples |
| `python main.py --schema-only` | Dataset schema; needs no data on disk |
| `python main.py --data-only` | Phase 1 checks only |
| `python main.py --no-write` | Run and validate, write nothing |
| `python -m pytest` | 684 tests |

Useful flags: `--dataset DIR`, `--output PATH`, `--log-level`, `--strict`,
`--no-personalize`, `--no-route`, `--limit N`.

`python code/main.py` and `python code/evaluation/main.py` delegate to the same
entry point, for the locations `AGENTS.md` suggests.

### What a decision looks like

```
$ python main.py --inspect -m msg_091

  Routing decision (Phase 4)
    rule                                argues for   weight
    -------------------------------------------------------
    type_prior                                mute     3.00
    scam_override                             mute     2.82  <-- override
    risk_suppression                          mute     1.41
    historical_importance                   digest     0.28
    counterparty_standing                   digest     0.24
    totals                   mute=7.23, digest=0.52, notify=0.00
    runner-up                digest (margin 6.72)

    ACTION                   MUTE
    confidence               0.95
    evidence                 message_0381;message_0238;message_0322
    evidence basis           3 comparable message(s) of the same type the user
                             reacted to consistently with mute.
    reason                   The message shows clear scam characteristics and is
                             unsafe to deliver, and multiple risk signals point to
                             unwanted or unsafe content. This is consistent with
                             how the user treated similar messages.
```

Every number there is traceable: the weights are what the rules returned, the
evidence is history this recipient actually reacted to, and the reason is
assembled from the same outcome objects that produced the score.

### Verifying the submission yourself

The run validates its own output before writing, but the checks are worth
reproducing independently:

```bash
python -c "import pandas as pd; o=pd.read_csv('output.csv'); m=pd.read_csv('dataset/messages.csv'); print(list(o.columns)); print(len(o)==len(m), o.message_id.is_unique, set(o.action)<={'notify','digest','mute'}, o.confidence.between(0,1).all())"
```

Expected: the six contract columns in order, then `True True True True`.
`python main.py --no-write` runs and validates everything without touching a
file, and `python main.py --strict` additionally treats every dataset warning as
blocking.

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
│   ├── media/                  multimodal seam — OCR / speech-to-text plug point
│   │   ├── content.py          attachment + recovered-content records
│   │   ├── understanding.py    the provider interface (null by default)
│   │   └── resolver.py         registry lookup, disk check, caching
│   ├── evaluation/             measurement against labelled rows
│   ├── pipeline.py             Phases 1–3
│   └── utils/                  coercion, text analysis
└── tests/                      684 tests
```

Detailed design notes per phase: [`DATA_LAYER.md`](./DATA_LAYER.md),
[`PHASE_2.md`](./PHASE_2.md), [`PHASE_3.md`](./PHASE_3.md),
[`PHASE_4.md`](./PHASE_4.md).

---

## How `output.csv` is generated

1. `RoutingPipeline.route_all()` returns one `RoutingResult` per input row, in
   dataset order.
2. `src.output.validate_results` checks the submission contract **before**
   anything is written:
   - **coverage** — exactly one prediction per input row, none extra, in order;
   - **format** — allowed actions and message types only, confidence a finite
     number in `[0,1]`, non-empty reasons, evidence either the `none` sentinel
     or a clean semicolon-separated list with no blanks or duplicates;
   - **truth** — every cited evidence id exists in `message_history.csv` **and
     belongs to the same recipient**. A fabricated citation and a citation of
     another user's history are both blocking errors, because a reason column
     that asserts something untrue is worse than one that asserts nothing.

   Structural faults abort the run and the previous submission is left intact.
3. `src.output.write_submission` writes through a temporary file that is renamed
   into place, so an interrupted run never leaves a half-written submission.

Columns, in order:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

**Why two files.** The brief names the deliverable `output.csv` without fixing a
directory, and the dataset ships an `output.csv` template listing every
`message_id` with blank predictions. Filling that template is the natural
reading; a grader running the project would reasonably look in the repository
root. Both are written, byte-identical, so the ambiguity costs nothing. An
explicit `--output PATH` writes exactly one file, where you asked for it.

---

## Multimodal: where OCR and Whisper plug in

23 of 110 incoming messages carry an image or a voice note, and 8 of those have
no text at all. Nothing here reads pixels or audio. What exists instead is a
finished seam with a null provider in it — the load-bearing decision being that
**recovered text is not a special case**. It is appended to the typed body
before feature extraction, so length, URLs, currency, scam vocabulary and
urgency all cover a transcript automatically.

```text
message.media_id
  → MediaResolver         registry lookup, then filesystem check     ✅ built
  → MediaUnderstanding    OCR / speech-to-text                       ⬅ plug here
  → MediaFeatures         provenance + recovered text                ✅ built
  → FeatureExtractor      recovered text joins the typed body        ✅ built
```

Installing a model is one class and one argument:

```python
class WhisperTranscriber:
    name = "whisper-small"

    def supports(self, modality):
        return modality is MediaModality.VOICE

    def understand(self, attachment):
        result = self._model.transcribe(str(attachment.file_path))
        return MediaContent(text=result["text"], provider=self.name, confidence=0.8)

RoutingPipeline.load(understanding=WhisperTranscriber())
```

No feature, rule, classifier, evidence or writer change. `tests/test_media.py`
proves it: a fake transcriber is written exactly as the real one would be, and
the tests assert its output reaches the keyword matcher and changes routing.

Three guarantees make the seam safe to rely on:

| Guarantee | Why it matters |
|---|---|
| A provider never sees an unreadable file | The resolver checks the registry *and* the filesystem first |
| A provider that raises cannot fail the run | `SafeUnderstanding` wraps every provider; a crash costs one transcript, not `output.csv` |
| Each attachment is read at most once | Results are cached by media id — `img_008` appears three times, and transcription is expensive |

**Until a model is installed, the confidence column says so.** A message whose
content could not be read at all takes an explicit opacity penalty
(`CalibrationModel.opacity_penalty`). Routing a voice note on sender history is
often right, but it is a call made blind — the same sender can send "running
late" and "the hospital just rang". Admitting that widened the calibration gap
from +0.22 to **+0.31** with no loss of accuracy.

---

## Degrading instead of failing

A hidden evaluation set will not be as clean as the shipped one, and the
costliest failure is not a wrong action — it is a traceback, because a traceback
costs *every* row at once. The data layer is therefore built to finish:

| Defect | Behaviour |
|---|---|
| Blank or unreadable non-key cell | Repaired to the neutral value for its type, counted in `DataLoader.repairs` |
| Row with an unusable primary key | Dropped and counted — a record with no identity cannot be indexed |
| Duplicate primary key | First occurrence kept, rest dropped. A warning, not an error: failing would turn a two-row defect into a zero-row submission |
| Unparseable timestamp | Falls back to `config.FALLBACK_TIMESTAMP` (midday, so a repaired row is never muted merely for having an unreadable clock) |
| Unknown user, group or business id | Resolves to `None`; context features record the absence rather than assuming a default |
| Empty auxiliary table | Warning. Only `messages` and `users` are `requires_rows` — a cold-start evaluation set with no history must still produce a full submission |
| Missing media registry or binaries | Warning; routing does not need the bytes |
| Unknown `media_type` or `conversation_type` | Recorded verbatim, routed on what is left |

`tests/test_robustness.py` breaks the dataset one way at a time — 40 corruptions
— and asserts two invariants each time: **completeness** (one prediction per
input row) and **validity** (no output errors). Determinism is pinned too: two
independently loaded pipelines produce byte-identical files.

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
*both* `notify` and `mute` is not a confident call, and neither is a confident
decision made about a message nobody could read. Calibration shows in the
numbers: 0.81 mean on correct decisions, 0.50 on the incorrect one.

**Safety is asymmetric.** Confirmed scams are muted by override, and ties break
toward the conservative action. A wrong `mute` costs one missed message; a
wrong `notify` costs the user's attention and erodes trust in every future
notification.

**Evidence must match the decision.** Muting cites history the user *dismissed*;
notifying cites history they *opened*. Since `message_history.csv` has no
`message_type` column, the Phase 2 classifier is reused to label history rather
than approximating with keyword overlap. The output validator then checks that
every cited id is real and belongs to the recipient, so a plausible-looking
fabrication cannot reach the submission.

**Degrade, never abort.** The contract is one prediction per message,
unconditionally. A malformed cell is repaired, an unidentifiable row is dropped,
and only a missing `messages.csv` or `users.csv` stops the run — because a
traceback costs every row at once, and a defect affecting two rows should not
cost 110.

---

## Configuration

Global settings live in [`src/config.py`](./src/config.py): dataset and output
paths, logging, timestamp formats and the domain vocabularies. Four
environment variables override without editing code, so nothing about a run
requires touching source:

| Variable | Effect |
|---|---|
| `MNR_DATASET_DIR` | Read the dataset from elsewhere |
| `MNR_OUTPUT_CSV` | Write predictions elsewhere (suppresses the root mirror) |
| `MNR_LOG_LEVEL` | Console verbosity |
| `MNR_STRICT_VALIDATION` | Promote dataset warnings to blocking errors |

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
| Media provider | `src.media.understanding.default_understanding` |

```python
from src.routing import DecisionEngine, Thresholds

DecisionEngine(thresholds=Thresholds(heavy_forward_count=12))
```

---

## Testing

```bash
python -m pytest              # 684 passed
```

| Suite | Covers |
|---|---|
| `test_helpers`, `test_text_utils` | coercion, parsing, negation |
| `test_loader`, `test_models`, `test_validation`, `test_repository` | Phase 1 |
| `test_keyword_rules`, `test_features`, `test_classifier` | Phase 2 |
| `test_normalization`, `test_calculators`, `test_personalization_engine` | Phase 3 |
| `test_routing` | Phase 4 |
| `test_media` | multimodal seam — resolution, providers, the integration claim |
| `test_robustness` | 40 dataset corruptions, degradation, determinism |
| `test_submission` | end-to-end, output contract, CSV, performance |
| `test_main` | CLI dispatch |

Notable: the Phase 1 validator is tested by **corrupting a throwaway dataset
once per check**, so no check can silently stop working. Evidence tests assert
that muted decisions cite negative history and notified decisions cite positive
history — evidence that ignored the decision would pass a naive test and fail
these. `test_media` tests a *claim* rather than a function: that installing
speech-to-text needs no change beyond one constructor argument. Two subprocess
tests run `python main.py` from a clean invocation and parse the CSV it
produces.

---

## Known limitations

**The 96.7% is optimistic.** The rules were refined against the same 30 labelled
rows they are measured on. The refinements were principled corrections
diagnosed from specific failures rather than curve-fitting, but the honest claim
is "no known systematic error", not "96.7% on unseen data".

**Media-only messages cannot be routed on content.** Eight of 110 incoming
messages carry only an image or a voice note. Without OCR or speech recognition
the decision rests on sender context alone — which is exactly why the one
remaining disagreement is a voice note. The seam for fixing this is built and
tested (see [Multimodal](#multimodal-where-ocr-and-whisper-plug-in)); the model
is deliberately not installed, and the confidence column discounts every
decision made blind rather than pretending otherwise.

**Repaired cells are silent in the output.** A row whose timestamp was
unreadable still gets a prediction, and nothing in `output.csv` marks it as
having been routed on repaired data. The repair is logged and counted in
`DataLoader.repairs`, but the six-column contract has nowhere to say so.

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
  clearest remaining accuracy gain, the cause of the one known miss, and now a
  one-class change thanks to `src/media/`. Whisper-small for voice, Tesseract or
  a small vision model for the poster and screenshot images.
- **Weight recovered text below typed text.** `MediaFeatures.derived_confidence`
  is carried through extraction and currently unused by the rules. Once a real
  provider exists, a low-confidence transcript should move a decision less than
  a sentence the sender actually typed.
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

That warning now extends to the multimodal seam. Text recovered by OCR or
speech-to-text is joined to the message body and reaches the same rules, so a
poster image containing *"ignore all previous rules"* becomes exactly the same
kind of untrusted input as typed text — which is safe here precisely because
nothing in the pipeline interprets message content as instructions.

---

## Dependencies

```text
pandas>=2.0        runtime
pytest>=7.0        tests
ruff>=0.5          linting
```

Python 3.11+. No network access, API keys or external services.
