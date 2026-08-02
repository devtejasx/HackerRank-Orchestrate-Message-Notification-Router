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
| **Action accuracy** vs the 30 labelled rows | **93.3%** (28/30) |
| **`message_type` accuracy** | **93.3%** (28/30) |
| Labelled scams delivered to a user | **0** |
| Confidence, correct vs incorrect decisions | **0.83 vs 0.59** (gap **+0.24**) |
| Evidence ids valid and correctly scoped | **77/77** |
| Evidence whose reaction fits the action | **94%** |
| Dataset corruptions survived with a complete, valid submission | **40/40** |
| Voice notes transcribed | **13/13** |
| Tests | **735 passing** |
| Full run over 110 messages | **~1 second** (1.9 ms/message, linear to 5,500) |

> Measured against the same 30 labelled rows the system was tuned on, so it is
> an optimistic estimate rather than held-out performance. The dataset provides
> no other labelled actions. See [Known limitations](#known-limitations).

**This number went down when Whisper was added — 96.7% to 93.3% — while actual
routing quality went up.** Both statements are true and the tension is worth
understanding before reading further:

- On the **110 real messages**, transcription changed 5 rows and 4 are clearly
  better: a genuine airport-booking change is no longer muted as spam, a
  same-day school transport update now interrupts, credit-card telemarketing is
  muted, and an OTP phishing attempt is now typed `scam` rather than `spam`.
- On the **30 labelled rows**, it cost one: `sample_msg_043` is a call-centre
  recording whose transcript contains the polite phrase *"who can help you
  out"*. `help` is an urgency keyword, so the row flips from `mute`/`spam` to
  `notify`/`urgent`.

The labelled set contains 3 voice notes; the dataset contains 13. A 30-row
proxy moved by one row is a weaker signal than 5 changed rows judged against
their own audio, which is why transcription ships enabled. The details, and
the measurement showing that removing `help` makes things *worse* overall, are
in [Known limitations](#known-limitations).

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

Ahead of Phase 2 sits the **multimodal seam** (`src/media/`): attachments are
resolved to files on disk and handed to a speech-to-text or OCR provider, and
whatever text comes back joins the message body before extraction runs. Voice
notes are transcribed with Whisper; images are not read yet. See
[Voice transcription](#voice-transcription).

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
| `python -m pytest` | 735 tests |

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
├── output.csv                  the submission (mirrored to dataset/)
├── transcripts.json            cached voice transcripts, keyed by media_id
├── requirements.txt
├── requirements-voice.txt      optional: faster-whisper + audio decoders
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
│   ├── media/                  multimodal — speech-to-text, and the OCR seam
│   │   ├── content.py          attachment + recovered-content records
│   │   ├── understanding.py    the provider interface + default resolution
│   │   ├── whisper.py          faster-whisper speech-to-text
│   │   ├── cache.py            transcripts.json, keyed by media_id
│   │   └── resolver.py         registry lookup, disk check, per-run caching
│   ├── evaluation/             measurement against labelled rows
│   ├── pipeline.py             Phases 1–3
│   └── utils/                  coercion, text analysis
└── tests/                      735 tests
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

## Voice transcription

23 of 110 incoming messages carry an image or a voice note, and 8 of those have
no text at all. **Voice notes are transcribed with Whisper** before they enter
the pipeline. Images are not read; OCR is still unimplemented.

The load-bearing decision is that **a transcript is not a special case**. It is
appended to the typed body before feature extraction, so length, URLs,
currency, scam vocabulary and urgency all cover it automatically. Nothing in
feature extraction, classification, personalisation or routing was changed to
support speech:

```text
voice note
  → MediaResolver          registry lookup, then filesystem check
  → WhisperTranscriber     faster-whisper
  → TranscriptCache        keyed by media_id, persisted to transcripts.json
  → MediaFeatures          transcript + provenance + confidence
  → FeatureExtractor       the transcript becomes the message body
  → classifier → personalisation → routing        (all unchanged)
```

`tests/test_whisper.py::TestVoiceEqualsText` is the guard on that claim: for
five different transcripts it asserts that a voice note transcribing to some
text produces the *same* tokens, keywords, category and action as a text
message containing that text. If a special case ever creeps in, those fail.

### Installation

Transcription is optional. The project runs, and routes voice notes, without it:

```bash
pip install -r requirements.txt          # core — always needed
pip install -r requirements-voice.txt    # optional — adds live transcription
```

Three levels, resolved automatically by `default_understanding()`:

| Available | Behaviour |
|---|---|
| `faster-whisper` installed | Voice notes are transcribed; results cached to `transcripts.json` |
| Not installed, but `transcripts.json` present | Committed transcripts are reused — full benefit, no dependency |
| Neither | Voice notes route on sender context alone, exactly as before Whisper |

**`transcripts.json` is committed deliberately.** It is a cache, not a label
file: derived from `dataset/media/audio/` by a documented command, keyed by
`media_id`, and fingerprinted against the audio so replacing a file invalidates
its entry. Committing it means a grader without `faster-whisper` still sees the
system as designed. Rebuild it any time with:

```bash
python main.py --refresh-transcripts
```

`faster-whisper` downloads model weights on first use (~150 MB for `base`) and
decodes audio with PyAV. Where PyAV cannot load — a locked-down host, a slim
container — `best_available_decoder()` falls back to ffmpeg or libsndfile
rather than losing transcription entirely.

### Controls

| Flag | Effect |
|---|---|
| `--no-transcribe` | Skip speech-to-text; voice notes route on context alone |
| `--whisper-model SIZE` | `tiny`, `base` (default), `small`, … |
| `--refresh-transcripts` | Discard the cache and re-transcribe from audio |

### What it changed

Transcription moved 5 of 110 rows, **all of them voice notes**:

| Message | Before | After | Transcript |
|---|---|---|---|
| `msg_081` | digest / personal | **notify / event** | *"this is from School Transport. Today's pickup will be from Gate 2…"* |
| `msg_084` | digest / business_update | **mute / promotion** | credit-card telemarketing, *"press 1 now"* |
| `msg_086` | mute / spam | **digest / business_update** | *"Your airport pickup for tomorrow has moved to 6.15 a.m."* |
| `msg_085` | mute / spam | **mute / scam** | *"Your bank account will be blocked today. Share the OTP…"* |
| `msg_087` | digest / personal | digest / event | real-estate robocall, *"dial eight to… unsubscribe"* |

Four are clear improvements — a real booking change is no longer muted as spam,
a same-day school logistics update now interrupts, telemarketing is muted, and
an OTP phishing attempt is now typed `scam` rather than `spam`. The fifth
(`msg_087`) is still wrong: it should be `mute`/`promotion`, and `event` is no
better than the `personal` it replaced.

**On the labelled set this reads as a regression**, and the honest number is
below: 96.7% → 93.3%. See [Known limitations](#known-limitations) for why the
two figures disagree.

Two second-order effects, both intended:

- **Historical voice notes become classifiable too.** The evidence engine
  classifies `message_history.csv` with the same classifier, so a transcribed
  historical voice note can now be cited as evidence — which is how one *text*
  message (`msg_089`) gained evidence it previously had none of. Its action and
  category are unchanged; a test pins that this is the only way transcription
  may touch a non-voice row.
- **The opacity penalty stops applying.** A message whose content could not be
  read takes an explicit confidence penalty
  (`CalibrationModel.opacity_penalty`). Transcribing a voice note removes it,
  because the decision is no longer being made blind.

### Guarantees

| Guarantee | Why it matters |
|---|---|
| A provider never sees an unreadable file | The resolver checks the registry *and* the filesystem first |
| A provider that raises cannot fail the run | `SafeUnderstanding` wraps every provider; a crash costs one transcript, not `output.csv` |
| Missing, corrupt and unsupported audio all return an empty transcript | The contract requires a prediction for every message, including ones no model could read |
| Each attachment is transcribed at most once, ever | In-memory within a run, `transcripts.json` across runs |
| A failed transcription is never cached | Failures are usually environmental; caching one would make a missing install permanent |
| A transcript never scores above 0.95 confidence | It is evidence about the message, not the message itself |

### Adding OCR

The seam is unchanged and still open. An image provider is the same one-class
change Whisper was:

```python
class TesseractOCR:
    name = "tesseract"

    def supports(self, modality):
        return modality is MediaModality.IMAGE

    def understand(self, attachment):
        return MediaContent(text=pytesseract.image_to_string(attachment.file_path),
                            provider=self.name, confidence=0.6)

RoutingPipeline.load(understanding=CompositeUnderstanding(WhisperTranscriber(), TesseractOCR()))
```

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
python -m pytest              # 735 passed
```

| Suite | Covers |
|---|---|
| `test_helpers`, `test_text_utils` | coercion, parsing, negation |
| `test_loader`, `test_models`, `test_validation`, `test_repository` | Phase 1 |
| `test_keyword_rules`, `test_features`, `test_classifier` | Phase 2 |
| `test_normalization`, `test_calculators`, `test_personalization_engine` | Phase 3 |
| `test_routing` | Phase 4 |
| `test_media` | multimodal seam — resolution, providers, the integration claim |
| `test_whisper` | transcriber, transcript cache, and voice/text equivalence |
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

**The labelled-set score and real routing quality disagree, and the labelled
score is the one that went down.** Detailed above; the short version is that
transcription improved 4 of the 5 rows it moved on the full dataset and cost 1
of 30 on the labelled proxy. Shipping it enabled is a judgement call, made on
the strength of the evidence rather than the headline number, and the headline
number is reported unadjusted.

**`sample_msg_043` is a real miss, not a measurement artefact.** A call-centre
recording transcribes to text containing *"who can help you out"*; `help` is an
urgency keyword, so it routes `notify`/`urgent` when it should be `mute`/`spam`
— the worst kind of error, interrupting a user with spam. The obvious fix is to
drop the bare keyword `help`, and that was measured rather than assumed: doing
so drops `both correct` from 93.3% to **90.0%**, because it breaks
`sample_msg_051`, where `help` is a genuine distress signal. The keyword stays.
Spoken language carries far more politeness filler than typed text, and the
urgency vocabulary was tuned on the latter; adapting it properly needs labelled
*speech*, which the dataset does not provide.

**Images are still not read.** 15 of 110 incoming messages carry an image, and
their captions are the only signal. OCR would use exactly the seam Whisper
does — see [Adding OCR](#adding-ocr) — and is the clearest remaining gain.

**Transcripts are cached, so a fresh dataset costs model time.** Voice notes
not in `transcripts.json` need `faster-whisper` installed to be transcribed;
without it they fall back to sender context and the confidence column discounts
them. A hidden evaluation set would be entirely uncached.

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

- **OCR for the 15 image messages** — the clearest remaining accuracy gain now
  that voice is handled, and the same one-class change through `src/media/`.
  Tesseract or a small vision model for the poster and screenshot images.
- **Weight recovered text below typed text.** `MediaFeatures.derived_confidence`
  is carried through extraction and still unused by the rules. A transcript
  scored 0.34 should move a decision less than one scored 0.75, and less than a
  sentence the sender actually typed. This is now concrete rather than
  hypothetical: real confidences on this dataset range 0.34–0.76.
- **An urgency vocabulary tuned for speech.** `sample_msg_043` misroutes
  because *"who can help you out"* is polite call-centre filler that the
  urgency keyword `help` reads as distress. Typed messages rarely contain such
  filler; transcripts are full of it. Fixing this properly needs labelled
  speech, and removing the keyword outright measurably makes things worse.
- **A larger Whisper model.** `base` runs in about a second per note on CPU;
  `small` would likely resolve the mis-hearings visible in the transcripts
  (*"Dad is on well"* → *"Dad is unwell"* was corrected by moving up from
  `tiny`).
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

faster-whisper     optional — live voice transcription
imageio-ffmpeg     optional — audio decoding where PyAV cannot load
soundfile          optional — lighter alternative to the above
```

Python 3.11+. No API keys or external services. The core run needs no network;
`faster-whisper` downloads model weights on first use, and never again.
