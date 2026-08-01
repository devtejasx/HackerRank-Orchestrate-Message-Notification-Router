# Hour 1 — Data Layer

A reusable, validated, indexed foundation for the Message Notification Router.

**Scope:** loading, typing, validating and indexing the dataset. Nothing else.
No classification, routing, spam detection, OCR, ASR, embeddings, LLMs or
`output.csv` generation — those belong to later phases and are deliberately
absent.

The contract for everything that comes next: **no module outside `src/data/`
ever reads a CSV.**

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python main.py
```

```python
from src.data import DataRepository

repo = DataRepository.load()                 # load + validate + index

user     = repo.get_user("u_001")            # -> User | None
groups   = repo.get_user_groups("u_001")     # -> tuple[Group, ...]
recent   = repo.get_user_history("u_001", limit=10, newest_first=True)
event    = repo.get_message_event("message_0001")
poster   = repo.get_media_path(repo.get_message("msg_005"))   # -> Path | None
```

### CLI

```bash
python main.py                  # schema summary, load, validate, index, lookups
python main.py --schema-only    # print the schema and exit
python main.py --strict         # treat validation warnings as failures
python main.py --log-level DEBUG
python -m pytest                # 387 tests (Phases 1 and 2)
```

Environment overrides: `MNR_DATASET_DIR`, `MNR_LOG_LEVEL`, `MNR_STRICT_VALIDATION`.

---

## Layout

```text
src/
├── config.py              # paths, logging, parse formats, domain vocabulary
├── data/
│   ├── schema.py          # ← declarative table registry (single source of truth)
│   ├── models.py          # frozen dataclasses, one per CSV row
│   ├── loader.py          # reads each CSV exactly once
│   ├── validation.py      # 15 checks, all driven by schema.py
│   ├── indexes.py         # 26 precomputed lookup tables
│   └── repository.py      # ← the public facade later phases should import
└── utils/
    └── helpers.py         # total coercion/parsing helpers
main.py                    # smoke test entry point
tests/                     # Phase 1 tests (see PHASE_2.md for the rest)
```

### Why `schema.py` exists

`schema.py` declares every table once — columns, types, nullability, primary
keys, foreign keys, closed value sets. The loader, validator and index builder
all read from it. Adding a column is a one-line change in one file rather than
four coordinated edits, and `test_loader.py` asserts the dataclasses never
drift from the declarations.

### Why `repository.py` was added

The requested structure had no obvious home for the helper methods. Putting
them on the index would have mixed "build lookups" with "serve queries", so
`DataRepository` is the single public entry point and `DataIndex` stays a pure
data structure.

---

## Dataset findings

Derived by profiling the CSVs, not assumed.

| File | Rows | Primary key | Notes |
|---|---:|---|---|
| `users.csv` | 54 | `user_id` | |
| `groups.csv` | 23 | `group_id` | |
| `group_members.csv` | 401 | (`group_id`, `user_id`) | composite |
| `business_accounts.csv` | 110 | `business_id` | 5 rows lack `official_domain` |
| `user_business_history.csv` | 106 | (`user_id`, `business_id`) | composite |
| `messages.csv` | 110 | `message_id` | ids are `msg_*` |
| `message_history.csv` | 412 | `message_id` | ids are `message_*` |
| `message_events.csv` | 412 | `message_id` | **1:1** with history |
| `images.csv` | 20 | `image_id` | |
| `voice_notes.csv` | 13 | `voice_note_id` | |
| `daily_notification_summary.csv` | 756 | (`user_id`, `date`) | composite |
| `sample_messages.csv` | 30 | `message_id` | optional, reference only |

Things worth knowing before building on this:

- **All 15 foreign keys resolve cleanly.** Zero violations in the shipped data.
- **`user_id` is the recipient; `sender_user_id` is the sender.** Different
  people. The indexes keep them strictly separate (`*_by_user` vs `*_by_sender`),
  and a test enforces it.
- **`messages` and `message_history` share an identical 11-column envelope.**
  Modelled once as `MessageRecord` and subclassed, so they cannot drift.
- **Incoming and historical ids live in separate namespaces** (`msg_023` vs
  `message_0107`), so `get_message` and `get_history_message` never collide.
- **Media keys are not called `media_id` at rest.** A message's `media_id`
  resolves to `images.image_id` or `voice_notes.voice_note_id` depending on
  `media_type`. `get_media_path()` hides this.
- **`conversation_type` strictly determines which reference columns are set:**
  `personal` → `sender_user_id`; `group` → `group_id` + `sender_user_id`;
  `business` → `business_id`. Holds across all 552 message rows and is
  validated.
- **`evidence_message_ids` uses the literal string `none` as a sentinel.** So
  `none`/`na`/`null` are deliberately *not* treated as null tokens anywhere;
  only empty cells and pandas artifacts are.
- **Two timestamp layouts:** `%Y-%m-%d %H:%M` and date-only `%Y-%m-%d`. `date`
  and `datetime` are kept distinct in the models.
- **`groups.member_count` exceeds the number of `group_members` rows** (e.g.
  group_005 declares 241 members, 30 rows ship). The CSV is a sample of
  memberships, not the full roster — do not treat the row count as group size.
- **`vn_001`–`vn_003` are referenced only by `sample_messages.csv`**, not by
  `messages.csv` or `message_history.csv`.

---

## Typing

Three views of every table, each cached and computed once:

| Accessor | Shape | Use for |
|---|---|---|
| `loader.raw_frame(t)` | all strings, blanks kept as `""` | validation |
| `loader.frame(t)` / `loader.messages` | `Int64`, `boolean`, `datetime64`, `string` | pandas analysis |
| `loader.records(t)` | frozen dataclasses | everything else |

Nullable extension dtypes throughout, so a missing integer stays missing
instead of silently becoming a float.

---

## Validation

15 checks, all schema-driven. Severity is the important distinction:

**ERROR — raises `DatasetValidationError`**

`missing_file` (required) · `missing_columns` · `empty_table` (required) ·
`duplicate_primary_key` · `blank_primary_key`

**WARNING — logged, load continues**

`missing_file` (optional) · `unexpected_columns` · `unexpected_null` ·
`malformed_int` / `malformed_float` / `malformed_bool` / `malformed_date` /
`malformed_timestamp` · `unexpected_value` · `broken_reference` ·
`conversation_reference_missing` · `conversation_reference_unexpected` ·
`media_pair_mismatch` · `unknown_media_id` · `missing_media_file`

A table that fails a structural check is skipped for the remaining checks, so
one missing column cannot produce a cascade of unrelated noise.

`--strict` promotes every warning to an error.

**The shipped dataset produces zero errors and zero warnings.** Because a
validator that never fires proves nothing, `tests/test_validation.py` corrupts
a throwaway copy of the dataset once per check and asserts each one fires.

---

## Indexes

26 lookup tables, built once, exposed read-only via `MappingProxyType`.
Downstream code should never filter a DataFrame in a loop.

- **Unique:** `users_by_id`, `groups_by_id`, `business_by_id`, `messages_by_id`,
  `history_by_id`, `images_by_id`, `voice_by_id`, `samples_by_id`,
  `group_member_by_key`, `user_business_by_key`, `event_by_message`,
  `notification_summary_by_key`
- **Grouped:** `messages_by_{user,sender,group,business}`,
  `history_by_{user,sender,group,business}`, `events_by_user`,
  `group_members_by_{group,user}`, `user_business_by_{user,business}`,
  `notification_summary_by_user`

Message collections are sorted oldest-first, with `message_id` as a tiebreaker,
so ordering is deterministic across runs.

### Return-value convention

- Single-entity getters return `None` when the id is unknown.
- Collection getters return an **empty tuple**, never `None`, so callers can
  iterate without a guard.

This deviates from a literal reading of "return None if not found", which would
force a `None` check before every loop. `get_message_events()` is provided in
the requested collection form alongside the singular `get_message_event()`,
since events are 1:1 with historical messages.

---

## Testing

```bash
python -m pytest              # 387 passed across both phases
```

| File | Covers |
|---|---|
| `test_helpers.py` | coercion, temporal parsing, null sentinels, grouping |
| `test_loader.py` | schema/model consistency, dtypes, caching, failure modes |
| `test_models.py` | construction from every real row, required fields, properties |
| `test_validation.py` | one purpose-built corruption per check |
| `test_repository.py` | index integrity, lookups, ordering, media resolution |
| `test_main.py` | CLI exit codes and output sections |

---

## Consuming this layer

Depend on `DataRepository` only.

```python
from src.data import DataRepository

repo = DataRepository.load()

for message in repo.get_messages():
    user      = repo.get_user(message.user_id)
    history   = repo.get_user_history(message.user_id, limit=20, newest_first=True)
    business  = repo.get_business(message.business_id) if message.business_id else None
    relation  = repo.get_user_business(message.user_id, message.business_id) if business else None
    member    = repo.get_group_member(message.group_id, message.user_id) if message.group_id else None
    media     = repo.get_media_path(message)      # OCR / ASR will read from here
```

Everything a routing decision needs is one dict lookup away.

**Phase 2 already builds on this**, turning each message into features and a
classification — see [`PHASE_2.md`](./PHASE_2.md). Code written after Phase 2
should generally depend on `MessagePipeline` rather than reaching for the
repository directly, and drop to `DataRepository` only for raw records the
feature layer does not carry.
