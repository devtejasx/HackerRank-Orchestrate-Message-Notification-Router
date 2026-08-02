"""Central configuration for the Message Notification Router.

Every path, tunable and shared constant lives here so that no other module
needs to hardcode a filesystem location or a magic value.

The dataset directory can be overridden with the ``MNR_DATASET_DIR``
environment variable, which keeps the package usable from tests, notebooks
and future evaluation harnesses without editing source.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root (``src/config.py`` -> ``src`` -> project root).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

_DATASET_DIR_ENV_VAR: Final[str] = "MNR_DATASET_DIR"
_DEFAULT_DATASET_DIRNAME: Final[str] = "dataset"


def resolve_dataset_dir() -> Path:
    """Return the dataset directory, honouring the environment override.

    Returns:
        Absolute path to the directory holding the participant-facing CSVs.
    """
    override = os.environ.get(_DATASET_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / _DEFAULT_DATASET_DIRNAME


#: Directory holding every participant-facing CSV.
DATASET_DIR: Final[Path] = resolve_dataset_dir()

_OUTPUT_PATH_ENV_VAR: Final[str] = "MNR_OUTPUT_CSV"
_DEFAULT_OUTPUT_FILENAME: Final[str] = "output.csv"


def resolve_output_path() -> Path:
    """Return the path predictions are written to.

    Overridable with ``MNR_OUTPUT_CSV`` so a run can write somewhere else
    without touching the shipped template.
    """
    override = os.environ.get(_OUTPUT_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return DATASET_DIR / _DEFAULT_OUTPUT_FILENAME


#: Where ``python main.py`` writes its predictions.
OUTPUT_CSV: Final[Path] = resolve_output_path()

#: Root of the binary media referenced by ``images.csv`` / ``voice_notes.csv``.
#: Both CSVs store paths relative to :data:`DATASET_DIR`, e.g. ``media/images/img_001.jpg``.
MEDIA_DIR: Final[Path] = DATASET_DIR / "media"
IMAGE_DIR: Final[Path] = MEDIA_DIR / "images"
AUDIO_DIR: Final[Path] = MEDIA_DIR / "audio"

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_LOG_LEVEL_ENV_VAR: Final[str] = "MNR_LOG_LEVEL"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """Install a single console handler for the whole application.

    Safe to call more than once; repeat calls only adjust the level rather
    than stacking duplicate handlers.

    Args:
        level: Explicit level name or number. Falls back to the
            ``MNR_LOG_LEVEL`` environment variable, then to
            :data:`DEFAULT_LOG_LEVEL`.
    """
    resolved = level or os.environ.get(_LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(resolved)
        return
    logging.basicConfig(level=resolved, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger under a common application prefix."""
    return logging.getLogger(f"mnr.{name}")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

#: Accepted datetime layouts, most specific first. The dataset uses
#: minute-precision timestamps; the seconds variant is tolerated defensively.
TIMESTAMP_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
)

#: Accepted date-only layouts (``joined_at``, ``groups.created_at``, ``date``).
DATE_FORMATS: Final[tuple[str, ...]] = ("%Y-%m-%d",)

#: Strings that mean "no value" once surrounding whitespace is stripped and
#: lowercased.
#:
#: Deliberately narrow. Every null in this dataset is an empty cell, so the
#: only other entries are stringified pandas sentinels. Words like ``none``
#: and ``na`` are *not* listed: ``sample_messages.evidence_message_ids`` uses
#: the literal string ``none`` as a meaningful sentinel, and treating such
#: words as null would silently corrupt free-text columns.
NULL_TOKENS: Final[frozenset[str]] = frozenset({"", "nan", "<na>", "nat"})

#: Literals accepted for the dataset's 0/1 integer booleans.
TRUE_TOKENS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y"})
FALSE_TOKENS: Final[frozenset[str]] = frozenset({"0", "false", "f", "no", "n"})

#: Separator used inside ``do_not_disturb_window`` (e.g. ``22:00-07:00``).
DND_WINDOW_SEPARATOR: Final[str] = "-"

# --------------------------------------------------------------------------- #
# Domain vocabulary
#
# Declared here for reuse by later phases (classification, routing, output).
# Hour 1 only uses these to validate the values actually present in the CSVs.
# --------------------------------------------------------------------------- #

CONVERSATION_TYPES: Final[tuple[str, ...]] = ("personal", "group", "business")
MEDIA_TYPES: Final[tuple[str, ...]] = ("image", "voice")
MEMBER_ROLES: Final[tuple[str, ...]] = ("admin", "member")

#: Routing decisions produced by a later phase. Unused in Hour 1.
ROUTING_ACTIONS: Final[tuple[str, ...]] = ("notify", "digest", "mute")

#: Message categories produced by a later phase. Unused in Hour 1.
MESSAGE_TYPES: Final[tuple[str, ...]] = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

# --------------------------------------------------------------------------- #
# Validation behaviour
# --------------------------------------------------------------------------- #

#: When true, warning-level validation issues are promoted to hard failures.
#: Structural problems (missing file/column, duplicate primary key, empty
#: required table) always raise regardless of this setting.
STRICT_VALIDATION: Final[bool] = os.environ.get("MNR_STRICT_VALIDATION", "0") in TRUE_TOKENS

#: Cap on how many example offenders a single validation issue reports.
MAX_ISSUE_EXAMPLES: Final[int] = 5

# --------------------------------------------------------------------------- #
# Output format
#
# The submission contract, fixed by the challenge. These are not tunable.
# --------------------------------------------------------------------------- #

#: Encoding used when writing predictions.
OUTPUT_ENCODING: Final[str] = "utf-8"

#: Line terminator. Explicit so output is byte-identical on every platform
#: rather than depending on the host's newline convention.
OUTPUT_LINE_TERMINATOR: Final[str] = "\n"

#: Decimal places kept on the confidence column.
CONFIDENCE_DECIMALS: Final[int] = 2

# --------------------------------------------------------------------------- #
# Where the remaining tunables live
#
# Deliberately *not* gathered here. Each group is a typed, documented dataclass
# next to the code it governs, which keeps every value beside the reasoning
# that justifies it and lets callers override a whole group by passing one
# object. A flat list of constants in this module would separate the numbers
# from their rationale and make that override impossible.
#
#   Classification weights .... src.classifier.rules.Weights
#   Classifier confidence ..... src.classifier.confidence.ConfidenceModel
#   Keyword vocabularies ...... src.classifier.keyword_rules.DEFAULT_KEYWORDS
#   Engagement weighting ...... src.features.historical_features.EngagementWeights
#   Routing rule thresholds ... src.routing.rules.Thresholds
#   Routing type priors ....... src.routing.rules.TYPE_PRIORS
#   Routing confidence ........ src.routing.confidence.CalibrationModel
#
# See the Configuration section of README.md.
# --------------------------------------------------------------------------- #
