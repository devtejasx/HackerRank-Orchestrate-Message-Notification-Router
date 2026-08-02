"""Evaluation entry point at the location ``AGENTS.md`` suggests.

Scores the system against the labelled examples in ``sample_messages.csv``:

    python code/evaluation/main.py

Equivalent to ``python main.py --evaluate``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts its own directory on sys.path rather than the
# repository root, so the project package would not otherwise be importable.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from main import main  # noqa: E402  (import must follow the path fix)

if __name__ == "__main__":
    sys.exit(main(["--evaluate", *sys.argv[1:]]))
