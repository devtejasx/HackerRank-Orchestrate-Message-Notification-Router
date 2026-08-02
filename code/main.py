"""Alternate entry point at the location ``AGENTS.md`` suggests.

The real entry point is ``main.py`` at the repository root; this delegates to
it so both invocations behave identically:

    python main.py
    python code/main.py

Arguments pass straight through, so every flag documented in the README works
here too.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts `code/` on sys.path rather than the
# repository root, so the project package would not otherwise be importable.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from main import main  # noqa: E402  (import must follow the path fix)

if __name__ == "__main__":
    sys.exit(main())
