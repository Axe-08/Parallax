#!/usr/bin/env python3
"""CLI wrapper to run the Parallax 200k medium benchmark with real-time logging."""

import sys
from pathlib import Path

# Ensure src is on python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from parallax.experiments.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
