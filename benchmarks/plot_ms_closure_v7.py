#!/usr/bin/env python3
"""Deprecated name: use ``plot_ms_closure_v7.2.py`` (MS + SI v7.2)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

print(
    "plot_ms_closure_v7.py is a stub; running plot_ms_closure_v7.2.py",
    file=sys.stderr,
)
runpy.run_path(str(Path(__file__).with_name("plot_ms_closure_v7.2.py")), run_name="__main__")
