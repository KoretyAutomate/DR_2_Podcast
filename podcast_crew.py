#!/usr/bin/env python3
"""
DR_2_Podcast — Research-driven debate podcast generator.

Thin shim that delegates to orchestrator.py.
Preserves backward compatibility with existing CLI usage:
  python podcast_crew.py --topic "..." --language en
  python podcast_crew.py --topic "..." --reuse-dir DIR --crew3-only
  python podcast_crew.py --topic "..." --reuse-dir DIR --check-supplemental

The actual implementation is in:
  orchestrator.py      — Central pipeline controller
  shared/              — Config, models, progress, tools, PDF utils
  flows/f01-f08        — Numbered sub-flow modules

Original monolithic code preserved at: podcast_crew_original.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from flows.f01_parameter_gathering import gather_params_from_cli
from orchestrator import run_pipeline

if __name__ == "__main__":
    params = gather_params_from_cli()
    run_pipeline(params)
