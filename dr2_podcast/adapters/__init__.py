"""Stage adapters: each phase as a function of ``(run_dir, run_config)``.

The shared state reconstruction lives in ``_common``; the adapters themselves are in
``research_stages`` (phases 0-3) and ``script_stages`` (phases 4-8), split only to keep each file
under the repo's size ceiling. Import everything from ``dr2_podcast.adapters``.

**Adapters fail closed.** The phases they replace do not, in places: ``phase_0_framing`` catches
every exception from the framing crew, logs "continuing", and returns an empty string, so a run
whose framing never happened proceeds to search for nothing in particular. A stage that produced
nothing is a failed stage.
"""

from __future__ import annotations

from dr2_podcast.adapters._common import (
    SESSION_ROLES_ARTIFACT,
    drop_unproduced_optional_outputs,
    promote,
    staging_dir,
)
from dr2_podcast.adapters.research_stages import framing, translate, url_validation
from dr2_podcast.adapters.script_stages import audio, audit, blueprint, draft, polish

def registered() -> tuple[str, ...]:
    """Stage names this module registers. Imported for its side effects; this makes that visible."""
    from dr2_podcast.stage import ADAPTERS

    return tuple(sorted(ADAPTERS))


__all__ = [
    "SESSION_ROLES_ARTIFACT",
    "drop_unproduced_optional_outputs",
    "promote",
    "staging_dir",
    "audio",
    "audit",
    "blueprint",
    "draft",
    "framing",
    "polish",
    "registered",
    "translate",
    "url_validation",
]
