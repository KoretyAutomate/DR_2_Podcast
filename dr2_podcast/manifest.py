"""The run manifest: what each stage consumed, produced, and ran under.

PLAN.md Step 8. The first draft of the plan asserted that the existing ``--resume`` and Prefect
caching "stay valid" for fourteen independently re-runnable stages. That is an assertion, not a
design, and it is not true as written: ``session_metadata.txt`` records no completed-phase list
(``pipeline.py:590``), Prefect caching is keyed around whole-phase tasks (``pipeline.py:2984``),
and legacy checkpoint read errors already fall back to empty state (``pipeline.py:502``). Re-running
``keywords`` would leave search, synthesis, blueprint and script artifacts falsely current.

So staleness is *derived*, never assumed. A stage is current only while every artifact it recorded
as an input still hashes to what it recorded, under the same model and configuration. Anything else
is stale, and stale propagates along the graph in :mod:`dr2_podcast.stages`.

Transport retries are recorded as attempts with outcome ``transport`` and are deliberately NOT
revision rounds: a malformed or empty response is retried without consuming one of the three rounds
a loop is bounded at, because conflating them silently shortens the loop.
"""

from __future__ import annotations

import ast
import hashlib
from functools import cache
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import ArtifactError, read_json_strict, sha256_file, write_json_atomic
from dr2_podcast.schemas import SchemaValidationError, schema_errors
from dr2_podcast.stages import MANIFEST_FILENAMES, downstream_of, get_stage, producer_of, resolve

MANIFEST_SCHEMA_VERSION = 1

#: Config attributes that are deliberately NOT part of a stage's identity, each with its reason.
#: Everything else public and uppercase on :mod:`dr2_podcast.config` IS hashed in.
#:
#: A hand-maintained allowlist was the first version of this and it was wrong within a day: it
#: named four settings and missed TTS_SPEED_SCALE, TTS_RANDOM_VOICE, TTS_INTONATION_OVERRIDES and
#: the rest, so an `.env` change that produces a different waveform left the audio stage "current".
#: An allowlist has to be remembered on every new setting; a denylist fails in the safe direction,
#: because over-invalidating costs a re-run while under-invalidating ships artifacts built under
#: settings that no longer hold. It also stopped naming environment variables: LLM_BASE_URL is
#: exposed as SMART_BASE_URL (config.py:10), and naming the env var hashed None forever.
CONFIG_IDENTITY_EXCLUDE = frozenset(
    {
        # Where runs are written, not what they contain. Hashing it would make every stage stale on
        # a machine with a different output root.
        "OUTPUT_DIR_OVERRIDE",
    }
)

#: Environment variables that change what a run produces but live nowhere in ``dr2_podcast.config``.
#: They are read directly — ``initialise_run_globals`` takes the channel brief and accessibility
#: level, ``assign_roles`` takes PODCAST_HOSTS, the audio engine takes TTS_GLOSSARY_ENABLED — so a
#: scan of the config module alone misses every one of them and a changed channel brief would leave
#: completed stages "current" while the prompts moved underneath.
CONTENT_ENV_KEYS = (
    "ACCESSIBILITY_LEVEL",
    # The model a Claude-authored stage runs on. Its output is a judgement, so the model that made
    # it is part of what the artifact IS — not a detail of how it was produced. Required rather
    # than optional (claude_runner.resolve_model): an unset value hashes the same before and after
    # a CLI default changes, so hashing it would only LOOK like recording the authoring model.
    "DR2_CLAUDE_MODEL",
    "PODCAST_CHANNEL_INTRO",
    "PODCAST_CHANNEL_MISSION",
    "PODCAST_CORE_TARGET",
    "PODCAST_HOSTS",
    "PODCAST_LENGTH",
    "SEARXNG_URL",
    "TTS_API_URL",
    "TTS_ENGINE_EN",
    "TTS_ENGINE_JA",
    "TTS_GLOSSARY_ENABLED",
    "TTS_INTONATION_OVERRIDES",
    "TTS_INTONATION_SCALE",
    "TTS_JUDGE_BASE_URL",
    "TTS_JUDGE_CONCURRENCY",
    "TTS_JUDGE_MODEL",
    "TTS_RANDOM_VOICE",
    "TTS_SPEED_OVERRIDES",
    "TTS_SPEED_SCALE",
    "TTS_HOST1_ID",
    "TTS_HOST2_ID",
    "VLLM_MAX_CONCURRENCY",
    "VOICE_DUCKING_DB",
    "MODEL_NAME",
    "LLM_BASE_URL",
)

#: Environment variables read somewhere in the package that deliberately do NOT participate, each
#: with its reason. ``test_every_environment_read_in_the_package_is_classified`` scans the source
#: and fails on anything absent from both tuples, so a new read has to be classified rather than
#: silently ignored — which is how PODCAST_HOSTS and TTS_GLOSSARY_ENABLED were missed the first time.
ENV_IDENTITY_EXCLUDE = {
    # Where the Claude CLI lives, not what it writes. Hashing it would make every Claude-authored
    # stage stale on a machine that installed the binary somewhere else.
    "DR2_CLAUDE_BINARY",
    # Credentials. Rotating one does not change what was produced.
    "BRAVE_API_KEY",
    "LLM_API_KEY",
    "PUBMED_API_KEY",
    "S2_API_KEY",
    "PODCAST_WEB_PASSWORD",
    "PODCAST_WEB_USER",
    # Contact addresses sent to APIs as politeness headers. They identify us, not the content.
    "CROSSREF_MAILTO",
    "OPENALEX_EMAIL",
    "UNPAYWALL_EMAIL",
    # Where things are written or served, not what they contain.
    "OUTPUT_DIR",
    "PODCAST_WEB_BIND",
    "PODCAST_WEB_PORT",
    # Where the searxng-deep-research checkout lives (tools/evidence_search.py).
    # A path to the library, not an input to it: the same claim searched on two
    # machines that keep that checkout in different places must produce the same
    # evidence bundle, or the bundle was never about the claim.
    "SEARXNG_DEEP_RESEARCH_DIR",
    # The run's own inputs, which a staged run carries in meta/run_config.json instead. They are
    # already part of identity through the run config; taking them from the environment as well
    # would invalidate stages over a variable the staged path never reads.
    "PODCAST_TOPIC",
    "PODCAST_LANGUAGE",
}

#: Types safe to render into a stable fingerprint. Anything else on the module (a callable, a
#: module, an object) is not configuration and is skipped.
_IDENTITY_TYPES = (str, int, float, bool, tuple, list, dict, type(None))


def _canonical(value: Any) -> str:
    """Render a config value deterministically, so dict ordering cannot move the fingerprint."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_canonical(v)}" for k, v in sorted(value.items(), key=repr)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_canonical(v) for v in value) + "]"
    return repr(value)


#: Which settings each stage's identity is built from. A stage absent from this map gets ALL of
#: them, which is the safe default: over-invalidating costs a re-run, under-invalidating ships
#: artifacts built under settings that no longer hold.
#:
#: Scoped because a global fingerprint couples unrelated stages — changing TTS_SPEED_SCALE made
#: `framing` and `research` non-current, and since a stage's producers must be current before it
#: runs, an audio-only tweak forced the whole ~28-minute research chain to re-run before anything
#: could render.
CONFIG_GROUPS: dict[str, tuple[str, ...]] = {
    "llm": ("SMART_MODEL", "SMART_BASE_URL", "VLLM_MAX_CONCURRENCY", "LLM_TIMEOUT",
            "env:MODEL_NAME", "env:LLM_BASE_URL"),
    "research": ("SCREENING_TOP_N", "TIER_CASCADE_THRESHOLD", "MIN_TIER3_STUDIES", "MAX_TIER3_RATIO",
                 "EVIDENCE_LIMITED_THRESHOLD", "SEARXNG_URL", "PUBMED_TIMEOUT", "SCRAPING_TIMEOUT",
                 "VALIDATION_TIMEOUT", "USER_AGENT", "env:SEARXNG_URL"),
    # Its OWN group, not "llm". A Claude-authored stage never touches the Smart/vLLM backend, so
    # changing MODEL_NAME or LLM_BASE_URL would restale a frozen prior — and the whole research
    # chain behind it — over a setting it does not read (prepush codex 2026-08-20).
    "claude": ("env:DR2_CLAUDE_MODEL",),
    "prompt": ("env:PODCAST_CHANNEL_INTRO", "env:PODCAST_CHANNEL_MISSION",
               "env:PODCAST_CORE_TARGET",
               "env:ACCESSIBILITY_LEVEL", "env:PODCAST_HOSTS", "env:PODCAST_LENGTH"),
    "tts": (),  # every TTS_* setting plus the ducking level; matched by prefix below
}

STAGE_CONFIG_GROUPS: dict[str, tuple[str, ...]] = {
    "framing": ("llm", "prompt"),
    # The prior is Claude's judgement from background knowledge: the Claude model that made it and
    # the channel prompt settings, and nothing about the Smart backend or the search parameters.
    "framing_prior": ("claude", "prompt"),
    # The strategy comes from the model and the framing prompt, and from the research settings that
    # decide what a tier is — not from anything TTS or audio touches.
    "plan_search": ("llm", "research", "prompt"),
    "research": ("llm", "research", "prompt"),
    "url_validation": ("research",),
    "translate": ("llm", "prompt"),
    "blueprint": ("llm", "prompt"),
    "draft": ("llm", "prompt"),
    "polish": ("llm", "prompt"),
    "audit": ("llm", "prompt"),
    "audio": ("tts", "prompt"),
}


#: Bundled data a stage's OUTPUT depends on, hashed into its identity. Configuration is not the
#: only thing that changes what a stage produces: the TTS glossary is applied inside
#: ``clean_script_for_tts`` (``audio/engine.py:873``), so editing it changes the rendered speech
#: while the script stays byte-identical — PLAN.md Step 12 makes the same point about hashing the
#: TTS input rather than the script. Paths are relative to the repository root.
STAGE_DATA_ASSETS: dict[str, tuple[str, ...]] = {
    "audio": ("dr2_podcast/data/tts_glossary.json",),
}


#: The ROOTS of the code each stage's output depends on — its adapter and the modules that do the
#: work. The set actually hashed is the import closure of these (see :func:`implementation_closure`).
#: The code each stage's output depends on. Configuration is not the only thing that decides what a
#: stage produces — the implementation does too, and without this a deployed change to an adapter or
#: to the phase behind it left every existing run's stage "current", so the runner skipped it and
#: the run kept artifacts the current code would not produce (prepush codex 2026-08-13).
#:
#: The roots are per stage, but MEASURE the closures before assuming that buys precision: they come
#: out at 39-40 files each out of ~45, because `pipeline.py` is in every stage's roots and imports
#: nearly the whole package. In practice a code change anywhere restales every stage — the same
#: effect a single build identifier would have, arrived at honestly rather than assumed away.
#:
#: That is over-invalidation, which is the direction this module chooses everywhere: over-
#: invalidating costs a re-run, under-invalidating ships artifacts built by code that no longer
#: exists. `--force` remains the human's answer to a change they know is inert. If the phase
#: functions ever move out of `pipeline.py` into per-stage modules, the roots below start meaning
#: what they say and the closures separate on their own.
#: Every stage runs through these, so every stage hashes them. `pipeline.py` holds the phase
#: functions the adapters call — `_run_sectional_draft`, `_run_polish_loop`, `_finalize_script`,
#: `_run_audio_pipeline` — and `_common.py` rebuilds the module state they run against, so a change
#: to either changes what any stage produces (prepush codex 2026-08-13).
_EVERY_STAGE: tuple[str, ...] = (
    "dr2_podcast/pipeline.py",
    "dr2_podcast/adapters/_common.py",
)

_STAGE_SPECIFIC: dict[str, tuple[str, ...]] = {
    "framing": ("dr2_podcast/adapters/research_stages.py", "dr2_podcast/research/domain_classifier.py"),
    "framing_prior": ("dr2_podcast/adapters/research_stages.py", "dr2_podcast/claude_runner.py"),
    "plan_search": ("dr2_podcast/adapters/research_stages.py", "dr2_podcast/research/clinical.py"),
    "research": (
        "dr2_podcast/adapters/research_stages.py",
        "dr2_podcast/research/clinical.py",
        "dr2_podcast/research/clinical_math.py",
        "dr2_podcast/research/effect_size_math.py",
        "dr2_podcast/research/confidence.py",
        "dr2_podcast/pipeline_sot.py",
    ),
    "url_validation": ("dr2_podcast/adapters/research_stages.py", "dr2_podcast/tools/link_validator.py"),
    "translate": ("dr2_podcast/adapters/research_stages.py", "dr2_podcast/pipeline_translation.py"),
    "blueprint": ("dr2_podcast/adapters/script_stages.py", "dr2_podcast/pipeline_crew.py"),
    "draft": ("dr2_podcast/adapters/script_stages.py", "dr2_podcast/pipeline_script.py"),
    "polish": ("dr2_podcast/adapters/script_stages.py", "dr2_podcast/pipeline_script.py"),
    "audit": (
        "dr2_podcast/adapters/script_stages.py",
        "dr2_podcast/pipeline_validators.py",
        "dr2_podcast/pipeline_flow.py",
    ),
    "audio": ("dr2_podcast/adapters/script_stages.py", "dr2_podcast/audio/engine.py"),
}

STAGE_IMPLEMENTATION: dict[str, tuple[str, ...]] = {
    stage: _EVERY_STAGE + files for stage, files in _STAGE_SPECIFIC.items()
}


def _source_hashes(stage: str | None, relatives: tuple[str, ...], prefix: str) -> dict[str, Any]:
    """Content hashes of files under the package root, keyed for the fingerprint."""
    root = Path(__file__).resolve().parent.parent
    values: dict[str, Any] = {}
    for relative in relatives:
        path = root / relative
        try:
            values[f"{prefix}:{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            # None, not a skip: a file that cannot be read is a fingerprint that CHANGED, and
            # silently omitting it would make a missing implementation look like an unchanged one.
            values[f"{prefix}:{relative}"] = None
    return values


def _data_asset_values(stage: str | None) -> dict[str, Any]:
    """Content hashes of the bundled data the named stage's output depends on."""
    return _source_hashes(stage, STAGE_DATA_ASSETS.get(stage or "", ()), "data")


@cache
def implementation_closure(stage: str) -> tuple[str, ...]:
    """Every ``dr2_podcast`` module reachable from a stage's roots, sorted.

    COMPUTED, not curated. Three rounds of review found three different files missing from the
    hand-written list — pipeline.py, then pipeline_flow.py, then prompt_strings.py — and the fourth
    would have been found the same way (prepush codex 2026-08-13). A list nobody can verify by
    reading it is not a guarantee; the import graph is the thing that actually decides what code
    runs, so it is what gets walked.

    Static, over the source: an import inside a function body counts, because the adapters import
    `pipeline_flow` exactly that way. Cached per stage — which files are reachable does not change
    while the process runs, and their CONTENT is hashed fresh on every call.
    """
    root = Path(__file__).resolve().parent.parent
    seen: set[str] = set()
    frontier = list(STAGE_IMPLEMENTATION.get(stage, ()))
    while frontier:
        relative = frontier.pop()
        if relative in seen:
            continue
        seen.add(relative)
        try:
            tree = ast.parse((root / relative).read_bytes())
        except (OSError, SyntaxError):
            continue
        for module in _imported_modules(tree):
            for candidate in _module_paths(module):
                if candidate not in seen and (root / candidate).exists():
                    frontier.append(candidate)
    return tuple(sorted(seen))


def _imported_modules(tree: ast.AST) -> set[str]:
    """Dotted ``dr2_podcast.*`` module names imported anywhere in a parsed source file."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("dr2_podcast"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dr2_podcast"):
            found.add(node.module)
            # `from dr2_podcast.adapters import _common` imports a MODULE, not a name.
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _module_paths(dotted: str) -> tuple[str, ...]:
    """The file a dotted module could live in — a module, or a package's __init__."""
    base = dotted.replace(".", "/")
    return (f"{base}.py", f"{base}/__init__.py")


def _implementation_values(stage: str | None) -> dict[str, Any]:
    """Content hashes of the code the named stage runs, transitively."""
    if not stage:
        return {}
    return _source_hashes(stage, implementation_closure(stage), "code")


def _in_group(name: str, group: str) -> bool:
    if group == "tts":
        return name.startswith(("TTS_", "env:TTS_")) or name in ("VOICE_DUCKING_DB", "env:VOICE_DUCKING_DB")
    return name in CONFIG_GROUPS[group]


def scoped_identity_values(values: dict[str, Any], stage: str | None) -> dict[str, Any]:
    """The subset of the configuration a given stage's identity is built from.

    An unmapped stage keeps everything: a new stage must over-invalidate rather than quietly ignore
    a setting nobody remembered to classify.
    """
    groups = STAGE_CONFIG_GROUPS.get(stage or "")
    if not groups:
        return values
    return {name: value for name, value in values.items() if any(_in_group(name, g) for g in groups)}


def config_identity_values() -> dict[str, Any]:
    """Every config attribute that participates in stage identity, read from the live module."""
    from dr2_podcast import config

    values = {
        name: getattr(config, name)
        for name in dir(config)
        if name.isupper()
        and not name.startswith("_")
        and name not in CONFIG_IDENTITY_EXCLUDE
        and isinstance(getattr(config, name), _IDENTITY_TYPES)
    }
    values.update({f"env:{key}": os.environ.get(key) for key in CONTENT_ENV_KEYS})
    return values


def manifest_errors(manifest: dict[str, Any]) -> list[str]:
    """Structural errors for a manifest document."""
    return schema_errors("manifest", manifest)


RUN_CONFIG_IDENTITY_KEYS = ("topic", "language", "target_length_minutes")

#: Environment variables whose EFFECTIVE value a run config supersedes. PODCAST_LENGTH picks a mode
#: from a table for the monolithic runner, but a staged run passes ``target_length_minutes`` into
#: ``initialise_run_globals`` and that argument overrides the lookup entirely — so with a run config
#: present the env var determines nothing, while hashing it would make framing and every stage
#: downstream of it non-current whenever an unrelated shell or Web-UI setting moved (prepush codex
#: 2026-08-13). It stays in the identity for the legacy manifest, which has no run config and really
#: is steered by it.
RUN_CONFIG_SUPERSEDES = frozenset({"env:PODCAST_LENGTH"})


def config_fingerprint(
    values: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    stage: str | None = None,
) -> str:
    """sha256 over everything that changes what a stage would produce.

    Reads :mod:`dr2_podcast.config` when given nothing, so callers do not have to assemble it, but
    accepts an explicit mapping so tests never depend on the machine's ``.env``.

    ``run_config`` belongs in here for the same reason the model does. Without it, rewriting
    ``meta/run_config.json`` with a new ``--topic`` leaves every completed stage "current", so the
    runner skips them and the run ends up with artifacts about the old topic and a config file
    describing the new one.
    """
    if values is None:
        values = config_identity_values()
    values = {
        **scoped_identity_values(values, stage),
        **_data_asset_values(stage),
        **_implementation_values(stage),
    }
    if run_config is not None:
        values = {k: v for k, v in values.items() if k not in RUN_CONFIG_SUPERSEDES}
    parts = [f"{key}={_canonical(values[key])}" for key in sorted(values)]
    if run_config is not None:
        parts += [f"run.{key}={run_config.get(key)!r}" for key in RUN_CONFIG_IDENTITY_KEYS]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Manifest:
    """Read/modify/write access to one run's manifest. Every write is atomic and schema-checked."""

    run_dir: Path
    mode: str
    document: dict[str, Any]

    @classmethod
    def path_for(cls, run_dir: Path, mode: str = "staged") -> Path:
        try:
            return run_dir / MANIFEST_FILENAMES[mode]
        except KeyError:
            raise KeyError(f"unknown mode {mode!r}; known: {', '.join(MANIFEST_FILENAMES)}") from None

    @classmethod
    def load(cls, run_dir: Path, mode: str = "staged") -> Manifest:
        """Load a run's manifest, or start an empty one. A corrupt manifest raises rather than resets.

        Resetting on a read error is exactly the ``pipeline.py:502`` behaviour this replaces: it
        turns "I cannot tell what ran" into "nothing ran", which then re-runs an 87-minute pipeline
        or, worse, marks completed work as pending and overwrites it.
        """
        path = cls.path_for(run_dir, mode)
        if not path.exists():
            return cls(run_dir, mode, {"schema_version": MANIFEST_SCHEMA_VERSION, "mode": mode, "stages": {}})
        document = read_json_strict(path, schema="manifest")
        if document["mode"] != mode:
            raise ArtifactError(f"{path} is a {document['mode']!r} manifest, opened as {mode!r}")
        return cls(run_dir, mode, document)

    def save(self) -> str:
        errors = manifest_errors(self.document)
        if errors:
            raise SchemaValidationError("manifest", errors)
        return write_json_atomic(self.path_for(self.run_dir, self.mode), self.document, schema="manifest")

    # -- reading ---------------------------------------------------------- #
    def record_for(self, stage: str) -> dict[str, Any]:
        get_stage(stage)
        return self.document["stages"].get(stage, {"status": "pending", "inputs": [], "outputs": [], "attempts": []})

    def status(self, stage: str) -> str:
        return str(self.record_for(stage).get("status", "pending"))

    def drift(self, stage: str) -> list[str]:
        """Why this stage is not current: each recorded artifact whose hash no longer matches disk.

        Missing counts as drift. An artifact that is gone has not been checked, and treating an
        unresolvable input as unchanged is how unverified state passes for verified.
        """
        record = self.record_for(stage)
        reasons: list[str] = []
        for kind in ("inputs", "outputs"):
            for ref in record.get(kind, []):
                path = self.run_dir / ref["artifact"]
                if not path.exists():
                    reasons.append(f"{kind[:-1]} {ref['artifact']} is missing")
                elif sha256_file(path) != ref["sha256"]:
                    reasons.append(f"{kind[:-1]} {ref['artifact']} changed since this stage ran")
        return reasons

    def is_current(self, stage: str, *, config_sha256: str | None = None) -> bool:
        """True only if the stage completed, nothing it touched has drifted, and config still matches."""
        record = self.record_for(stage)
        if record.get("status") != "complete" or self.drift(stage):
            return False
        if config_sha256 is None:
            return True
        return record.get("identity", {}).get("config_sha256") == config_sha256

    # -- writing ---------------------------------------------------------- #
    def _stage_record(self, stage: str) -> dict[str, Any]:
        return self.document["stages"].setdefault(
            stage,
            {"status": "pending", "inputs": [], "outputs": [], "attempts": [], "identity": {}},
        )

    def start(self, stage: str, *, model: str, config_sha256: str) -> None:
        """Mark a stage running and stamp the identity it is running under."""
        record = self._stage_record(stage)
        record.update(
            status="running",
            identity={"model": model, "config_sha256": config_sha256},
            started_at=_now(),
            finished_at=None,
            stale_reason=None,
        )

    def record_attempt(self, stage: str, outcome: str, detail: str | None = None) -> None:
        """Append an attempt. ``transport`` is a retry and is not a revision round."""
        record = self._stage_record(stage)
        attempts = record.setdefault("attempts", [])
        attempts.append({"number": len(attempts) + 1, "outcome": outcome, "at": _now(), "detail": detail})

    def transport_retries(self, stage: str) -> int:
        return sum(1 for a in self.record_for(stage).get("attempts", []) if a["outcome"] == "transport")

    def complete(self, stage: str, substitutions: dict[str, str] | None = None) -> tuple[str, ...]:
        """Hash the stage's declared inputs and outputs, mark it complete, stale its downstream.

        Returns the stages marked stale. Marking happens on EVERY completion, not only on a
        re-run: the point is that a downstream stage can never be silently reused across a change
        to something it consumed.
        """
        definition = get_stage(stage)
        record = self._stage_record(stage)
        inputs = [self._ref(name, required=True) for name in definition.consumes]
        optional_inputs = resolve(definition.optional_consumes, substitutions)
        inputs += [ref for name in optional_inputs if (ref := self._ref(name, required=False))]
        record["inputs"] = inputs
        outputs = [self._ref(name, required=True) for name in definition.produces]
        optional_outputs = resolve(definition.optional_outputs, substitutions)
        outputs += [ref for name in optional_outputs if (ref := self._ref(name, required=False))]
        record["outputs"] = outputs
        record.update(status="complete", finished_at=_now(), stale_reason=None)
        return self.invalidate_downstream(stage)

    def fail(self, stage: str, detail: str) -> None:
        record = self._stage_record(stage)
        record.update(status="failed", finished_at=_now(), stale_reason=detail)

    def invalidate_downstream(self, stage: str) -> tuple[str, ...]:
        """Mark every stage this one invalidated as stale, and say why.

        Two ways to be invalidated, and the second is the one a purely hash-based rule misses. A
        stage is stale if an artifact it recorded has drifted — or if a stage it consumes from is
        not current, even though nothing on disk has moved yet. Consistency with an artifact that
        is known to be out of date is not currency: that upstream stage is going to re-run and
        change the very input this one was built on.

        The producer test is *currency*, not "did this call stale it", so it covers a producer that
        is stale, failed, or was never run. That distinction is what makes this correct on the
        failure path: a rerun of ``research`` that fails after rewriting one output leaves ``sot``
        stale by drift, but ``blueprint`` — whose own input may not have changed — would otherwise
        stay falsely current behind it.

        ``downstream_of`` returns declaration order, which is run order, so a producer is always
        marked before its consumers are examined.
        """
        marked: list[str] = []
        for name in downstream_of(stage):
            record = self.document["stages"].get(name)
            if record is None or record.get("status") not in ("complete", "running"):
                continue
            # The producers of what this stage RECORDED consuming, not every producer the graph
            # allows. An English blueprint records no translated SOT, so `translate` never produced
            # anything it read — checking it anyway staled the whole script chain the next time an
            # unrelated stage completed, purely because `translate` sits pending forever.
            consumed = {producer_of(ref["artifact"]) for ref in record.get("inputs", [])}
            reasons = self.drift(name)
            reasons += [
                f"{producer} is not current"
                for producer in sorted(p for p in consumed if p)
                if not self.is_current(producer)
            ]
            if reasons:
                record.update(status="stale", stale_reason="; ".join(reasons))
                marked.append(name)
        return tuple(marked)

    def _ref(self, artifact: str, *, required: bool) -> dict[str, str] | None:
        path = self.run_dir / artifact
        if not path.exists():
            if required:
                raise ArtifactError(f"stage declared it produces {artifact}, but it is not on disk")
            return None
        return {"artifact": artifact, "sha256": sha256_file(path)}
