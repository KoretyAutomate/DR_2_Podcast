"""Evidence search behind a mandatory domain allowlist — PLAN.md Step 1.

A generic web search answers "confidence interval meaning" with blogs and
review articles, which is exactly what the project objective bans. This wraps
`~/Project/searxng-deep-research/search_agent.py` and drops every hit whose
host is not named in `dr2_podcast/data/evidence_domains.json`.

An ALLOWLIST, distinct from and additional to the existing BLOCKlists
(`research/clinical.py:JUNK_DOMAINS`, `fulltext_fetcher.py:_SCRAPE_BLOCKED_DOMAINS`).
A blocklist answers "is this one known to be bad"; only an allowlist can answer
"has anyone vouched for this at all", which is the question a citation needs.

FILTER BEFORE SCRAPE. `deep_dive()` searches and then scrapes every result,
so using it would fetch the banned pages before discarding them. This calls
`SearxngClient.search()`, applies the allowlist, and only then scrapes the
survivors with `DeepResearch.fetch_page_content()` — so a rejected domain is
never requested, and the tool cannot be the reason a blog is downloaded.

The tiers are selectable because Step 6's guideline reviewer needs a
guideline-only allowlist, not the full set.

Scraping is BEST-EFFORT and often fails on exactly the domains that matter:
ncbi.nlm.nih.gov and doi.org answer a scraper with HTTP 403 (measured
2026-08-16). Each failure is recorded per page and printed, so an empty
`scraped[]` never reads as "full text obtained and it said nothing". The hit's
title, snippet and URL are what identify the source; resolving it properly is
Step 2's citation validator via `metadata_clients.py`, which uses the APIs
these sites publish instead of scraping the pages they defend.

Usage:
    python -m dr2_podcast.tools.evidence_search "CLAIM" [--json OUT]
    python -m dr2_podcast.tools.evidence_search "CLAIM" -q "query one" -q "query two"
    python -m dr2_podcast.tools.evidence_search "CLAIM" --tier guidelines      # Step 6
    python -m dr2_podcast.tools.evidence_search --list-tiers

Exit status: 0 when at least one allowlisted hit was found, 1 when none was
(a claim nobody authoritative discusses is a finding, not a crash), 2 on a
usage or connectivity error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "evidence_domains.json"

# search_agent.py lives outside this package, in its own repo.
SEARCH_AGENT_DIR = Path(
    os.environ.get("SEARXNG_DEEP_RESEARCH_DIR", Path.home() / "Project" / "searxng-deep-research")
)

# Keys inside a tier that document it rather than allow a domain.
_META_KEYS = {"_why"}

DEFAULT_NUM_RESULTS = 10
DEFAULT_SCRAPE = 5

# search_agent defaults to google/bing/brave. For a claim like "statins reduce
# LDL" those answer with webmd, drugs.com and mayoclinic — measured 2026-08-16:
# ten hits, eight consumer-health domains, zero primary literature. Filtering
# that to PubMed afterwards yields an empty bundle, because the question was
# put to the wrong index.
SCHOLARLY_ENGINES = ("pubmed", "semantic scholar", "google scholar", "openairepublications")
WEB_ENGINES = ("google", "duckduckgo", "brave")

# Every tier asks both sets. It is tempting to send only the scholarly engines
# for primary_literature and only the web ones for guidelines — WHO pages are
# not in any paper index — but on this instance the general engines are refused
# outright (measured 2026-08-16: brave "Suspended: too many requests",
# duckduckgo "CAPTCHA", google "Suspended: access denied"). A tier routed to
# them alone returns nothing at all, so the union is what actually retrieves,
# and the allowlist is what discriminates. Override with --engines.
DEFAULT_ENGINES = tuple(dict.fromkeys(SCHOLARLY_ENGINES + WEB_ENGINES))
# Bundles are pasted into a refuter's prompt (PLAN.md: Codex is text-in/text-out
# and cannot fetch anything itself), so an unbounded page would crowd out the
# script excerpt it is meant to be judged against.
DEFAULT_MAX_CHARS = 12000


class EvidenceSearchError(RuntimeError):
    """Setup or connectivity failure — distinct from 'no evidence found'."""


@dataclass(frozen=True)
class SearchOptions:
    """How to search — grouped so the knobs travel together instead of as
    eight positional parameters threaded through two functions."""

    num_results: int = DEFAULT_NUM_RESULTS
    scrape: int = DEFAULT_SCRAPE
    language: str = "en"
    max_chars: int = DEFAULT_MAX_CHARS
    # None = let the selected tiers choose (see engines_for)
    engines: tuple[str, ...] | None = None


# ------------------------------------------------------------------ allowlist


@dataclass(frozen=True)
class Allowlist:
    """Domains that may enter an evidence bundle, and which tier vouched.

    `tier_of` always holds EVERY tier's domains, even when only some tiers are
    selected, because classification and admission are different questions.
    The tiers overlap by design — `nih.gov` is a guideline publisher and
    `ncbi.nlm.nih.gov` sits under it — so a PubMed paper is still a PubMed
    paper when the guidelines tier is the one being asked for. Narrowing the
    map instead of the admission made `--tier guidelines` answer "what do
    current guidelines say" with three research papers (measured 2026-08-16),
    which is the failure Step 6 exists to avoid.
    """

    tier_of: dict[str, str]
    selected: frozenset[str]
    why: dict[str, str] = field(default_factory=dict)

    @property
    def tiers(self) -> list[str]:
        return sorted(self.selected)

    def classify(self, url: str) -> tuple[str, str] | None:
        """(domain, tier) for this URL against the WHOLE allowlist, or None.

        Resolves to the most specific domain that covers the host, so
        `ncbi.nlm.nih.gov` wins over `nih.gov` regardless of dict order.

        Matches the host itself or any subdomain of it. The leading dot is what
        makes that safe: without it "pubmed.ncbi.nlm.nih.gov.attacker.test"
        would read as PubMed.
        """
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
        if not host:
            return None
        hits = [
            (domain, tier)
            for domain, tier in self.tier_of.items()
            if host == domain or host.endswith(f".{domain}")
        ]
        if not hits:
            return None
        return max(hits, key=lambda pair: len(pair[0]))

    def match(self, url: str) -> tuple[str, str] | None:
        """(domain, tier) if this URL is admitted under the selected tiers."""
        found = self.classify(url)
        return found if found and found[1] in self.selected else None


def load_allowlist(tiers: list[str] | None = None, path: Path = DATA_FILE) -> Allowlist:
    """Read evidence_domains.json, optionally narrowed to some tiers."""
    if not path.exists():
        raise EvidenceSearchError(f"allowlist missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    available = [k for k, v in raw.items() if isinstance(v, dict)]
    wanted = tiers if tiers else available
    unknown = [t for t in wanted if t not in available]
    if unknown:
        raise EvidenceSearchError(f"unknown tier(s) {unknown}; available: {available}")

    # Every tier is classified; only `wanted` is admitted (see Allowlist).
    tier_of: dict[str, str] = {}
    why: dict[str, str] = {}
    for tier in available:
        why[tier] = raw[tier].get("_why", "")
        for domain in raw[tier]:
            if domain not in _META_KEYS:
                tier_of.setdefault(domain.lower(), tier)

    if not any(tier in wanted for tier in tier_of.values()):
        raise EvidenceSearchError(f"no domains in tier(s) {wanted} — refusing to search unfiltered")
    return Allowlist(tier_of=tier_of, selected=frozenset(wanted), why=why)


# --------------------------------------------------------------------- search


def _load_search_agent():
    """Import search_agent.py from its own repo (it is not an installed package)."""
    if not (SEARCH_AGENT_DIR / "search_agent.py").exists():
        raise EvidenceSearchError(
            f"search_agent.py not found under {SEARCH_AGENT_DIR} "
            "— set SEARXNG_DEEP_RESEARCH_DIR to its checkout"
        )
    if str(SEARCH_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(SEARCH_AGENT_DIR))
    try:
        # imported late on purpose: sys.path is only correct from here on
        import search_agent
    except ImportError as exc:  # pragma: no cover — env problem, not logic
        raise EvidenceSearchError(f"could not import search_agent: {exc}") from exc
    return search_agent


async def _gather(
    claim: str, queries: list[str], allow: Allowlist, opts: SearchOptions
) -> dict[str, Any]:
    agent = _load_search_agent()
    # search_evidence resolves this; defaulting again keeps _gather callable
    # directly and keeps the type honest (engines is Optional on the dataclass).
    engines = list(opts.engines or DEFAULT_ENGINES)

    allowed: list[dict[str, Any]] = []
    seen_count = 0
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    errors: list[str] = []

    async with agent.SearxngClient() as client:
        for query in queries:
            try:
                hits = await client.search(
                    query=query, num_results=opts.num_results,
                    language=opts.language, engines=engines,
                )
            # broad on purpose: one engine refusing must not lose the other queries
            except Exception as exc:
                errors.append(f"search failed for {query!r}: {exc}")
                continue
            for hit in hits:
                seen_count += 1
                verdict = allow.match(hit.url)
                if verdict is None:
                    host = (urlsplit(hit.url).hostname or "?").lower()
                    rejected[host] += 1
                    continue
                if hit.url in seen:
                    continue
                seen.add(hit.url)
                domain, tier = verdict
                allowed.append(
                    {
                        "title": hit.title,
                        "url": hit.url,
                        "snippet": hit.snippet,
                        "engine": hit.engine,
                        "domain": domain,
                        "tier": tier,
                        "query": query,
                    }
                )

        # Scrape only what survived the allowlist — a rejected domain is never fetched.
        scraped: list[dict[str, Any]] = []
        targets = allowed[: opts.scrape] if opts.scrape > 0 else []
        if targets:
            async with agent.DeepResearch(client) as research:
                pages = await asyncio.gather(
                    *(research.fetch_page_content(h["url"]) for h in targets),
                    return_exceptions=True,
                )
            for hit, page in zip(targets, pages, strict=True):
                if isinstance(page, BaseException):
                    scraped.append({"url": hit["url"], "error": str(page)})
                    continue
                text = page.content or ""
                scraped.append(
                    {
                        "url": page.url,
                        "title": page.title,
                        "tier": hit["tier"],
                        "word_count": page.word_count,
                        "truncated": len(text) > opts.max_chars,
                        "content": text[: opts.max_chars],
                        "error": page.error,
                    }
                )

    return {
        "claim": claim,
        "queries": queries,
        "engines": engines,
        "tiers": allow.tiers,
        "allowed_hits": allowed,
        # counts, not a bare list: "13 hits from one blog" and "13 blogs" are
        # different problems with the query, and the fix differs
        "rejected_domains": dict(rejected.most_common()),
        # zero hits SEEN means the engines answered with nothing (all suspended,
        # or the query is hopeless) — a different problem from hits that were
        # all rejected, and one the allowlist cannot be blamed for
        "hits_seen": seen_count,
        "scraped": scraped,
        "errors": errors,
    }


def search_evidence(
    claim: str,
    queries: list[str] | None = None,
    tiers: list[str] | None = None,
    opts: SearchOptions | None = None,
) -> dict[str, Any]:
    """Run the searches and return the evidence bundle.

    The bundle's invariant, and the reason this module exists: every entry in
    `allowed_hits` is vouched for by a named domain in a named tier.
    """
    allow = load_allowlist(tiers)
    opts = opts or SearchOptions()
    if opts.engines is None:
        opts = replace(opts, engines=DEFAULT_ENGINES)
    bundle = asyncio.run(_gather(claim, queries or [claim], allow, opts))
    # Belt and braces: the filter above is the gate, this proves it held.
    stowaways = [h["url"] for h in bundle["allowed_hits"] if allow.match(h["url"]) is None]
    if stowaways:  # pragma: no cover — would be a logic error, not an input problem
        raise EvidenceSearchError(f"allowlist leak: {stowaways}")
    return bundle


# ------------------------------------------------------------------------ CLI


def _summarise(bundle: dict[str, Any]) -> str:
    lines = [f"claim: {bundle['claim']}", f"tiers: {', '.join(bundle['tiers'])}"]
    by_tier = Counter(h["tier"] for h in bundle["allowed_hits"])
    lines.append(f"allowed hits: {len(bundle['allowed_hits'])}" + (f"  ({dict(by_tier)})" if by_tier else ""))
    for hit in bundle["allowed_hits"]:
        lines.append(f"  [{hit['tier']}] {hit['domain']}  {hit['title'][:70]}")
        lines.append(f"        {hit['url']}")
    if not bundle.get("hits_seen"):
        lines.append("  ! the engines returned NO hits at all — check SearXNG engine status")
    dropped = bundle["rejected_domains"]
    lines.append(f"rejected: {sum(dropped.values())} hit(s) from {len(dropped)} domain(s)")
    for host, n in list(dropped.items())[:10]:
        lines.append(f"  ✗ {host} ({n})")
    scraped_ok = [s for s in bundle["scraped"] if not s.get("error")]
    lines.append(f"scraped: {len(scraped_ok)}/{len(bundle['scraped'])}")
    # A 403 from an index is normal and must not read as full text obtained.
    for page in bundle["scraped"]:
        if page.get("error"):
            lines.append(f"  ✗ {page['url']}  {page['error']}")
    for err in bundle["errors"]:
        lines.append(f"  ! {err}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dr2_podcast.tools.evidence_search",
        description="Search for evidence behind a mandatory domain allowlist (PLAN.md Step 1).",
    )
    ap.add_argument("claim", nargs="?", help="the claim being checked")
    ap.add_argument("-q", "--query", action="append", dest="queries",
                    help="search query (repeatable; defaults to the claim itself)")
    ap.add_argument("--tier", action="append", dest="tiers",
                    help="restrict to a tier (repeatable; default: all)")
    ap.add_argument("--num-results", type=int, default=DEFAULT_NUM_RESULTS, help="hits per query")
    ap.add_argument("--scrape", type=int, default=DEFAULT_SCRAPE,
                    help="how many allowlisted hits to fetch (0 = none)")
    ap.add_argument("--engines", help="comma-separated SearXNG engines (default: chosen by --tier)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help="per-page content cap in the bundle")
    ap.add_argument("--json", dest="json_out", type=Path, help="write the bundle here")
    ap.add_argument("--list-tiers", action="store_true", help="print the tiers and exit")
    args = ap.parse_args(argv)

    try:
        if args.list_tiers:
            allow = load_allowlist()
            for tier in allow.tiers:
                domains = sorted(d for d, t in allow.tier_of.items() if t == tier)
                print(f"{tier}  ({len(domains)} domains)")
                print(f"  why: {allow.why.get(tier, '')}")
                print(f"  {', '.join(domains)}\n")
            return 0

        if not args.claim:
            ap.error("a claim is required (or use --list-tiers)")

        bundle = search_evidence(
            claim=args.claim, queries=args.queries, tiers=args.tiers,
            opts=SearchOptions(
                num_results=args.num_results, scrape=args.scrape,
                language=args.language, max_chars=args.max_chars,
                engines=tuple(e.strip() for e in args.engines.split(",")) if args.engines else None,
            ),
        )
    except EvidenceSearchError as exc:
        print(f"evidence_search: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"bundle → {args.json_out}")
    print(_summarise(bundle))
    return 0 if bundle["allowed_hits"] else 1


if __name__ == "__main__":
    sys.exit(main())
