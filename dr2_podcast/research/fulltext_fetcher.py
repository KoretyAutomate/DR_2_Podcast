"""
Full-text article fetcher with 5-tier fallback:
1. PubMed Central — NCBI ELink + EFetch (modern eutils API)
2. Europe PMC REST API
3. Unpaywall API
4. NCBI EFetch abstract XML (reliable API fallback)
5. Publisher page scrape (last resort)

Used by the deep research pipeline (Step 4) to retrieve full-text
articles for the top-20 screened studies.
"""

import asyncio
import logging
import os
from urllib.parse import quote as _url_quote
import re
import defusedxml.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

from dr2_podcast.config import SCRAPING_TIMEOUT, USER_AGENT
from dr2_podcast.utils import extract_content_from_html, is_safe_url

logger = logging.getLogger(__name__)
MAX_TEXT_CHARS = 128_000  # ~32K tokens


@dataclass
class FullTextArticle:
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    full_text: str
    source: str         # "pmc", "europepmc", "unpaywall", "ncbi_abstract", "scrape"
    word_count: int
    url: str
    error: Optional[str] = None


class FullTextFetcher:
    """Fetch full-text content for studies identified by PMID/DOI."""

    def __init__(self, max_concurrent: int = 5, cache=None):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.unpaywall_email = os.getenv("UNPAYWALL_EMAIL")  # Set in .env; required for Unpaywall OA lookup
        self.ncbi_api_key = os.getenv("PUBMED_API_KEY", "")
        self.cache = cache  # PageCache instance (optional)

    async def fetch_fulltext(self, pmid: Optional[str], doi: Optional[str],
                             title: str, url: str) -> FullTextArticle:
        """Try PMC EFetch → Europe PMC → Unpaywall → NCBI Abstract → publisher scrape."""
        async with self.semaphore:
            # Tier 1: PMC full text via NCBI ELink + EFetch
            if pmid:
                text = await self._try_pmc(pmid)
                if text:
                    return FullTextArticle(
                        pmid=pmid, doi=doi, title=title, full_text=text,
                        source="pmc", word_count=len(text.split()), url=url
                    )

            # Tier 2: Europe PMC
            if pmid:
                text = await self._try_europepmc(pmid)
                if text:
                    return FullTextArticle(
                        pmid=pmid, doi=doi, title=title, full_text=text,
                        source="europepmc", word_count=len(text.split()), url=url
                    )

            # Tier 3: Unpaywall
            if doi:
                text = await self._try_unpaywall(doi)
                if text:
                    return FullTextArticle(
                        pmid=pmid, doi=doi, title=title, full_text=text,
                        source="unpaywall", word_count=len(text.split()), url=url
                    )

            # Tier 4: NCBI EFetch abstract (reliable API, no bot detection)
            if pmid:
                text = await self._try_ncbi_abstract(pmid)
                if text:
                    return FullTextArticle(
                        pmid=pmid, doi=doi, title=title, full_text=text,
                        source="ncbi_abstract", word_count=len(text.split()), url=url
                    )

            # Tier 5: Publisher page scrape (last resort)
            text = await self._try_scrape(url)
            if text:
                return FullTextArticle(
                    pmid=pmid, doi=doi, title=title, full_text=text,
                    source="scrape", word_count=len(text.split()), url=url
                )

            return FullTextArticle(
                pmid=pmid, doi=doi, title=title, full_text="",
                source="none", word_count=0, url=url,
                error="All full-text sources exhausted"
            )

    async def fetch_all(self, records) -> List[FullTextArticle]:
        """Parallel fetch from a list of WideNetRecord objects."""
        tasks = [
            self.fetch_fulltext(
                pmid=getattr(r, 'pmid', None),
                doi=getattr(r, 'doi', None),
                title=getattr(r, 'title', ''),
                url=getattr(r, 'url', ''),
            )
            for r in records
        ]
        return await asyncio.gather(*tasks)

    async def _try_pmc(self, pmid: str) -> Optional[str]:
        """Tier 1: PMC full text via NCBI ELink + EFetch (replaces broken OAI endpoint)."""
        try:
            base_params = {"api_key": self.ncbi_api_key} if self.ncbi_api_key else {}
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
                # Step 1: Find PMCID via ELink
                resp = await http.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
                    params={"dbfrom": "pubmed", "db": "pmc", "id": pmid,
                            "retmode": "json", **base_params}
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                pmcid = None
                for ls in data.get("linksets", []):
                    for lsdb in ls.get("linksetdbs", []):
                        if lsdb.get("dbto") == "pmc":
                            links = lsdb.get("links", [])
                            if links:
                                pmcid = str(links[0])
                                break
                    if pmcid:
                        break
                if not pmcid:
                    return None  # Article not in PMC OA

                # Step 2: Fetch full-text XML via EFetch
                resp2 = await http.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pmc", "id": pmcid, "rettype": "full",
                            "retmode": "xml", **base_params}
                )
                if resp2.status_code != 200:
                    return None
                text = self._extract_text_from_xml(resp2.text)
                return text[:MAX_TEXT_CHARS] if text else None
        except Exception as e:
            logger.warning(f"PMC EFetch failed for PMID {pmid}: {e}")
            return None

    async def _try_europepmc(self, pmid: str) -> Optional[str]:
        """Tier 2: Europe PMC REST API — full-text XML for OA articles."""
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
                resp = await http.get(
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmid}/fullTextXML"
                )
                if resp.status_code != 200:
                    return None
                text = self._extract_text_from_xml(resp.text)
                return text[:MAX_TEXT_CHARS] if text else None
        except Exception as e:
            logger.warning(f"Europe PMC fetch failed for PMID {pmid}: {e}")
            return None

    async def _try_unpaywall(self, doi: str) -> Optional[str]:
        """Tier 3: Unpaywall API — find OA PDF URL, then scrape it."""
        if not self.unpaywall_email:
            return None
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
                resp = await http.get(
                    f"https://api.unpaywall.org/v2/{_url_quote(doi, safe='')}",
                    params={"email": self.unpaywall_email}
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                best_oa = data.get("best_oa_location")
                if not best_oa:
                    return None

                # Prefer PDF URL, fall back to landing page
                pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")
                if not pdf_url:
                    return None

                # Scrape the OA URL
                return await self._try_scrape(pdf_url)
        except Exception as e:
            logger.warning(f"Unpaywall fetch failed for DOI {doi}: {e}")
            return None

    async def _try_ncbi_abstract(self, pmid: str) -> Optional[str]:
        """Tier 4: NCBI EFetch abstract XML — reliable fallback when full text unavailable."""
        try:
            params: dict = {"db": "pubmed", "id": pmid, "rettype": "abstract",
                            "retmode": "xml"}
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
                resp = await http.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params=params
                )
                if resp.status_code != 200:
                    return None
                text = self._extract_text_from_xml(resp.text)
                return text[:MAX_TEXT_CHARS] if text else None
        except Exception as e:
            logger.warning(f"NCBI abstract fetch failed for PMID {pmid}: {e}")
            return None

    async def _try_scrape(self, url: str) -> Optional[str]:
        """Tier 5: Publisher page scrape (last resort)."""
        if not url:
            return None
        if not is_safe_url(url):
            logger.warning(f"Blocked SSRF-unsafe URL: {url}")
            return None
        try:
            # Check cache first
            if self.cache:
                cached = self.cache.get(url)
                if cached and cached.content:
                    return cached.content[:MAX_TEXT_CHARS]

            headers = {"User-Agent": USER_AGENT}
            async with httpx.AsyncClient(
                timeout=SCRAPING_TIMEOUT, follow_redirects=True, headers=headers
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
                text = extract_content_from_html(soup, max_chars=MAX_TEXT_CHARS)
                return text if text else None
        except Exception as e:
            logger.warning(f"Scrape failed for {url}: {e}")
            return None

    def _extract_text_from_xml(self, xml_text: str) -> Optional[str]:
        """Extract readable text from PMC/Europe PMC/NCBI EFetch XML."""
        try:
            root = ET.fromstring(xml_text)
            # Collect text from body paragraphs — strip namespace prefixes before matching
            parts = []
            for elem in root.iter():
                local_tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if local_tag in ("p", "title", "sec", "abstract", "body", "AbstractText"):
                    text = "".join(elem.itertext()).strip()
                    if text:
                        parts.append(text)
            full_text = "\n\n".join(parts)
            return full_text if len(full_text) > 100 else None
        except ET.ParseError:
            # If XML parsing fails, try extracting raw text
            try:
                soup = BeautifulSoup(xml_text, "lxml-xml")
                text = soup.get_text(separator="\n", strip=True)
                return text if len(text) > 100 else None
            except Exception:
                return None
