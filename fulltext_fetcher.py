"""
Full-text article fetcher with 4-tier fallback:
1. PubMed Central Open Access API
2. Europe PMC REST API
3. Unpaywall API
4. Publisher page scrape (reuses ContentFetcher logic)

Used by the deep research pipeline (Step 4) to retrieve full-text
articles for the top-20 screened studies.
"""

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SCRAPING_TIMEOUT = 20.0
MAX_TEXT_CHARS = 128_000  # ~32K tokens


@dataclass
class FullTextArticle:
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    full_text: str
    source: str         # "pmc", "europepmc", "unpaywall", "scrape"
    word_count: int
    url: str
    error: Optional[str] = None


class FullTextFetcher:
    """Fetch full-text content for studies identified by PMID/DOI."""

    def __init__(self, max_concurrent: int = 5, cache=None):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "research@example.com")
        self.cache = cache  # PageCache instance (optional)

    async def fetch_fulltext(self, pmid: Optional[str], doi: Optional[str],
                             title: str, url: str) -> FullTextArticle:
        """Try PMC → Europe PMC → Unpaywall → publisher scrape."""
        async with self.semaphore:
            # Tier 1: PubMed Central OA
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

            # Tier 4: Publisher page scrape
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
        """Tier 1: PubMed Central Open Access — get full-text XML."""
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                # Check if article is in PMC
                resp = await http.get(
                    "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                    params={"ids": pmid, "format": "json"}
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                records = data.get("records", [])
                if not records or "pmcid" not in records[0]:
                    return None

                pmcid = records[0]["pmcid"]

                # Fetch full-text XML from PMC
                resp = await http.get(
                    f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi",
                    params={
                        "verb": "GetRecord",
                        "identifier": f"oai:pubmedcentral.nih.gov:{pmcid.replace('PMC', '')}",
                        "metadataPrefix": "pmc"
                    }
                )
                if resp.status_code != 200:
                    return None

                # Extract text from XML
                text = self._extract_text_from_xml(resp.text)
                return text[:MAX_TEXT_CHARS] if text else None
        except Exception as e:
            logger.debug(f"PMC fetch failed for PMID {pmid}: {e}")
            return None

    async def _try_europepmc(self, pmid: str) -> Optional[str]:
        """Tier 2: Europe PMC REST API — full-text XML for OA articles."""
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.get(
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmid}/fullTextXML"
                )
                if resp.status_code != 200:
                    return None
                text = self._extract_text_from_xml(resp.text)
                return text[:MAX_TEXT_CHARS] if text else None
        except Exception as e:
            logger.debug(f"Europe PMC fetch failed for PMID {pmid}: {e}")
            return None

    async def _try_unpaywall(self, doi: str) -> Optional[str]:
        """Tier 3: Unpaywall API — find OA PDF URL, then scrape it."""
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.get(
                    f"https://api.unpaywall.org/v2/{doi}",
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
            logger.debug(f"Unpaywall fetch failed for DOI {doi}: {e}")
            return None

    async def _try_scrape(self, url: str) -> Optional[str]:
        """Tier 4: Publisher page scrape (same logic as ContentFetcher)."""
        if not url:
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
                for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                    tag.decompose()
                content_el = (
                    soup.find("main") or soup.find("article") or
                    soup.find("div", class_=re.compile(r"content|main-content|post-content|article")) or
                    soup.find("body")
                )
                text = content_el.get_text(separator=" ", strip=True) if content_el else ""
                return text[:MAX_TEXT_CHARS] if text else None
        except Exception as e:
            logger.debug(f"Scrape failed for {url}: {e}")
            return None

    def _extract_text_from_xml(self, xml_text: str) -> Optional[str]:
        """Extract readable text from PMC/Europe PMC XML."""
        try:
            root = ET.fromstring(xml_text)
            # Collect text from body paragraphs
            parts = []
            for elem in root.iter():
                if elem.tag in ("p", "title", "sec", "abstract", "body"):
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
