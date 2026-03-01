"""
Deep Research Tool - Self-hosted search using SearXNG with async web scraping.

This module provides a complete deep research solution that:
- Connects to a local SearXNG instance (avoiding paid APIs)
- Performs web searches across multiple engines
- Scrapes and cleans content from top search results
- Uses async operations for maximum performance

Author: AI Assistant
License: MIT
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from dr2_podcast.config import SCRAPING_TIMEOUT, USER_AGENT, SEARXNG_URL
from dr2_podcast.utils import extract_content_from_html

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 10.0
MAX_CONTENT_LENGTH = 8000  # ~2000 tokens (4 chars per token estimate)


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    url: str
    snippet: str
    engine: Optional[str] = None
    score: Optional[float] = None


@dataclass
class ScrapedContent:
    """Represents scraped and cleaned content from a webpage."""

    url: str
    title: str
    content: str
    word_count: int
    error: Optional[str] = None


@dataclass
class ResearchResult:
    """Complete research result including search results and scraped content."""

    query: str
    search_results: List[SearchResult]
    scraped_pages: List[ScrapedContent]
    total_results: int
    total_scraped: int
    errors: List[str] = field(default_factory=list)


class SearxngClient:
    """
    Client for interacting with a local SearXNG search instance.

    This client handles connection management, search queries, and error handling
    for a self-hosted SearXNG server.

    Attributes:
        base_url: Base URL of the SearXNG instance (default: http://localhost:8080)
        client: Async HTTP client for making requests
    """

    def __init__(
        self,
        base_url: str = SEARXNG_URL,
        timeout: float = DEFAULT_TIMEOUT
    ):
        """
        Initialize the SearXNG client.

        Args:
            base_url: Base URL of the SearXNG instance
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def validate_connection(self) -> bool:
        """
        Validate that the SearXNG instance is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.client:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(self.base_url)
                    return response.status_code == 200
            else:
                response = await self.client.get(self.base_url)
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Connection error: {e}")
            logger.error("Is SearXNG Docker container running? Try: docker ps | grep searxng")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection validation: {e}")
            return False

    async def search(
        self,
        query: str,
        engines: List[str] = None,
        num_results: int = 5,
        language: str = "en"
    ) -> List[SearchResult]:
        """
        Perform a search query using SearXNG.

        Args:
            query: Search query string
            engines: List of search engines to use (default: ['google', 'bing', 'brave'])
            num_results: Maximum number of results to return
            language: Search language code (default: 'en')

        Returns:
            List of SearchResult objects

        Raises:
            httpx.RequestError: If the request fails
        """
        if engines is None:
            engines = ['google', 'bing', 'brave']

        # Build search parameters
        params = {
            'q': query,
            'format': 'json',
            'language': language,
            'engines': ','.join(engines)
        }

        try:
            if not self.client:
                raise RuntimeError("Client not initialized. Use 'async with' context manager.")

            response = await self.client.get(
                f"{self.base_url}/search",
                params=params
            )
            response.raise_for_status()

            data = response.json()
            results = []

            # Parse search results
            for item in data.get('results', [])[:num_results]:
                result = SearchResult(
                    title=item.get('title', 'No title'),
                    url=item.get('url', ''),
                    snippet=item.get('content', ''),
                    engine=item.get('engine'),
                    score=item.get('score')
                )
                results.append(result)

            logger.info(f"Found {len(results)} results for query: {query}")
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during search: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error during search: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise


class DeepResearch:
    """
    Deep research engine that scrapes and analyzes web content.

    This class performs comprehensive research by:
    1. Searching for relevant content using SearXNG
    2. Scraping the top N URLs in parallel
    3. Cleaning and extracting meaningful content
    4. Limiting content to manageable token counts

    Attributes:
        searxng_client: SearXNG client instance for searching
        scraping_client: Separate HTTP client for web scraping
    """

    def __init__(self, searxng_client: SearxngClient):
        """
        Initialize the deep research engine.

        Args:
            searxng_client: Initialized SearXNG client
        """
        self.searxng_client = searxng_client
        self.scraping_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.scraping_client = httpx.AsyncClient(
            timeout=SCRAPING_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.scraping_client:
            await self.scraping_client.aclose()

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract meaningful content from HTML using the shared utility.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Extracted and cleaned text content, truncated to MAX_CONTENT_LENGTH
        """
        return extract_content_from_html(soup, max_chars=MAX_CONTENT_LENGTH)

    async def fetch_page_content(self, url: str) -> ScrapedContent:
        """
        Fetch and extract content from a single webpage.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent object with extracted content or error information
        """
        try:
            if not self.scraping_client:
                raise RuntimeError("Scraping client not initialized. Use 'async with' context manager.")

            logger.info(f"Scraping: {url}")

            response = await self.scraping_client.get(url)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else urlparse(url).netloc

            # Extract content
            content = self._extract_content(soup)
            word_count = len(content.split())

            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                word_count=word_count
            )

        except httpx.TimeoutException:
            error_msg = f"Timeout while scraping {url}"
            logger.warning(error_msg)
            return ScrapedContent(
                url=url,
                title="Timeout Error",
                content="",
                word_count=0,
                error=error_msg
            )
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code} error for {url}"
            logger.warning(error_msg)
            return ScrapedContent(
                url=url,
                title="HTTP Error",
                content="",
                word_count=0,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Error scraping {url}: {str(e)}"
            logger.warning(error_msg)
            return ScrapedContent(
                url=url,
                title="Scraping Error",
                content="",
                word_count=0,
                error=error_msg
            )

    async def deep_dive(
        self,
        query: str,
        top_n: int = 5,
        engines: List[str] = None
    ) -> ResearchResult:
        """
        Perform deep research on a query.

        This method:
        1. Searches for the query using SearXNG
        2. Extracts top N URLs from results
        3. Scrapes all URLs in parallel
        4. Returns comprehensive research data

        Args:
            query: Research query
            top_n: Number of top results to scrape (default: 5)
            engines: List of search engines to use

        Returns:
            ResearchResult containing search results and scraped content
        """
        logger.info(f"Starting deep research for: {query}")

        # Perform search
        search_results = await self.searxng_client.search(
            query=query,
            engines=engines,
            num_results=top_n
        )

        if not search_results:
            logger.warning("No search results found")
            return ResearchResult(
                query=query,
                search_results=[],
                scraped_pages=[],
                total_results=0,
                total_scraped=0,
                errors=["No search results found"]
            )

        # Extract URLs
        urls = [result.url for result in search_results]

        # Scrape all URLs in parallel
        logger.info(f"Scraping {len(urls)} URLs in parallel...")
        scraping_tasks = [self.fetch_page_content(url) for url in urls]
        scraped_contents = await asyncio.gather(*scraping_tasks, return_exceptions=False)

        # Collect errors
        errors = [content.error for content in scraped_contents if content.error]
        successful_scrapes = [content for content in scraped_contents if not content.error]

        logger.info(f"Successfully scraped {len(successful_scrapes)}/{len(urls)} pages")

        return ResearchResult(
            query=query,
            search_results=search_results,
            scraped_pages=scraped_contents,
            total_results=len(search_results),
            total_scraped=len(successful_scrapes),
            errors=errors
        )


async def main():
    """
    Example usage of the deep research tool.

    This demonstrates:
    1. Connection validation
    2. Simple search
    3. Deep research with scraping
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("=" * 80)
    logger.info("Deep Research Tool - Example Usage")
    logger.info("=" * 80)

    # Initialize SearXNG client
    async with SearxngClient() as searxng_client:
        # Validate connection
        logger.info("\n1. Validating connection to SearXNG...")
        is_connected = await searxng_client.validate_connection()

        if not is_connected:
            logger.error("Cannot connect to SearXNG at http://localhost:8080")
            logger.error("Make sure SearXNG is running. Start it with:")
            logger.error("docker run -d --name searxng -p 8080:8080 searxng/searxng:latest")
            return

        logger.info("Connected to SearXNG successfully!")

        # Perform a simple search
        logger.info("\n2. Performing search for 'Python asyncio tutorial'...")
        search_results = await searxng_client.search(
            query="Python asyncio tutorial",
            num_results=5
        )

        logger.info("\nFound %d results:", len(search_results))
        for i, result in enumerate(search_results, 1):
            logger.info("\n%d. %s", i, result.title)
            logger.info("   URL: %s", result.url)
            logger.info("   Snippet: %s...", result.snippet[:100])

        # Perform deep research
        logger.info("\n3. Performing deep research (scraping top 3 URLs)...")
        async with DeepResearch(searxng_client) as research:
            deep_results = await research.deep_dive(
                query="Python asyncio tutorial",
                top_n=3
            )

            logger.info("\nDeep Research Complete!")
            logger.info("   Query: %s", deep_results.query)
            logger.info("   Total search results: %d", deep_results.total_results)
            logger.info("   Successfully scraped: %d/%d", deep_results.total_scraped, len(deep_results.scraped_pages))

            if deep_results.errors:
                logger.warning("\nErrors encountered: %d", len(deep_results.errors))
                for error in deep_results.errors:
                    logger.warning("   - %s", error)

            logger.info("\nScraped Content:")
            for i, content in enumerate(deep_results.scraped_pages, 1):
                if not content.error:
                    logger.info("\n%d. %s", i, content.title)
                    logger.info("   URL: %s", content.url)
                    logger.info("   Word count: %d", content.word_count)
                    logger.info("   Preview: %s...", content.content[:200])
                else:
                    logger.error("\n%d. %s", i, content.url)
                    logger.error("   Error: %s", content.error)

    logger.info("\n" + "=" * 80)
    logger.info("Example complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
