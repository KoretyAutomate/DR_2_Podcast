"""Shared utility functions for DR_2_Podcast pipeline."""
import re

from bs4 import BeautifulSoup


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (Qwen3 thinking mode safety net)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_content_from_html(soup: BeautifulSoup, max_chars: int = 8000) -> str:
    """Extract meaningful text content from parsed HTML.

    Uses a priority-based extraction strategy:
    1. <main> tag
    2. <article> tag
    3. <div> with content-related classes
    4. <body> tag (fallback)

    Removes script, style, nav, footer, header, aside, iframe tags first.

    Args:
        soup: BeautifulSoup parsed HTML (will be modified in-place by decompose).
        max_chars: Maximum characters to return.

    Returns:
        Extracted and cleaned text content, truncated to max_chars.
    """
    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    # Priority extraction
    content_element = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|main-content|post-content|article-content", re.I))
        or soup.find("body")
    )

    if content_element:
        text = content_element.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars] if text else ""
