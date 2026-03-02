"""Shared utility functions for DR_2_Podcast pipeline."""
import ipaddress
import re
import socket
from urllib.parse import urlparse

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


def is_safe_url(url: str) -> bool:
    """Check that a URL does not target private/link-local IP ranges (SSRF guard).

    Returns True if the URL resolves to a public IP or cannot be resolved.
    Returns False for RFC-1918, link-local (169.254.x.x), loopback, and IPv6 private addresses.
    Only blocks HTTP/HTTPS schemes.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Resolve hostname to IP(s) and check each
        for info in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (socket.gaierror, ValueError, OSError):
        # DNS resolution failure — allow (will fail at fetch time anyway)
        return True
    return True


def safe_float(v):
    """Safely convert value to float, returning None on failure."""
    if v is None or v == "null":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def safe_int(v):
    """Safely convert value to int, returning None on failure."""
    if v is None or v == "null":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def safe_str(v):
    """Safely convert value to string, returning None for empty/null."""
    if v is None or v == "null" or v == "":
        return None
    return str(v)
