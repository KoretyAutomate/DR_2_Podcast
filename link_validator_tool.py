"""
Link Validation Tool for Scientific Source Verification
========================================================

CrewAI tool that validates URLs are accessible and returns status information.
Used by the Scientific Auditor agent to verify research citations.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from typing import Optional


class LinkValidatorTool(BaseTool):
    """
    Tool for validating that scientific source URLs are accessible.

    Checks if a URL returns a successful HTTP response (200 OK) or is protected (403).
    Essential for ensuring research citations point to valid, accessible sources.
    """

    name: str = "Link Validator"
    description: str = (
        "Checks if a URL is valid and accessible. "
        "Returns 'Valid Link' for 200 OK, 'Link is protected (403)' for protected content that exists, "
        "or 'Broken Link' for other status codes. "
        "Use this to verify all scientific source URLs before including them in research reports."
    )

    def _run(self, url: str) -> str:
        """
        Validate a single URL.

        Args:
            url: The URL to validate

        Returns:
            Status string indicating link validity
        """
        try:
            # Use HEAD request for efficiency (doesn't download body)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)

            if response.status_code == 200:
                return f"✓ Valid Link (Status: 200 OK)"
            elif response.status_code == 403:
                # Protected but exists (common for paywalled journals)
                return f"⚠ Link is protected (403), but likely exists. May require institutional access."
            elif response.status_code == 404:
                return f"✗ Broken Link (Status: 404 Not Found)"
            elif response.status_code >= 500:
                return f"⚠ Server error (Status: {response.status_code}). Link may be valid but server is down."
            else:
                return f"⚠ Unexpected status (Status: {response.status_code})"

        except requests.exceptions.Timeout:
            return "⚠ Timeout: Server did not respond within 5 seconds. Link may be valid but slow."
        except requests.exceptions.TooManyRedirects:
            return "✗ Invalid URL: Too many redirects. Possible broken link."
        except requests.exceptions.RequestException as e:
            return f"✗ Invalid URL or Server Down: {str(e)[:100]}"
        except Exception as e:
            return f"✗ ERROR: {str(e)[:100]}"


# Batch validation function for processing multiple URLs
def validate_multiple_urls(urls: list[str]) -> dict[str, str]:
    """
    Validate multiple URLs in batch.

    Args:
        urls: List of URLs to validate

    Returns:
        Dictionary mapping each URL to its validation status
    """
    validator = LinkValidatorTool()
    results = {}

    for url in urls:
        results[url] = validator._run(url)

    return results


# Test function
if __name__ == "__main__":
    test_urls = [
        "https://www.nature.com",  # Should be valid
        "https://pubmed.ncbi.nlm.nih.gov/",  # Should be valid
        "https://example.com/nonexistent-page-12345",  # Should be 404
        "https://invalid-domain-that-does-not-exist-123456.com",  # Should fail
    ]

    print("Testing Link Validator Tool\n" + "="*60)

    validator = LinkValidatorTool()
    for url in test_urls:
        result = validator._run(url)
        print(f"\nURL: {url}")
        print(f"Result: {result}")

    print("\n" + "="*60)
