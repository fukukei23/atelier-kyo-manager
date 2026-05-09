"""
navigation_helpers.py - Navigation helper functions for BrowserUseAgent

CR-ATELIER-003 Phase C: Navigation-related helper functions extracted from BrowserUseAgent

This module handles:
- URL normalization
- Redirect detection
- Locale handling
- URL validation
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunparse, urlunsplit


def normalize_url(url: str, site_config: dict[str, Any]) -> str:
    """
    Normalize URL based on site_config rules.

    CR-ATELIER-003 Phase C: Extracted from BrowserUseAgent._normalize_url

    Args:
        url: URL to normalize
        site_config: Site configuration

    Returns:
        str: Normalized URL
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        query_params = dict(parse_qsl(parsed.query))

        # Get normalization rules from site_config
        normalize_rules = site_config.get("navigation", {}).get("normalize_rules", [])

        # Apply normalization rules
        for rule in normalize_rules:
            if rule.get("type") == "locale_path":
                preferred_locale = rule.get("preferred", "en-int")
                # Replace locale in path
                path = re.sub(r"/en-[a-z]{2}/", f"/{preferred_locale}/", path, flags=re.IGNORECASE)
            elif rule.get("type") == "query_param":
                param_name = rule.get("param")
                param_value = rule.get("value")
                if param_name and param_value:
                    query_params[param_name] = param_value

        # Reconstruct URL
        new_query = urlencode(query_params) if query_params else ""
        normalized = urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, new_query, parsed.fragment))

        return normalized
    except Exception:
        return url


def normalize_abs_url(base_url: str, href: str) -> str:
    """
    Normalize absolute URL from base URL and relative href.

    CR-ATELIER-003 Phase C: Extracted from BrowserUseAgent._normalize_abs_url

    Args:
        base_url: Base URL
        href: Relative or absolute href

    Returns:
        str: Normalized absolute URL
    """
    try:
        absu = urljoin(base_url, href)
        parts = list(urlsplit(absu))
        if parts[2].endswith("/"):
            parts[2] = parts[2].rstrip("/")
        parts[3] = ""  # params
        parts[4] = ""  # query & fragment
        return urlunsplit(parts)
    except Exception:
        return href


def is_redirect_loop(url: str, history: list[str], max_history: int = 5) -> bool:
    """
    Check if URL is part of a redirect loop.

    CR-ATELIER-003 Phase C: New helper function

    Args:
        url: Current URL
        history: List of previous URLs
        max_history: Maximum history length to check

    Returns:
        bool: True if redirect loop detected
    """
    if not history:
        return False

    # Check if URL appears multiple times in recent history
    recent_history = history[-max_history:]
    return url in recent_history and recent_history.count(url) >= 2


def is_wrong_locale(url: str, site_config: dict[str, Any]) -> bool:
    """
    Check if URL has wrong locale based on site_config.

    CR-ATELIER-003 Phase C: New helper function

    Args:
        url: URL to check
        site_config: Site configuration

    Returns:
        bool: True if locale is wrong
    """
    try:
        parsed = urlparse(url)
        path = parsed.path

        preferred_locale = site_config.get("navigation", {}).get("locale", {}).get("preferred", "en-int")

        # Check if path contains a different locale than preferred
        if f"/{preferred_locale}/" not in path and re.search(r"/en-[a-z]{2}/", path, re.IGNORECASE):
            return True

        return False
    except Exception:
        return False
