"""
url_rules.py — URL検証・分類・正規化の pure functions

navigation_driver.py から抽出（P1-1 Phase 2）。
全て self 参照なしの pure function。
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from app.agents.browser.nav_types import LinkCandidate, RejectReason

logger = logging.getLogger(__name__)


def extract_etld_plus_one(hostname: str) -> str | None:
    """eTLD+1を抽出（例: www.moncler.com -> moncler.com）"""
    if not hostname:
        return None
    hostname = hostname.lower().strip()
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def is_same_site(url1: str, url2: str) -> bool:
    """2つのURLが同じサイト（eTLD+1）か判定"""
    try:
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)
        etld1 = extract_etld_plus_one(parsed1.netloc)
        etld2 = extract_etld_plus_one(parsed2.netloc)
        return etld1 and etld2 and etld1 == etld2
    except Exception:
        return False


def check_origin_allowed(
    url: str,
    base_url: str,
    site_config: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """originが許可されているかチェック（eTLD+1判定対応）"""
    try:
        parsed = urlparse(url)
        base_parsed = urlparse(base_url)
        url_origin = (parsed.scheme, parsed.netloc)
        base_origin = (base_parsed.scheme, base_parsed.netloc)

        if url_origin == base_origin:
            return True, None

        if is_same_site(url, base_url):
            if parsed.netloc.lower() != base_parsed.netloc.lower():
                return True, None
            return True, None

        if site_config:
            allowed_origins = site_config.get("allowed_origins", [])
            blocked_origins = site_config.get("blocked_origins", [])
            allowed_host_suffixes = site_config.get("allowed_host_suffixes", [])
            allowed_domains = site_config.get("allowed_domains", [])
            if not allowed_domains:
                allowed_domain = site_config.get("allowed_domain")
                if allowed_domain:
                    allowed_domains = [allowed_domain]
        else:
            allowed_origins = []
            blocked_origins = []
            allowed_host_suffixes = []
            allowed_domains = []

        if blocked_origins:
            for blocked in blocked_origins:
                if blocked in parsed.netloc.lower():
                    return False, RejectReason.BLOCKED_ORIGIN.value

        if allowed_domains:
            url_etld = extract_etld_plus_one(parsed.netloc)
            for allowed_domain in allowed_domains:
                allowed_etld = extract_etld_plus_one(allowed_domain)
                if url_etld and allowed_etld and url_etld == allowed_etld:
                    return True, None

        if allowed_origins:
            for allowed in allowed_origins:
                if allowed in parsed.netloc.lower():
                    return True, None

        if allowed_host_suffixes:
            for suffix in allowed_host_suffixes:
                if parsed.netloc.lower().endswith(suffix):
                    return True, None

        url_etld = extract_etld_plus_one(parsed.netloc)
        base_etld = extract_etld_plus_one(base_parsed.netloc)
        if url_etld and base_etld and url_etld != base_etld:
            return False, RejectReason.DIFFERENT_DOMAIN.value
        elif parsed.netloc.lower() != base_parsed.netloc.lower():
            return False, RejectReason.DIFFERENT_SUBDOMAIN.value

        return False, RejectReason.DIFFERENT_ORIGIN.value
    except Exception:
        return False, RejectReason.VALIDATION_ERROR.value


def extract_origin(url: str) -> str | None:
    """URLからoriginを抽出"""
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return None


def normalize_candidate_url(
    url: str,
    base_url: str,
    site_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """候補URLを正規化"""
    normalization_info = {
        "original": url,
        "was_relative": False,
        "removed_query_params": [],
        "removed_fragment": False,
        "normalized": url,
    }

    try:
        if url.startswith("//"):
            base_parsed = urlparse(base_url)
            normalized = f"{base_parsed.scheme}:{url}"
            normalization_info["was_relative"] = True
        elif url.startswith("/") or not url.startswith("http"):
            normalized = urljoin(base_url, url)
            normalization_info["was_relative"] = True
        else:
            normalized = url

        parsed = urlparse(normalized)

        query_params = {}
        if parsed.query:
            query_dict = parse_qs(parsed.query)

            remove_params = []
            if site_config:
                url_rules = site_config.get("url_rules", {})
                if isinstance(url_rules, dict):
                    normalize_rules = url_rules.get("normalize_rules", {})
                    if isinstance(normalize_rules, dict):
                        remove_params = normalize_rules.get("remove_query_params", [])

            for key, values in query_dict.items():
                if key not in remove_params:
                    query_params[key] = values
                else:
                    normalization_info["removed_query_params"].append(key)

        fragment = ""
        if site_config:
            url_rules = site_config.get("url_rules", {})
            if isinstance(url_rules, dict):
                normalize_rules = url_rules.get("normalize_rules", {})
                if isinstance(normalize_rules, dict) and not normalize_rules.get("remove_fragment", True):
                    fragment = parsed.fragment
                else:
                    normalization_info["removed_fragment"] = True

        normalized_query = urlencode(query_params, doseq=True) if query_params else ""
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, normalized_query, fragment))

        normalization_info["normalized"] = normalized
        return normalized, normalization_info
    except Exception as e:
        logger.debug(f"[URL Normalize] Failed to normalize URL: {e}")
        normalization_info["error"] = str(e)
        return url, normalization_info


def validate_candidate_url(
    url: str,
    normalized_url: str,
    base_url: str,
    site_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """候補URLを検証し、判定根拠を返す"""
    result = {
        "domain_allowed": False,
        "allow_path_matched": False,
        "allow_path_pattern": None,
        "forbidden_path_matched": False,
        "forbidden_path_pattern": None,
        "locale_ok": False,
        "final_decision": "reject",
        "reject_reasons": [],
        "product_url_rules": {},
    }

    try:
        parsed = urlparse(normalized_url)
        path = parsed.path or ""
        host = parsed.netloc.lower() if parsed.netloc else ""

        # 1. ドメイン許可チェック
        if site_config:
            allowed_domains = site_config.get("allowed_domains", [])
            if not allowed_domains:
                allowed_domain = site_config.get("allowed_domain")
                if allowed_domain:
                    allowed_domains = [allowed_domain]

            domain_allowed = False
            if allowed_domains:
                url_etld = extract_etld_plus_one(host)
                for allowed_domain in allowed_domains:
                    allowed_etld = extract_etld_plus_one(allowed_domain)
                    if url_etld and allowed_etld and url_etld == allowed_etld:
                        domain_allowed = True
                        break

            result["domain_allowed"] = domain_allowed

            if not domain_allowed:
                result["reject_reasons"].append(RejectReason.DIFFERENT_DOMAIN.value)
                result["final_decision"] = "reject"
                return result
        else:
            base_parsed = urlparse(base_url)
            base_host = base_parsed.netloc.lower() if base_parsed.netloc else ""
            if host and base_host:
                url_etld = extract_etld_plus_one(host)
                base_etld = extract_etld_plus_one(base_host)
                result["domain_allowed"] = url_etld and base_etld and url_etld == base_etld
                if not result["domain_allowed"]:
                    result["reject_reasons"].append(RejectReason.DIFFERENT_DOMAIN.value)
                    result["final_decision"] = "reject"
                    return result
            else:
                result["domain_allowed"] = True

        # 2. forbidden_path_patternsチェック
        if site_config:
            url_rules = site_config.get("url_rules", {})
            if isinstance(url_rules, dict):
                forbidden_path_patterns = url_rules.get("forbidden_path_patterns", [])
                for pattern in forbidden_path_patterns:
                    if re.search(pattern, path, re.I):
                        result["forbidden_path_matched"] = True
                        result["forbidden_path_pattern"] = pattern
                        result["reject_reasons"].append(RejectReason.NOISE_PATTERN.value)
                        result["final_decision"] = "reject"
                        return result

                allow_path_patterns = url_rules.get("allow_path_patterns", [])
                if allow_path_patterns:
                    for pattern in allow_path_patterns:
                        if re.search(pattern, path, re.I):
                            result["allow_path_matched"] = True
                            result["allow_path_pattern"] = pattern
                            break

                if allow_path_patterns and not result["allow_path_matched"]:
                    result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                    result["final_decision"] = "reject"
                    return result
                elif not allow_path_patterns:
                    if not re.search(r"/products?/", path, re.I):
                        result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                        result["final_decision"] = "reject"
                        return result
            else:
                if not re.search(r"/products?/", path, re.I):
                    result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                    result["final_decision"] = "reject"
                    return result
        else:
            if not re.search(r"/products?/", path, re.I):
                result["reject_reasons"].append(RejectReason.NO_PRODUCTS_PATH.value)
                result["final_decision"] = "reject"
                return result
            else:
                result["allow_path_matched"] = True

        result["locale_ok"] = path.lower().startswith("/en-int/")

        if not result["allow_path_matched"]:
            from app.agents.browser.extractor import looks_like_product_url

            looks_like_product = looks_like_product_url(normalized_url)
            if not looks_like_product:
                result["reject_reasons"].append(RejectReason.NOT_PRODUCT_URL.value)
                result["final_decision"] = "reject"
                return result

        if not result["reject_reasons"]:
            result["final_decision"] = "accept"

        result["product_url_rules"] = {
            "domain_allowed": result["domain_allowed"],
            "allow_path_matched": result["allow_path_matched"],
            "allow_path_pattern": result["allow_path_pattern"],
            "forbidden_path_matched": result["forbidden_path_matched"],
            "forbidden_path_pattern": result["forbidden_path_pattern"],
            "locale_ok": result["locale_ok"],
            "final_decision": result["final_decision"],
            "reject_reasons": result["reject_reasons"],
        }

        return result
    except Exception as e:
        logger.warning(f"[Validate Candidate URL] Failed to validate URL: {e}", exc_info=True)
        result["reject_reasons"].append(RejectReason.VALIDATION_ERROR.value)
        result["final_decision"] = "reject"
        result["error"] = str(e)
        return result


def classify_candidate(
    candidate: LinkCandidate,
    base_url: str,
    site_config: dict[str, Any] | None = None,
) -> LinkCandidate:
    """候補を分類し、reject理由を記録"""
    if candidate.normalized_url:
        candidate.origin = extract_origin(candidate.normalized_url)

    parsed_url = urlparse(candidate.normalized_url)
    if parsed_url.scheme and parsed_url.scheme.lower() not in ("http", "https", ""):
        candidate.reject_reasons.append(RejectReason.NO_HREF.value)
        candidate.notes = f"invalid scheme: {parsed_url.scheme}"
        candidate.product_url_rules = {
            "domain_allowed": False,
            "allow_path_matched": False,
            "forbidden_path_matched": False,
            "locale_ok": False,
            "final_decision": "reject",
            "reject_reasons": [RejectReason.NO_HREF.value],
            "error": "invalid scheme",
        }
        return candidate

    if "onetrust.com" in parsed_url.netloc.lower():
        candidate.reject_reasons.append(RejectReason.BLOCKED_DOMAIN.value)
        candidate.notes = "blocked domain: onetrust.com"
        candidate.product_url_rules = {
            "domain_allowed": False,
            "allow_path_matched": False,
            "forbidden_path_matched": False,
            "locale_ok": False,
            "final_decision": "reject",
            "reject_reasons": [RejectReason.BLOCKED_DOMAIN.value],
        }
        return candidate

    validation_result = validate_candidate_url(
        url=candidate.url,
        normalized_url=candidate.normalized_url,
        base_url=base_url,
        site_config=site_config,
    )

    candidate.reject_reasons.extend(validation_result["reject_reasons"])
    candidate.product_url_rules = validation_result.get("product_url_rules", {})

    if validation_result["forbidden_path_matched"]:
        candidate.notes = f"forbidden path pattern: {validation_result.get('forbidden_path_pattern', 'unknown')}"
    elif not validation_result["domain_allowed"]:
        candidate.notes = f"different domain (eTLD+1): {candidate.origin}"
    elif not validation_result["allow_path_matched"]:
        candidate.notes = "path did not match allow_path_patterns"
    elif not validation_result["locale_ok"]:
        candidate.notes = "locale mismatch"

    if validation_result["final_decision"] == "accept":
        candidate.accepted = True

    return candidate
