"""
nav_types.py — NavigationDriver データクラス・型定義

navigation_driver.py から抽出した型定義。
P1-1 リファクタリング: 4,212行モノリスの分割（Phase 1）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ロケールセグメント判定用の正規表現
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)

# trap 判定関数の型定義
TrapCheckerFn = Callable[[str], bool]


class RejectReason(str, Enum):
    """PDPリンク候補のreject理由"""

    NO_HREF = "no_href"
    DIFFERENT_ORIGIN = "different_origin"
    DIFFERENT_DOMAIN = "different_domain"
    DIFFERENT_SUBDOMAIN = "different_subdomain"
    BLOCKED_ORIGIN = "blocked_origin"
    NOT_PRODUCT_URL = "not_product_url"
    NO_PRODUCTS_PATH = "no_products_path"
    BLOCKED_DOMAIN = "blocked_domain"
    TRAP_PATTERN = "trap_pattern"
    NOISE_PATTERN = "noise_pattern"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


@dataclass
class LinkCandidate:
    """PDPリンク候補"""

    url: str
    phase: str  # "1a", "1b", "2", "moncler"
    normalized_url: str
    reject_reasons: list[str] = field(default_factory=list)
    accepted: bool = False
    source_selector: str | None = None
    origin: str | None = None
    notes: str | None = None
    product_url_rules: dict[str, Any] | None = None


class TrapPageDetected(Exception):
    """PLP ではなく trap ページが検出されたことを示す例外"""

    def __init__(self, trap_type: str, reason: str, url: str):
        self.trap_type = trap_type
        self.reason = reason
        self.url = url
        message = f"Trap page detected: {trap_type} - {reason} (URL: {url})"
        super().__init__(message)


@dataclass
class NavigationContext:
    """ナビゲーション実行時のコンテキスト情報"""

    site: str
    query: str
    site_config: dict[str, Any]
    settings: dict[str, Any]
    run_context: Any  # RunContext を直接 import しない（循環回避）
    start_t: float
    budget_ms: int
    entry_url: str | None = None
    context: Any = None
    link_collection_summary: dict[str, Any] | None = None


@dataclass
class NavigationOutcome:
    """ナビゲーション実行結果"""

    entry_url: str
    plp_materialized: bool = False
    trap_detected: bool = False
    trap_reason: str | None = None
    recovered: bool = False
    pdp_links: list[str] | None = None
    fallback_used: str | None = None
    locale_corrections: int = 0
    moncler_outcome: dict[str, Any] | None = None

    def __post_init__(self):
        if self.pdp_links is None:
            self.pdp_links = []
