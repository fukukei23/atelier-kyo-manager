"""
Moncler Navigation Policy

CR-ATELIER-003 Phase B: Moncler 専用のリダイレクト・ロケール制御を集約

このモジュールは、Moncler サイト専用のナビゲーションポリシーを担当します：
- ロケール調整（二重ロケール修正、期待ロケール誘導）
- URL バリデーション（PDP/PLP URL の検証）
"""

import logging
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

logger = logging.getLogger(__name__)


class MonclerNavigationPolicy:
    """
    Moncler 専用ナビゲーションポリシー

    CR-ATELIER-003 Phase B: BrowserUseAgent から Moncler 固有ロジックを分離
    """

    # 期待されるロケール
    EXPECTED_LOCALE = "en-int"
    EXPECTED_COUNTRY = "GB"

    # ブロック対象ドメイン
    BLOCKED_DOMAINS = {
        "onetrust.com",
        "monclergroup.com",
    }

    @staticmethod
    def adjust_locale(url: str) -> str:
        """
        Moncler の二重ロケール修正 / 期待ロケール誘導

        Args:
            url: 現在のURL

        Returns:
            str: 修正後のURL
        """
        parsed = urlparse(url)
        path = parsed.path

        # 二重ロケールパターンを検出（例: /en-lt/en-int/）
        double_locale_pattern = re.compile(r"/(en-[a-z]{2})/(en-[a-z]{2})/")
        match = double_locale_pattern.search(path)

        if match:
            # 二重ロケールを検出した場合、期待ロケールに修正
            logger.warning(f"[MonclerNavigationPolicy] Double locale detected: {path}")
            path = re.sub(r"/(en-[a-z]{2})/(en-[a-z]{2})/", f"/{MonclerNavigationPolicy.EXPECTED_LOCALE}/", path)

        # パスが期待ロケールで始まっていない場合、修正
        if not path.startswith(f"/{MonclerNavigationPolicy.EXPECTED_LOCALE}/"):
            # 既存のロケールを置換
            path = re.sub(r"^/(en-[a-z]{2})/", f"/{MonclerNavigationPolicy.EXPECTED_LOCALE}/", path)
            if not path.startswith(f"/{MonclerNavigationPolicy.EXPECTED_LOCALE}/"):
                # ロケールがない場合は挿入
                path = f"/{MonclerNavigationPolicy.EXPECTED_LOCALE}{path}"

        # クエリパラメータを追加/更新
        query_params = parse_qs(parsed.query)
        query_params["forceLocale"] = [MonclerNavigationPolicy.EXPECTED_LOCALE]
        query_params["shipToCountry"] = [MonclerNavigationPolicy.EXPECTED_COUNTRY]

        # URL を再構築
        new_query = urlencode(query_params, doseq=True)
        new_url = urljoin(url, path)
        if new_query:
            new_url = f"{new_url.split('?')[0]}?{new_query}"

        if new_url != url:
            logger.info(f"[MonclerNavigationPolicy] Adjusted locale: {url} -> {new_url}")

        return new_url

    @staticmethod
    def is_valid_moncler_url(url: str) -> bool:
        """
        Moncler の PDP/PLP URL バリデーション

        Args:
            url: 検証対象のURL

        Returns:
            bool: 有効な Moncler URL の場合 True
        """
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            path = parsed.path

            # ホストチェック（Moncler 本体のドメインのみ）
            if not host.endswith("moncler.com"):
                logger.debug(f"[MonclerNavigationPolicy] Invalid host: {host}")
                return False

            # ブロック対象ドメインをチェック
            for blocked_domain in MonclerNavigationPolicy.BLOCKED_DOMAINS:
                if blocked_domain in host:
                    logger.debug(f"[MonclerNavigationPolicy] Blocked domain: {host}")
                    return False

            # 二重ロケールパターンをチェック
            double_locale_pattern = re.compile(r"/(en-[a-z]{2})/(en-[a-z]{2})/")
            if double_locale_pattern.search(path):
                logger.debug(f"[MonclerNavigationPolicy] Double locale pattern detected: {path}")
                return False

            # Trap ページパターンをチェック
            trap_patterns = ["/search", "/404", "/client-service", "/not-found"]
            for trap_pattern in trap_patterns:
                if trap_pattern in path:
                    logger.debug(f"[MonclerNavigationPolicy] Trap pattern detected: {path}")
                    return False

            # PDP URL の場合は /products/ を含む必要がある
            if "/products/" in path:
                # PDP URL として有効
                return True

            # PLP URL の場合は /en-int/ を含む必要がある
            if f"/{MonclerNavigationPolicy.EXPECTED_LOCALE}/" in path:
                return True

            logger.debug(f"[MonclerNavigationPolicy] URL does not match expected patterns: {path}")
            return False

        except Exception as e:
            logger.debug(f"[MonclerNavigationPolicy] URL validation failed: {e}")
            return False
