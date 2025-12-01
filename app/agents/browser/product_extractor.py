# -*- coding: utf-8 -*-
# ==============================================================================
# File: app/agents/browser/product_extractor.py
# Version: 1.0.0 (Task D: ProductExtractor の site_config 駆動化)
# Purpose: PDP 抽出ロジックを site_config JSON で駆動する汎用 Extractor
# ==============================================================================
"""
Task D: Stage 5 – Extractor の site_config 駆動化

PDP 抽出ロジックを「サイトごとにハードコード」するのではなく、
site_config JSON で定義されたセレクタ・正規化ルールに基づいて抽出する汎用 Extractor。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from playwright.async_api import Page, BrowserContext

if TYPE_CHECKING:
    from app.core.run_context import RunContext

logger = logging.getLogger(__name__)

# デフォルトセレクタ（site_config に定義がない場合のフォールバック）
DEFAULT_PRICE_SELECTORS = [
    "meta[property='product:price:amount']",
    "meta[itemprop='price']",
    "[itemprop=price]",
    "span:has(meta[itemprop='price'])",
    "[data-testid*=price]",
    "[class*=price]",
]

DEFAULT_TITLE_SELECTORS = [
    "h1[itemprop='name']",
    "h1.product-title",
    "h1",
    "meta[property='og:title']",
]

DEFAULT_SIZE_BUTTON_SELECTORS = [
    "button[aria-disabled='false'][data-size]",
    "button[aria-disabled='false'][data-testid*='size']",
    "button:not([disabled])[data-size]",
    "li[data-size] button:not([disabled])",
    "button[class*='size']:not([disabled])",
]


@dataclass
class ProductInfo:
    """抽出された商品情報"""
    title: Optional[str] = None
    price: Optional[float] = None  # Task 2: float に変更
    currency: Optional[str] = None
    images: List[str] = field(default_factory=list)
    sizes: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    description: Optional[str] = None
    raw_html_path: Optional[str] = None
    url: Optional[str] = None
    brand: Optional[str] = None
    list_price: Optional[float] = None  # Task 2: float に変更
    discount_pct: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # Task 2: metadata フィールドを追加


@dataclass
class PriceRules:
    """価格正規化ルール"""
    strip_chars: List[str] = field(default_factory=lambda: ["¥", ",", " "])
    decimal_separator: str = "."
    thousands_separator: str = ","


class ProductExtractor:
    """
    PDP から商品情報を抽出する汎用 Extractor。
    site_config で定義されたセレクタと正規化ルールに基づいて抽出する。
    
    Stage 5: site_config 駆動に完全移行
    """
    
    def __init__(
        self,
        site_config: Dict[str, Any],
        run_context: Optional["RunContext"] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.site_config = site_config
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
        
        # Stage 5: config getter のキャッシュ用
        self._pdp_config: Optional[Dict[str, Any]] = None
        self._price_rules: Optional[Dict[str, Any]] = None
    
    def _get_pdp_config(self) -> Dict[str, Any]:
        """
        site_config から PDP 設定を取得（後方互換性維持）
        
        Stage 5: 新しい selectors.pdp.* スキーマを優先しつつ、
        既存の selectors.pdp.* やデフォルト値からもフォールバック
        """
        if self._pdp_config is not None:
            return self._pdp_config
        
        selectors = (self.site_config.get("selectors") or {})
        pdp_cfg = selectors.get("pdp") or {}
        
        # 後方互換性: 既存のスキーマからもフォールバック
        self._pdp_config = {
            "title": pdp_cfg.get("title") or DEFAULT_TITLE_SELECTORS,
            "price": pdp_cfg.get("price") or DEFAULT_PRICE_SELECTORS,
            "list_price": pdp_cfg.get("list_price") or [],
            "currency": pdp_cfg.get("currency") or [
                "meta[property='product:price:currency']",
                "meta[itemprop='priceCurrency']",
            ],
            "images": self._normalize_images_config(pdp_cfg.get("images")) or [
                ".product-images img",
                ".product-gallery img",
                "[data-testid*='image'] img",
                "img[itemprop='image']",
            ],
            "sizes": pdp_cfg.get("size") or pdp_cfg.get("sizes") or [
                ".size-selector option",
                "button[data-size]",
                "[role='radiogroup'] [role='radio']",
            ],
            "colors": pdp_cfg.get("color") or pdp_cfg.get("colors") or [
                ".color-selector .swatch",
                "button[data-color]",
                "[data-testid*='color']",
            ],
            "description": pdp_cfg.get("description") or [
                ".product-description",
                "[itemprop='description']",
                "meta[property='og:description']",
            ],
            "brand": pdp_cfg.get("brand") or [
                "meta[property='og:site_name']",
                "[itemprop='brand']",
                ".product-brand",
            ],
            "sku": pdp_cfg.get("sku") or [],
            "availability": self._normalize_availability_config(pdp_cfg.get("availability")) or [],
            "breadcrumbs": pdp_cfg.get("breadcrumbs") or [],
            "size_button": pdp_cfg.get("size_button") or DEFAULT_SIZE_BUTTON_SELECTORS,
            "size_select_policy": pdp_cfg.get("size_select_policy") or {
                "mode": "off",
                "prefer_labels": [],
                "price_wait_ms": 4000,
            },
            "visible_price_selectors": pdp_cfg.get("visible_price_selectors") or DEFAULT_PRICE_SELECTORS,
            "image_attr": self._get_image_attr(pdp_cfg.get("images")) or pdp_cfg.get("image_attr", "src"),
            "image_base_url": self._get_image_base_url(pdp_cfg.get("images")) or pdp_cfg.get("image_base_url"),
            "raw_html_capture": pdp_cfg.get("raw_html_capture", {
                "enabled": True,
                "filename": "pdp_raw.html",
            }),
            "json_ld": pdp_cfg.get("json_ld", {
                "enabled": True,
                "paths": {
                    "price": ["offers.price", "offers[0].price"],
                    "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
                    "title": ["name"],
                    "description": ["description"],
                },
            }),
            "meta_fallback": pdp_cfg.get("meta_fallback", {
                "enabled": True,
                "selectors": [
                    "meta[property='og:price:amount']",
                    "meta[name='twitter:data1']",
                ],
            }),
            "availability_patterns": self._get_availability_patterns(pdp_cfg.get("availability"), pdp_cfg.get("availability_patterns")) or [
                "out of stock",
                "在庫なし",
            ],
        }
        
        return self._pdp_config
    
    def _normalize_images_config(self, images_cfg: Any) -> Optional[List[str]]:
        """
        images 設定をリスト形式に正規化する
        
        Stage 5: 辞書形式（{"selectors": [...]}）とリスト形式の両方に対応
        """
        if not images_cfg:
            return None
        
        if isinstance(images_cfg, list):
            return images_cfg
        
        if isinstance(images_cfg, dict):
            return images_cfg.get("selectors")
        
        return None
    
    def _get_image_attr(self, images_cfg: Any) -> Optional[str]:
        """images 設定から image_attr を取得"""
        if isinstance(images_cfg, dict):
            return images_cfg.get("image_attr")
        return None
    
    def _get_image_base_url(self, images_cfg: Any) -> Optional[str]:
        """images 設定から image_base_url を取得"""
        if isinstance(images_cfg, dict):
            return images_cfg.get("base_url")
        return None
    
    def _normalize_availability_config(self, availability_cfg: Any) -> Optional[List[str]]:
        """
        availability 設定をリスト形式に正規化する
        
        Stage 5: 辞書形式（{"selectors": [...]}）とリスト形式の両方に対応
        """
        if not availability_cfg:
            return None
        
        if isinstance(availability_cfg, list):
            return availability_cfg
        
        if isinstance(availability_cfg, dict):
            return availability_cfg.get("selectors")
        
        return None
    
    def _get_availability_patterns(self, availability_cfg: Any, availability_patterns: Any) -> Optional[List[str]]:
        """
        availability_patterns を取得する
        
        Stage 5: 辞書形式の availability から patterns を取得するか、直接 availability_patterns を取得
        """
        # まず辞書形式の availability から patterns を取得
        if isinstance(availability_cfg, dict):
            patterns = availability_cfg.get("patterns")
            if patterns:
                return patterns
        
        # 直接 availability_patterns が設定されている場合
        if availability_patterns:
            return availability_patterns
        
        return None
    
    def _get_price_rules(self) -> Dict[str, Any]:
        """
        site_config から価格正規化ルールを取得
        
        Stage 5: selectors.pdp.price.normalize_rules を優先しつつ、
        トップレベルの price_rules からもフォールバック
        """
        if self._price_rules is not None:
            return self._price_rules
        
        pdp_cfg = self._get_pdp_config()
        price_cfg = pdp_cfg.get("price")
        
        # 新スキーマ: selectors.pdp.price.normalize_rules
        normalize_rules = {}
        if isinstance(price_cfg, dict) and "normalize_rules" in price_cfg:
            normalize_rules = price_cfg["normalize_rules"]
        else:
            # 旧スキーマ: トップレベルの price_rules
            normalize_rules = self.site_config.get("price_rules", {})
        
        self._price_rules = {
            "strip_chars": normalize_rules.get("strip_chars", ["¥", ",", " "]),
            "thousands_separator": normalize_rules.get("thousands_separator", ","),
            "decimal_separator": normalize_rules.get("decimal_separator", "."),
            "currency_fallback": normalize_rules.get("currency_fallback", "JPY"),
            "price_pattern": normalize_rules.get("price_pattern", r"[\d.,]+"),
            "currency_symbols": normalize_rules.get("currency_symbols", {
                "¥": "JPY",
                "$": "USD",
                "€": "EUR",
                "£": "GBP",
            }),
        }
        
        return self._price_rules
    
    async def extract(
        self,
        page: Page,
        *,
        context: Optional[BrowserContext] = None,
        prepare_page: Optional[Any] = None,
    ) -> ProductInfo:
        """
        PDP から商品情報を抽出する。
        
        Args:
            page: Playwright Page オブジェクト
            context: BrowserContext（オプション）
            prepare_page: ページ準備用のコルーチン（オプション）
            
        Returns:
            ProductInfo: 抽出された商品情報
        """
        if prepare_page:
            try:
                await prepare_page(page)
            except Exception as prep_e:
                self.logger.debug(f"[ProductExtractor] prepare_page skipped: {prep_e}")
        
        # Stage 5: config getter を使用
        pdp_config = self._get_pdp_config()
        price_rules = self._get_price_rules()
        
        product_info = ProductInfo(url=page.url)
        
        # 1. タイトル抽出
        product_info.title = await self._extract_title(page, pdp_config)
        
        # 2. 価格抽出（サイズ選択を試行）
        product_info.price = await self._extract_price_with_size_option(page, pdp_config, price_rules)
        
        # 3. 通貨抽出
        product_info.currency = await self._extract_currency(page, pdp_config)
        
        # 4. 画像抽出
        product_info.images = await self._extract_images(page, pdp_config)
        
        # 5. サイズ抽出
        product_info.sizes = await self._extract_sizes(page, pdp_config)
        
        # 6. カラー抽出
        product_info.colors = await self._extract_colors(page, pdp_config)
        
        # 7. 説明抽出
        product_info.description = await self._extract_description(page, pdp_config)
        
        # 8. ブランド抽出
        product_info.brand = await self._extract_brand(page, pdp_config)
        
        # 9. 定価・割引率抽出
        list_price, discount_pct = await self._extract_list_price_and_discount(page, pdp_config, price_rules)
        product_info.list_price = list_price
        product_info.discount_pct = discount_pct
        
        # 10. フォールバック: JSON-LD / Meta タグ
        if product_info.price is None:
            fallback_data = await self._extract_from_json_ld_or_meta(page, pdp_config)
            if fallback_data:
                if product_info.price is None and fallback_data.get("price"):
                    # フォールバックの価格も float に変換
                    price_str = str(fallback_data["price"])
                    product_info.price = self._normalize_price_to_float(price_str, price_rules)
                if not product_info.currency and fallback_data.get("currency"):
                    product_info.currency = fallback_data["currency"]
        
        # 11. メタデータを収集
        product_info.metadata = {
            "extraction_timestamp": time.time(),
            "url": product_info.url,
            "has_title": product_info.title is not None,
            "has_price": product_info.price is not None,
            "has_currency": product_info.currency is not None,
            "image_count": len(product_info.images),
            "size_count": len(product_info.sizes),
            "color_count": len(product_info.colors),
        }
        
        # 12. HTML 保存（オプション）
        html_capture = pdp_config.get("raw_html_capture", {})
        if html_capture.get("enabled", True) and self.run_context:
            try:
                html_content = await page.content()
                filename = html_capture.get("filename", "pdp_raw.html")
                self.run_context.save_content(filename, html_content)
                product_info.raw_html_path = str(self.run_context.get_path(filename))
            except Exception as e:
                self.logger.warning(f"[ProductExtractor] Failed to save HTML: {e}")
        
        return product_info
    
    async def _extract_title(self, page: Page, pdp_config: Dict[str, Any]) -> Optional[str]:
        """タイトルを抽出する（Stage 5: pdp_config を引数として受け取る）"""
        title_selectors = pdp_config.get("title", [])
        
        for selector in title_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                
                if selector.startswith("meta["):
                    content = await locator.get_attribute("content")
                else:
                    content = await locator.inner_text()
                
                title = (content or "").strip()
                if title:
                    self.logger.debug(f"[ProductExtractor] Title found via: {selector}")
                    return title
            except Exception:
                continue
        
        return None
    
    async def _extract_price_with_size_option(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
        price_rules: Dict[str, Any],
    ) -> Optional[float]:
        """価格を抽出する（サイズ選択を試行）。Stage 5: pdp_config と price_rules を引数として受け取る"""
        # まず通常の価格抽出を試行
        price = await self._extract_price(page, pdp_config, price_rules)
        if price is not None:
            return price
        
        # サイズ選択を試行
        size_select_policy = pdp_config.get("size_select_policy", {}) or {}
        
        if size_select_policy.get("mode") == "off":
            return None
        
        self.logger.debug("[ProductExtractor] Price not found initially, attempting size selection...")
        
        if await self._click_size_to_reveal_price(page, pdp_config, size_select_policy):
            price = await self._extract_price(page, pdp_config, price_rules)
            if price is not None:
                self.logger.debug("[ProductExtractor] Price found after size selection.")
                return price
        
        return None
    
    async def _extract_price(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
        price_rules: Dict[str, Any],
    ) -> Optional[float]:
        """価格を抽出する（Stage 5: pdp_config と price_rules を引数として受け取る）"""
        price_selectors = pdp_config.get("price", [])
        
        # price が dict の場合、selectors キーから取得
        if isinstance(price_selectors, dict):
            price_selectors = price_selectors.get("selectors", [])
        
        for selector in price_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                
                if selector.startswith("meta["):
                    content = await locator.get_attribute("content")
                else:
                    content = await locator.inner_text()
                
                price_text = (content or "").strip()
                if price_text:
                    # 価格正規化と float 変換
                    price_float = self._normalize_price_to_float(price_text, price_rules)
                    if price_float is not None:
                        self.logger.debug(f"[ProductExtractor] Price found via: {selector} = {price_float}")
                        return price_float
            except Exception:
                continue
        
        return None
    
    def _normalize_price(self, price_text: str, price_rules: Dict[str, Any]) -> Optional[str]:
        """価格テキストを正規化する（文字列として）。Stage 5: price_rules を引数として受け取る"""
        if not price_text:
            return None
        
        # strip_chars を削除
        normalized = price_text
        for char in price_rules.get("strip_chars", []):
            normalized = normalized.replace(char, "")
        
        # 正規表現パターンで数値部分を抽出
        pattern = price_rules.get("price_pattern", r"[\d.,]+")
        match = re.search(pattern, normalized)
        if match:
            return match.group(0)
        
        return normalized.strip()
    
    def _normalize_price_to_float(self, price_text: str, price_rules: Dict[str, Any]) -> Optional[float]:
        """
        価格テキストを正規化して float に変換する。
        Stage 5: price_rules を引数として受け取る
        """
        if not price_text:
            return None
        
        try:
            # まず文字列として正規化
            normalized_str = self._normalize_price(price_text, price_rules)
            if not normalized_str:
                return None
            
            # thousands_separator を削除
            thousands_sep = price_rules.get("thousands_separator", ",")
            if thousands_sep:
                normalized_str = normalized_str.replace(thousands_sep, "")
            
            # decimal_separator を "." に統一（既に "." の場合はそのまま）
            decimal_sep = price_rules.get("decimal_separator", ".")
            if decimal_sep != ".":
                normalized_str = normalized_str.replace(decimal_sep, ".")
            
            # float に変換
            price_float = float(normalized_str)
            return price_float
        except (ValueError, TypeError) as e:
            self.logger.warning(f"[ProductExtractor] Failed to parse price '{price_text}': {e}")
            return None
    
    async def _extract_currency(self, page: Page, pdp_config: Dict[str, Any]) -> Optional[str]:
        """通貨を抽出する（Stage 5: pdp_config を引数として受け取る）"""
        currency_selectors = pdp_config.get("currency", [])
        
        for selector in currency_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                
                content = await locator.get_attribute("content")
                currency = (content or "").strip()
                if currency:
                    self.logger.debug(f"[ProductExtractor] Currency found via: {selector}")
                    return currency
            except Exception:
                continue
        
        return None
    
    async def _extract_images(self, page: Page, pdp_config: Dict[str, Any]) -> List[str]:
        """商品画像を抽出する（Stage 5: pdp_config を引数として受け取る）"""
        image_selectors = pdp_config.get("images", [])
        image_attr = pdp_config.get("image_attr", "src")
        image_base_url = pdp_config.get("image_base_url")
        
        images = []
        for selector in image_selectors:
            try:
                locators = page.locator(selector)
                count = await locators.count()
                for i in range(count):
                    try:
                        img = locators.nth(i)
                        src = await img.get_attribute(image_attr)
                        if src:
                            # 相対URLを絶対URLに変換
                            src = self._normalize_image_url(src, page.url, image_base_url)
                            if src:
                                images.append(src)
                    except Exception:
                        continue
            except Exception:
                continue
        
        # 重複除去
        return list(dict.fromkeys(images))
    
    def _normalize_image_url(
        self,
        src: str,
        page_url: str,
        base_url: Optional[str] = None,
    ) -> Optional[str]:
        """画像 URL を正規化する（Stage 5: site_config から base_url を取得）"""
        if not src:
            return None
        
        # プロトコル相対 URL (//example.com/image.jpg)
        if src.startswith("//"):
            return f"https:{src}"
        
        # 絶対 URL
        if src.startswith("http://") or src.startswith("https://"):
            return src
        
        # 相対 URL
        if src.startswith("/"):
            base = base_url or page_url
            try:
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(base)
                return urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    src,
                    "",
                    "",
                    "",
                ))
            except Exception:
                return None
        
        return src
    
    async def _extract_sizes(self, page: Page, pdp_config: Dict[str, Any]) -> List[str]:
        """サイズオプションを抽出する（Stage 5: pdp_config を引数として受け取る）"""
        size_selectors = pdp_config.get("sizes", [])
        
        sizes = []
        for selector in size_selectors:
            try:
                locators = page.locator(selector)
                count = await locators.count()
                for i in range(count):
                    try:
                        el = locators.nth(i)
                        size_text = await el.inner_text()
                        if size_text:
                            sizes.append(size_text.strip())
                    except Exception:
                        continue
            except Exception:
                continue
        
        # 重複除去
        return list(dict.fromkeys(sizes))
    
    async def _extract_colors(self, page: Page, pdp_config: Dict[str, Any]) -> List[str]:
        """カラーオプションを抽出する（Stage 5: pdp_config を引数として受け取る）"""
        color_selectors = pdp_config.get("colors", [])
        
        colors = []
        for selector in color_selectors:
            try:
                locators = page.locator(selector)
                count = await locators.count()
                for i in range(count):
                    try:
                        el = locators.nth(i)
                        color_text = await el.get_attribute("aria-label") or await el.inner_text()
                        if color_text:
                            colors.append(color_text.strip())
                    except Exception:
                        continue
            except Exception:
                continue
        
        # 重複除去
        return list(dict.fromkeys(colors))
    
    async def _extract_description(self, page: Page, pdp_config: Dict[str, Any]) -> Optional[str]:
        """商品説明を抽出する（Stage 5: pdp_config を引数として受け取る）"""
        description_selectors = pdp_config.get("description", [])
        
        for selector in description_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                
                if selector.startswith("meta["):
                    content = await locator.get_attribute("content")
                else:
                    content = await locator.inner_text()
                
                description = (content or "").strip()
                if description:
                    self.logger.debug(f"[ProductExtractor] Description found via: {selector}")
                    return description
            except Exception:
                continue
        
        return None
    
    async def _extract_brand(self, page: Page, pdp_config: Dict[str, Any]) -> Optional[str]:
        """ブランドを抽出する（Stage 5: pdp_config を引数として受け取る）"""
        brand_selectors = pdp_config.get("brand", [])
        
        for selector in brand_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                
                if selector.startswith("meta["):
                    content = await locator.get_attribute("content")
                else:
                    content = await locator.inner_text()
                
                brand = (content or "").strip()
                if brand:
                    self.logger.debug(f"[ProductExtractor] Brand found via: {selector}")
                    return brand
            except Exception:
                continue
        
        return None
    
    async def _extract_list_price_and_discount(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
        price_rules: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """定価と割引率を抽出する（Stage 5: pdp_config と price_rules を引数として受け取る）"""
        list_price_selectors = pdp_config.get("list_price", [])
        
        list_price = None
        discount_pct = None
        
        if list_price_selectors:
            for selector in list_price_selectors:
                try:
                    locator = page.locator(selector).first
                    if await locator.count() == 0:
                        continue
                    
                    content = await locator.inner_text()
                    list_price_text = (content or "").strip()
                    if list_price_text:
                        # Stage 5: float に変換（price_rules を使用）
                        list_price = self._normalize_price_to_float(list_price_text, price_rules)
                        if list_price is not None:
                            break
                except Exception:
                    continue
        
        # 割引率の計算（定価と現在価格がある場合）
        if list_price is not None:
            current_price = await self._extract_price(page, pdp_config, price_rules)
            if current_price is not None:
                try:
                    if list_price > current_price:
                        discount_pct = ((list_price - current_price) / list_price) * 100
                except Exception:
                    pass
        
        return list_price, discount_pct
    
    async def _click_size_to_reveal_price(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
        size_select_policy: Dict[str, Any],
    ) -> bool:
        """サイズをクリックして価格を表示させる（Stage 5: pdp_config を引数として受け取る）"""
        size_button_selectors = pdp_config.get("size_button", [])
        # Stage 5: availability_patterns は _get_pdp_config() で既にデフォルト値が設定されている
        availability_patterns = pdp_config.get("availability_patterns", [])
        
        try:
            buttons = page.locator(", ".join(size_button_selectors))
            count = await buttons.count()
        except Exception:
            return False
        
        if count == 0:
            return False
        
        mode = size_select_policy.get("mode", "first_instock")
        prefer_labels = size_select_policy.get("prefer_labels", [])
        
        clicked = False
        
        # 優先ラベルでクリック
        if mode == "by_label" and prefer_labels:
            prefer = [s.strip().upper() for s in prefer_labels if s.strip()]
            for i in range(count):
                try:
                    btn = buttons.nth(i)
                    if not await btn.is_visible() or await btn.is_disabled():
                        continue
                    
                    label = (await btn.get_attribute("data-size") or
                            await btn.get_attribute("aria-label") or
                            await btn.inner_text() or "").strip().upper()
                    
                    if label and any(label == pref for pref in prefer):
                        await btn.click()
                        clicked = True
                        break
                except Exception:
                    continue
        
        # 最初の在庫ありサイズをクリック
        if not clicked and mode in ("first_instock", "by_label"):
            for i in range(count):
                try:
                    btn = buttons.nth(i)
                    if not await btn.is_visible() or await btn.is_disabled():
                        continue
                    
                    if await btn.get_attribute("aria-disabled") == "true":
                        continue
                    
                    text = (await btn.inner_text() or "").lower()
                    # Stage 5: 在庫チェックテキストを site_config から取得
                    if any(pattern.lower() in text for pattern in availability_patterns):
                        continue
                    
                    await btn.click()
                    clicked = True
                    break
                except Exception:
                    continue
        
        if clicked:
            try:
                await page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                self.logger.debug("[ProductExtractor] networkidle timeout after size click.")
            
            try:
                wait_ms = size_select_policy.get("price_wait_ms", 4000)
                visible_price_selectors = pdp_config.get("visible_price_selectors", DEFAULT_PRICE_SELECTORS)
                sel = ", ".join(visible_price_selectors)
                await page.wait_for_selector(sel, state="visible", timeout=wait_ms)
            except Exception:
                self.logger.debug("[ProductExtractor] Price selector did not become visible after size click.")
            
            await page.wait_for_timeout(500)
        
        return clicked
    
    async def _extract_from_json_ld_or_meta(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """JSON-LD または Meta タグから価格情報を抽出する（フォールバック）。Stage 5: pdp_config を引数として受け取る"""
        json_ld_cfg = pdp_config.get("json_ld", {})
        meta_fallback_cfg = pdp_config.get("meta_fallback", {})
        
        # JSON-LD から抽出
        if json_ld_cfg.get("enabled", True):
            try:
                scripts = await page.query_selector_all("script[type='application/ld+json']")
                # Stage 5: paths は _get_pdp_config() で既にデフォルト値が設定されている
                paths = json_ld_cfg.get("paths", {})
                
                for script in scripts:
                    try:
                        raw = await script.inner_text()
                        data = json.loads(raw or "{}")
                    except Exception:
                        continue
                    
                    # 設定されたパスから価格と通貨を取得（デフォルト値は _get_pdp_config() で設定済み）
                    price_paths = paths.get("price", [])
                    currency_paths = paths.get("currency", [])
                    
                    def get_nested_value(obj: Any, path: str) -> Any:
                        """ネストされたパスから値を取得（例: "offers.price"）"""
                        parts = path.split(".")
                        current = obj
                        for part in parts:
                            if part.endswith("]") and "[" in part:
                                # 配列アクセス (例: "offers[0]")
                                key = part[:part.index("[")]
                                idx = int(part[part.index("[") + 1:part.index("]")])
                                current = current.get(key) if isinstance(current, dict) else None
                                if isinstance(current, list) and 0 <= idx < len(current):
                                    current = current[idx]
                                else:
                                    return None
                            else:
                                current = current.get(part) if isinstance(current, dict) else None
                            if current is None:
                                return None
                        return current
                    
                    price = None
                    currency = None
                    
                    # dict または list から抽出
                    if isinstance(data, dict):
                        for path in price_paths:
                            price = get_nested_value(data, path)
                            if price is not None:
                                break
                        for path in currency_paths:
                            currency = get_nested_value(data, path)
                            if currency is not None:
                                break
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                for path in price_paths:
                                    price = get_nested_value(item, path)
                                    if price is not None:
                                        break
                                for path in currency_paths:
                                    currency = get_nested_value(item, path)
                                    if currency is not None:
                                        break
                            if price is not None:
                                break
                    
                    if price is not None:
                        return {"price": str(price), "currency": currency}
            except Exception:
                pass
        
        # Meta タグから抽出
        if meta_fallback_cfg.get("enabled", True):
            # Stage 5: selectors は _get_pdp_config() で既にデフォルト値が設定されている
            meta_selectors = meta_fallback_cfg.get("selectors", [])
            for selector in meta_selectors:
                try:
                    node = page.locator(selector).first
                    if await node.count() == 0:
                        continue
                    content = await node.get_attribute("content")
                    if content:
                        return {"price": content.strip()}
                except Exception:
                    continue
        
        return None

