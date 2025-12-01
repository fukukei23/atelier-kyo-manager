# Stage 5 Task C: ProductExtractor リファクタリング方針と具体案

## 1. リファクタリング方針の整理

### 1.1 責務の分離

#### ProductExtractor の責務
- `site_config`（`selectors.pdp.*`, `price_rules` 等）を読み込む
- Page/DOM から値を取得 → ProductInfo に詰める
- 各抽出メソッドは `site_config` にキーが無い場合は graceful に None / [] を返す（例外は基本的に投げない）

#### site_config 側の責務
- 「どのセレクタを使うか」を定義
- 「価格文字列をどうパースするか（strip_chars, separator, 小数点など）」を定義
- 「JSON-LD のパス」「Meta タグのセレクタ」を定義

#### BrowserExtractionService の責務
- RunContext / Telemetry / HTML 保存パスの指定
- ProductExtractor の呼び出しと dict 変換
- 既存の Moncler 専用抽出やフォールバックロジックの維持

### 1.2 設計原則

1. **Graceful Degradation**: セレクタが見つからない場合は None / [] を返し、例外を投げない
2. **後方互換性**: 既存の `selectors.pdp.*` スキーマと互換性を保つ
3. **設定の優先順位**: 新スキーマ > 旧スキーマ > デフォルト値
4. **拡張性**: 将来的に新しいフィールドや抽出ロジックを追加しやすい構造

## 2. 実際のコード変更案

### 2.1 ProductExtractor クラスの構造変更

```python
class ProductExtractor:
    def __init__(
        self,
        site_config: Dict[str, Any],
        run_context: Optional["RunContext"] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.site_config = site_config
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
        
        # Stage 5: config getter を追加
        self._pdp_config = None  # キャッシュ用
        self._price_rules = None  # キャッシュ用
    
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
            "images": pdp_cfg.get("images") or [
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
            "availability": pdp_cfg.get("availability") or [],
            "breadcrumbs": pdp_cfg.get("breadcrumbs") or [],
            "size_button": pdp_cfg.get("size_button") or DEFAULT_SIZE_BUTTON_SELECTORS,
            "size_select_policy": pdp_cfg.get("size_select_policy") or {
                "mode": "off",
                "prefer_labels": [],
                "price_wait_ms": 4000,
            },
            "visible_price_selectors": pdp_cfg.get("visible_price_selectors") or DEFAULT_PRICE_SELECTORS,
            "image_attr": pdp_cfg.get("image_attr", "src"),
            "image_base_url": pdp_cfg.get("image_base_url"),
            "raw_html_capture": pdp_cfg.get("raw_html_capture", {
                "enabled": True,
                "filename": "pdp_raw.html",
            }),
            "json_ld": pdp_cfg.get("json_ld", {
                "enabled": True,
                "paths": {
                    "price": ["offers.price", "offers[0].price"],
                    "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
                },
            }),
            "meta_fallback": pdp_cfg.get("meta_fallback", {
                "enabled": True,
                "selectors": [
                    "meta[property='og:price:amount']",
                    "meta[name='twitter:data1']",
                ],
            }),
        }
        
        return self._pdp_config
    
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
                    price_float = self._normalize_price_to_float(price_text, price_rules)
                    if price_float is not None:
                        self.logger.debug(f"[ProductExtractor] Price found via: {selector} = {price_float}")
                        return price_float
            except Exception:
                continue
        
        return None
    
    def _normalize_price_to_float(
        self,
        price_text: str,
        price_rules: Dict[str, Any],
    ) -> Optional[float]:
        """
        価格テキストを正規化して float に変換する（Stage 5: price_rules を引数として受け取る）
        """
        if not price_text:
            return None
        
        try:
            # strip_chars を削除
            normalized = price_text
            for char in price_rules.get("strip_chars", []):
                normalized = normalized.replace(char, "")
            
            # 正規表現パターンで数値部分を抽出
            pattern = price_rules.get("price_pattern", r"[\d.,]+")
            match = re.search(pattern, normalized)
            if not match:
                return None
            
            normalized_str = match.group(0)
            
            # thousands_separator を削除
            thousands_sep = price_rules.get("thousands_separator", ",")
            if thousands_sep:
                normalized_str = normalized_str.replace(thousands_sep, "")
            
            # decimal_separator を "." に統一
            decimal_sep = price_rules.get("decimal_separator", ".")
            if decimal_sep != ".":
                normalized_str = normalized_str.replace(decimal_sep, ".")
            
            # float に変換
            price_float = float(normalized_str)
            return price_float
        except (ValueError, TypeError) as e:
            self.logger.warning(f"[ProductExtractor] Failed to parse price '{price_text}': {e}")
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
    
    async def _extract_images(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
    ) -> List[str]:
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
    
    async def _extract_colors(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
    ) -> List[str]:
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
    
    async def _extract_sizes(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
    ) -> List[str]:
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
    
    async def _extract_description(
        self,
        page: Page,
        pdp_config: Dict[str, Any],
    ) -> Optional[str]:
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
    
    async def extract(
        self,
        page: Page,
        *,
        context: Optional[BrowserContext] = None,
        prepare_page: Optional[Any] = None,
    ) -> ProductInfo:
        """
        PDP から商品情報を抽出する（Stage 5: config getter を使用）
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
        product_info.price = await self._extract_price_with_size_option(
            page, pdp_config, price_rules
        )
        
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
        list_price, discount_pct = await self._extract_list_price_and_discount(
            page, pdp_config, price_rules
        )
        product_info.list_price = list_price
        product_info.discount_pct = discount_pct
        
        # 10. フォールバック: JSON-LD / Meta タグ
        if product_info.price is None:
            fallback_data = await self._extract_from_json_ld_or_meta(
                page, pdp_config
            )
            if fallback_data:
                if product_info.price is None and fallback_data.get("price"):
                    price_str = str(fallback_data["price"])
                    product_info.price = self._normalize_price_to_float(
                        price_str, price_rules
                    )
                if not product_info.currency and fallback_data.get("currency"):
                    product_info.currency = fallback_data["currency"]
        
        # 11. メタデータを収集
        product_info.metadata = self._build_metadata(product_info, pdp_config)
        
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
```

### 2.2 BrowserExtractionService 側の調整案

```python
# app/agents/browser/extractor.py の _extract_from_pdp() メソッド

async def _extract_from_pdp(
    self,
    *,
    page: Page,
    url: str,
    context: Optional[BrowserContext],
    site: str,
    settings: Dict[str, Any],
    site_config: Dict[str, Any],
    timeout_override: Optional[int] = None,
    prepare_page: PreparePageCallable = None,
    run_context: Optional[RunContext] = None,
) -> Optional[Dict[str, Any]]:
    """
    Task D: ProductExtractor を使用して PDP から商品情報を抽出する。
    既存の Moncler 専用抽出やフォールバックロジックも維持。
    """
    goto_timeout = timeout_override or int(settings.get("timeout_sec", 60)) * 1000
    if page.url != url:
        await page.goto(url=url, wait_until="domcontentloaded", timeout=goto_timeout)

    # Task D: ProductExtractor を使用
    try:
        product_extractor = ProductExtractor(
            site_config=site_config,
            run_context=run_context,
            logger=self.logger,
        )
        product_info = await product_extractor.extract(
            page=page,
            context=context,
            prepare_page=prepare_page,
        )
        
        # Stage 5: ProductInfo を Dict に変換（すべてのフィールドを含める）
        data = {
            "title": product_info.title,
            "price": product_info.price,  # float
            "currency": product_info.currency,
            "url": product_info.url or page.url,
            "images": product_info.images,
            "sizes": product_info.sizes,
            "colors": product_info.colors,
            "description": product_info.description,
            "brand": product_info.brand,
            "list_price": product_info.list_price,  # float
            "discount_pct": product_info.discount_pct,
            "raw_html_path": product_info.raw_html_path,  # Stage 5: HTML パス
            "metadata": product_info.metadata,  # Stage 5: metadata
        }
        
        # price が None でも返す（Stage 5: graceful degradation）
        self.logger.debug(f"[Extractor] ProductExtractor succeeded for {url}")
        return data
    except Exception as pe_e:
        self.logger.warning(f"[Extractor] ProductExtractor failed, falling back to legacy: {pe_e}")

    # フォールバック: 既存の Moncler 専用抽出
    if site.upper() == "MONCLER_OFFICIAL":
        enriched = await self.moncler_extractor.extract(page=page, context=context)
        if enriched:
            return enriched

    # フォールバック: 既存の価格抽出ロジック
    # ... (既存のコード)
```

## 3. 変更のまとめ

### 主な変更点

1. **Config Getter の追加**
   - `_get_pdp_config()`: PDP 設定を取得（新スキーマ優先、旧スキーマフォールバック）
   - `_get_price_rules()`: 価格正規化ルールを取得（新スキーマ優先、旧スキーマフォールバック）

2. **抽出メソッドのシグネチャ変更**
   - すべての `_extract_*()` メソッドが `pdp_config` を引数として受け取る
   - `_extract_price()`, `_extract_list_price_and_discount()` が `price_rules` も受け取る

3. **価格正規化の改善**
   - `_normalize_price_to_float()` が `price_rules` を引数として受け取る
   - 正規表現パターンが設定可能に

4. **画像 URL 正規化の改善**
   - `_normalize_image_url()` メソッドを追加
   - `image_base_url` 設定に対応

5. **JSON-LD / Meta タグフォールバックの改善**
   - `_extract_from_json_ld_or_meta()` が `pdp_config` を引数として受け取る
   - JSON-LD のパスが設定可能に

6. **メタデータ収集の改善**
   - `_build_metadata()` メソッドを追加（設定可能なメタデータキーに対応）

### 後方互換性

- 既存の `selectors.pdp.*` スキーマは引き続き動作
- トップレベルの `price_rules` も引き続き動作
- デフォルトセレクタは引き続き使用される（site_config に定義がない場合）

