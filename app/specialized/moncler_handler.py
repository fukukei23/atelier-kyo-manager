# -*- coding: utf-8 -*-
"""
MONCLER専用のBot対策突破モジュール (DrissionPage版)

このモジュールは、MONCLER のような高度なBot対策（Cloudflare/Akamai）を持つサイトに対して、
DrissionPage という Bot 検知に強いライブラリを使用して商品情報を取得します。
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# DrissionPage のインポート（未導入時は None）
try:
    from DrissionPage import ChromiumPage
except ImportError:
    ChromiumPage = None

# 既存のモデルとコンテキスト
try:
    from app.models.result_models import DiscoveryResult
    from app.core.run_context import RunContext
except ImportError:
    # フォールバック
    DiscoveryResult = None  # type: ignore
    RunContext = None  # type: ignore

logger = logging.getLogger(__name__)


class MonclerDrissionHandler:
    """
    MONCLER サイト専用の DrissionPage ベースハンドラ
    
    このクラスは同期関数として実装されており、
    BrowserUseAgent からは asyncio.to_thread() 経由で呼び出されます。
    """
    
    def __init__(
        self,
        *,
        runtime_kwargs: dict | None = None,
        user_data_path: str | None = None,
        debug: bool = False,
    ) -> None:
        """
        初期化
        
        Args:
            runtime_kwargs: 実行時の追加パラメータ
            user_data_path: Chrome のユーザーデータディレクトリパス
                           (None の場合は "user_data/moncler_profile" を使用)
            debug: 診断モードを有効にする（HTML、スクリーンショット、JSON を保存）
        """
        self.runtime_kwargs = runtime_kwargs or {}
        self.user_data_path = user_data_path or "user_data/moncler_profile"
        self.debug = debug
        self.page: Optional[Any] = None  # ChromiumPage インスタンス
        self._diag_dir: Optional[Path] = None  # 診断用ディレクトリ
        
        if ChromiumPage is None:
            raise ImportError(
                "DrissionPage がインストールされていません。"
                "以下のコマンドでインストールしてください: pip install DrissionPage"
            )
    
    def run(
        self,
        query: str,
        site_config: dict,
        run_context: "RunContext",
        target_url: str | None = None,
    ) -> "DiscoveryResult":
        """
        同期関数として実装されたメイン処理
        
        BrowserUseAgent 側からは asyncio.to_thread(...) 経由で呼び出されます。
        
        Args:
            query: 商品名・検索キーワード
            site_config: MONCLER 用の設定（selectors, navigation など）
            run_context: 既存の実行コンテキスト（ログや保存用）
            target_url: 直接 PLP URL が指定される場合に利用
            
        Returns:
            DiscoveryResult: 取得した商品情報を含む結果オブジェクト
        """
        site = "MONCLER_OFFICIAL"
        
        # 診断モードの初期化
        if self.debug:
            base_dir = self.runtime_kwargs.get("diagnostics_dir", "artifacts/moncler_drission")
            run_id = self.runtime_kwargs.get("run_id")
            self._diag_dir = self._ensure_diag_dir(base_dir, run_id)
            logger.info(f"[MonclerDrissionHandler] 診断モード有効: {self._diag_dir}")
        
        try:
            logger.info(f"[MonclerDrissionHandler] 処理開始: query={query}, target_url={target_url}")
            
            # ブラウザ起動
            self._start_browser()
            
            # アクセスと検索
            plp_url = self._navigate_to_plp(query=query, target_url=target_url, site_config=site_config)
            
            # 商品情報の取得
            items = self._extract_products(plp_url=plp_url, site_config=site_config, max_items=5)
            
            if not items:
                logger.warning("[MonclerDrissionHandler] 商品が見つかりませんでした")
                
                # 診断モード: 失敗時のスナップショット
                if self.debug and self._diag_dir is not None and self.page is not None:
                    self._save_diag_snapshot(
                        self._diag_dir,
                        page=self.page,
                        name_prefix="error_no_items",
                        payload={
                            "error": "商品が見つかりませんでした",
                            "url": plp_url or "",
                            "query": query,
                        },
                    )
                
                return DiscoveryResult(
                    ok=False,
                    site=site,
                    query=query,
                    message="商品が見つかりませんでした",
                    evidence={"extracted_data": [], "final_url": plp_url or ""},
                )
            
            logger.info(f"[MonclerDrissionHandler] {len(items)} 件の商品を取得しました")
            
            # 診断モード: 成功時のスナップショット
            if self.debug and self._diag_dir is not None and self.page is not None:
                self._save_diag_snapshot(
                    self._diag_dir,
                    page=self.page,
                    name_prefix="success_plp",
                    payload={
                        "products": items,
                        "url": plp_url or "",
                        "query": query,
                        "count": len(items),
                    },
                )
            
            # スクリーンショット保存（可能であれば）
            try:
                screenshot_path = self._save_screenshot(run_context)
            except Exception as e:
                logger.debug(f"[MonclerDrissionHandler] スクリーンショット保存スキップ: {e}")
                screenshot_path = None
            
            # 結果を返す
            return DiscoveryResult(
                ok=True,
                site=site,
                query=query,
                message=f"{len(items)} 件の商品を取得しました",
                evidence={
                    "extracted_data": items,
                    "final_url": plp_url or "",
                },
                screenshot=screenshot_path,
            )
            
        except Exception as e:
            logger.error(f"[MonclerDrissionHandler] エラー発生: {e}", exc_info=True)
            
            # 診断モード: エラー時のスナップショット
            if self.debug and self._diag_dir is not None and self.page is not None:
                try:
                    self._save_diag_snapshot(
                        self._diag_dir,
                        page=self.page,
                        name_prefix="error_plp",
                        payload={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "url": getattr(self.page, "url", None),
                            "query": query,
                        },
                    )
                except Exception as diag_error:
                    logger.warning(f"[MonclerDrissionHandler] 診断スナップショット保存失敗: {diag_error}")
            
            # エラー時のスクリーンショット保存
            try:
                screenshot_path = self._save_screenshot(run_context, suffix="_error")
            except Exception:
                screenshot_path = None
            
            # 失敗結果を返す
            return DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=f"エラー: {str(e)}",
                evidence={"error": str(e), "extracted_data": []},
                screenshot=screenshot_path,
            )
            
        finally:
            # ブラウザを閉じる
            self._close_browser()
    
    def _start_browser(self) -> None:
        """ブラウザを起動"""
        if self.page is not None:
            return
        
        logger.info(f"[MonclerDrissionHandler] ブラウザ起動中 (user_data_path={self.user_data_path})")
        
        # ユーザーデータディレクトリの作成
        user_data_dir = Path(self.user_data_path)
        user_data_dir.mkdir(parents=True, exist_ok=True)
        
        # ChromiumPage を初期化
        # DrissionPage では、自動操作フラグの抑制が自動的に行われる
        self.page = ChromiumPage(
            user_data_path=str(user_data_dir),
            # headless=False にするとブラウザが表示される（デバッグ用）
            # headless=True にすると非表示モード
            headless=self.runtime_kwargs.get("headless", True),
        )
        
        logger.info("[MonclerDrissionHandler] ブラウザ起動完了")
    
    def _navigate_to_plp(
        self,
        query: str,
        target_url: str | None,
        site_config: dict,
    ) -> str:
        """
        PLP (Product Listing Page) に遷移
        
        Args:
            query: 検索キーワード
            target_url: 直接指定された PLP URL
            site_config: サイト設定
            
        Returns:
            str: 最終的な PLP URL
        """
        base_url = site_config.get("base_url", "https://www.moncler.com")
        
        if target_url:
            # target_url が指定されている場合は、その URL に直接アクセス
            logger.info(f"[MonclerDrissionHandler] 直接 URL にアクセス: {target_url}")
            self.page.get(target_url)
            return target_url
        
        # target_url が指定されていない場合は、検索を行う
        logger.info(f"[MonclerDrissionHandler] 検索を実行: {query}")
        
        # MONCLER トップページにアクセス
        top_url = f"{base_url}/en-int"
        self.page.get(top_url)
        time.sleep(2)  # ページ読み込み待機
        
        # ロケール選択やクッキー同意を処理（必要に応じて）
        self._handle_cookies_and_locale()
        
        # 検索窓を探して検索を実行
        search_input_selector = self._get_search_selector(site_config)
        if search_input_selector:
            try:
                search_input = self.page.ele(search_input_selector, timeout=5)
                if search_input:
                    search_input.input(query)
                    time.sleep(1)
                    
                    # 検索ボタンをクリックまたは Enter キーを押す
                    submit_selector = site_config.get("navigation", {}).get("header_search", {}).get("submit_selector")
                    if submit_selector:
                        submit_btn = self.page.ele(submit_selector, timeout=3)
                        if submit_btn:
                            submit_btn.click()
                        else:
                            # Enter キーで送信
                            search_input.input("\n")
                    else:
                        # Enter キーで送信
                        search_input.input("\n")
                    
                    time.sleep(3)  # 検索結果の読み込み待機
                    
                    current_url = self.page.url
                    logger.info(f"[MonclerDrissionHandler] 検索結果ページ: {current_url}")
                    return current_url
                    
            except Exception as e:
                logger.warning(f"[MonclerDrissionHandler] 検索入力に失敗: {e}")
        
        # 検索に失敗した場合は、検索URLを直接構築
        logger.info("[MonclerDrissionHandler] 検索URLを直接構築")
        search_url = f"{base_url}/en-int/search?q={query}"
        self.page.get(search_url)
        time.sleep(3)
        
        return self.page.url
    
    def _handle_cookies_and_locale(self) -> None:
        """クッキー同意とロケール選択を処理"""
        try:
            # クッキーバナーの処理
            cookie_selectors = [
                "button#onetrust-accept-btn-handler",
                "button[data-test='accept-all']",
                ".cookie-accept button",
            ]
            
            for selector in cookie_selectors:
                try:
                    cookie_btn = self.page.ele(selector, timeout=2)
                    if cookie_btn:
                        cookie_btn.click()
                        time.sleep(1)
                        logger.debug("[MonclerDrissionHandler] クッキー同意ボタンをクリック")
                        break
                except Exception:
                    continue
            
            # ロケール/地理的モーダルの処理
            geo_selectors = [
                "div[data-test='geo-modal'] button[data-test='close']",
                ".geo-selector button.close",
            ]
            
            for selector in geo_selectors:
                try:
                    geo_btn = self.page.ele(selector, timeout=2)
                    if geo_btn:
                        geo_btn.click()
                        time.sleep(1)
                        logger.debug("[MonclerDrissionHandler] 地理的モーダルを閉じた")
                        break
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"[MonclerDrissionHandler] クッキー/ロケール処理スキップ: {e}")
    
    def _get_search_selector(self, site_config: dict) -> Optional[str]:
        """検索入力欄のセレクタを取得"""
        nav_config = site_config.get("navigation", {})
        header_search = nav_config.get("header_search", {})
        search_selector = header_search.get("search_input_selector")
        
        if search_selector:
            # 複数のセレクタがカンマ区切りで指定されている場合は最初のものを使用
            return search_selector.split(",")[0].strip()
        
        # デフォルトのセレクタ
        return "input[name='q'], input[type='search'], input[data-test='search-input']"
    
    def _extract_products(
        self,
        plp_url: str,
        site_config: dict,
        max_items: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        PLP から商品情報を抽出
        
        Args:
            plp_url: PLP の URL
            site_config: サイト設定
            max_items: 取得する最大商品数
            
        Returns:
            List[Dict[str, Any]]: 商品情報のリスト
        """
        items: List[Dict[str, Any]] = []
        
        try:
            # セレクタを取得
            plp_config = site_config.get("selectors", {}).get("plp", {})
            pdp_link_selectors = plp_config.get("pdp_link_selectors", [])
            card_selectors = plp_config.get("card_selectors", [])
            
            # デフォルトのセレクタ
            if not pdp_link_selectors:
                pdp_link_selectors = [
                    "div[data-test='product-card'] a",
                    "li.product-grid__item a.product-tile__link",
                    ".product-cell a",
                    "a[href*='/products/']",
                    "a[href*='/product/']",
                ]
            
            logger.info(f"[MonclerDrissionHandler] 商品情報を抽出中 (最大 {max_items} 件)")
            
            # 商品リンクを取得
            found_links: List[str] = []
            found_elements = []
            
            for selector in pdp_link_selectors:
                try:
                    elements = self.page.eles(selector)
                    for elem in elements:
                        href = elem.attr("href")
                        if href:
                            # 相対URLを絶対URLに変換
                            if href.startswith("/"):
                                base_url = site_config.get("base_url", "https://www.moncler.com")
                                full_url = f"{base_url}{href}"
                            elif href.startswith("http"):
                                full_url = href
                            else:
                                continue
                            
                            # 重複チェック
                            if full_url not in found_links:
                                found_links.append(full_url)
                                found_elements.append(elem)
                                
                                if len(found_links) >= max_items:
                                    break
                    
                    if len(found_links) >= max_items:
                        break
                        
                except Exception as e:
                    logger.debug(f"[MonclerDrissionHandler] セレクタ '{selector}' でエラー: {e}")
                    continue
            
            logger.info(f"[MonclerDrissionHandler] {len(found_links)} 個の商品リンクを発見")
            
            # 各商品から情報を抽出
            for i, (link, elem) in enumerate(zip(found_links[:max_items], found_elements[:max_items])):
                try:
                    item = self._extract_product_info(elem, link, site_config)
                    if item:
                        items.append(item)
                        logger.debug(f"[MonclerDrissionHandler] 商品 {i+1}: {item.get('title', 'N/A')}")
                except Exception as e:
                    logger.warning(f"[MonclerDrissionHandler] 商品情報抽出エラー (リンク {i+1}): {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[MonclerDrissionHandler] 商品抽出エラー: {e}", exc_info=True)
        
        return items
    
    def _extract_product_info(
        self,
        element: Any,
        url: str,
        site_config: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        1つの商品要素から情報を抽出
        
        Args:
            element: DrissionPage の要素オブジェクト
            url: 商品URL
            site_config: サイト設定
            
        Returns:
            Dict[str, Any]: 商品情報
        """
        try:
            # タイトルを取得
            title = None
            title_selectors = ["h2", "h3", "[data-test='product-title']", ".product-title"]
            for selector in title_selectors:
                try:
                    title_elem = element.ele(selector, timeout=1)
                    if title_elem:
                        title = title_elem.text.strip()
                        break
                except Exception:
                    continue
            
            # タイトルが見つからない場合は、alt 属性から取得
            if not title:
                try:
                    img = element.ele("img", timeout=1)
                    if img:
                        title = img.attr("alt") or ""
                except Exception:
                    pass
            
            # 価格を取得
            price = None
            price_selectors = [
                "[data-test='price']",
                ".product-price",
                ".price",
                "[class*='price']",
            ]
            
            for selector in price_selectors:
                try:
                    price_elem = element.ele(selector, timeout=1)
                    if price_elem:
                        price_text = price_elem.text.strip()
                        # 価格テキストから数値と通貨を抽出
                        price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(",", ""))
                        if price_match:
                            price = price_text.strip()
                            break
                except Exception:
                    continue
            
            # 画像URLを取得
            image_url = None
            try:
                img = element.ele("img", timeout=1)
                if img:
                    image_url = img.attr("src") or img.attr("data-src")
                    if image_url and image_url.startswith("//"):
                        image_url = f"https:{image_url}"
            except Exception:
                pass
            
            # 最小限の情報があれば返す
            if title or price or url:
                return {
                    "title": title or "N/A",
                    "price": price or "N/A",
                    "url": url,
                    "image": image_url,
                }
            
        except Exception as e:
            logger.debug(f"[MonclerDrissionHandler] 商品情報抽出エラー: {e}")
        
        return None
    
    def _save_screenshot(self, run_context: "RunContext", suffix: str = "") -> Optional[str]:
        """
        スクリーンショットを保存
        
        Args:
            run_context: RunContext インスタンス
            suffix: ファイル名のサフィックス
            
        Returns:
            Optional[str]: 保存されたファイルパス（失敗時は None）
        """
        if self.page is None:
            return None
        
        try:
            screenshot_path = f"screenshot_drission{suffix}.png"
            full_path = Path(run_context.run_path) / screenshot_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # DrissionPage でスクリーンショットを取得
            self.page.get_screenshot(path=str(full_path))
            
            logger.debug(f"[MonclerDrissionHandler] スクリーンショット保存: {full_path}")
            return str(full_path)
            
        except Exception as e:
            logger.debug(f"[MonclerDrissionHandler] スクリーンショット保存失敗: {e}")
            return None
    
    def _close_browser(self) -> None:
        """ブラウザを閉じる"""
        if self.page is not None:
            try:
                self.page.close()
                logger.info("[MonclerDrissionHandler] ブラウザを閉じました")
            except Exception as e:
                logger.warning(f"[MonclerDrissionHandler] ブラウザクローズエラー: {e}")
            finally:
                self.page = None
    
    def _ensure_diag_dir(self, base_dir: str | Path, run_id: str | None = None) -> Path:
        """
        診断用ディレクトリを確保
        
        Args:
            base_dir: ベースディレクトリ
            run_id: 実行ID（None の場合はタイムスタンプを使用）
            
        Returns:
            Path: 診断用ディレクトリのパス
        """
        base = Path(base_dir)
        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S")
        out_dir = base / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    
    def _save_diag_snapshot(
        self,
        out_dir: Path,
        *,
        page: "ChromiumPage",
        name_prefix: str,
        payload: dict | None = None,
    ) -> None:
        """
        診断用に HTML, PNG, JSON をまとめて保存する
        
        Args:
            out_dir: 保存先ディレクトリ
            page: ChromiumPage インスタンス
            name_prefix: ファイル名のプレフィックス（例: "success_plp", "error_plp"）
            payload: 保存する JSON データ
        """
        # HTML
        html_path = out_dir / f"{name_prefix}.html"
        try:
            html = page.html
            html_path.write_text(html, encoding="utf-8")
            logger.debug(f"[MonclerDrissionHandler] HTML保存: {html_path}")
        except Exception as e:
            logger.warning(f"[MonclerDrissionHandler] HTML保存失敗: {e}")
        
        # Screenshot
        try:
            png_path = out_dir / f"{name_prefix}.png"
            # DrissionPage の screenshot API
            page.get_screenshot(path=str(png_path))
            logger.debug(f"[MonclerDrissionHandler] スクリーンショット保存: {png_path}")
        except Exception as e:
            logger.warning(f"[MonclerDrissionHandler] スクリーンショット保存失敗: {e}")
        
        # JSON payload
        if payload is not None:
            json_path = out_dir / f"{name_prefix}.json"
            try:
                json_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                logger.debug(f"[MonclerDrissionHandler] JSON保存: {json_path}")
            except Exception as e:
                logger.warning(f"[MonclerDrissionHandler] JSON保存失敗: {e}")

