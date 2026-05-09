# ======================================================================
# ファイル名: shipping_agent.py
# 場所: app/utils/shipping_agent.py
# 役割: Buyandship(転送倉庫サービス)にログインして倉庫一覧を取得するエージェント。
# - 永続コンテキストで「一度ログインしたら次回以降は自動ログイン不要」
# - ログインUIは「日本語プレースホルダ/ラベル/ボタン文言」優先で堅牢化
# - セレクタ配列の段階フォールバック + パネル起動 + Cookie同意対応
# - iframeフォールバック + デバッグ用スクショ/HTML保存
# - 起動タブ選定/空白タブ除去/URLオープン再試行で about:blank 問題を回避
# 更新日: 2025-08-25
#
# --- 操作するソフト/前提 ---
# - Python 3.10 以上
# - 任意のエディタ/IDE（VS Code 推奨）
#
# --- 依存ライブラリのインストール（ターミナル） ---
# pip install "playwright==1.42.0" python-dotenv
# playwright install
#
# --- 環境変数（.env をプロジェクトルートに作成） ---
# BUYANDSHIP_EMAIL="your_email@example.com"
# BUYANDSHIP_PASSWORD="your_password"
#
# --- 初回のみ（推奨: 手動ログインでセッション保存）---
# [Windows PowerShell]
# $env:PLAYWRIGHT_HEADFUL="1"
# python -m app.utils.shipping_agent
#
# [macOS/Linux]
# PLAYWRIGHT_HEADFUL=1 python -m app.utils.shipping_agent
#
# - ブラウザが起動したら、Buyandshipにログイン（CAPTCHA/2FA対応可）。
# - 以降はヘッドレスで自動的にログイン状態が維持されます（永続プロファイル）。
#
# --- API/確認 ---
# - 例: /api/warehouses?country=HK
# ログに「メールアドレス → 次へ進む → パスワード → ログイン」の順で進行が出ればOK。
#
# --- デバッグ ---
# - 自動ログインに失敗した場合、instance/login_debug.png と instance/login_debug.html を保存。
# - 動作をゆっくりに: PLAYWRIGHT_SLOWMO=500 等を環境変数で指定。
# - ステップ実行: 一時的に $env:PWDEBUG="1" で実行 or page.pause() を挿入。
# ======================================================================
from __future__ import annotations

import os
import asyncio
import logging
import re
from typing import List, Dict, Optional

from playwright.async_api import async_playwright, Page, BrowserContext
from flask import current_app

# --- 定数定義 ---
WAREHOUSE_LIST_URL = "https://www.buyandship.co.jp/account/v2020/warehouse/"
# 永続プロファイル（セッション情報）の保存先
USER_DATA_DIR = os.path.abspath("instance/pw_profile")

# --- 追加ヘルパ（UIテキスト優先で操作。候補セレクタ配列を順に試す） ---
async def _click_first(page_or_frame, selectors: list[str], timeout_ms: int = 8000) -> bool:
    """指定されたセレクタリストに一致する最初の可視要素をクリックする"""
    for sel in selectors:
        try:
            loc = page_or_frame.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                await loc.click(timeout=timeout_ms)
                return True
        except Exception:
            pass
    return False

async def _fill_first(page_or_frame, selectors: list[str], text: str, timeout_ms: int = 8000) -> bool:
    """指定されたセレクタリストに一致する最初の可視要素にテキストを入力する"""
    for sel in selectors:
        try:
            loc = page_or_frame.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                await loc.fill(text, timeout=timeout_ms)
                return True
        except Exception:
            pass
    return False

class ShippingAgent:
    """転送倉庫の情報を取得するエージェント（Playwright, Async, 永続セッション対応, UIテキスト優先ログイン）"""

    def __init__(
        self,
        headless: Optional[bool] = None,  # None の場合は環境変数 PLAYWRIGHT_HEADFUL で制御
        state_path: str = "instance/playwright_state.json",  # デバッグ/互換用（現行は永続プロファイル優先）
        base_url: str = WAREHOUSE_LIST_URL,
        timeout_ms: int = 60_000,
    ) -> None:
        # headless の決定: 引数 > 環境変数 > デフォルト(True)
        if headless is None:
            self.headless = os.getenv("PLAYWRIGHT_HEADFUL") != "1"
        else:
            self.headless = headless

        self.state_path = state_path
        self.base_url = base_url
        self.timeout_ms = timeout_ms
        self.logger = logging.getLogger(__name__)
        self.email: Optional[str] = None
        self.password: Optional[str] = None

        # 保存先ディレクトリ作成
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        os.makedirs(USER_DATA_DIR, exist_ok=True)

    # ----------------------------
    # 資格情報の取得（環境変数優先）
    # ----------------------------
    def _get_credentials(self) -> Dict[str, str]:
        """
        優先度:
          1) env: BUYANDSHIP_EMAIL / BUYANDSHIP_PASSWORD
          2) Flask config: BUYANDSHIP_EMAIL / BUYANDSHIP_PASSWORD
          3) 後方互換 env/config: BNS_EMAIL / BNS_PASSWORD
        """
        email = os.getenv("BUYANDSHIP_EMAIL")
        password = os.getenv("BUYANDSHIP_PASSWORD")

        if not email or not password:
            cfg = getattr(current_app, "config", None)
            if cfg:
                email = email or cfg.get("BUYANDSHIP_EMAIL")
                password = password or cfg.get("BUYANDSHIP_PASSWORD")

        if not email:
            email = os.getenv("BNS_EMAIL") or (getattr(current_app, "config", {}) or {}).get("BNS_EMAIL")
            if email:
                self.logger.warning("Using legacy env BNS_EMAIL (rename to BUYANDSHIP_EMAIL).")
        if not password:
            password = os.getenv("BNS_PASSWORD") or (getattr(current_app, "config", {}) or {}).get("BNS_PASSWORD")
            if password:
                self.logger.warning("Using legacy env BNS_PASSWORD (rename to BUYANDSHIP_PASSWORD).")

        if not email or not password:
            raise RuntimeError("BUYANDSHIP_EMAIL / BUYANDSHIP_PASSWORD が見つかりません。.env に設定してください。")

        self.email, self.password = email, password
        return {"email": email, "password": password}

    # ----------------------------
    # 追加: 起動後に“実際に使うべきページ”を選ぶヘルパ
    # ----------------------------
    async def _pick_working_page(self, context: BrowserContext) -> Page:
        """
        永続プロファイル起動時に混在するページから、仕事に使うべき page を選ぶ（about:blank を避ける）。
        なければ新規作成して返す。
        """
        # 1) 既存タブのうち、about:blank 以外＆未クローズを優先
        for p in context.pages:
            try:
                if not p.is_closed() and p.url and not p.url.startswith("about:blank"):
                    await p.bring_to_front()
                    return p
            except Exception:
                continue

        # 2) それでも無ければ新規ページ
        page = await context.new_page()
        await page.bring_to_front()
        return page

    # ----------------------------
    # 追加: 不要な空タブを閉じる（任意）
    # ----------------------------
    async def _close_blank_tabs(self, context: BrowserContext) -> None:
        """不要な about:blank タブを閉じる"""
        # pagesリストのコピーに対してループすることで、安全にページを閉じることができる
        for p in list(context.pages):
            try:
                if not p.is_closed() and (not p.url or p.url.startswith("about:blank")):
                    await p.close()
            except Exception:
                pass

    # ----------------------------
    # 追加: URL を開く（作業ページ選択 + 再試行）
    # ----------------------------
    async def _open_target_url(self, context: BrowserContext, url: str) -> Page:
        """
        作業用ページを取得し、対象URLに遷移。
        まれに bring_to_front 直後に URL 未解決の空タブを掴むことがあるため再試行。
        """
        page = await self._pick_working_page(context)
        for _ in range(2): # 最大2回試行
            try:
                await page.goto(url, wait_until="load", timeout=self.timeout_ms)
                if page.url and not page.url.startswith("about:blank"):
                    return page
            except Exception:
                pass
            # ダメなら新しいタブで再試行
            self.logger.warning("ページの読み込みに失敗したため、新しいタブで再試行します。")
            page = await context.new_page()

        # 最終試行
        await page.goto(url, wait_until="load", timeout=self.timeout_ms)
        return page

    # ----------------------------
    # ログイン（必要時のみ）- UIテキスト優先 + 段階フォールバック + パネル起動 + 同意対応
    # ----------------------------
    async def _login_if_needed(self, page: Page, context: BrowserContext) -> None:
        """
        Buyandship ログインが必要なら行う（UI変化に強いセレクタで対応）
        """
        # 0) 既ログイン簡易チェック（見出し/ユーザーメニュー）
        try:
            if await page.locator("text=Buyandship倉庫住所一覧, text=倉庫住所一覧, text=倉庫住所").first.is_visible(timeout=3000):
                self.logger.info("既にログイン済みと判断しました。")
                return
        except Exception:
            pass
        try:
            if await page.locator("button:has-text('マイアカウント'), a:has-text('マイアカウント')").first.is_visible(timeout=2000):
                self.logger.info("既にログイン済み（ユーザーメニュー検出）と判断しました。")
                return
        except Exception:
            pass

        self.logger.info("ログインが必要です。UIテキスト優先のセレクタでログインを試みます。")

        # 1) Cookie 同意/ダイアログのクローズ
        await _click_first(page, [
            "button:has-text('同意')", "button:has-text('Accept')",
            "button[aria-label='Close']", "button.bs-cookie-consent-accept",
        ])

        # 2) ログインパネルの起動
        try:
            login_panel_open = await page.locator("input[placeholder='メールアドレス']").first.is_visible(timeout=2000)
        except Exception:
            login_panel_open = False

        if not login_panel_open:
            if await _click_first(page, [
                "a:has-text('ログイン')", "a:has-text('登録 / ログイン')",
                "button:has-text('ログイン')", "button:has-text('登録 / ログイン')",
            ]):
                await page.wait_for_timeout(500)

        # 3) メール入力
        creds = self._get_credentials()
        EMAIL_SELECTORS = [
            "input[placeholder='メールアドレス']", "input[aria-label='メールアドレス']",
            "input[type='email']", "input[name='email']", "#email, #user_email, input[id*='email']",
        ]
        await page.wait_for_timeout(300)
        if not await _fill_first(page, EMAIL_SELECTORS, creds["email"], timeout_ms=12000):
            filled_in_frame = any(await _fill_first(frame, EMAIL_SELECTORS, creds["email"], timeout_ms=8000) for frame in page.frames)
            if not filled_in_frame:
                raise RuntimeError("メール入力欄が見つかりませんでした。UIが変わっている可能性があります。")

        # 4) 「次へ進む」
        await _click_first(page, [
            "button:has-text('次へ進む')", "button:has-text('次へ')",
            "button:has-text('ログイン')", "button[type='submit']",
        ])

        # 5) パスワード入力
        PASS_SELECTORS = [
            "input[placeholder*='パスワード']", "input[aria-label*='パスワード']",
            "input[type='password']", "input[name='password']", "#password, input[id*='pass']",
        ]
        await page.wait_for_timeout(600)
        if not await _fill_first(page, PASS_SELECTORS, creds["password"], timeout_ms=15000):
            try:
                filled_in_frame = any(await _fill_first(frame, PASS_SELECTORS, creds["password"], timeout_ms=8000) for frame in page.frames)
                if not filled_in_frame:
                    raise RuntimeError("パスワード入力欄が見つかりませんでした。")
            except Exception:
                try:
                    debug_dir = os.path.dirname(self.state_path)
                    await page.screenshot(path=os.path.join(debug_dir, "login_debug.png"), full_page=True)
                    html = await page.content()
                    with open(os.path.join(debug_dir, "login_debug.html"), "w", encoding="utf-8") as f:
                        f.write(html)
                    self.logger.error(f"デバッグ情報: {os.path.join(debug_dir, 'login_debug.*')} を確認してください。")
                except Exception as e:
                    self.logger.error(f"デバッグ情報の保存中にエラー: {e}")
                raise

        # 6) 送信（ログイン）
        await _click_first(page, ["button:has-text('ログイン')", "button[type='submit']", "button:has-text('次へ')"])

        # 7) 完了待ち
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(800)
        try:
            if await page.locator("text=Buyandship倉庫住所一覧, text=倉庫住所").first.is_visible():
                self.logger.info("ログイン完了を検出しました。")
            else:
                self.logger.warning("ログイン完了を検出できませんでしたが処理を続行します。")
        except Exception:
            self.logger.warning("ログイン完了を検出できませんでしたが処理を続行します。")

    # ----------------------------
    # データ抽出
    # ----------------------------
    async def _extract_warehouses(self, page: Page, country_code: str) -> List[Dict]:
        """倉庫カード要素から情報を抽出する。"""
        warehouses: List[Dict] = []
        card_sel = "div.warehouse-card, article.warehouse, li.warehouse, div[class*='warehouse-item']"
        cards = page.locator(card_sel)
        count = await cards.count()
        self.logger.info(f"'{country_code}' の倉庫カード候補要素数: {count}")
        for i in range(count):
            card = cards.nth(i)
            country_text = ""
            for sel in [".country-name", '[data-testid="country"]', ".warehouse-country"]:
                if await card.locator(sel).count() > 0:
                    country_text = (await card.locator(sel).first.inner_text()).strip()
                    break
            if country_code.upper() not in country_text.upper():
                continue
            name = (await card.locator("h3, .warehouse-name").first.inner_text()).strip()
            address = (await card.locator("address, [class*='address-detail']").first.inner_text()).strip()
            fee = None
            if await card.locator(".warehouse-fee").count() > 0:
                raw = (await card.locator(".warehouse-fee").first.inner_text()).strip()
                m = re.search(r"(\d+(?:\.\d+)?)", raw.replace(",", ""))
                if m:
                    fee = float(m.group(1))
            if name and address:
                warehouses.append({"name": name, "address": address, "country": country_code.upper(), "fee": fee})
        return warehouses

    # ----------------------------
    # メイン処理（非同期）- 永続セッション対応
    # ----------------------------
    async def _get_warehouses_async(self, country_code: str = "US") -> List[Dict]:
        """永続コンテキストを使用してブラウザを起動し、倉庫情報を取得する。"""
        self.logger.info(f"{country_code} の倉庫情報取得を開始します (Headless: {self.headless})...")
        self.logger.info(f"ユーザープロファイル: {USER_DATA_DIR}")
        async with async_playwright() as p:
            context = None
            try:
                context = await p.chromium.launch_persistent_context(
                    USER_DATA_DIR,
                    headless=self.headless,
                    slow_mo=int(os.getenv("PLAYWRIGHT_SLOWMO", "0") or 0),
                    channel="chrome",
                    args=["--restore-last-session", "--no-first-run", "--no-default-browser-check", "--disable-session-crashed-bubble"],
                )
                await self._close_blank_tabs(context)
                page = await self._open_target_url(context, self.base_url)
                page.set_default_timeout(self.timeout_ms)
                await self._login_if_needed(page, context)
                if self.base_url not in page.url:
                    self.logger.info("再度、倉庫ページへアクセスします。")
                    page = await self._open_target_url(context, self.base_url)
                data = await self._extract_warehouses(page, country_code=country_code.upper())
                if not data:
                    self.logger.warning(f"{country_code} の倉庫情報が抽出できませんでした。")
                else:
                    self.logger.info(f"{len(data)}件の倉庫情報を取得しました。")
                return data
            except Exception as e:
                self.logger.error(f"Playwright操作中にエラーが発生しました: {e}", exc_info=True)
                return []
            finally:
                if context:
                    await context.close()
                self.logger.info("ブラウザセッションを終了しました。")

    # ----------------------------
    # 同期ラッパー
    # ----------------------------
    def get_warehouses_by_country(self, country_code: str = "US") -> List[Dict]:
        """Flaskなどから同期的に呼び出すためのラッパー関数"""
        return asyncio.run(self._get_warehouses_async(country_code))

    def fetch_warehouses(self, country_code: str) -> List[Dict]:
        """get_warehouses_by_country のエイリアス"""
        return self.get_warehouses_by_country(country_code)

# --- 単体実行テスト ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info(".envファイルを読み込みました。")
    except ImportError:
        logger.warning("python-dotenvが未インストールのため、.envファイルは読み込まれません。")

    # headless=None にすると環境変数(PLAYWRIGHT_HEADFUL)で制御
    agent = ShippingAgent(headless=None)

    logger.info("--- アメリカ(US)の倉庫情報を取得 ---")
    us_result = agent.get_warehouses_by_country("US")
    logger.info(f"Result: {us_result}")

    logger.info("--- 香港(HK)の倉庫情報を取得 ---")
    hk_result = agent.get_warehouses_by_country("HK")
    logger.info(f"Result: {hk_result}")

    logger.info("--- イギリス(UK)の倉庫情報を取得 ---")
    uk_result = agent.get_warehouses_by_country("UK")
    logger.info(f"Result: {uk_result}")
