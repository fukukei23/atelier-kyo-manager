"""
BUYMA カタログマネージャ — ブラウザ自動操作によるカタログ画像収集

Playwright sync API によるスクレイピング + CatalogStorage への保存委譲。
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from app.utils.buyma_catalog_storage import CONFIG, CatalogStorage


class BUYMACatalogManager:
    def __init__(self):
        self.pw = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.request_count = 0
        self.error_count = 0
        self.stop_flag = False
        self.storage = CatalogStorage()

    def _init_driver(self):
        profile_dir = Path(CONFIG["profile_path"])
        profile_dir.mkdir(parents=True, exist_ok=True)

        self.pw = sync_playwright().start()
        self.context = self.pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            args=["--disable-blink-features=AutomationControlled"],
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['ja-JP', 'ja', 'en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()

    def _human_like_delay(self):
        if self._check_response_time():
            delay = random.uniform(*CONFIG["safety"]["request_interval"]) * 1.5
        else:
            delay = random.uniform(*CONFIG["safety"]["request_interval"])
        time.sleep(delay)

    def _check_response_time(self):
        try:
            nav_start = self.page.evaluate("return window.performance.timing.navigationStart")
            resp_start = self.page.evaluate("return window.performance.timing.responseStart")
            return (resp_start - nav_start) / 1000 > CONFIG["safety"]["response_time_threshold"]
        except Exception:
            return False

    def _force_close_modals(self):
        try:
            close_selectors = [
                ".catalogs-modal__close",
                ".modal-close",
            ]
            for selector in close_selectors:
                try:
                    elements = self.page.query_selector_all(selector)
                    for element in elements:
                        if element.is_visible() and element.is_enabled():
                            element.click()
                            self._human_like_delay()
                            return True
                except Exception:
                    continue
            self.page.evaluate("""
                document.querySelectorAll('.catalogs-modal-table, .modal, .modal-backdrop').forEach(e => e.remove());
            """)
            return True
        except Exception as e:
            print(f"モーダル閉じエラー: {str(e)[:30]}")
            return False

    def get_popular_brands(self, limit=50):
        return {
            203: "GUCCI",
            290: "PRADA",
            142: "CHANEL",
            180: "HERMES",
            195: "LOUIS VUITTON",
            215: "BOTTEGA VENETA",
            183: "CELINE",
            186: "BALENCIAGA",
            164: "SAINT LAURENT",
            202: "DIOR",
            147: "FENDI",
            167: "VALENCIAGA",
            144: "COACH",
            155: "BURBERRY",
            172: "MONCLER",
            149: "MARC JACOBS",
            222: "MARNI",
            146: "CHLOE",
            209: "TORY BURCH",
            176: "JIMMY CHOO",
            191: "MICHAEL KORS",
            214: "VERSACE",
            141: "BVLGARI",
            227: "ROBINMAY",
            204: "LONGCHAMP",
            225: "GIANNI CHIARINI",
            228: "MM6 MAISON MARGIELA",
            148: "MAISON MARGIELA",
            150: "HAY",
            299: "NIKE",
            300: "adidas",
            301: "PUMA",
        }

    def process_catalog(self, row, brand_name):
        try:
            self._force_close_modals()
            catalog_id_el = row.query_selector("span.catalogs-table__contents-id")
            catalog_id = catalog_id_el.inner_text().strip() if catalog_id_el else ""

            image_cell = self.page.wait_for_selector(
                ".catalogs-table__image-item > .catalogs-table__image", timeout=10000
            )
            image_cell.hover()
            time.sleep(0.5)
            image_cell.click()

            self.page.wait_for_selector(".catalogs-modal-table", state="visible", timeout=15000)

            download_link = self.page.wait_for_selector(
                "a.catalogs-modal-table__link", state="visible", timeout=15000
            )
            download_url = download_link.get_attribute("href")

            cookies = self.context.cookies()
            user_agent = self.page.evaluate("return navigator.userAgent;")
            success, record = self.storage.download_file(
                download_url, brand_name, catalog_id,
                cookies=cookies, referer=self.page.url, user_agent=user_agent,
            )
            if success:
                print(f"成功: {brand_name} {catalog_id} (画像{record['image_count']}枚)")
            else:
                print(f"スキップ: {brand_name} {catalog_id}")

            self._force_close_modals()
            return True

        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.page.screenshot(path=os.path.join(CONFIG["screenshot_dir"], f"error_{timestamp}.png"))
            with open(os.path.join(CONFIG["screenshot_dir"], f"error_{timestamp}.html"), "w", encoding="utf-8") as f:
                f.write(self.page.content())
            print(f"エラー発生: {str(e)[:100]}")
            return False

    def process_pagination(self, base_url, brand_name):
        page_num = 1
        while not self.stop_flag:
            try:
                self.page.goto(
                    f"{base_url}&page={page_num}" if "?brand_id=" in base_url else f"{base_url}?page={page_num}"
                )
                self.page.wait_for_selector("tr.catalogs-table__row", timeout=20000)

                rows = self.page.query_selector_all("tr.catalogs-table__row")
                if not rows:
                    break

                for row in rows:
                    if self.stop_flag:
                        return
                    self.process_catalog(row, brand_name)
                    self._human_like_delay()

                try:
                    next_btn = self.page.wait_for_selector(
                        "a.pagination__next:not([disabled])", state="visible", timeout=10000
                    )
                    page_num += 1
                    self._human_like_delay()
                except Exception:
                    print(f"{brand_name} の最終ページに到達")
                    break
            except Exception as e:
                print(f"ページ処理エラー（{brand_name}）: {str(e)}")
                self.page.screenshot(
                    path=os.path.join(CONFIG["screenshot_dir"], f"pagination_error_{brand_name}.png")
                )
                break

    def main_flow(self):
        try:
            if self.storage.worksheet:
                try:
                    headers = [
                        "Brand", "Catalog_ID", "ZIP_Path", "Extracted_Dir",
                        "Download_Date", "Image_Count", "File_Size_Bytes",
                        "First_Image_Path", "All_Image_Paths", "Status",
                    ]
                    self.storage.worksheet.clear()
                    self.storage.worksheet.append_row(headers)
                    print("Googleスプレッドシートヘッダー設定完了")
                except Exception as e:
                    print(f"ヘッダー設定エラー: {e}")

            self._init_driver()
            self.page.goto("https://www.buyma.com/login/")
            input("手動ログイン後、Enterを押してください...\n（途中で止めたい場合はCtrl+C）")

            popular_brands = self.get_popular_brands(30)
            print(f"処理対象ブランド: {list(popular_brands.values())}")

            for brand_id, brand_name in popular_brands.items():
                if self._safety_check():
                    break

                print(f"\n{brand_name}の処理を開始します...")
                catalog_url = f"https://www.buyma.com/my/sell/catalogs?brand_id={brand_id}"
                self.process_pagination(catalog_url, brand_name)

        except KeyboardInterrupt:
            print("\nユーザー要求により停止しました")
            self.stop_flag = True
        except Exception as e:
            print(f"致命的エラー: {str(e)}")
        finally:
            self.cleanup()

    def _safety_check(self):
        self.request_count += 1
        if self.request_count >= CONFIG["safety"]["max_daily_requests"]:
            print("1日のリクエスト上限に達しました")
            return True
        if self.error_count >= CONFIG["safety"]["error_threshold"]:
            print("エラーが多発したため停止します")
            return True
        return False

    def cleanup(self):
        self.storage.save_csv_summary()
        for resource in [self.context, self.browser]:
            if resource:
                try:
                    resource.close()
                except Exception:
                    pass
        if self.pw:
            try:
                self.pw.stop()
            except Exception:
                pass
        self.page = None
        self.context = None
        self.browser = None
        self.pw = None
        print("リソースを解放しました")


# --- 実行部分 ---
if __name__ == "__main__":
    print("BUYMA画像自動収集ツール（SDカードD:ドライブ対応版）")
    print("必要な設定:")
    print("1. SDカードがD:ドライブとして認識されていること")
    print("2. credentials.jsonがD:ルートにあること")
    print("3. スプレッドシートの共有設定（サービスアカウント追加）")
    print("-" * 50)

    manager = BUYMACatalogManager()
    manager.main_flow()
