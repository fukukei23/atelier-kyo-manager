
# === atelier-kyo-manager/config.py ===
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'kenshin4416' # 本番環境では強力なキーに変更
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///site.db' # SQLiteデータベースを使用
    SQLALCHEMY_TRACK_MODIFICATIONS = False

# === atelier-kyo-manager/app\utils\buyma_catalog_manager.py ===
import os
import csv
import time
import random
import requests
import hashlib
import zipfile
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# --- 設定（SDカードD:ドライブ・スプレッドシートID設定済み）---
CONFIG = {
    'profile_path': r"C:/Users/USER/AppData/Local/Google/Chrome/SeleniumProfile",
    'base_dir': 'D:/catalog_images',
    'screenshot_dir': 'D:/screenshots',
    'csv_path': 'D:/catalog_data.csv',
    'extracted_images_dir': 'D:/extracted_images',
    'google_credentials': 'D:/credentials.json',
    'spreadsheet_id': '1z9_lczAbnbsMYpAEslamfekEMrPQVIM1rfHqNbzze_Y',
    'worksheet_name': 'catalog_data',
    'safety': {
        'max_daily_requests': 500,
        'request_interval': (5, 10),
        'error_threshold': 10,
        'response_time_threshold': 8.0
    }
}

class BUYMACatalogManager:
    def __init__(self):
        self.driver = self._init_driver()
        self.request_count = 0
        self.error_count = 0
        self.downloaded_hashes = set()
        self.downloaded_catalog_ids = set()
        self.csv_records = []
        self._setup_directories()
        self._init_google_sheets()
        self.stop_flag = False

    def _setup_directories(self):
        directories = [
            CONFIG['screenshot_dir'],
            CONFIG['base_dir'],
            CONFIG['extracted_images_dir']
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"ディレクトリ作成: {directory}")

    def _init_google_sheets(self):
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_file(
                CONFIG['google_credentials'], scopes=scope
            )
            self.gc = gspread.authorize(creds)
            self.worksheet = self.gc.open_by_key(CONFIG['spreadsheet_id']).worksheet(CONFIG['worksheet_name'])
            print("Googleスプレッドシート接続成功")
        except Exception as e:
            print(f"Googleスプレッドシート接続エラー: {e}")
            self.gc = None
            self.worksheet = None

    def _init_driver(self):
        chrome_options = Options()
        chrome_options.add_argument(f"--user-data-dir={CONFIG['profile_path']}")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--lang=ja")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        stealth(driver,
            languages=["ja-JP", "ja"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        return driver

    def _human_like_delay(self):
        if self._check_response_time():
            delay = random.uniform(*CONFIG['safety']['request_interval']) * 1.5
        else:
            delay = random.uniform(*CONFIG['safety']['request_interval'])
        time.sleep(delay)

    def _check_response_time(self):
        try:
            navigation_start = self.driver.execute_script("return window.performance.timing.navigationStart")
            response_start = self.driver.execute_script("return window.performance.timing.responseStart")
            return (response_start - navigation_start) / 1000 > CONFIG['safety']['response_time_threshold']
        except Exception:
            return False

    def _force_close_modals(self):
        try:
            close_selectors = [
                ".catalogs-modal__close",
                ".modal-close",
                "//button[contains(text(), '閉じる')]",
                "//button[contains(text(), 'キャンセル')]"
            ]
            for selector in close_selectors:
                try:
                    if selector.startswith("//"):
                        elements = self.driver.find_elements(By.XPATH, selector)
                    else:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            element.click()
                            self._human_like_delay()
                            return True
                except Exception:
                    continue
            self.driver.execute_script("""
                document.querySelectorAll('.catalogs-modal-table, .modal, .modal-backdrop').forEach(e => e.remove());
            """)
            return True
        except Exception as e:
            print(f"モーダル閉じエラー: {str(e)[:30]}")
            return False

    def _extract_images_from_zip(self, zip_path, brand_name, catalog_id):
        try:
            extract_dir = os.path.join(CONFIG['extracted_images_dir'], brand_name, catalog_id)
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            image_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        old_path = os.path.join(root, file)
                        new_filename = f"{brand_name}_{catalog_id}_{len(image_files)+1}_{file}"
                        new_path = os.path.join(extract_dir, new_filename)
                        os.rename(old_path, new_path)
                        image_files.append(new_path)

            return len(image_files), image_files
        except Exception as e:
            print(f"ZIP解凍エラー: {e}")
            return 0, []

    def _download_file(self, url, brand_name, catalog_id):
        session = requests.Session()
        for c in self.driver.get_cookies():
            session.cookies.set(c['name'], c['value'])
        headers = {
            'Referer': self.driver.current_url,
            'User-Agent': self.driver.execute_script("return navigator.userAgent;")
        }

        response = session.get(url, headers=headers)
        if response.status_code == 200:
            file_hash = hashlib.md5(response.content).hexdigest()
            if catalog_id in self.downloaded_catalog_ids or file_hash in self.downloaded_hashes:
                return False, None

            save_dir = os.path.join(CONFIG['base_dir'], brand_name, catalog_id)
            os.makedirs(save_dir, exist_ok=True)
            zip_path = os.path.join(save_dir, f'catalog_{catalog_id}.zip')

            with open(zip_path, 'wb') as f:
                f.write(response.content)

            image_count, image_files = self._extract_images_from_zip(zip_path, brand_name, catalog_id)

            record = {
                'brand': brand_name,
                'catalog_id': catalog_id,
                'zip_path': zip_path,
                'extracted_dir': os.path.join(CONFIG['extracted_images_dir'], brand_name, catalog_id),
                'download_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_count': image_count,
                'file_size': os.path.getsize(zip_path),
                'first_image_path': image_files[0] if image_files else '',
                'all_image_paths': '|'.join(image_files),
                'status': 'success'
            }

            self.csv_records.append(record)
            self.downloaded_hashes.add(file_hash)
            self.downloaded_catalog_ids.add(catalog_id)

            self._add_to_google_sheet(record)

            return True, record
        return False, None

    def _add_to_google_sheet(self, record):
        if self.worksheet:
            try:
                row_data = [
                    record['brand'],
                    record['catalog_id'],
                    record['zip_path'],
                    record['extracted_dir'],
                    record['download_date'],
                    record['image_count'],
                    record['file_size'],
                    record['first_image_path'],
                    record['all_image_paths'],
                    record['status']
                ]
                self.worksheet.append_row(row_data)
                print(f"Googleスプレッドシートに追加: {record['brand']} {record['catalog_id']}")
            except Exception as e:
                print(f"Googleスプレッドシート追加エラー: {e}")

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
            167: "VALENTINO",
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
            catalog_id = row.find_element(By.CSS_SELECTOR, 'span.catalogs-table__contents-id').text.strip()

            image_cell = WebDriverWait(row, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.catalogs-table__image-item > .catalogs-table__image'))
            )
            ActionChains(self.driver).move_to_element(image_cell).pause(0.5).click().perform()

            WebDriverWait(self.driver, 15).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".catalogs-modal-table"))
            )

            download_link = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.catalogs-modal-table__link"))
            )
            download_url = download_link.get_attribute('href')

            success, record = self._download_file(download_url, brand_name, catalog_id)
            if success:
                print(f"成功: {brand_name} {catalog_id} (画像{record['image_count']}枚)")
            else:
                print(f"スキップ: {brand_name} {catalog_id}")

            self._force_close_modals()
            return True

        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.driver.save_screenshot(os.path.join(CONFIG['screenshot_dir'], f'error_{timestamp}.png'))
            with open(os.path.join(CONFIG['screenshot_dir'], f'error_{timestamp}.html'), 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            print(f"エラー発生: {str(e)[:100]}")
            return False

    def process_pagination(self, base_url, brand_name):
        page_num = 1
        while not self.stop_flag:
            try:
                self.driver.get(f"{base_url}&page={page_num}" if "?brand_id=" in base_url else f"{base_url}?page={page_num}")
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'tr.catalogs-table__row'))
                )

                rows = self.driver.find_elements(By.CSS_SELECTOR, 'tr.catalogs-table__row')
                if not rows:
                    break

                for row in rows:
                    if self.stop_flag:
                        return
                    self.process_catalog(row, brand_name)
                    self._human_like_delay()

                try:
                    next_btn = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.pagination__next:not([disabled])'))
                    )
                    page_num += 1
                    self._human_like_delay()
                except (NoSuchElementException, TimeoutException):
                    print(f"{brand_name} の最終ページに到達")
                    break
            except Exception as e:
                print(f"ページ処理エラー（{brand_name}）: {str(e)}")
                self.driver.save_screenshot(os.path.join(CONFIG['screenshot_dir'], f'pagination_error_{brand_name}.png'))
                break

    def _save_csv_summary(self):
        if not self.csv_records:
            return

        fieldnames = [
            'brand', 'catalog_id', 'zip_path', 'extracted_dir',
            'download_date', 'image_count', 'file_size',
            'first_image_path', 'all_image_paths', 'status'  # ← ここに'all_image_paths'を追加
        ]

        with open(CONFIG['csv_path'], 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_records)

        print(f"詳細レポート保存: {CONFIG['csv_path']}")

        total_downloads = len(self.csv_records)
        total_images = sum(record['image_count'] for record in self.csv_records)
        total_size_mb = sum(record['file_size'] for record in self.csv_records) / (1024 * 1024)

        summary = f"""
=== ダウンロード完了サマリー ===
総ダウンロード数: {total_downloads}件
総画像数: {total_images}枚
総ファイルサイズ: {total_size_mb:.2f}MB
保存先: {CONFIG['base_dir']}
解凍画像: {CONFIG['extracted_images_dir']}
CSV詳細: {CONFIG['csv_path']}
Googleスプレッドシート: {'連携済み' if self.worksheet else '未接続'}
        """
        print(summary)

    def main_flow(self):
        try:
            if self.worksheet:
                try:
                    headers = ['Brand', 'Catalog_ID', 'ZIP_Path', 'Extracted_Dir',
                             'Download_Date', 'Image_Count', 'File_Size_Bytes',
                             'First_Image_Path', 'All_Image_Paths', 'Status']
                    self.worksheet.clear()
                    self.worksheet.append_row(headers)
                    print("Googleスプレッドシートヘッダー設定完了")
                except Exception as e:
                    print(f"ヘッダー設定エラー: {e}")

            self.driver.get('https://www.buyma.com/login/')
            input("手動ログイン後、Enterを押してください...\n（途中で止めたい場合はCtrl+C）")

            popular_brands = self.get_popular_brands(30)
            print(f"処理対象ブランド: {list(popular_brands.values())}")

            for brand_id, brand_name in popular_brands.items():
                if self._safety_check():
                    break

                print(f"\n{brand_name}の処理を開始します...")
                catalog_url = f'https://www.buyma.com/my/sell/catalogs?brand_id={brand_id}'
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
        if self.request_count >= CONFIG['safety']['max_daily_requests']:
            print("1日のリクエスト上限に達しました")
            return True
        if self.error_count >= CONFIG['safety']['error_threshold']:
            print("エラーが多発したため停止します")
            return True
        return False

    def cleanup(self):
        self._save_csv_summary()
        self.driver.quit()
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

# === atelier-kyo-manager/app\utils\utils\buyma_catalog_manager.py ===
import os
import csv
import time
import random
import requests
import hashlib
import zipfile
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# --- 設定（SDカードD:ドライブ・スプレッドシートID設定済み）---
CONFIG = {
    'profile_path': r"C:/Users/USER/AppData/Local/Google/Chrome/SeleniumProfile",
    'base_dir': 'D:/catalog_images',
    'screenshot_dir': 'D:/screenshots',
    'csv_path': 'D:/catalog_data.csv',
    'extracted_images_dir': 'D:/extracted_images',
    'google_credentials': 'D:/credentials.json',
    'spreadsheet_id': '1z9_lczAbnbsMYpAEslamfekEMrPQVIM1rfHqNbzze_Y',
    'worksheet_name': 'catalog_data',
    'safety': {
        'max_daily_requests': 500,
        'request_interval': (5, 10),
        'error_threshold': 10,
        'response_time_threshold': 8.0
    }
}

class BUYMACatalogManager:
    def __init__(self):
        self.driver = self._init_driver()
        self.request_count = 0
        self.error_count = 0
        self.downloaded_hashes = set()
        self.downloaded_catalog_ids = set()
        self.csv_records = []
        self._setup_directories()
        self._init_google_sheets()
        self.stop_flag = False

    def _setup_directories(self):
        directories = [
            CONFIG['screenshot_dir'],
            CONFIG['base_dir'],
            CONFIG['extracted_images_dir']
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"ディレクトリ作成: {directory}")

    def _init_google_sheets(self):
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_file(
                CONFIG['google_credentials'], scopes=scope
            )
            self.gc = gspread.authorize(creds)
            self.worksheet = self.gc.open_by_key(CONFIG['spreadsheet_id']).worksheet(CONFIG['worksheet_name'])
            print("Googleスプレッドシート接続成功")
        except Exception as e:
            print(f"Googleスプレッドシート接続エラー: {e}")
            self.gc = None
            self.worksheet = None

    def _init_driver(self):
        chrome_options = Options()
        chrome_options.add_argument(f"--user-data-dir={CONFIG['profile_path']}")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--lang=ja")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        stealth(driver,
            languages=["ja-JP", "ja"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        return driver

    def _human_like_delay(self):
        if self._check_response_time():
            delay = random.uniform(*CONFIG['safety']['request_interval']) * 1.5
        else:
            delay = random.uniform(*CONFIG['safety']['request_interval'])
        time.sleep(delay)

    def _check_response_time(self):
        try:
            navigation_start = self.driver.execute_script("return window.performance.timing.navigationStart")
            response_start = self.driver.execute_script("return window.performance.timing.responseStart")
            return (response_start - navigation_start) / 1000 > CONFIG['safety']['response_time_threshold']
        except Exception:
            return False

    def _force_close_modals(self):
        try:
            close_selectors = [
                ".catalogs-modal__close",
                ".modal-close",
                "//button[contains(text(), '閉じる')]",
                "//button[contains(text(), 'キャンセル')]"
            ]
            for selector in close_selectors:
                try:
                    if selector.startswith("//"):
                        elements = self.driver.find_elements(By.XPATH, selector)
                    else:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            element.click()
                            self._human_like_delay()
                            return True
                except Exception:
                    continue
            self.driver.execute_script("""
                document.querySelectorAll('.catalogs-modal-table, .modal, .modal-backdrop').forEach(e => e.remove());
            """)
            return True
        except Exception as e:
            print(f"モーダル閉じエラー: {str(e)[:30]}")
            return False

    def _extract_images_from_zip(self, zip_path, brand_name, catalog_id):
        try:
            extract_dir = os.path.join(CONFIG['extracted_images_dir'], brand_name, catalog_id)
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            image_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        old_path = os.path.join(root, file)
                        new_filename = f"{brand_name}_{catalog_id}_{len(image_files)+1}_{file}"
                        new_path = os.path.join(extract_dir, new_filename)
                        os.rename(old_path, new_path)
                        image_files.append(new_path)

            return len(image_files), image_files
        except Exception as e:
            print(f"ZIP解凍エラー: {e}")
            return 0, []

    def _download_file(self, url, brand_name, catalog_id):
        session = requests.Session()
        for c in self.driver.get_cookies():
            session.cookies.set(c['name'], c['value'])
        headers = {
            'Referer': self.driver.current_url,
            'User-Agent': self.driver.execute_script("return navigator.userAgent;")
        }

        response = session.get(url, headers=headers)
        if response.status_code == 200:
            file_hash = hashlib.md5(response.content).hexdigest()
            if catalog_id in self.downloaded_catalog_ids or file_hash in self.downloaded_hashes:
                return False, None

            save_dir = os.path.join(CONFIG['base_dir'], brand_name, catalog_id)
            os.makedirs(save_dir, exist_ok=True)
            zip_path = os.path.join(save_dir, f'catalog_{catalog_id}.zip')

            with open(zip_path, 'wb') as f:
                f.write(response.content)

            image_count, image_files = self._extract_images_from_zip(zip_path, brand_name, catalog_id)

            record = {
                'brand': brand_name,
                'catalog_id': catalog_id,
                'zip_path': zip_path,
                'extracted_dir': os.path.join(CONFIG['extracted_images_dir'], brand_name, catalog_id),
                'download_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_count': image_count,
                'file_size': os.path.getsize(zip_path),
                'first_image_path': image_files[0] if image_files else '',
                'all_image_paths': '|'.join(image_files),
                'status': 'success'
            }

            self.csv_records.append(record)
            self.downloaded_hashes.add(file_hash)
            self.downloaded_catalog_ids.add(catalog_id)

            self._add_to_google_sheet(record)

            return True, record
        return False, None

    def _add_to_google_sheet(self, record):
        if self.worksheet:
            try:
                row_data = [
                    record['brand'],
                    record['catalog_id'],
                    record['zip_path'],
                    record['extracted_dir'],
                    record['download_date'],
                    record['image_count'],
                    record['file_size'],
                    record['first_image_path'],
                    record['all_image_paths'],
                    record['status']
                ]
                self.worksheet.append_row(row_data)
                print(f"Googleスプレッドシートに追加: {record['brand']} {record['catalog_id']}")
            except Exception as e:
                print(f"Googleスプレッドシート追加エラー: {e}")

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
            167: "VALENTINO",
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
            catalog_id = row.find_element(By.CSS_SELECTOR, 'span.catalogs-table__contents-id').text.strip()

            image_cell = WebDriverWait(row, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.catalogs-table__image-item > .catalogs-table__image'))
            )
            ActionChains(self.driver).move_to_element(image_cell).pause(0.5).click().perform()

            WebDriverWait(self.driver, 15).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".catalogs-modal-table"))
            )

            download_link = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.catalogs-modal-table__link"))
            )
            download_url = download_link.get_attribute('href')

            success, record = self._download_file(download_url, brand_name, catalog_id)
            if success:
                print(f"成功: {brand_name} {catalog_id} (画像{record['image_count']}枚)")
            else:
                print(f"スキップ: {brand_name} {catalog_id}")

            self._force_close_modals()
            return True

        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.driver.save_screenshot(os.path.join(CONFIG['screenshot_dir'], f'error_{timestamp}.png'))
            with open(os.path.join(CONFIG['screenshot_dir'], f'error_{timestamp}.html'), 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            print(f"エラー発生: {str(e)[:100]}")
            return False

    def process_pagination(self, base_url, brand_name):
        page_num = 1
        while not self.stop_flag:
            try:
                self.driver.get(f"{base_url}&page={page_num}" if "?brand_id=" in base_url else f"{base_url}?page={page_num}")
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'tr.catalogs-table__row'))
                )

                rows = self.driver.find_elements(By.CSS_SELECTOR, 'tr.catalogs-table__row')
                if not rows:
                    break

                for row in rows:
                    if self.stop_flag:
                        return
                    self.process_catalog(row, brand_name)
                    self._human_like_delay()

                try:
                    next_btn = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.pagination__next:not([disabled])'))
                    )
                    page_num += 1
                    self._human_like_delay()
                except (NoSuchElementException, TimeoutException):
                    print(f"{brand_name} の最終ページに到達")
                    break
            except Exception as e:
                print(f"ページ処理エラー（{brand_name}）: {str(e)}")
                self.driver.save_screenshot(os.path.join(CONFIG['screenshot_dir'], f'pagination_error_{brand_name}.png'))
                break

    def _save_csv_summary(self):
        if not self.csv_records:
            return

        fieldnames = [
            'brand', 'catalog_id', 'zip_path', 'extracted_dir',
            'download_date', 'image_count', 'file_size',
            'first_image_path', 'all_image_paths', 'status'  # ← ここに'all_image_paths'を追加
        ]

        with open(CONFIG['csv_path'], 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_records)

        print(f"詳細レポート保存: {CONFIG['csv_path']}")

        total_downloads = len(self.csv_records)
        total_images = sum(record['image_count'] for record in self.csv_records)
        total_size_mb = sum(record['file_size'] for record in self.csv_records) / (1024 * 1024)

        summary = f"""
=== ダウンロード完了サマリー ===
総ダウンロード数: {total_downloads}件
総画像数: {total_images}枚
総ファイルサイズ: {total_size_mb:.2f}MB
保存先: {CONFIG['base_dir']}
解凍画像: {CONFIG['extracted_images_dir']}
CSV詳細: {CONFIG['csv_path']}
Googleスプレッドシート: {'連携済み' if self.worksheet else '未接続'}
        """
        print(summary)

    def main_flow(self):
        try:
            if self.worksheet:
                try:
                    headers = ['Brand', 'Catalog_ID', 'ZIP_Path', 'Extracted_Dir',
                             'Download_Date', 'Image_Count', 'File_Size_Bytes',
                             'First_Image_Path', 'All_Image_Paths', 'Status']
                    self.worksheet.clear()
                    self.worksheet.append_row(headers)
                    print("Googleスプレッドシートヘッダー設定完了")
                except Exception as e:
                    print(f"ヘッダー設定エラー: {e}")

            self.driver.get('https://www.buyma.com/login/')
            input("手動ログイン後、Enterを押してください...\n（途中で止めたい場合はCtrl+C）")

            popular_brands = self.get_popular_brands(30)
            print(f"処理対象ブランド: {list(popular_brands.values())}")

            for brand_id, brand_name in popular_brands.items():
                if self._safety_check():
                    break

                print(f"\n{brand_name}の処理を開始します...")
                catalog_url = f'https://www.buyma.com/my/sell/catalogs?brand_id={brand_id}'
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
        if self.request_count >= CONFIG['safety']['max_daily_requests']:
            print("1日のリクエスト上限に達しました")
            return True
        if self.error_count >= CONFIG['safety']['error_threshold']:
            print("エラーが多発したため停止します")
            return True
        return False

    def cleanup(self):
        self._save_csv_summary()
        self.driver.quit()
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_function_base_impl.py ===
import builtins
import collections.abc
import functools
import re
import sys
import warnings

import numpy as np
import numpy._core.numeric as _nx
from numpy._core import overrides, transpose
from numpy._core._multiarray_umath import _array_converter
from numpy._core.fromnumeric import any, mean, nonzero, partition, ravel, sum
from numpy._core.multiarray import _monotonicity, _place, bincount, normalize_axis_index
from numpy._core.multiarray import interp as compiled_interp
from numpy._core.multiarray import interp_complex as compiled_interp_complex
from numpy._core.numeric import (
    absolute,
    arange,
    array,
    asanyarray,
    asarray,
    concatenate,
    dot,
    empty,
    integer,
    intp,
    isscalar,
    ndarray,
    ones,
    take,
    where,
    zeros_like,
)
from numpy._core.numerictypes import typecodes
from numpy._core.umath import (
    add,
    arctan2,
    cos,
    exp,
    frompyfunc,
    less_equal,
    minimum,
    mod,
    not_equal,
    pi,
    sin,
    sqrt,
    subtract,
)
from numpy._utils import set_module

# needed in this module for compatibility
from numpy.lib._histograms_impl import histogram, histogramdd  # noqa: F401
from numpy.lib._twodim_base_impl import diag

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'select', 'piecewise', 'trim_zeros', 'copy', 'iterable', 'percentile',
    'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'flip',
    'rot90', 'extract', 'place', 'vectorize', 'asarray_chkfinite', 'average',
    'bincount', 'digitize', 'cov', 'corrcoef',
    'median', 'sinc', 'hamming', 'hanning', 'bartlett',
    'blackman', 'kaiser', 'trapezoid', 'trapz', 'i0',
    'meshgrid', 'delete', 'insert', 'append', 'interp',
    'quantile'
    ]

# _QuantileMethods is a dictionary listing all the supported methods to
# compute quantile/percentile.
#
# Below virtual_index refers to the index of the element where the percentile
# would be found in the sorted sample.
# When the sample contains exactly the percentile wanted, the virtual_index is
# an integer to the index of this element.
# When the percentile wanted is in between two elements, the virtual_index
# is made of a integer part (a.k.a 'i' or 'left') and a fractional part
# (a.k.a 'g' or 'gamma')
#
# Each method in _QuantileMethods has two properties
# get_virtual_index : Callable
#   The function used to compute the virtual_index.
# fix_gamma : Callable
#   A function used for discrete methods to force the index to a specific value.
_QuantileMethods = {
    # --- HYNDMAN and FAN METHODS
    # Discrete methods
    'inverted_cdf': {
        'get_virtual_index': lambda n, quantiles: _inverted_cdf(n, quantiles),  # noqa: PLW0108
        'fix_gamma': None,  # should never be called
    },
    'averaged_inverted_cdf': {
        'get_virtual_index': lambda n, quantiles: (n * quantiles) - 1,
        'fix_gamma': lambda gamma, _: _get_gamma_mask(
            shape=gamma.shape,
            default_value=1.,
            conditioned_value=0.5,
            where=gamma == 0),
    },
    'closest_observation': {
        'get_virtual_index': lambda n, quantiles: _closest_observation(n, quantiles),  # noqa: PLW0108
        'fix_gamma': None,  # should never be called
    },
    # Continuous methods
    'interpolated_inverted_cdf': {
        'get_virtual_index': lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0, 1),
        'fix_gamma': lambda gamma, _: gamma,
    },
    'hazen': {
        'get_virtual_index': lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0.5, 0.5),
        'fix_gamma': lambda gamma, _: gamma,
    },
    'weibull': {
        'get_virtual_index': lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0, 0),
        'fix_gamma': lambda gamma, _: gamma,
    },
    # Default method.
    # To avoid some rounding issues, `(n-1) * quantiles` is preferred to
    # `_compute_virtual_index(n, quantiles, 1, 1)`.
    # They are mathematically equivalent.
    'linear': {
        'get_virtual_index': lambda n, quantiles: (n - 1) * quantiles,
        'fix_gamma': lambda gamma, _: gamma,
    },
    'median_unbiased': {
        'get_virtual_index': lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 1 / 3.0, 1 / 3.0),
        'fix_gamma': lambda gamma, _: gamma,
    },
    'normal_unbiased': {
        'get_virtual_index': lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 3 / 8.0, 3 / 8.0),
        'fix_gamma': lambda gamma, _: gamma,
    },
    # --- OTHER METHODS
    'lower': {
        'get_virtual_index': lambda n, quantiles: np.floor(
            (n - 1) * quantiles).astype(np.intp),
        'fix_gamma': None,  # should never be called, index dtype is int
    },
    'higher': {
        'get_virtual_index': lambda n, quantiles: np.ceil(
            (n - 1) * quantiles).astype(np.intp),
        'fix_gamma': None,  # should never be called, index dtype is int
    },
    'midpoint': {
        'get_virtual_index': lambda n, quantiles: 0.5 * (
                np.floor((n - 1) * quantiles)
                + np.ceil((n - 1) * quantiles)),
        'fix_gamma': lambda gamma, index: _get_gamma_mask(
            shape=gamma.shape,
            default_value=0.5,
            conditioned_value=0.,
            where=index % 1 == 0),
    },
    'nearest': {
        'get_virtual_index': lambda n, quantiles: np.around(
            (n - 1) * quantiles).astype(np.intp),
        'fix_gamma': None,
        # should never be called, index dtype is int
    }}


def _rot90_dispatcher(m, k=None, axes=None):
    return (m,)


@array_function_dispatch(_rot90_dispatcher)
def rot90(m, k=1, axes=(0, 1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.

    Rotation direction is from the first towards the second axis.
    This means for a 2D array with the default `k` and `axes`, the
    rotation will be counterclockwise.

    Parameters
    ----------
    m : array_like
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes : (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.

    Returns
    -------
    y : ndarray
        A rotated view of `m`.

    See Also
    --------
    flip : Reverse the order of elements in an array along the given axis.
    fliplr : Flip an array horizontally.
    flipud : Flip an array vertically.

    Notes
    -----
    ``rot90(m, k=1, axes=(1,0))``  is the reverse of
    ``rot90(m, k=1, axes=(0,1))``

    ``rot90(m, k=1, axes=(1,0))`` is equivalent to
    ``rot90(m, k=-1, axes=(0,1))``

    Examples
    --------
    >>> import numpy as np
    >>> m = np.array([[1,2],[3,4]], int)
    >>> m
    array([[1, 2],
           [3, 4]])
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]])
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]])
    >>> m = np.arange(8).reshape((2,2,2))
    >>> np.rot90(m, 1, (1,2))
    array([[[1, 3],
            [0, 2]],
           [[5, 7],
            [4, 6]]])

    """
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    m = asanyarray(m)

    if axes[0] == axes[1] or absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    if (axes[0] >= m.ndim or axes[0] < -m.ndim
        or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError(f"Axes={axes} out of range for array of ndim={m.ndim}.")

    k %= 4

    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])

    axes_list = arange(0, m.ndim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                axes_list[axes[0]])

    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    else:
        # k == 3
        return flip(transpose(m, axes_list), axes[1])


def _flip_dispatcher(m, axis=None):
    return (m,)


@array_function_dispatch(_flip_dispatcher)
def flip(m, axis=None):
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple of ints, optional
         Axis or axes along which to flip over. The default,
         axis=None, will flip over all of the axes of the input array.
         If axis is negative it counts from the last to the first axis.

         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.

    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).

    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).

    flip(m, 1) is equivalent to fliplr(m).

    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.

    flip(m) corresponds to ``m[::-1,::-1,...,::-1]`` with ``::-1`` at all
    positions.

    flip(m, (0, 1)) corresponds to ``m[::-1,::-1,...]`` with ``::-1`` at
    position 0 and position 1.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> np.flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> np.flip(A)
    array([[[7, 6],
            [5, 4]],
           [[3, 2],
            [1, 0]]])
    >>> np.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
           [[1, 0],
            [3, 2]]])
    >>> rng = np.random.default_rng()
    >>> A = rng.normal(size=(3,4,5))
    >>> np.all(np.flip(A,2) == A[:,:,::-1,...])
    True
    """
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    if axis is None:
        indexer = (np.s_[::-1],) * m.ndim
    else:
        axis = _nx.normalize_axis_tuple(axis, m.ndim)
        indexer = [np.s_[:]] * m.ndim
        for ax in axis:
            indexer[ax] = np.s_[::-1]
        indexer = tuple(indexer)
    return m[indexer]


@set_module('numpy')
def iterable(y):
    """
    Check whether or not an object can be iterated over.

    Parameters
    ----------
    y : object
      Input object.

    Returns
    -------
    b : bool
      Return ``True`` if the object has an iterator method or is a
      sequence and ``False`` otherwise.


    Examples
    --------
    >>> import numpy as np
    >>> np.iterable([1, 2, 3])
    True
    >>> np.iterable(2)
    False

    Notes
    -----
    In most cases, the results of ``np.iterable(obj)`` are consistent with
    ``isinstance(obj, collections.abc.Iterable)``. One notable exception is
    the treatment of 0-dimensional arrays::

        >>> from collections.abc import Iterable
        >>> a = np.array(1.0)  # 0-dimensional numpy array
        >>> isinstance(a, Iterable)
        True
        >>> np.iterable(a)
        False

    """
    try:
        iter(y)
    except TypeError:
        return False
    return True


def _weights_are_valid(weights, a, axis):
    """Validate weights array.

    We assume, weights is not None.
    """
    wgt = np.asanyarray(weights)

    # Sanity checks
    if a.shape != wgt.shape:
        if axis is None:
            raise TypeError(
                "Axis must be specified when shapes of a and weights "
                "differ.")
        if wgt.shape != tuple(a.shape[ax] for ax in axis):
            raise ValueError(
                "Shape of weights must be consistent with "
                "shape of a along specified axis.")

        # setup wgt to broadcast along axis
        wgt = wgt.transpose(np.argsort(axis))
        wgt = wgt.reshape(tuple((s if ax in axis else 1)
                                for ax, s in enumerate(a.shape)))
    return wgt


def _average_dispatcher(a, axis=None, weights=None, returned=None, *,
                        keepdims=None):
    return (a, weights)


@array_function_dispatch(_average_dispatcher)
def average(a, axis=None, weights=None, returned=False, *,
            keepdims=np._NoValue):
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged. If `a` is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        `axis=None`, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The array of weights must be the same shape as `a` if no axis is
        specified, otherwise the weights must have dimensions and shape
        consistent with `a` along the specified axis.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        The calculation is::

            avg = sum(a * weights) / sum(weights)

        where the sum is over all included elements.
        The only constraint on the values of `weights` is that `sum(weights)`
        must not be 0.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        *Note:* `keepdims` will not work with instances of `numpy.matrix`
        or other classes whose methods do not support `keepdims`.

        .. versionadded:: 1.23.0

    Returns
    -------
    retval, [sum_of_weights] : array_type or double
        Return the average along the specified axis. When `returned` is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. `sum_of_weights` is of the
        same type as `retval`. The result dtype follows a general pattern.
        If `weights` is None, the result dtype will be that of `a` , or ``float64``
        if `a` is integral. Otherwise, if `weights` is not None and `a` is non-
        integral, the result type will be the type of lowest precision capable of
        representing values of both `a` and `weights`. If `a` happens to be
        integral, the previous rules still applies but the result dtype will
        at least be ``float64``.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When `weights` does not have the same shape as `a`, and `axis=None`.
    ValueError
        When `weights` does not have dimensions and shape consistent with `a`
        along specified `axis`.

    See Also
    --------
    mean

    ma.average : average for masked arrays -- useful if your data contains
                 "missing" values
    numpy.result_type : Returns the type that results from applying the
                        numpy type promotion rules to the arguments.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(1, 5)
    >>> data
    array([1, 2, 3, 4])
    >>> np.average(data)
    2.5
    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))
    4.0

    >>> data = np.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> np.average(data, axis=1, weights=[1./4, 3./4])
    array([0.75, 2.75, 4.75])
    >>> np.average(data, weights=[1./4, 3./4])
    Traceback (most recent call last):
        ...
    TypeError: Axis must be specified when shapes of a and weights differ.

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.average(data, axis=1, keepdims=True)
    array([[0.5],
           [2.5],
           [4.5]])

    >>> data = np.arange(8).reshape((2, 2, 2))
    >>> data
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
    array([3.4, 4.4])
    >>> np.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
    Traceback (most recent call last):
        ...
    ValueError: Shape of weights must be consistent
    with shape of a along specified axis.
    """
    a = np.asanyarray(a)

    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")

    if keepdims is np._NoValue:
        # Don't pass on the keepdims argument if one wasn't given.
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        avg_as_array = np.asanyarray(avg)
        scl = avg_as_array.dtype.type(a.size / avg_as_array.size)
    else:
        wgt = _weights_are_valid(weights=weights, a=a, axis=axis)

        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = avg_as_array = np.multiply(a, wgt,
                          dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != avg_as_array.shape:
            scl = np.broadcast_to(scl, avg_as_array.shape).copy()
        return avg, scl
    else:
        return avg


@set_module('numpy')
def asarray_chkfinite(a, dtype=None, order=None):
    """Convert the input to an array, checking for NaNs or Infs.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.  Success requires no NaNs or Infs.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray.  If `a` is a subclass of ndarray, a base
        class ndarray is returned.

    Raises
    ------
    ValueError
        Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

    See Also
    --------
    asarray : Create and array.
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    >>> import numpy as np

    Convert a list into an array. If all elements are finite, then
    ``asarray_chkfinite`` is identical to ``asarray``.

    >>> a = [1, 2]
    >>> np.asarray_chkfinite(a, dtype=float)
    array([1., 2.])

    Raises ValueError if array_like contains Nans or Infs.

    >>> a = [1, 2, np.inf]
    >>> try:
    ...     np.asarray_chkfinite(a)
    ... except ValueError:
    ...     print('ValueError')
    ...
    ValueError

    """
    a = asarray(a, dtype=dtype, order=order)
    if a.dtype.char in typecodes['AllFloat'] and not np.isfinite(a).all():
        raise ValueError(
            "array must not contain infs or NaNs")
    return a


def _piecewise_dispatcher(x, condlist, funclist, *args, **kw):
    yield x
    # support the undocumented behavior of allowing scalars
    if np.iterable(condlist):
        yield from condlist


@array_function_dispatch(_piecewise_dispatcher)
def piecewise(x, condlist, funclist, *args, **kw):
    """
    Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions, evaluate each
    function on the input data wherever its condition is true.

    Parameters
    ----------
    x : ndarray or scalar
        The input domain.
    condlist : list of bool arrays or bool scalars
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x)` is used as the output value.

        Each boolean array in `condlist` selects a piece of `x`,
        and should therefore be of the same shape as `x`.

        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if
        ``len(funclist) == len(condlist) + 1``, then that extra function
        is the default value, used wherever all conditions are false.
    funclist : list of callables, f(x,*args,**kw), or scalars
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take a 1d array as input and give an 1d
        array or a scalar value as output.  If, instead of a callable,
        a scalar is provided then a constant function (``lambda x: scalar``) is
        assumed.
    args : tuple, optional
        Any further arguments given to `piecewise` are passed to the functions
        upon execution, i.e., if called ``piecewise(..., ..., 1, 'a')``, then
        each function is called as ``f(x, 1, 'a')``.
    kw : dict, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(..., ..., alpha=1)``, then each function is called as
        ``f(x, alpha=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have a default value of 0.


    See Also
    --------
    choose, select, where

    Notes
    -----
    This is similar to choose or select, except that functions are
    evaluated on elements of `x` that satisfy the corresponding condition from
    `condlist`.

    The result is::

            |--
            |funclist[0](x[condlist[0]])
      out = |funclist[1](x[condlist[1]])
            |...
            |funclist[n2](x[condlist[n2]])
            |--

    Examples
    --------
    >>> import numpy as np

    Define the signum function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.

    >>> x = np.linspace(-2.5, 2.5, 6)
    >>> np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    array([-1., -1., -1.,  1.,  1.,  1.])

    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for
    ``x >= 0``.

    >>> np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    array([2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    Apply the same function to a scalar value.

    >>> y = -2
    >>> np.piecewise(y, [y < 0, y >= 0], [lambda x: -x, lambda x: x])
    array(2)

    """
    x = asanyarray(x)
    n2 = len(funclist)

    # undocumented: single condition is promoted to a list of one condition
    if isscalar(condlist) or (
            not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        condlist = [condlist]

    condlist = asarray(condlist, dtype=bool)
    n = len(condlist)

    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            f"with {n} condition(s), either {n} or {n + 1} functions are expected"
        )

    y = zeros_like(x)
    for cond, func in zip(condlist, funclist):
        if not isinstance(func, collections.abc.Callable):
            y[cond] = func
        else:
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, *args, **kw)

    return y


def _select_dispatcher(condlist, choicelist, default=None):
    yield from condlist
    yield from choicelist


@array_function_dispatch(_select_dispatcher)
def select(condlist, choicelist, default=0):
    """
    Return an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of bool ndarrays
        The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    choicelist : list of ndarrays
        The list of arrays from which the output elements are taken. It has
        to be of the same length as `condlist`.
    default : scalar, optional
        The element inserted in `output` when all conditions evaluate to False.

    Returns
    -------
    output : ndarray
        The output at position m is the m-th element of the array in
        `choicelist` where the m-th element of the corresponding array in
        `condlist` is True.

    See Also
    --------
    where : Return elements from one of two arrays depending on condition.
    take, choose, compress, diag, diagonal

    Examples
    --------
    >>> import numpy as np

    Beginning with an array of integers from 0 to 5 (inclusive),
    elements less than ``3`` are negated, elements greater than ``3``
    are squared, and elements not meeting either of these conditions
    (exactly ``3``) are replaced with a `default` value of ``42``.

    >>> x = np.arange(6)
    >>> condlist = [x<3, x>3]
    >>> choicelist = [-x, x**2]
    >>> np.select(condlist, choicelist, 42)
    array([ 0,  -1,  -2, 42, 16, 25])

    When multiple conditions are satisfied, the first one encountered in
    `condlist` is used.

    >>> condlist = [x<=4, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 55)
    array([ 0,  1,  2,  3,  4, 25])

    """
    # Check the size of condlist and choicelist are the same, or abort.
    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')

    # Now that the dtype is known, handle the deprecated select([], []) case
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    # TODO: This preserves the Python int, float, complex manually to get the
    #       right `result_type` with NEP 50.  Most likely we will grow a better
    #       way to spell this (and this can be replaced).
    choicelist = [
        choice if type(choice) in (int, float, complex) else np.asarray(choice)
        for choice in choicelist]
    choicelist.append(default if type(default) in (int, float, complex)
                      else np.asarray(default))

    try:
        dtype = np.result_type(*choicelist)
    except TypeError as e:
        msg = f'Choicelist and default value do not have a common dtype: {e}'
        raise TypeError(msg) from None

    # Convert conditions to arrays and broadcast conditions and choices
    # as the shape is needed for the result. Doing it separately optimizes
    # for example when all choices are scalars.
    condlist = np.broadcast_arrays(*condlist)
    choicelist = np.broadcast_arrays(*choicelist)

    # If cond array is not an ndarray in boolean format or scalar bool, abort.
    for i, cond in enumerate(condlist):
        if cond.dtype.type is not np.bool:
            raise TypeError(
                f'invalid entry {i} in condlist: should be boolean ndarray')

    if choicelist[0].ndim == 0:
        # This may be common, so avoid the call.
        result_shape = condlist[0].shape
    else:
        result_shape = np.broadcast_arrays(condlist[0], choicelist[0])[0].shape

    result = np.full(result_shape, choicelist[-1], dtype)

    # Use np.copyto to burn each choicelist array onto result, using the
    # corresponding condlist as a boolean mask. This is done in reverse
    # order since the first choice should take precedence.
    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        np.copyto(result, choice, where=cond)

    return result


def _copy_dispatcher(a, order=None, subok=None):
    return (a,)


@array_function_dispatch(_copy_dispatcher)
def copy(a, order='K', subok=False):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :meth:`ndarray.copy` are very
        similar, but have different default values for their order=
        arguments.)
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the
        returned array will be forced to be a base-class array (defaults to False).

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    See Also
    --------
    ndarray.copy : Preferred method for creating an array copy

    Notes
    -----
    This is equivalent to:

    >>> np.array(a, copy=True)  #doctest: +SKIP

    The copy made of the data is shallow, i.e., for arrays with object dtype,
    the new array will point to the same objects.
    See Examples from `ndarray.copy`.

    Examples
    --------
    >>> import numpy as np

    Create an array x, with a reference y and a copy z:

    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when we modify x, y changes, but not z:

    >>> x[0] = 10
    >>> x[0] == y[0]
    True
    >>> x[0] == z[0]
    False

    Note that, np.copy clears previously set WRITEABLE=False flag.

    >>> a = np.array([1, 2, 3])
    >>> a.flags["WRITEABLE"] = False
    >>> b = np.copy(a)
    >>> b.flags["WRITEABLE"]
    True
    >>> b[0] = 3
    >>> b
    array([3, 2, 3])
    """
    return array(a, order=order, subok=subok, copy=True)

# Basic operations


def _gradient_dispatcher(f, *varargs, axis=None, edge_order=None):
    yield f
    yield from varargs


@array_function_dispatch(_gradient_dispatcher)
def gradient(f, *varargs, axis=None, edge_order=1):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    Parameters
    ----------
    f : array_like
        An N-dimensional array containing samples of a scalar function.
    varargs : list of scalar or array, optional
        Spacing between f values. Default unitary spacing for all dimensions.
        Spacing can be specified using:

        1. single scalar to specify a sample distance for all dimensions.
        2. N scalars to specify a constant sample distance for each dimension.
           i.e. `dx`, `dy`, `dz`, ...
        3. N arrays to specify the coordinates of the values along each
           dimension of F. The length of the array must match the size of
           the corresponding dimension
        4. Any combination of N scalars/arrays with the meaning of 2. and 3.

        If `axis` is given, the number of varargs must equal the number of axes
        specified in the axis parameter.
        Default: 1. (see Examples below).

    edge_order : {1, 2}, optional
        Gradient is calculated using N-th order accurate differences
        at the boundaries. Default: 1.
    axis : None or int or tuple of ints, optional
        Gradient is calculated only along the given axis or axes
        The default (axis = None) is to calculate the gradient for all the axes
        of the input array. axis may be negative, in which case it counts from
        the last to the first axis.

    Returns
    -------
    gradient : ndarray or tuple of ndarray
        A tuple of ndarrays (or a single ndarray if there is only one
        dimension) corresponding to the derivatives of f with respect
        to each dimension. Each derivative has the same shape as f.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.array([1, 2, 4, 7, 11, 16])
    >>> np.gradient(f)
    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
    >>> np.gradient(f, 2)
    array([0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])

    Spacing can be also specified with an array that represents the coordinates
    of the values F along the dimensions.
    For instance a uniform spacing:

    >>> x = np.arange(f.size)
    >>> np.gradient(f, x)
    array([1. ,  1.5,  2.5,  3.5,  4.5,  5. ])

    Or a non uniform one:

    >>> x = np.array([0., 1., 1.5, 3.5, 4., 6.])
    >>> np.gradient(f, x)
    array([1. ,  3. ,  3.5,  6.7,  6.9,  2.5])

    For two dimensional arrays, the return will be two arrays ordered by
    axis. In this example the first array stands for the gradient in
    rows and the second one in columns direction:

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]))
    (array([[ 2.,  2., -1.],
            [ 2.,  2., -1.]]),
     array([[1. , 2.5, 4. ],
            [1. , 1. , 1. ]]))

    In this example the spacing is also specified:
    uniform for axis=0 and non uniform for axis=1

    >>> dx = 2.
    >>> y = [1., 1.5, 3.5]
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]), dx, y)
    (array([[ 1. ,  1. , -0.5],
            [ 1. ,  1. , -0.5]]),
     array([[2. , 2. , 2. ],
            [2. , 1.7, 0.5]]))

    It is possible to specify how boundaries are treated using `edge_order`

    >>> x = np.array([0, 1, 2, 3, 4])
    >>> f = x**2
    >>> np.gradient(f, edge_order=1)
    array([1.,  2.,  4.,  6.,  7.])
    >>> np.gradient(f, edge_order=2)
    array([0., 2., 4., 6., 8.])

    The `axis` keyword can be used to specify a subset of axes of which the
    gradient is calculated

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]]), axis=0)
    array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]])

    The `varargs` argument defines the spacing between sample points in the
    input array. It can take two forms:

    1. An array, specifying coordinates, which may be unevenly spaced:

    >>> x = np.array([0., 2., 3., 6., 8.])
    >>> y = x ** 2
    >>> np.gradient(y, x, edge_order=2)
    array([ 0.,  4.,  6., 12., 16.])

    2. A scalar, representing the fixed sample distance:

    >>> dx = 2
    >>> x = np.array([0., 2., 4., 6., 8.])
    >>> y = x ** 2
    >>> np.gradient(y, dx, edge_order=2)
    array([ 0.,  4.,  8., 12., 16.])

    It's possible to provide different data for spacing along each dimension.
    The number of arguments must match the number of dimensions in the input
    data.

    >>> dx = 2
    >>> dy = 3
    >>> x = np.arange(0, 6, dx)
    >>> y = np.arange(0, 9, dy)
    >>> xs, ys = np.meshgrid(x, y)
    >>> zs = xs + 2 * ys
    >>> np.gradient(zs, dy, dx)  # Passing two scalars
    (array([[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]]),
     array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]))

    Mixing scalars and arrays is also allowed:

    >>> np.gradient(zs, y, dx)  # Passing one array and one scalar
    (array([[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]]),
     array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]))

    Notes
    -----
    Assuming that :math:`f\\in C^{3}` (i.e., :math:`f` has at least 3 continuous
    derivatives) and let :math:`h_{*}` be a non-homogeneous stepsize, we
    minimize the "consistency error" :math:`\\eta_{i}` between the true gradient
    and its estimate from a linear combination of the neighboring grid-points:

    .. math::

        \\eta_{i} = f_{i}^{\\left(1\\right)} -
                    \\left[ \\alpha f\\left(x_{i}\\right) +
                            \\beta f\\left(x_{i} + h_{d}\\right) +
                            \\gamma f\\left(x_{i}-h_{s}\\right)
                    \\right]

    By substituting :math:`f(x_{i} + h_{d})` and :math:`f(x_{i} - h_{s})`
    with their Taylor series expansion, this translates into solving
    the following the linear system:

    .. math::

        \\left\\{
            \\begin{array}{r}
                \\alpha+\\beta+\\gamma=0 \\\\
                \\beta h_{d}-\\gamma h_{s}=1 \\\\
                \\beta h_{d}^{2}+\\gamma h_{s}^{2}=0
            \\end{array}
        \\right.

    The resulting approximation of :math:`f_{i}^{(1)}` is the following:

    .. math::

        \\hat f_{i}^{(1)} =
            \\frac{
                h_{s}^{2}f\\left(x_{i} + h_{d}\\right)
                + \\left(h_{d}^{2} - h_{s}^{2}\\right)f\\left(x_{i}\\right)
                - h_{d}^{2}f\\left(x_{i}-h_{s}\\right)}
                { h_{s}h_{d}\\left(h_{d} + h_{s}\\right)}
            + \\mathcal{O}\\left(\\frac{h_{d}h_{s}^{2}
                                + h_{s}h_{d}^{2}}{h_{d}
                                + h_{s}}\\right)

    It is worth noting that if :math:`h_{s}=h_{d}`
    (i.e., data are evenly spaced)
    we find the standard second order approximation:

    .. math::

        \\hat f_{i}^{(1)}=
            \\frac{f\\left(x_{i+1}\\right) - f\\left(x_{i-1}\\right)}{2h}
            + \\mathcal{O}\\left(h^{2}\\right)

    With a similar procedure the forward/backward approximations used for
    boundaries can be derived.

    References
    ----------
    .. [1]  Quarteroni A., Sacco R., Saleri F. (2007) Numerical Mathematics
            (Texts in Applied Mathematics). New York: Springer.
    .. [2]  Durran D. R. (1999) Numerical Methods for Wave Equations
            in Geophysical Fluid Dynamics. New York: Springer.
    .. [3]  Fornberg B. (1988) Generation of Finite Difference Formulas on
            Arbitrarily Spaced Grids,
            Mathematics of Computation 51, no. 184 : 699-706.
            `PDF <https://www.ams.org/journals/mcom/1988-51-184/
            S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_.
    """
    f = np.asanyarray(f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _nx.normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = np.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            if np.issubdtype(distances.dtype, np.integer):
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    otype = f.dtype
    if otype.type is np.datetime64:
        # the timedelta dtype with the same unit information
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        # view as timedelta to allow addition
        f = f.view(otype)
    elif otype.type is np.timedelta64:
        pass
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        # All other types convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required.")
        # result allocation
        out = np.empty_like(f, dtype=otype)

        # spacing for the current axis
        uniform_spacing = np.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2) / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = shape
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] \
                                                + c * f[tuple(slice4)]

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2. / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2. * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = - dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] \
                                                        + c * f[tuple(slice4)]

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2. / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = - (dx2 + dx1) / (dx1 * dx2)
                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] \
                                                        + c * f[tuple(slice4)]

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    return tuple(outvals)


def _diff_dispatcher(a, n=None, axis=None, prepend=None, append=None):
    return (a, prepend, append)


@array_function_dispatch(_diff_dispatcher)
def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """
    Calculate the n-th discrete difference along the given axis.

    The first difference is given by ``out[i] = a[i+1] - a[i]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes.  Otherwise the
        dimension and shape must match `a` except along axis.

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.

    See Also
    --------
    gradient, ediff1d, cumsum

    Notes
    -----
    Type is preserved for boolean arrays, so the result will contain
    `False` when consecutive elements are the same and `True` when they
    differ.

    For unsigned integer arrays, the results will also be unsigned. This
    should not be surprising, as the result is consistent with
    calculating the difference directly:

    >>> u8_arr = np.array([1, 0], dtype=np.uint8)
    >>> np.diff(u8_arr)
    array([255], dtype=uint8)
    >>> u8_arr[1,...] - u8_arr[0,...]
    np.uint8(255)

    If this is not desirable, then the array should be cast to a larger
    integer type first:

    >>> i16_arr = u8_arr.astype(np.int16)
    >>> np.diff(i16_arr)
    array([-1], dtype=int16)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.diff(x)
    array([ 1,  2,  3, -7])
    >>> np.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> np.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> np.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    >>> x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)
    >>> np.diff(x)
    array([1, 1], dtype='timedelta64[D]')

    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))

    a = asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = normalize_axis_index(axis, nd)

    combined = []
    if prepend is not np._NoValue:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not np._NoValue:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = not_equal if a.dtype == np.bool else subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a


def _interp_dispatcher(x, xp, fp, left=None, right=None, period=None):
    return (x, xp, fp)


@array_function_dispatch(_interp_dispatcher)
def interp(x, xp, fp, left=None, right=None, period=None):
    """
    One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.

    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument
        `period` is not specified. Otherwise, `xp` is internally sorted after
        normalizing the periodic boundaries with ``xp = xp % period``.

    fp : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as `xp`.

    left : optional float or complex corresponding to fp
        Value to return for `x < xp[0]`, default is `fp[0]`.

    right : optional float or complex corresponding to fp
        Value to return for `x > xp[-1]`, default is `fp[-1]`.

    period : None or float, optional
        A period for the x-coordinates. This parameter allows the proper
        interpolation of angular x-coordinates. Parameters `left` and `right`
        are ignored if `period` is specified.

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different length
        If `xp` or `fp` are not 1-D sequences
        If `period == 0`

    See Also
    --------
    scipy.interpolate

    Warnings
    --------
    The x-coordinate sequence is expected to be increasing, but this is not
    explicitly enforced.  However, if the sequence `xp` is non-increasing,
    interpolation results are meaningless.

    Note that, since NaN is unsortable, `xp` also cannot contain NaNs.

    A simple check for `xp` being strictly increasing is::

        np.all(np.diff(xp) > 0)

    Examples
    --------
    >>> import numpy as np
    >>> xp = [1, 2, 3]
    >>> fp = [3, 2, 0]
    >>> np.interp(2.5, xp, fp)
    1.0
    >>> np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
    array([3.  , 3.  , 2.5 , 0.56, 0.  ])
    >>> UNDEF = -99.0
    >>> np.interp(3.14, xp, fp, right=UNDEF)
    -99.0

    Plot an interpolant to the sine function:

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.sin(x)
    >>> xvals = np.linspace(0, 2*np.pi, 50)
    >>> yinterp = np.interp(xvals, x, y)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(xvals, yinterp, '-x')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    Interpolation with periodic x-coordinates:

    >>> x = [-180, -170, -185, 185, -10, -5, 0, 365]
    >>> xp = [190, -190, 350, -350]
    >>> fp = [5, 10, 3, 4]
    >>> np.interp(x, xp, fp, period=360)
    array([7.5 , 5.  , 8.75, 6.25, 3.  , 3.25, 3.5 , 3.75])

    Complex interpolation:

    >>> x = [1.5, 4.0]
    >>> xp = [2,3,5]
    >>> fp = [1.0j, 0, 2+3j]
    >>> np.interp(x, xp, fp)
    array([0.+1.j , 1.+1.5j])

    """

    fp = np.asarray(fp)

    if np.iscomplexobj(fp):
        interp_func = compiled_interp_complex
        input_dtype = np.complex128
    else:
        interp_func = compiled_interp
        input_dtype = np.float64

    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None

        x = np.asarray(x, dtype=np.float64)
        xp = np.asarray(xp, dtype=np.float64)
        fp = np.asarray(fp, dtype=input_dtype)

        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError("Data points must be 1-D sequences")
        if xp.shape[0] != fp.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        # normalizing periodic boundaries
        x = x % period
        xp = xp % period
        asort_xp = np.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = np.concatenate((xp[-1:] - period, xp, xp[0:1] + period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))

    return interp_func(x, xp, fp, left, right)


def _angle_dispatcher(z, deg=None):
    return (z,)


@array_function_dispatch(_angle_dispatcher)
def angle(z, deg=False):
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Return angle in degrees if True, radians if False (default).

    Returns
    -------
    angle : ndarray or scalar
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.

    See Also
    --------
    arctan2
    absolute

    Notes
    -----
    This function passes the imaginary and real parts of the argument to
    `arctan2` to compute the result; consequently, it follows the convention
    of `arctan2` when the magnitude of the argument is zero. See example.

    Examples
    --------
    >>> import numpy as np
    >>> np.angle([1.0, 1.0j, 1+1j])               # in radians
    array([ 0.        ,  1.57079633,  0.78539816]) # may vary
    >>> np.angle(1+1j, deg=True)                  # in degrees
    45.0
    >>> np.angle([0., -0., complex(0., -0.), complex(-0., -0.)])  # convention
    array([ 0.        ,  3.14159265, -0.        , -3.14159265])

    """
    z = asanyarray(z)
    if issubclass(z.dtype.type, _nx.complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        zimag = 0
        zreal = z

    a = arctan2(zimag, zreal)
    if deg:
        a *= 180 / pi
    return a


def _unwrap_dispatcher(p, discont=None, axis=None, *, period=None):
    return (p,)


@array_function_dispatch(_unwrap_dispatcher)
def unwrap(p, discont=None, axis=-1, *, period=2 * pi):
    r"""
    Unwrap by taking the complement of large deltas with respect to the period.

    This unwraps a signal `p` by changing elements which have an absolute
    difference from their predecessor of more than ``max(discont, period/2)``
    to their `period`-complementary values.

    For the default case where `period` is :math:`2\pi` and `discont` is
    :math:`\pi`, this unwraps a radian phase `p` such that adjacent differences
    are never greater than :math:`\pi` by adding :math:`2k\pi` for some
    integer :math:`k`.

    Parameters
    ----------
    p : array_like
        Input array.
    discont : float, optional
        Maximum discontinuity between values, default is ``period/2``.
        Values below ``period/2`` are treated as if they were ``period/2``.
        To have an effect different from the default, `discont` should be
        larger than ``period/2``.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.
    period : float, optional
        Size of the range over which the input wraps. By default, it is
        ``2 pi``.

        .. versionadded:: 1.21.0

    Returns
    -------
    out : ndarray
        Output array.

    See Also
    --------
    rad2deg, deg2rad

    Notes
    -----
    If the discontinuity in `p` is smaller than ``period/2``,
    but larger than `discont`, no unwrapping is done because taking
    the complement would only make the discontinuity larger.

    Examples
    --------
    >>> import numpy as np
    >>> phase = np.linspace(0, np.pi, num=5)
    >>> phase[3:] += np.pi
    >>> phase
    array([ 0.        ,  0.78539816,  1.57079633,  5.49778714,  6.28318531]) # may vary
    >>> np.unwrap(phase)
    array([ 0.        ,  0.78539816,  1.57079633, -0.78539816,  0.        ]) # may vary
    >>> np.unwrap([0, 1, 2, -1, 0], period=4)
    array([0, 1, 2, 3, 4])
    >>> np.unwrap([ 1, 2, 3, 4, 5, 6, 1, 2, 3], period=6)
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.unwrap([2, 3, 4, 5, 2, 3, 4, 5], period=4)
    array([2, 3, 4, 5, 6, 7, 8, 9])
    >>> phase_deg = np.mod(np.linspace(0 ,720, 19), 360) - 180
    >>> np.unwrap(phase_deg, period=360)
    array([-180., -140., -100.,  -60.,  -20.,   20.,   60.,  100.,  140.,
            180.,  220.,  260.,  300.,  340.,  380.,  420.,  460.,  500.,
            540.])
    """
    p = asarray(p)
    nd = p.ndim
    dd = diff(p, axis=axis)
    if discont is None:
        discont = period / 2
    slice1 = [slice(None, None)] * nd     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    dtype = np.result_type(dd, period)
    if _nx.issubdtype(dtype, _nx.integer):
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = mod(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        # for `mask = (abs(dd) == period/2)`, the above line made
        # `ddmod[mask] == -period/2`. correct these such that
        # `ddmod[mask] == sign(dd[mask])*period/2`.
        _nx.copyto(ddmod, interval_high,
                   where=(ddmod == interval_low) & (dd > 0))
    ph_correct = ddmod - dd
    _nx.copyto(ph_correct, 0, where=abs(dd) < discont)
    up = array(p, copy=True, dtype=dtype)
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up


def _sort_complex(a):
    return (a,)


@array_function_dispatch(_sort_complex)
def sort_complex(a):
    """
    Sort a complex array using the real part first, then the imaginary part.

    Parameters
    ----------
    a : array_like
        Input array

    Returns
    -------
    out : complex ndarray
        Always returns a sorted complex array.

    Examples
    --------
    >>> import numpy as np
    >>> np.sort_complex([5, 3, 6, 2, 1])
    array([1.+0.j, 2.+0.j, 3.+0.j, 5.+0.j, 6.+0.j])

    >>> np.sort_complex([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
    array([1.+2.j,  2.-1.j,  3.-3.j,  3.-2.j,  3.+5.j])

    """
    b = array(a, copy=True)
    b.sort()
    if not issubclass(b.dtype.type, _nx.complexfloating):
        if b.dtype.char in 'bhBH':
            return b.astype('F')
        elif b.dtype.char == 'g':
            return b.astype('G')
        else:
            return b.astype('D')
    else:
        return b


def _arg_trim_zeros(filt):
    """Return indices of the first and last non-zero element.

    Parameters
    ----------
    filt : array_like
        Input array.

    Returns
    -------
    start, stop : ndarray
        Two arrays containing the indices of the first and last non-zero
        element in each dimension.

    See also
    --------
    trim_zeros

    Examples
    --------
    >>> import numpy as np
    >>> _arg_trim_zeros(np.array([0, 0, 1, 1, 0]))
    (array([2]), array([3]))
    """
    nonzero = (
        np.argwhere(filt)
        if filt.dtype != np.object_
        # Historically, `trim_zeros` treats `None` in an object array
        # as non-zero while argwhere doesn't, account for that
        else np.argwhere(filt != 0)
    )
    if nonzero.size == 0:
        start = stop = np.array([], dtype=np.intp)
    else:
        start = nonzero.min(axis=0)
        stop = nonzero.max(axis=0)
    return start, stop


def _trim_zeros(filt, trim=None, axis=None):
    return (filt,)


@array_function_dispatch(_trim_zeros)
def trim_zeros(filt, trim='fb', axis=None):
    """Remove values along a dimension which are zero along all other.

    Parameters
    ----------
    filt : array_like
        Input array.
    trim : {"fb", "f", "b"}, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back. By default, zeros are trimmed on both sides.
        Front and back refer to the edges of a dimension, with "front" referring
        to the side with the lowest index 0, and "back" referring to the highest
        index (or index -1).
    axis : int or sequence, optional
        If None, `filt` is cropped such that the smallest bounding box is
        returned that still contains all values which are not zero.
        If an axis is specified, `filt` will be sliced in that dimension only
        on the sides specified by `trim`. The remaining area will be the
        smallest that still contains all values wich are not zero.

        .. versionadded:: 2.2.0

    Returns
    -------
    trimmed : ndarray or sequence
        The result of trimming the input. The number of dimensions and the
        input data type are preserved.

    Notes
    -----
    For all-zero arrays, the first axis is trimmed first.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    >>> np.trim_zeros(a)
    array([1, 2, 3, 0, 2, 1])

    >>> np.trim_zeros(a, trim='b')
    array([0, 0, 0, ..., 0, 2, 1])

    Multiple dimensions are supported.

    >>> b = np.array([[0, 0, 2, 3, 0, 0],
    ...               [0, 1, 0, 3, 0, 0],
    ...               [0, 0, 0, 0, 0, 0]])
    >>> np.trim_zeros(b)
    array([[0, 2, 3],
           [1, 0, 3]])

    >>> np.trim_zeros(b, axis=-1)
    array([[0, 2, 3],
           [1, 0, 3],
           [0, 0, 0]])

    The input data type is preserved, list/tuple in means list/tuple out.

    >>> np.trim_zeros([0, 1, 2, 0])
    [1, 2]

    """
    filt_ = np.asarray(filt)

    trim = trim.lower()
    if trim not in {"fb", "bf", "f", "b"}:
        raise ValueError(f"unexpected character(s) in `trim`: {trim!r}")

    start, stop = _arg_trim_zeros(filt_)
    stop += 1  # Adjust for slicing

    if start.size == 0:
        # filt is all-zero -> assign same values to start and stop so that
        # resulting slice will be empty
        start = stop = np.zeros(filt_.ndim, dtype=np.intp)
    else:
        if 'f' not in trim:
            start = (None,) * filt_.ndim
        if 'b' not in trim:
            stop = (None,) * filt_.ndim

    if len(start) == 1:
        # filt is 1D -> don't use multi-dimensional slicing to preserve
        # non-array input types
        sl = slice(start[0], stop[0])
    elif axis is None:
        # trim all axes
        sl = tuple(slice(*x) for x in zip(start, stop))
    else:
        # only trim single axis
        axis = normalize_axis_index(axis, filt_.ndim)
        sl = (slice(None),) * axis + (slice(start[axis], stop[axis]),) + (...,)

    trimmed = filt[sl]
    return trimmed


def _extract_dispatcher(condition, arr):
    return (condition, arr)


@array_function_dispatch(_extract_dispatcher)
def extract(condition, arr):
    """
    Return the elements of an array that satisfy some condition.

    This is equivalent to ``np.compress(ravel(condition), ravel(arr))``.  If
    `condition` is boolean ``np.extract`` is equivalent to ``arr[condition]``.

    Note that `place` does the exact opposite of `extract`.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of `arr`
        to extract.
    arr : array_like
        Input array of the same size as `condition`.

    Returns
    -------
    extract : ndarray
        Rank 1 array of values from `arr` where `condition` is True.

    See Also
    --------
    take, put, copyto, compress, place

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(12).reshape((3, 4))
    >>> arr
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> condition = np.mod(arr, 3)==0
    >>> condition
    array([[ True, False, False,  True],
           [False, False,  True, False],
           [False,  True, False, False]])
    >>> np.extract(condition, arr)
    array([0, 3, 6, 9])


    If `condition` is boolean:

    >>> arr[condition]
    array([0, 3, 6, 9])

    """
    return _nx.take(ravel(arr), nonzero(ravel(condition))[0])


def _place_dispatcher(arr, mask, vals):
    return (arr, mask, vals)


@array_function_dispatch(_place_dispatcher)
def place(arr, mask, vals):
    """
    Change elements of an array based on conditional and input values.

    Similar to ``np.copyto(arr, vals, where=mask)``, the difference is that
    `place` uses the first N elements of `vals`, where N is the number of
    True values in `mask`, while `copyto` uses the elements where `mask`
    is True.

    Note that `extract` does the exact opposite of `place`.

    Parameters
    ----------
    arr : ndarray
        Array to put data into.
    mask : array_like
        Boolean mask array. Must have the same size as `a`.
    vals : 1-D sequence
        Values to put into `a`. Only the first N elements are used, where
        N is the number of True values in `mask`. If `vals` is smaller
        than N, it will be repeated, and if elements of `a` are to be masked,
        this sequence must be non-empty.

    See Also
    --------
    copyto, put, take, extract

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(6).reshape(2, 3)
    >>> np.place(arr, arr>2, [44, 55])
    >>> arr
    array([[ 0,  1,  2],
           [44, 55, 44]])

    """
    return _place(arr, mask, vals)


def disp(mesg, device=None, linefeed=True):
    """
    Display a message on a device.

    .. deprecated:: 2.0
        Use your own printing function instead.

    Parameters
    ----------
    mesg : str
        Message to display.
    device : object
        Device to write message. If None, defaults to ``sys.stdout`` which is
        very similar to ``print``. `device` needs to have ``write()`` and
        ``flush()`` methods.
    linefeed : bool, optional
        Option whether to print a line feed or not. Defaults to True.

    Raises
    ------
    AttributeError
        If `device` does not have a ``write()`` or ``flush()`` method.

    Examples
    --------
    >>> import numpy as np

    Besides ``sys.stdout``, a file-like object can also be used as it has
    both required methods:

    >>> from io import StringIO
    >>> buf = StringIO()
    >>> np.disp('"Display" in a file', device=buf)
    >>> buf.getvalue()
    '"Display" in a file\\n'

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`disp` is deprecated, "
        "use your own printing function instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    if device is None:
        device = sys.stdout
    if linefeed:
        device.write(f'{mesg}\n')
    else:
        device.write(f'{mesg}')
    device.flush()


# See https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = f'(?:{_DIMENSION_NAME}(?:,{_DIMENSION_NAME})*)?'
_ARGUMENT = fr'\({_CORE_DIMENSION_LIST}\)'
_ARGUMENT_LIST = f'{_ARGUMENT}(?:,{_ARGUMENT})*'
_SIGNATURE = f'^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$'


def _parse_gufunc_signature(signature):
    """
    Parse string signatures for a generalized universal function.

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
        for ``np.matmul``.

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]].
    """
    signature = re.sub(r'\s+', '', signature)

    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            f'not a valid gufunc signature: {signature}')
    return tuple([tuple(re.findall(_DIMENSION_NAME, arg))
                  for arg in re.findall(_ARGUMENT, arg_list)]
                 for arg_list in signature.split('->'))


def _update_dim_sizes(dim_sizes, arg, core_dims):
    """
    Incrementally check and update core dimension sizes for a single argument.

    Arguments
    ---------
    dim_sizes : Dict[str, int]
        Sizes of existing core dimensions. Will be updated in-place.
    arg : ndarray
        Argument to examine.
    core_dims : Tuple[str, ...]
        Core dimensions for this argument.
    """
    if not core_dims:
        return

    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            '%d-dimensional argument does not have enough '
            'dimensions for all core dimensions %r'
            % (arg.ndim, core_dims))

    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError(
                    'inconsistent size for core dimension %r: %r vs %r'
                    % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size


def _parse_input_dimensions(args, input_core_dims):
    """
    Parse broadcast and core dimensions for vectorize with a signature.

    Arguments
    ---------
    args : Tuple[ndarray, ...]
        Tuple of input arguments to examine.
    input_core_dims : List[Tuple[str, ...]]
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib._stride_tricks_impl._broadcast_shape(
        *broadcast_args
    )
    return broadcast_shape, dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    """Helper for calculating broadcast shapes with core dimensions."""
    return [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
            for core_dims in list_of_core_dims]


def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes,
                   results=None):
    """Helper for creating output arrays in vectorize."""
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    if dtypes is None:
        dtypes = [None] * len(shapes)
    if results is None:
        arrays = tuple(np.empty(shape=shape, dtype=dtype)
                       for shape, dtype in zip(shapes, dtypes))
    else:
        arrays = tuple(np.empty_like(result, shape=shape, dtype=dtype)
                       for result, shape, dtype
                       in zip(results, shapes, dtypes))
    return arrays


def _get_vectorize_dtype(dtype):
    if dtype.char in "SU":
        return dtype.char
    return dtype


@set_module('numpy')
class vectorize:
    """
    vectorize(pyfunc=np._NoValue, otypes=None, doc=None, excluded=None,
    cache=False, signature=None)

    Returns an object that acts like pyfunc, but takes arrays as input.

    Define a vectorized function which takes a nested sequence of objects or
    numpy arrays as inputs and returns a single numpy array or a tuple of numpy
    arrays. The vectorized function evaluates `pyfunc` over successive tuples
    of the input arrays like the python map function, except it uses the
    broadcasting rules of numpy.

    The data type of the output of `vectorized` is determined by calling
    the function with the first element of the input.  This can be avoided
    by specifying the `otypes` argument.

    Parameters
    ----------
    pyfunc : callable, optional
        A python function or method.
        Can be omitted to produce a decorator with keyword arguments.
    otypes : str or list of dtypes, optional
        The output data type. It must be specified as either a string of
        typecode characters or a list of data type specifiers. There should
        be one data type specifier for each output.
    doc : str, optional
        The docstring for the function. If None, the docstring will be the
        ``pyfunc.__doc__``.
    excluded : set, optional
        Set of strings or integers representing the positional or keyword
        arguments for which the function will not be vectorized. These will be
        passed directly to `pyfunc` unmodified.

    cache : bool, optional
        If `True`, then cache the first function call that determines the number
        of outputs if `otypes` is not provided.

    signature : string, optional
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
        be called with (and expected to return) arrays with shapes given by the
        size of corresponding core dimensions. By default, ``pyfunc`` is
        assumed to take scalars as input and output.

    Returns
    -------
    out : callable
        A vectorized function if ``pyfunc`` was provided,
        a decorator otherwise.

    See Also
    --------
    frompyfunc : Takes an arbitrary Python function and returns a ufunc

    Notes
    -----
    The `vectorize` function is provided primarily for convenience, not for
    performance. The implementation is essentially a for loop.

    If `otypes` is not specified, then a call to the function with the
    first argument will be used to determine the number of outputs.  The
    results of this call will be cached if `cache` is `True` to prevent
    calling the function twice.  However, to implement the cache, the
    original function must be wrapped which will slow down subsequent
    calls, so only do this if your function is expensive.

    The new keyword argument interface and `excluded` argument support
    further degrades performance.

    References
    ----------
    .. [1] :doc:`/reference/c-api/generalized-ufuncs`

    Examples
    --------
    >>> import numpy as np
    >>> def myfunc(a, b):
    ...     "Return a-b if a>b, otherwise return a+b"
    ...     if a > b:
    ...         return a - b
    ...     else:
    ...         return a + b

    >>> vfunc = np.vectorize(myfunc)
    >>> vfunc([1, 2, 3, 4], 2)
    array([3, 4, 1, 2])

    The docstring is taken from the input function to `vectorize` unless it
    is specified:

    >>> vfunc.__doc__
    'Return a-b if a>b, otherwise return a+b'
    >>> vfunc = np.vectorize(myfunc, doc='Vectorized `myfunc`')
    >>> vfunc.__doc__
    'Vectorized `myfunc`'

    The output type is determined by evaluating the first element of the input,
    unless it is specified:

    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <class 'numpy.int64'>
    >>> vfunc = np.vectorize(myfunc, otypes=[float])
    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <class 'numpy.float64'>

    The `excluded` argument can be used to prevent vectorizing over certain
    arguments.  This can be useful for array-like arguments of a fixed length
    such as the coefficients for a polynomial as in `polyval`:

    >>> def mypolyval(p, x):
    ...     _p = list(p)
    ...     res = _p.pop(0)
    ...     while _p:
    ...         res = res*x + _p.pop(0)
    ...     return res

    Here, we exclude the zeroth argument from vectorization whether it is
    passed by position or keyword.

    >>> vpolyval = np.vectorize(mypolyval, excluded={0, 'p'})
    >>> vpolyval([1, 2, 3], x=[0, 1])
    array([3, 6])
    >>> vpolyval(p=[1, 2, 3], x=[0, 1])
    array([3, 6])

    The `signature` argument allows for vectorizing functions that act on
    non-scalar arrays of fixed length. For example, you can use it for a
    vectorized calculation of Pearson correlation coefficient and its p-value:

    >>> import scipy.stats
    >>> pearsonr = np.vectorize(scipy.stats.pearsonr,
    ...                 signature='(n),(n)->(),()')
    >>> pearsonr([[0, 1, 2, 3]], [[1, 2, 3, 4], [4, 3, 2, 1]])
    (array([ 1., -1.]), array([ 0.,  0.]))

    Or for a vectorized convolution:

    >>> convolve = np.vectorize(np.convolve, signature='(n),(m)->(k)')
    >>> convolve(np.eye(4), [1, 2, 1])
    array([[1., 2., 1., 0., 0., 0.],
           [0., 1., 2., 1., 0., 0.],
           [0., 0., 1., 2., 1., 0.],
           [0., 0., 0., 1., 2., 1.]])

    Decorator syntax is supported.  The decorator can be called as
    a function to provide keyword arguments:

    >>> @np.vectorize
    ... def identity(x):
    ...     return x
    ...
    >>> identity([0, 1, 2])
    array([0, 1, 2])
    >>> @np.vectorize(otypes=[float])
    ... def as_float(x):
    ...     return x
    ...
    >>> as_float([0, 1, 2])
    array([0., 1., 2.])
    """
    def __init__(self, pyfunc=np._NoValue, otypes=None, doc=None,
                 excluded=None, cache=False, signature=None):

        if (pyfunc != np._NoValue) and (not callable(pyfunc)):
            # Splitting the error message to keep
            # the length below 79 characters.
            part1 = "When used as a decorator, "
            part2 = "only accepts keyword arguments."
            raise TypeError(part1 + part2)

        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature
        if pyfunc != np._NoValue and hasattr(pyfunc, '__name__'):
            self.__name__ = pyfunc.__name__

        self._ufunc = {}    # Caching to improve default performance
        self._doc = None
        self.__doc__ = doc
        if doc is None and hasattr(pyfunc, '__doc__'):
            self.__doc__ = pyfunc.__doc__
        else:
            self._doc = doc

        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes['All']:
                    raise ValueError(f"Invalid otype specified: {char}")
        elif iterable(otypes):
            otypes = [_get_vectorize_dtype(_nx.dtype(x)) for x in otypes]
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes

        # Excluded variable support
        if excluded is None:
            excluded = set()
        self.excluded = set(excluded)

        if signature is not None:
            self._in_and_out_core_dims = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dims = None

    def _init_stage_2(self, pyfunc, *args, **kwargs):
        self.__name__ = pyfunc.__name__
        self.pyfunc = pyfunc
        if self._doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = self._doc

    def _call_as_normal(self, *args, **kwargs):
        """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
        excluded = self.excluded
        if not kwargs and not excluded:
            func = self.pyfunc
            vargs = args
        else:
            # The wrapper accepts only positional arguments: we use `names` and
            # `inds` to mutate `the_args` and `kwargs` to pass to the original
            # function.
            nargs = len(args)

            names = [_n for _n in kwargs if _n not in excluded]
            inds = [_i for _i in range(nargs) if _i not in excluded]
            the_args = list(args)

            def func(*vargs):
                for _n, _i in enumerate(inds):
                    the_args[_i] = vargs[_n]
                kwargs.update(zip(names, vargs[len(inds):]))
                return self.pyfunc(*the_args, **kwargs)

            vargs = [args[_i] for _i in inds]
            vargs.extend([kwargs[_n] for _n in names])

        return self._vectorize_call(func=func, args=vargs)

    def __call__(self, *args, **kwargs):
        if self.pyfunc is np._NoValue:
            self._init_stage_2(*args, **kwargs)
            return self

        return self._call_as_normal(*args, **kwargs)

    def _get_ufunc_and_otypes(self, func, args):
        """Return (ufunc, otypes)."""
        # frompyfunc will fail if args is empty
        if not args:
            raise ValueError('args can not be empty')

        if self.otypes is not None:
            otypes = self.otypes

            # self._ufunc is a dictionary whose keys are the number of
            # arguments (i.e. len(args)) and whose values are ufuncs created
            # by frompyfunc. len(args) can be different for different calls if
            # self.pyfunc has parameters with default values.  We only use the
            # cache when func is self.pyfunc, which occurs when the call uses
            # only positional arguments and no arguments are excluded.

            nin = len(args)
            nout = len(self.otypes)
            if func is not self.pyfunc or nin not in self._ufunc:
                ufunc = frompyfunc(func, nin, nout)
            else:
                ufunc = None  # We'll get it from self._ufunc
            if func is self.pyfunc:
                ufunc = self._ufunc.setdefault(nin, ufunc)
        else:
            # Get number of outputs and output types by calling the function on
            # the first entries of args.  We also cache the result to prevent
            # the subsequent call when the ufunc is evaluated.
            # Assumes that ufunc first evaluates the 0th elements in the input
            # arrays (the input values are not checked to ensure this)
            if builtins.any(arg.size == 0 for arg in args):
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')

            inputs = [arg.flat[0] for arg in args]
            outputs = func(*inputs)

            # Performance note: profiling indicates that -- for simple
            # functions at least -- this wrapping can almost double the
            # execution time.
            # Hence we make it optional.
            if self.cache:
                _cache = [outputs]

                def _func(*vargs):
                    if _cache:
                        return _cache.pop()
                    else:
                        return func(*vargs)
            else:
                _func = func

            if isinstance(outputs, tuple):
                nout = len(outputs)
            else:
                nout = 1
                outputs = (outputs,)

            otypes = ''.join([asarray(outputs[_k]).dtype.char
                              for _k in range(nout)])

            # Performance note: profiling indicates that creating the ufunc is
            # not a significant cost compared with wrapping so it seems not
            # worth trying to cache this.
            ufunc = frompyfunc(_func, len(args), nout)

        return ufunc, otypes

    def _vectorize_call(self, func, args):
        """Vectorized call to `func` over positional `args`."""
        if self.signature is not None:
            res = self._vectorize_call_with_signature(func, args)
        elif not args:
            res = func()
        else:
            args = [asanyarray(a, dtype=object) for a in args]
            ufunc, otypes = self._get_ufunc_and_otypes(func=func, args=args)
            outputs = ufunc(*args, out=...)

            if ufunc.nout == 1:
                res = asanyarray(outputs, dtype=otypes[0])
            else:
                res = tuple(asanyarray(x, dtype=t)
                            for x, t in zip(outputs, otypes))
        return res

    def _vectorize_call_with_signature(self, func, args):
        """Vectorized call over positional arguments with a signature."""
        input_core_dims, output_core_dims = self._in_and_out_core_dims

        if len(args) != len(input_core_dims):
            raise TypeError('wrong number of positional arguments: '
                            'expected %r, got %r'
                            % (len(input_core_dims), len(args)))
        args = tuple(asanyarray(arg) for arg in args)

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims)
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes,
                                         input_core_dims)
        args = [np.broadcast_to(arg, shape, subok=True)
                for arg, shape in zip(args, input_shapes)]

        outputs = None
        otypes = self.otypes
        nout = len(output_core_dims)

        for index in np.ndindex(*broadcast_shape):
            results = func(*(arg[index] for arg in args))

            n_results = len(results) if isinstance(results, tuple) else 1

            if nout != n_results:
                raise ValueError(
                    'wrong number of outputs from pyfunc: expected %r, got %r'
                    % (nout, n_results))

            if nout == 1:
                results = (results,)

            if outputs is None:
                for result, core_dims in zip(results, output_core_dims):
                    _update_dim_sizes(dim_sizes, result, core_dims)

                outputs = _create_arrays(broadcast_shape, dim_sizes,
                                         output_core_dims, otypes, results)

            for output, result in zip(outputs, results):
                output[index] = result

        if outputs is None:
            # did not call the function even once
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            if builtins.any(dim not in dim_sizes
                            for dims in output_core_dims
                            for dim in dims):
                raise ValueError('cannot call `vectorize` with a signature '
                                 'including new output dimensions on size 0 '
                                 'inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     output_core_dims, otypes)

        return outputs[0] if nout == 1 else outputs


def _cov_dispatcher(m, y=None, rowvar=None, bias=None, ddof=None,
                    fweights=None, aweights=None, *, dtype=None):
    return (m, y, fweights, aweights)


@array_function_dispatch(_cov_dispatcher)
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None, *, dtype=None):
    """
    Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    See the notes for an outline of the algorithm.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average. See the notes for the details. The default value
        is ``None``.
    fweights : array_like, int, optional
        1-D array of integer frequency weights; the number of times each
        observation vector should be repeated.
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `numpy.float64` precision.

        .. versionadded:: 1.20

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    See Also
    --------
    corrcoef : Normalized covariance matrix

    Notes
    -----
    Assume that the observations are in the columns of the observation
    array `m` and let ``f = fweights`` and ``a = aweights`` for brevity. The
    steps to compute the weighted covariance are as follows::

        >>> m = np.arange(10, dtype=np.float64)
        >>> f = np.arange(10) * 2
        >>> a = np.arange(10) ** 2.
        >>> ddof = 1
        >>> w = f * a
        >>> v1 = np.sum(w)
        >>> v2 = np.sum(w * a)
        >>> m -= np.sum(m * w, axis=None, keepdims=True) / v1
        >>> cov = np.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)

    Note that when ``a == 1``, the normalization factor
    ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (np.sum(f) - ddof)``
    as it should.

    Examples
    --------
    >>> import numpy as np

    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.stack((x, y), axis=0)
    >>> np.cov(X)
    array([[11.71      , -4.286     ], # may vary
           [-4.286     ,  2.144133]])
    >>> np.cov(x, y)
    array([[11.71      , -4.286     ], # may vary
           [-4.286     ,  2.144133]])
    >>> np.cov(x)
    array(11.71)

    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)

    X = array(m, ndmin=2, dtype=dtype)
    if not rowvar and m.ndim != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = array(y, copy=None, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    c = dot(X, X_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()


def _corrcoef_dispatcher(x, y=None, rowvar=None, bias=None, ddof=None, *,
                         dtype=None):
    return (x, y)


@array_function_dispatch(_corrcoef_dispatcher)
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, *,
             dtype=None):
    """
    Return Pearson product-moment correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, `R`, and the
    covariance matrix, `C`, is

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }

    The values of `R` are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    ddof : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `numpy.float64` precision.

        .. versionadded:: 1.20

    Returns
    -------
    R : ndarray
        The correlation coefficient matrix of the variables.

    See Also
    --------
    cov : Covariance matrix

    Notes
    -----
    Due to floating point rounding the resulting array may not be Hermitian,
    the diagonal elements may not be 1, and the elements may not satisfy the
    inequality abs(a) <= 1. The real and imaginary parts are clipped to the
    interval [-1,  1] in an attempt to improve on that situation but is not
    much help in the complex case.

    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    Examples
    --------
    >>> import numpy as np

    In this example we generate two random arrays, ``xarr`` and ``yarr``, and
    compute the row-wise and column-wise Pearson correlation coefficients,
    ``R``. Since ``rowvar`` is  true by  default, we first find the row-wise
    Pearson correlation coefficients between the variables of ``xarr``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> xarr = rng.random((3, 3))
    >>> xarr
    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],
           [0.7611397 , 0.78606431, 0.12811363]])
    >>> R1 = np.corrcoef(xarr)
    >>> R1
    array([[ 1.        ,  0.99256089, -0.68080986],
           [ 0.99256089,  1.        , -0.76492172],
           [-0.68080986, -0.76492172,  1.        ]])

    If we add another set of variables and observations ``yarr``, we can
    compute the row-wise Pearson correlation coefficients between the
    variables in ``xarr`` and ``yarr``.

    >>> yarr = rng.random((3, 3))
    >>> yarr
    array([[0.45038594, 0.37079802, 0.92676499],
           [0.64386512, 0.82276161, 0.4434142 ],
           [0.22723872, 0.55458479, 0.06381726]])
    >>> R2 = np.corrcoef(xarr, yarr)
    >>> R2
    array([[ 1.        ,  0.99256089, -0.68080986,  0.75008178, -0.934284  ,
            -0.99004057],
           [ 0.99256089,  1.        , -0.76492172,  0.82502011, -0.97074098,
            -0.99981569],
           [-0.68080986, -0.76492172,  1.        , -0.99507202,  0.89721355,
             0.77714685],
           [ 0.75008178,  0.82502011, -0.99507202,  1.        , -0.93657855,
            -0.83571711],
           [-0.934284  , -0.97074098,  0.89721355, -0.93657855,  1.        ,
             0.97517215],
           [-0.99004057, -0.99981569,  0.77714685, -0.83571711,  0.97517215,
             1.        ]])

    Finally if we use the option ``rowvar=False``, the columns are now
    being treated as the variables and we will find the column-wise Pearson
    correlation coefficients between variables in ``xarr`` and ``yarr``.

    >>> R3 = np.corrcoef(xarr, yarr, rowvar=False)
    >>> R3
    array([[ 1.        ,  0.77598074, -0.47458546, -0.75078643, -0.9665554 ,
             0.22423734],
           [ 0.77598074,  1.        , -0.92346708, -0.99923895, -0.58826587,
            -0.44069024],
           [-0.47458546, -0.92346708,  1.        ,  0.93773029,  0.23297648,
             0.75137473],
           [-0.75078643, -0.99923895,  0.93773029,  1.        ,  0.55627469,
             0.47536961],
           [-0.9665554 , -0.58826587,  0.23297648,  0.55627469,  1.        ,
            -0.46666491],
           [ 0.22423734, -0.44069024,  0.75137473,  0.47536961, -0.46666491,
             1.        ]])

    """
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning, stacklevel=2)
    c = cov(x, y, rowvar, dtype=dtype)
    try:
        d = diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c


@set_module('numpy')
def blackman(M):
    """
    Return the Blackman window.

    The Blackman window is a taper formed by using the first three
    terms of a summation of cosines. It was designed to have close to the
    minimal leakage possible.  It is close to optimal, only slightly worse
    than a Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    bartlett, hamming, hanning, kaiser

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \\cos(2\\pi n/M) + 0.08 \\cos(4\\pi n/M)

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the kaiser window.

    References
    ----------
    Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,
    Dover Publications, New York.

    Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
    Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.blackman(12)
    array([-1.38777878e-17,   3.26064346e-02,   1.59903635e-01, # may vary
            4.14397981e-01,   7.36045180e-01,   9.67046769e-01,
            9.67046769e-01,   7.36045180e-01,   4.14397981e-01,
            1.59903635e-01,   3.26064346e-02,  -1.38777878e-17])

    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.blackman(51)
        plt.plot(window)
        plt.title("Blackman window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()  # doctest: +SKIP

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Blackman window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return 0.42 + 0.5 * cos(pi * n / (M - 1)) + 0.08 * cos(2.0 * pi * n / (M - 1))


@set_module('numpy')
def bartlett(M):
    """
    Return the Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : array
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with
        the first and last samples equal to zero.

    See Also
    --------
    blackman, hamming, hanning, kaiser

    Notes
    -----
    The Bartlett window is defined as

    .. math:: w(n) = \\frac{2}{M-1} \\left(
              \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|
              \\right)

    Most references to the Bartlett window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  Note that convolution with this window produces linear
    interpolation.  It is also known as an apodization (which means "removing
    the foot", i.e. smoothing discontinuities at the beginning and end of the
    sampled signal) or tapering function. The Fourier transform of the
    Bartlett window is the product of two sinc functions. Note the excellent
    discussion in Kanasewich [2]_.

    References
    ----------
    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika 37, 1-16, 1950.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 109-110.
    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
           Processing", Prentice-Hall, 1999, pp. 468-471.
    .. [4] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 429.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.bartlett(12)
    array([ 0.        ,  0.18181818,  0.36363636,  0.54545455,  0.72727273, # may vary
            0.90909091,  0.90909091,  0.72727273,  0.54545455,  0.36363636,
            0.18181818,  0.        ])

    Plot the window and its frequency response (requires SciPy and matplotlib).

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.bartlett(51)
        plt.plot(window)
        plt.title("Bartlett window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()
        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Bartlett window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return where(less_equal(n, 0), 1 + n / (M - 1), 1 - n / (M - 1))


@set_module('numpy')
def hanning(M):
    """
    Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    bartlett, blackman, hamming, kaiser

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hanning was named for Julius von Hann, an Austrian meteorologist.
    It is also known as the Cosine Bell. Some authors prefer that it be
    called a Hann window, to help avoid confusion with the very similar
    Hamming window.

    Most references to the Hanning window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> import numpy as np
    >>> np.hanning(12)
    array([0.        , 0.07937323, 0.29229249, 0.57115742, 0.82743037,
           0.97974649, 0.97974649, 0.82743037, 0.57115742, 0.29229249,
           0.07937323, 0.        ])

    Plot the window and its frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.hanning(51)
        plt.plot(window)
        plt.title("Hann window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of the Hann window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return 0.5 + 0.5 * cos(pi * n / (M - 1))


@set_module('numpy')
def hamming(M):
    """
    Return the Hamming window.

    The Hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hanning, kaiser

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
    and is described in Blackman and Tukey. It was recommended for
    smoothing the truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> import numpy as np
    >>> np.hamming(12)
    array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594, # may vary
            0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,
            0.15302337,  0.08      ])

    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.hamming(51)
        plt.plot(window)
        plt.title("Hamming window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Hamming window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return 0.54 + 0.46 * cos(pi * n / (M - 1))


## Code from cephes for i0

_i0A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
    ]

_i0B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
    ]


def _chbevl(x, vals):
    b0 = vals[0]
    b1 = 0.0

    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + vals[i]

    return 0.5 * (b0 - b2)


def _i0_1(x):
    return exp(x) * _chbevl(x / 2.0 - 2, _i0A)


def _i0_2(x):
    return exp(x) * _chbevl(32.0 / x - 2.0, _i0B) / sqrt(x)


def _i0_dispatcher(x):
    return (x,)


@array_function_dispatch(_i0_dispatcher)
def i0(x):
    """
    Modified Bessel function of the first kind, order 0.

    Usually denoted :math:`I_0`.

    Parameters
    ----------
    x : array_like of float
        Argument of the Bessel function.

    Returns
    -------
    out : ndarray, shape = x.shape, dtype = float
        The modified Bessel function evaluated at each of the elements of `x`.

    See Also
    --------
    scipy.special.i0, scipy.special.iv, scipy.special.ive

    Notes
    -----
    The scipy implementation is recommended over this function: it is a
    proper ufunc written in C, and more than an order of magnitude faster.

    We use the algorithm published by Clenshaw [1]_ and referenced by
    Abramowitz and Stegun [2]_, for which the function domain is
    partitioned into the two intervals [0,8] and (8,inf), and Chebyshev
    polynomial expansions are employed in each interval. Relative error on
    the domain [0,30] using IEEE arithmetic is documented [3]_ as having a
    peak of 5.8e-16 with an rms of 1.4e-16 (n = 30000).

    References
    ----------
    .. [1] C. W. Clenshaw, "Chebyshev series for mathematical functions", in
           *National Physical Laboratory Mathematical Tables*, vol. 5, London:
           Her Majesty's Stationery Office, 1962.
    .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical
           Functions*, 10th printing, New York: Dover, 1964, pp. 379.
           https://personal.math.ubc.ca/~cbm/aands/page_379.htm
    .. [3] https://metacpan.org/pod/distribution/Math-Cephes/lib/Math/Cephes.pod#i0:-Modified-Bessel-function-of-order-zero

    Examples
    --------
    >>> import numpy as np
    >>> np.i0(0.)
    array(1.0)
    >>> np.i0([0, 1, 2, 3])
    array([1.        , 1.26606588, 2.2795853 , 4.88079259])

    """
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError("i0 not supported for complex values")
    if x.dtype.kind != 'f':
        x = x.astype(float)
    x = np.abs(x)
    return piecewise(x, [x <= 8.0], [_i0_1, _i0_2])

## End of cephes code for i0


@set_module('numpy')
def kaiser(M, beta):
    """
    Return the Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    beta : float
        Shape parameter for window.

    Returns
    -------
    out : array
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hamming, hanning

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
               \\right)/I_0(\\beta)

    with

    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple
    approximation to the DPSS window based on Bessel functions.  The Kaiser
    window is a very good approximation to the Digital Prolate Spheroidal
    Sequence, or Slepian window, which is the transform which maximizes the
    energy in the main lobe of the window relative to total energy.

    The Kaiser can approximate many other windows by varying the beta
    parameter.

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hanning
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    get returned.

    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.kaiser(12, 14)
     array([7.72686684e-06, 3.46009194e-03, 4.65200189e-02, # may vary
            2.29737120e-01, 5.99885316e-01, 9.45674898e-01,
            9.45674898e-01, 5.99885316e-01, 2.29737120e-01,
            4.65200189e-02, 3.46009194e-03, 7.72686684e-06])


    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.kaiser(51, 14)
        plt.plot(window)
        plt.title("Kaiser window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Kaiser window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.  (Simplified result_type with 0.0
    # strongly typed.  result-type is not/less order sensitive, but that mainly
    # matters for integers anyway.)
    values = np.array([0.0, M, beta])
    M = values[1]
    beta = values[2]

    if M == 1:
        return np.ones(1, dtype=values.dtype)
    n = arange(0, M)
    alpha = (M - 1) / 2.0
    return i0(beta * sqrt(1 - ((n - alpha) / alpha)**2.0)) / i0(beta)


def _sinc_dispatcher(x):
    return (x,)


@array_function_dispatch(_sinc_dispatcher)
def sinc(x):
    r"""
    Return the normalized sinc function.

    The sinc function is equal to :math:`\sin(\pi x)/(\pi x)` for any argument
    :math:`x\ne 0`. ``sinc(0)`` takes the limit value 1, making ``sinc`` not
    only everywhere continuous but also infinitely differentiable.

    .. note::

        Note the normalization factor of ``pi`` used in the definition.
        This is the most commonly used definition in signal processing.
        Use ``sinc(x / np.pi)`` to obtain the unnormalized sinc function
        :math:`\sin(x)/x` that is more common in mathematics.

    Parameters
    ----------
    x : ndarray
        Array (possibly multi-dimensional) of values for which to calculate
        ``sinc(x)``.

    Returns
    -------
    out : ndarray
        ``sinc(x)``, which has the same shape as the input.

    Notes
    -----
    The name sinc is short for "sine cardinal" or "sinus cardinalis".

    The sinc function is used in various signal processing applications,
    including in anti-aliasing, in the construction of a Lanczos resampling
    filter, and in interpolation.

    For bandlimited interpolation of discrete-time signals, the ideal
    interpolation kernel is proportional to the sinc function.

    References
    ----------
    .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
           Resource. https://mathworld.wolfram.com/SincFunction.html
    .. [2] Wikipedia, "Sinc function",
           https://en.wikipedia.org/wiki/Sinc_function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-4, 4, 41)
    >>> np.sinc(x)
     array([-3.89804309e-17,  -4.92362781e-02,  -8.40918587e-02, # may vary
            -8.90384387e-02,  -5.84680802e-02,   3.89804309e-17,
            6.68206631e-02,   1.16434881e-01,   1.26137788e-01,
            8.50444803e-02,  -3.89804309e-17,  -1.03943254e-01,
            -1.89206682e-01,  -2.16236208e-01,  -1.55914881e-01,
            3.89804309e-17,   2.33872321e-01,   5.04551152e-01,
            7.56826729e-01,   9.35489284e-01,   1.00000000e+00,
            9.35489284e-01,   7.56826729e-01,   5.04551152e-01,
            2.33872321e-01,   3.89804309e-17,  -1.55914881e-01,
           -2.16236208e-01,  -1.89206682e-01,  -1.03943254e-01,
           -3.89804309e-17,   8.50444803e-02,   1.26137788e-01,
            1.16434881e-01,   6.68206631e-02,   3.89804309e-17,
            -5.84680802e-02,  -8.90384387e-02,  -8.40918587e-02,
            -4.92362781e-02,  -3.89804309e-17])

    >>> plt.plot(x, np.sinc(x))
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("Sinc Function")
    Text(0.5, 1.0, 'Sinc Function')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("X")
    Text(0.5, 0, 'X')
    >>> plt.show()

    """
    x = np.asanyarray(x)
    x = pi * x
    # Hope that 1e-20 is sufficient for objects...
    eps = np.finfo(x.dtype).eps if x.dtype.kind == "f" else 1e-20
    y = where(x, x, eps)
    return sin(y) / y


def _ureduce(a, func, keepdims=False, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.

    Returns result and a.shape with axis dims set to 1.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.

    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.

    """
    a = np.asanyarray(a)
    axis = kwargs.get('axis')
    out = kwargs.get('out')

    if keepdims is np._NoValue:
        keepdims = False

    nd = a.ndim
    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, nd)

        if keepdims and out is not None:
            index_out = tuple(
                0 if i in axis else slice(None) for i in range(nd))
            kwargs['out'] = out[(Ellipsis, ) + index_out]

        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
    elif keepdims and out is not None:
        index_out = (0, ) * nd
        kwargs['out'] = out[(Ellipsis, ) + index_out]

    r = func(a, **kwargs)

    if out is not None:
        return out

    if keepdims:
        if axis is None:
            index_r = (np.newaxis, ) * nd
        else:
            index_r = tuple(
                np.newaxis if i in axis else slice(None)
                for i in range(nd))
        r = r[(Ellipsis, ) + index_r]

    return r


def _median_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)


@array_function_dispatch(_median_dispatcher)
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default,
        axis=None, will compute the median along a flattened version of
        the array. If a sequence of axes, the array is first flattened
        along the given axes, then the median is computed along the
        resulting flattened axis.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean, percentile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i
    e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
    two middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.median(a)
    np.float64(3.5)
    >>> np.median(a, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.median(a, axis=1)
    array([7.,  2.])
    >>> np.median(a, axis=(0, 1))
    np.float64(3.5)
    >>> m = np.median(a, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.median(a, axis=0, out=m)
    array([6.5,  4.5,  2.5])
    >>> m
    array([6.5,  4.5,  2.5])
    >>> b = a.copy()
    >>> np.median(b, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.median(b, axis=None, overwrite_input=True)
    np.float64(3.5)
    >>> assert not np.all(a==b)

    """
    return _ureduce(a, func=_median, keepdims=keepdims, axis=axis, out=out,
                    overwrite_input=overwrite_input)


def _median(a, axis=None, out=None, overwrite_input=False):
    # can't be reasonably be implemented in terms of percentile as we have to
    # call mean to not break astropy
    a = np.asanyarray(a)

    # Set the partition indexes
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    # We have to check for NaNs (as of writing 'M' doesn't actually work).
    supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
    if supports_nans:
        kth.append(-1)

    if overwrite_input:
        if axis is None:
            part = a.ravel()
            part.partition(kth)
        else:
            a.partition(kth, axis=axis)
            part = a
    else:
        part = partition(a, kth, axis=axis)

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)
    indexer = tuple(indexer)

    # Use mean in both odd and even case to coerce data type,
    # using out array if needed.
    rout = mean(part[indexer], axis=axis, out=out)
    if supports_nans and sz > 0:
        # If nans are possible, warn and replace by nans like mean would.
        rout = np.lib._utils_impl._median_nancheck(part, rout, axis)

    return rout


def _percentile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                           method=None, keepdims=None, *, weights=None,
                           interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_percentile_dispatcher)
def percentile(a,
               q,
               axis=None,
               out=None,
               overwrite_input=False,
               method="linear",
               keepdims=False,
               *,
               weights=None,
               interpolation=None):
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like of real numbers
        Input array or object that can be converted to an array.
    q : array_like of float
        Percentage or sequence of percentages for the percentiles to compute.
        Values must be between 0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

     weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the percentile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.
        See the notes for more details.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    median : equivalent to ``percentile(..., 50)``
    nanpercentile
    quantile : equivalent to percentile, except q in the range [0, 1].

    Notes
    -----
    The behavior of `numpy.percentile` with percentage `q` is
    that of `numpy.quantile` with argument ``q/100``.
    For more information, please see `numpy.quantile`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.percentile(a, 50)
    3.5
    >>> np.percentile(a, 50, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.percentile(a, 50, axis=1)
    array([7.,  2.])
    >>> np.percentile(a, 50, axis=1, keepdims=True)
    array([[7.],
           [2.]])

    >>> m = np.percentile(a, 50, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.percentile(a, 50, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])

    >>> b = a.copy()
    >>> np.percentile(b, 50, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a == b)

    The different methods can be visualized graphically:

    .. plot::

        import matplotlib.pyplot as plt

        a = np.arange(4)
        p = np.linspace(0, 100, 6001)
        ax = plt.gca()
        lines = [
            ('linear', '-', 'C0'),
            ('inverted_cdf', ':', 'C1'),
            # Almost the same as `inverted_cdf`:
            ('averaged_inverted_cdf', '-.', 'C1'),
            ('closest_observation', ':', 'C2'),
            ('interpolated_inverted_cdf', '--', 'C1'),
            ('hazen', '--', 'C3'),
            ('weibull', '-.', 'C4'),
            ('median_unbiased', '--', 'C5'),
            ('normal_unbiased', '-.', 'C6'),
            ]
        for method, style, color in lines:
            ax.plot(
                p, np.percentile(a, p, method=method),
                label=method, linestyle=style, color=color)
        ax.set(
            title='Percentiles for different methods and data: ' + str(a),
            xlabel='Percentile',
            ylabel='Estimated percentile value',
            yticks=a)
        ax.legend(bbox_to_anchor=(1.03, 1))
        plt.tight_layout()
        plt.show()

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "percentile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float)
    # by making the divisor have the dtype of the data array.
    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100, out=...)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _quantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                         method=None, keepdims=None, *, weights=None,
                         interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_quantile_dispatcher)
def quantile(a,
             q,
             axis=None,
             out=None,
             overwrite_input=False,
             method="linear",
             keepdims=False,
             *,
             weights=None,
             interpolation=None):
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : array_like of real numbers
        Input array or object that can be converted to an array.
    q : array_like of float
        Probability or sequence of probabilities of the quantiles to compute.
        Values must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is
        to compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and buffer length as the expected output, but the
        type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  There are many different methods, some unique to NumPy.
        The recommended options, numbered as they appear in [1]_, are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous. For backward compatibility
        with previous versions of NumPy, the following discontinuous variations
        of the default 'linear' (7.) option are available:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        See Notes for details.

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the quantile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.
        See the notes for more details.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single probability and `axis=None`, then the result
        is a scalar. If multiple probability levels are given, first axis
        of the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    percentile : equivalent to quantile, but with q in the range [0, 100].
    median : equivalent to ``quantile(..., 0.5)``
    nanquantile

    Notes
    -----
    Given a sample `a` from an underlying distribution, `quantile` provides a
    nonparametric estimate of the inverse cumulative distribution function.

    By default, this is done by interpolating between adjacent elements in
    ``y``, a sorted copy of `a`::

        (1-g)*y[j] + g*y[j+1]

    where the index ``j`` and coefficient ``g`` are the integral and
    fractional components of ``q * (n-1)``, and ``n`` is the number of
    elements in the sample.

    This is a special case of Equation 1 of H&F [1]_. More generally,

    - ``j = (q*n + m - 1) // 1``, and
    - ``g = (q*n + m - 1) % 1``,

    where ``m`` may be defined according to several different conventions.
    The preferred convention may be selected using the ``method`` parameter:

    =============================== =============== ===============
    ``method``                      number in H&F   ``m``
    =============================== =============== ===============
    ``interpolated_inverted_cdf``   4               ``0``
    ``hazen``                       5               ``1/2``
    ``weibull``                     6               ``q``
    ``linear`` (default)            7               ``1 - q``
    ``median_unbiased``             8               ``q/3 + 1/3``
    ``normal_unbiased``             9               ``q/4 + 3/8``
    =============================== =============== ===============

    Note that indices ``j`` and ``j + 1`` are clipped to the range ``0`` to
    ``n - 1`` when the results of the formula would be outside the allowed
    range of non-negative indices. The ``- 1`` in the formulas for ``j`` and
    ``g`` accounts for Python's 0-based indexing.

    The table above includes only the estimators from H&F that are continuous
    functions of probability `q` (estimators 4-9). NumPy also provides the
    three discontinuous estimators from H&F (estimators 1-3), where ``j`` is
    defined as above, ``m`` is defined as follows, and ``g`` is a function
    of the real-valued ``index = q*n + m - 1`` and ``j``.

    1. ``inverted_cdf``: ``m = 0`` and ``g = int(index - j > 0)``
    2. ``averaged_inverted_cdf``: ``m = 0`` and
       ``g = (1 + int(index - j > 0)) / 2``
    3. ``closest_observation``: ``m = -1/2`` and
       ``g = 1 - int((index == j) & (j%2 == 1))``

    For backward compatibility with previous versions of NumPy, `quantile`
    provides four additional discontinuous estimators. Like
    ``method='linear'``, all have ``m = 1 - q`` so that ``j = q*(n-1) // 1``,
    but ``g`` is defined as follows.

    - ``lower``: ``g = 0``
    - ``midpoint``: ``g = 0.5``
    - ``higher``: ``g = 1``
    - ``nearest``: ``g = (q*(n-1) % 1) > 0.5``

    **Weighted quantiles:**
    More formally, the quantile at probability level :math:`q` of a cumulative
    distribution function :math:`F(y)=P(Y \\leq y)` with probability measure
    :math:`P` is defined as any number :math:`x` that fulfills the
    *coverage conditions*

    .. math:: P(Y < x) \\leq q \\quad\\text{and}\\quad P(Y \\leq x) \\geq q

    with random variable :math:`Y\\sim P`.
    Sample quantiles, the result of `quantile`, provide nonparametric
    estimation of the underlying population counterparts, represented by the
    unknown :math:`F`, given a data vector `a` of length ``n``.

    Some of the estimators above arise when one considers :math:`F` as the
    empirical distribution function of the data, i.e.
    :math:`F(y) = \\frac{1}{n} \\sum_i 1_{a_i \\leq y}`.
    Then, different methods correspond to different choices of :math:`x` that
    fulfill the above coverage conditions. Methods that follow this approach
    are ``inverted_cdf`` and ``averaged_inverted_cdf``.

    For weighted quantiles, the coverage conditions still hold. The
    empirical cumulative distribution is simply replaced by its weighted
    version, i.e.
    :math:`P(Y \\leq t) = \\frac{1}{\\sum_i w_i} \\sum_i w_i 1_{x_i \\leq t}`.
    Only ``method="inverted_cdf"`` supports weights.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.quantile(a, 0.5)
    3.5
    >>> np.quantile(a, 0.5, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.quantile(a, 0.5, axis=1)
    array([7.,  2.])
    >>> np.quantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.quantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.quantile(a, 0.5, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    >>> np.quantile(b, 0.5, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a == b)

    See also `numpy.percentile` for a visualization of most methods.

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "quantile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float).
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)

    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _quantile_unchecked(a,
                        q,
                        axis=None,
                        out=None,
                        overwrite_input=False,
                        method="linear",
                        keepdims=False,
                        weights=None):
    """Assumes that q is in [0, 1], and is an ndarray"""
    return _ureduce(a,
                    func=_quantile_ureduce_func,
                    q=q,
                    weights=weights,
                    keepdims=keepdims,
                    axis=axis,
                    out=out,
                    overwrite_input=overwrite_input,
                    method=method)


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    elif not (q.min() >= 0 and q.max() <= 1):
        return False
    return True


def _check_interpolation_as_method(method, interpolation, fname):
    # Deprecated NumPy 1.22, 2021-11-08
    warnings.warn(
        f"the `interpolation=` argument to {fname} was renamed to "
        "`method=`, which has additional options.\n"
        "Users of the modes 'nearest', 'lower', 'higher', or "
        "'midpoint' are encouraged to review the method they used. "
        "(Deprecated NumPy 1.22)",
        DeprecationWarning, stacklevel=4)
    if method != "linear":
        # sanity check, we assume this basically never happens
        raise TypeError(
            "You shall not pass both `method` and `interpolation`!\n"
            "(`interpolation` is Deprecated in favor of `method`)")
    return interpolation


def _compute_virtual_index(n, quantiles, alpha: float, beta: float):
    """
    Compute the floating point indexes of an array for the linear
    interpolation of quantiles.
    n : array_like
        The sample sizes.
    quantiles : array_like
        The quantiles values.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.

    alpha and beta values depend on the chosen method
    (see quantile documentation)

    Reference:
    Hyndman&Fan paper "Sample Quantiles in Statistical Packages",
    DOI: 10.1080/00031305.1996.10473566
    """
    return n * quantiles + (
            alpha + quantiles * (1 - alpha - beta)
    ) - 1


def _get_gamma(virtual_indexes, previous_indexes, method):
    """
    Compute gamma (a.k.a 'm' or 'weight') for the linear interpolation
    of quantiles.

    virtual_indexes : array_like
        The indexes where the percentile is supposed to be found in the sorted
        sample.
    previous_indexes : array_like
        The floor values of virtual_indexes.
    interpolation : dict
        The interpolation method chosen, which may have a specific rule
        modifying gamma.

    gamma is usually the fractional part of virtual_indexes but can be modified
    by the interpolation method.
    """
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    gamma = method["fix_gamma"](gamma, virtual_indexes)
    # Ensure both that we have an array, and that we keep the dtype
    # (which may have been matched to the input array).
    return np.asanyarray(gamma, dtype=virtual_indexes.dtype)


def _lerp(a, b, t, out=None):
    """
    Compute the linear interpolation weighted by gamma on each point of
    two same shape array.

    a : array_like
        Left bound.
    b : array_like
        Right bound.
    t : array_like
        The interpolation weight.
    out : array_like
        Output array.
    """
    diff_b_a = subtract(b, a)
    # asanyarray is a stop-gap until gh-13105
    lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t >= 0.5,
             casting='unsafe', dtype=type(lerp_interpolation.dtype))
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    return lerp_interpolation


def _get_gamma_mask(shape, default_value, conditioned_value, where):
    out = np.full(shape, default_value)
    np.copyto(out, conditioned_value, where=where, casting="unsafe")
    return out


def _discrete_interpolation_to_boundaries(index, gamma_condition_fun):
    previous = np.floor(index)
    next = previous + 1
    gamma = index - previous
    res = _get_gamma_mask(shape=index.shape,
                          default_value=next,
                          conditioned_value=previous,
                          where=gamma_condition_fun(gamma, index)
                          ).astype(np.intp)
    # Some methods can lead to out-of-bound integers, clip them:
    res[res < 0] = 0
    return res


def _closest_observation(n, quantiles):
    # "choose the nearest even order statistic at g=0" (H&F (1996) pp. 362).
    # Order is 1-based so for zero-based indexing round to nearest odd index.
    gamma_fun = lambda gamma, index: (gamma == 0) & (np.floor(index) % 2 == 1)
    return _discrete_interpolation_to_boundaries((n * quantiles) - 1 - 0.5,
                                                 gamma_fun)


def _inverted_cdf(n, quantiles):
    gamma_fun = lambda gamma, _: (gamma == 0)
    return _discrete_interpolation_to_boundaries((n * quantiles) - 1,
                                                 gamma_fun)


def _quantile_ureduce_func(
        a: np.array,
        q: np.array,
        weights: np.array,
        axis: int | None = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
) -> np.array:
    if q.ndim > 2:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
            wgt = None if weights is None else weights.ravel()
        else:
            arr = a
            wgt = weights
    elif axis is None:
        axis = 0
        arr = a.flatten()
        wgt = None if weights is None else weights.flatten()
    else:
        arr = a.copy()
        wgt = weights
    result = _quantile(arr,
                       quantiles=q,
                       axis=axis,
                       method=method,
                       out=out,
                       weights=wgt)
    return result


def _get_indexes(arr, virtual_indexes, valid_values_count):
    """
    Get the valid indexes of arr neighbouring virtual_indexes.
    Note
    This is a companion function to linear interpolation of
    Quantiles

    Returns
    -------
    (previous_indexes, next_indexes): Tuple
        A Tuple of virtual_indexes neighbouring indexes
    """
    previous_indexes = np.asanyarray(np.floor(virtual_indexes))
    next_indexes = np.asanyarray(previous_indexes + 1)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # When indexes is above max index, take the max value of the array
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    # When indexes is below min index, take the min value of the array
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if np.issubdtype(arr.dtype, np.inexact):
        # After the sort, slices having NaNs will have for last element a NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return previous_indexes, next_indexes


def _quantile(
        arr: np.array,
        quantiles: np.array,
        axis: int = -1,
        method="linear",
        out=None,
        weights=None,
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    It computes the quantiles of the array for the given axis.
    A linear interpolation is performed based on the `interpolation`.

    By default, the method is "linear" where alpha == beta == 1 which
    performs the 7th method of Hyndman&Fan.
    With "median_unbiased" we get alpha == beta == 1/3
    thus the 8th method of Hyndman&Fan.
    """
    # --- Setup
    arr = np.asanyarray(arr)
    values_count = arr.shape[axis]
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    if axis != 0:  # But moveaxis is slow, so only call it if necessary.
        arr = np.moveaxis(arr, axis, destination=0)
    supports_nans = (
        np.issubdtype(arr.dtype, np.inexact) or arr.dtype.kind in 'Mm'
    )

    if weights is None:
        # --- Computation of indexes
        # Index where to find the value in the sorted array.
        # Virtual because it is a floating point value, not an valid index.
        # The nearest neighbours are used for interpolation
        try:
            method_props = _QuantileMethods[method]
        except KeyError:
            raise ValueError(
                f"{method!r} is not a valid method. Use one of: "
                f"{_QuantileMethods.keys()}") from None
        virtual_indexes = method_props["get_virtual_index"](values_count,
                                                            quantiles)
        virtual_indexes = np.asanyarray(virtual_indexes)

        if method_props["fix_gamma"] is None:
            supports_integers = True
        else:
            int_virtual_indices = np.issubdtype(virtual_indexes.dtype,
                                                np.integer)
            supports_integers = method == 'linear' and int_virtual_indices

        if supports_integers:
            # No interpolation needed, take the points along axis
            if supports_nans:
                # may contain nan, which would sort to the end
                arr.partition(
                    concatenate((virtual_indexes.ravel(), [-1])), axis=0,
                )
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                # cannot contain nan
                arr.partition(virtual_indexes.ravel(), axis=0)
                slices_having_nans = np.array(False, dtype=bool)
            result = take(arr, virtual_indexes, axis=0, out=out)
        else:
            previous_indexes, next_indexes = _get_indexes(arr,
                                                          virtual_indexes,
                                                          values_count)
            # --- Sorting
            arr.partition(
                np.unique(np.concatenate(([0, -1],
                                          previous_indexes.ravel(),
                                          next_indexes.ravel(),
                                          ))),
                axis=0)
            if supports_nans:
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                slices_having_nans = None
            # --- Get values from indexes
            previous = arr[previous_indexes]
            next = arr[next_indexes]
            # --- Linear interpolation
            gamma = _get_gamma(virtual_indexes, previous_indexes, method_props)
            result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
            gamma = gamma.reshape(result_shape)
            result = _lerp(previous,
                        next,
                        gamma,
                        out=out)
    else:
        # Weighted case
        # This implements method="inverted_cdf", the only supported weighted
        # method, which needs to sort anyway.
        weights = np.asanyarray(weights)
        if axis != 0:
            weights = np.moveaxis(weights, axis, destination=0)
        index_array = np.argsort(arr, axis=0, kind="stable")

        # arr = arr[index_array, ...]  # but this adds trailing dimensions of
        # 1.
        arr = np.take_along_axis(arr, index_array, axis=0)
        if weights.shape == arr.shape:
            weights = np.take_along_axis(weights, index_array, axis=0)
        else:
            # weights is 1d
            weights = weights.reshape(-1)[index_array, ...]

        if supports_nans:
            # may contain nan, which would sort to the end
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            # cannot contain nan
            slices_having_nans = np.array(False, dtype=bool)

        # We use the weights to calculate the empirical cumulative
        # distribution function cdf
        cdf = weights.cumsum(axis=0, dtype=np.float64)
        cdf /= cdf[-1, ...]  # normalization to 1
        # Search index i such that
        #   sum(weights[j], j=0..i-1) < quantile <= sum(weights[j], j=0..i)
        # is then equivalent to
        #   cdf[i-1] < quantile <= cdf[i]
        # Unfortunately, searchsorted only accepts 1-d arrays as first
        # argument, so we will need to iterate over dimensions.

        # Without the following cast, searchsorted can return surprising
        # results, e.g.
        #   np.searchsorted(np.array([0.2, 0.4, 0.6, 0.8, 1.]),
        #                   np.array(0.4, dtype=np.float32), side="left")
        # returns 2 instead of 1 because 0.4 is not binary representable.
        if quantiles.dtype.kind == "f":
            cdf = cdf.astype(quantiles.dtype)
        # Weights must be non-negative, so we might have zero weights at the
        # beginning leading to some leading zeros in cdf. The call to
        # np.searchsorted for quantiles=0 will then pick the first element,
        # but should pick the first one larger than zero. We
        # therefore simply set 0 values in cdf to -1.
        if np.any(cdf[0, ...] == 0):
            cdf[cdf == 0] = -1

        def find_cdf_1d(arr, cdf):
            indices = np.searchsorted(cdf, quantiles, side="left")
            # We might have reached the maximum with i = len(arr), e.g. for
            # quantiles = 1, and need to cut it to len(arr) - 1.
            indices = minimum(indices, values_count - 1)
            result = take(arr, indices, axis=0)
            return result

        r_shape = arr.shape[1:]
        if quantiles.ndim > 0:
            r_shape = quantiles.shape + r_shape
        if out is None:
            result = np.empty_like(arr, shape=r_shape)
        else:
            if out.shape != r_shape:
                msg = (f"Wrong shape of argument 'out', shape={r_shape} is "
                       f"required; got shape={out.shape}.")
                raise ValueError(msg)
            result = out

        # See apply_along_axis, which we do for axis=0. Note that Ni = (,)
        # always, so we remove it here.
        Nk = arr.shape[1:]
        for kk in np.ndindex(Nk):
            result[(...,) + kk] = find_cdf_1d(
                arr[np.s_[:, ] + kk], cdf[np.s_[:, ] + kk]
            )

        # Make result the same as in unweighted inverted_cdf.
        if result.shape == () and result.dtype == np.dtype("O"):
            result = result.item()

    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            # can't write to a scalar, but indexing will be correct
            result = arr[-1]
        else:
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    return result


def _trapezoid_dispatcher(y, x=None, dx=None, axis=None):
    return (y, x)


@array_function_dispatch(_trapezoid_dispatcher)
def trapezoid(y, x=None, dx=1.0, axis=-1):
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    .. versionadded:: 2.0.0

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapezoid : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    See Also
    --------
    sum, cumsum

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.


    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    >>> import numpy as np

    Use the trapezoidal rule on evenly spaced points:

    >>> np.trapezoid([1, 2, 3])
    4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> np.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> np.trapezoid([1, 2, 3], dx=2)
    8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> np.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> np.trapezoid(y, x)
    0.33340274885464394

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> np.trapezoid(np.cos(theta), x=np.sin(theta))
    3.141571941375841

    ``np.trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> np.trapezoid(a, axis=1)
    array([2.,  8.])
    """

    y = asanyarray(y)
    if x is None:
        d = dx
    else:
        x = asanyarray(x)
        if x.ndim == 1:
            d = diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret


@set_module('numpy')
def trapz(y, x=None, dx=1.0, axis=-1):
    """
    `trapz` is deprecated in NumPy 2.0.

    Please use `trapezoid` instead, or one of the numerical integration
    functions in `scipy.integrate`.
    """
    # Deprecated in NumPy 2.0, 2023-08-18
    warnings.warn(
        "`trapz` is deprecated. Use `trapezoid` instead, or one of the "
        "numerical integration functions in `scipy.integrate`.",
        DeprecationWarning,
        stacklevel=2
    )
    return trapezoid(y, x=x, dx=dx, axis=axis)


def _meshgrid_dispatcher(*xi, copy=None, sparse=None, indexing=None):
    return xi


# Based on scitools meshgrid
@array_function_dispatch(_meshgrid_dispatcher)
def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """
    Return a tuple of coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
        If True the shape of the returned coordinate array for dimension *i*
        is reduced from ``(N1, ..., Ni, ... Nn)`` to
        ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
        intended to be used with :ref:`basics.broadcasting`.  When all
        coordinates are used in an expression, broadcasting still leads to a
        fully-dimensonal result array.

        Default is False.

    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.

    Returns
    -------
    X1, X2,..., XN : tuple of ndarrays
        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``,
        returns ``(N1, N2, N3,..., Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,..., Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::

        xv, yv = np.meshgrid(x, y, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = np.meshgrid(x, y, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

    See Also
    --------
    mgrid : Construct a multi-dimensional "meshgrid" using indexing notation.
    ogrid : Construct an open multi-dimensional "meshgrid" using indexing
            notation.
    :ref:`how-to-index`

    Examples
    --------
    >>> import numpy as np
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = np.meshgrid(x, y)
    >>> xv
    array([[0. , 0.5, 1. ],
           [0. , 0.5, 1. ]])
    >>> yv
    array([[0.,  0.,  0.],
           [1.,  1.,  1.]])

    The result of `meshgrid` is a coordinate grid:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(xv, yv, marker='o', color='k', linestyle='none')
    >>> plt.show()

    You can create sparse output arrays to save memory and computation time.

    >>> xv, yv = np.meshgrid(x, y, sparse=True)
    >>> xv
    array([[0. ,  0.5,  1. ]])
    >>> yv
    array([[0.],
           [1.]])

    `meshgrid` is very useful to evaluate functions on a grid. If the
    function depends on all coordinates, both dense and sparse outputs can be
    used.

    >>> x = np.linspace(-5, 5, 101)
    >>> y = np.linspace(-5, 5, 101)
    >>> # full coordinate arrays
    >>> xx, yy = np.meshgrid(x, y)
    >>> zz = np.sqrt(xx**2 + yy**2)
    >>> xx.shape, yy.shape, zz.shape
    ((101, 101), (101, 101), (101, 101))
    >>> # sparse coordinate arrays
    >>> xs, ys = np.meshgrid(x, y, sparse=True)
    >>> zs = np.sqrt(xs**2 + ys**2)
    >>> xs.shape, ys.shape, zs.shape
    ((1, 101), (101, 1), (101, 101))
    >>> np.array_equal(zz, zs)
    True

    >>> h = plt.contourf(x, y, zs)
    >>> plt.axis('scaled')
    >>> plt.colorbar()
    >>> plt.show()
    """
    ndim = len(xi)

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
              for i, x in enumerate(xi)]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0].shape = (1, -1) + s0[2:]
        output[1].shape = (-1, 1) + s0[2:]

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = np.broadcast_arrays(*output, subok=True)

    if copy:
        output = tuple(x.copy() for x in output)

    return output


def _delete_dispatcher(arr, obj, axis=None):
    return (arr, obj)


@array_function_dispatch(_delete_dispatcher)
def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    `arr[obj]`.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int, array-like of ints or bools
        Indicate indices of sub-arrays to remove along the specified axis.

        .. versionchanged:: 1.19.0
            Boolean indices are now treated as a mask of elements to remove,
            rather than being cast to the integers 0 and 1.

    axis : int, optional
        The axis along which to delete the subarray defined by `obj`.
        If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : ndarray
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    append : Append elements at the end of an array.

    Notes
    -----
    Often it is preferable to use a boolean mask. For example:

    >>> arr = np.arange(12) + 1
    >>> mask = np.ones(len(arr), dtype=bool)
    >>> mask[[0,2,4]] = False
    >>> result = arr[mask,...]

    Is equivalent to ``np.delete(arr, [0,2,4], axis=0)``, but allows further
    use of `mask`.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> np.delete(arr, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])

    >>> np.delete(arr, np.s_[::2], 1)
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> np.delete(arr, [1,3,5], None)
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])

    """
    conv = _array_converter(arr)
    arr, = conv.as_arrays(subok=False)

    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        # needed for np.matrix, which is still not 1d after being ravelled
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)

    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        start, stop, step = obj.indices(N)
        xr = range(start, stop, step)
        numtodel = len(xr)

        if numtodel <= 0:
            return conv.wrap(arr.copy(order=arrorder), to_scalar=False)

        # Invert if step is negative:
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1

        newshape[axis] -= numtodel
        new = empty(newshape, arr.dtype, arrorder)
        # copy initial chunk
        if start == 0:
            pass
        else:
            slobj[axis] = slice(None, start)
            new[tuple(slobj)] = arr[tuple(slobj)]
        # copy end chunk
        if stop == N:
            pass
        else:
            slobj[axis] = slice(stop - numtodel, None)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(stop, None)
            new[tuple(slobj)] = arr[tuple(slobj2)]
        # copy middle pieces
        if step == 1:
            pass
        else:  # use array indexing.
            keep = ones(stop - start, dtype=bool)
            keep[:stop - start:step] = False
            slobj[axis] = slice(start, stop - numtodel)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(start, stop)
            arr = arr[tuple(slobj2)]
            slobj2[axis] = keep
            new[tuple(slobj)] = arr[tuple(slobj2)]

        return conv.wrap(new, to_scalar=False)

    if isinstance(obj, (int, integer)) and not isinstance(obj, bool):
        single_value = True
    else:
        single_value = False
        _obj = obj
        obj = np.asarray(obj)
        # `size == 0` to allow empty lists similar to indexing, but (as there)
        # is really too generic:
        if obj.size == 0 and not isinstance(_obj, np.ndarray):
            obj = obj.astype(intp)
        elif obj.size == 1 and obj.dtype.kind in "ui":
            # For a size 1 integer array we can use the single-value path
            # (most dtypes, except boolean, should just fail later).
            obj = obj.item()
            single_value = True

    if single_value:
        # optimization for a single value
        if (obj < -N or obj >= N):
            raise IndexError(
                f"index {obj} is out of bounds for axis {axis} with "
                f"size {N}")
        if (obj < 0):
            obj += N
        newshape[axis] -= 1
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, obj)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(obj + 1, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
    else:
        if obj.dtype == bool:
            if obj.shape != (N,):
                raise ValueError('boolean array argument obj to delete '
                                 'must be one dimensional and match the axis '
                                 f'length of {N}')

            # optimization, the other branch is slower
            keep = ~obj
        else:
            keep = ones(N, dtype=bool)
            keep[obj,] = False

        slobj[axis] = keep
        new = arr[tuple(slobj)]

    return conv.wrap(new, to_scalar=False)


def _insert_dispatcher(arr, obj, values, axis=None):
    return (arr, obj, values)


@array_function_dispatch(_insert_dispatcher)
def insert(arr, obj, values, axis=None):
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int, array-like of ints or bools
        Object that defines the index or indices before which `values` is
        inserted.

        .. versionchanged:: 2.1.2
            Boolean indices are now treated as a mask of elements to insert,
            rather than being cast to the integers 0 and 1.

        Support for multiple insertions when `obj` is a single scalar or a
        sequence with one element (similar to calling insert multiple
        times).
    values : array_like
        Values to insert into `arr`. If the type of `values` is different
        from that of `arr`, `values` is converted to the type of `arr`.
        `values` should be shaped so that ``arr[...,obj,...] = values``
        is legal.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then `arr`
        is flattened first.

    Returns
    -------
    out : ndarray
        A copy of `arr` with `values` inserted.  Note that `insert`
        does not occur in-place: a new array is returned. If
        `axis` is None, `out` is a flattened array.

    See Also
    --------
    append : Append elements at the end of an array.
    concatenate : Join a sequence of arrays along an existing axis.
    delete : Delete elements from an array.

    Notes
    -----
    Note that for higher dimensional inserts ``obj=0`` behaves very different
    from ``obj=[0]`` just like ``arr[:,0,:] = values`` is different from
    ``arr[:,[0],:] = values``. This is because of the difference between basic
    and advanced :ref:`indexing <basics.indexing>`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(6).reshape(3, 2)
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> np.insert(a, 1, 6)
    array([0, 6, 1, 2, 3, 4, 5])
    >>> np.insert(a, 1, 6, axis=1)
    array([[0, 6, 1],
           [2, 6, 3],
           [4, 6, 5]])

    Difference between sequence and scalars,
    showing how ``obj=[1]`` behaves different from ``obj=1``:

    >>> np.insert(a, [1], [[7],[8],[9]], axis=1)
    array([[0, 7, 1],
           [2, 8, 3],
           [4, 9, 5]])
    >>> np.insert(a, 1, [[7],[8],[9]], axis=1)
    array([[0, 7, 8, 9, 1],
           [2, 7, 8, 9, 3],
           [4, 7, 8, 9, 5]])
    >>> np.array_equal(np.insert(a, 1, [7, 8, 9], axis=1),
    ...                np.insert(a, [1], [[7],[8],[9]], axis=1))
    True

    >>> b = a.flatten()
    >>> b
    array([0, 1, 2, 3, 4, 5])
    >>> np.insert(b, [2, 2], [6, 7])
    array([0, 1, 6, 7, 2, 3, 4, 5])

    >>> np.insert(b, slice(2, 4), [7, 8])
    array([0, 1, 7, 2, 8, 3, 4, 5])

    >>> np.insert(b, [2, 2], [7.13, False]) # type casting
    array([0, 1, 7, 0, 2, 3, 4, 5])

    >>> x = np.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> np.insert(x, idx, 999, axis=1)
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])

    """
    conv = _array_converter(arr)
    arr, = conv.as_arrays(subok=False)

    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        # needed for np.matrix, which is still not 1d after being ravelled
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)
    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        # turn it into a range object
        indices = arange(*obj.indices(N), dtype=intp)
    else:
        # need to copy obj, because indices will be changed in-place
        indices = np.array(obj)
        if indices.dtype == bool:
            if obj.ndim != 1:
                raise ValueError('boolean array argument obj to insert '
                                'must be one dimensional')
            indices = np.flatnonzero(obj)
        elif indices.ndim > 1:
            raise ValueError(
                "index array argument obj to insert must be one dimensional "
                "or scalar")
    if indices.size == 1:
        index = indices.item()
        if index < -N or index > N:
            raise IndexError(f"index {obj} is out of bounds for axis {axis} "
                             f"with size {N}")
        if (index < 0):
            index += N

        # There are some object array corner cases here, but we cannot avoid
        # that:
        values = array(values, copy=None, ndmin=arr.ndim, dtype=arr.dtype)
        if indices.ndim == 0:
            # broadcasting is very different here, since a[:,0,:] = ... behaves
            # very different from a[:,[0],:] = ...! This changes values so that
            # it works likes the second case. (here a[:,0:1,:])
            values = np.moveaxis(values, 0, axis)
        numnew = values.shape[axis]
        newshape[axis] += numnew
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index + numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index + numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]

        return conv.wrap(new, to_scalar=False)

    elif indices.size == 0 and not isinstance(obj, np.ndarray):
        # Can safely cast the empty list to intp
        indices = indices.astype(intp)

    indices[indices < 0] += N

    numnew = len(indices)
    order = indices.argsort(kind='mergesort')   # stable sort
    indices[order] += np.arange(numnew)

    newshape[axis] += numnew
    old_mask = ones(newshape[axis], dtype=bool)
    old_mask[indices] = False

    new = empty(newshape, arr.dtype, arrorder)
    slobj2 = [slice(None)] * ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask
    new[tuple(slobj)] = values
    new[tuple(slobj2)] = arr

    return conv.wrap(new, to_scalar=False)


def _append_dispatcher(arr, values, axis=None):
    return (arr, values)


@array_function_dispatch(_append_dispatcher)
def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.

    Returns
    -------
    append : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that
        `append` does not occur in-place: a new array is allocated and
        filled.  If `axis` is None, `out` is a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    delete : Delete elements from an array.

    Examples
    --------
    >>> import numpy as np
    >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, ..., 7, 8, 9])

    When `axis` is specified, `values` must have the correct shape.

    >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
    Traceback (most recent call last):
        ...
    ValueError: all the input arrays must have same number of dimensions, but
    the array at index 0 has 2 dimension(s) and the array at index 1 has 1
    dimension(s)

    >>> a = np.array([1, 2], dtype=int)
    >>> c = np.append(a, [])
    >>> c
    array([1., 2.])
    >>> c.dtype
    float64

    Default dtype for empty ndarrays is `float64` thus making the output of dtype
    `float64` when appended with dtype `int64`

    """
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(values)
        axis = arr.ndim - 1
    return concatenate((arr, values), axis=axis)


def _digitize_dispatcher(x, bins, right=None):
    return (x, bins)


@array_function_dispatch(_digitize_dispatcher)
def digitize(x, bins, right=False):
    """
    Return the indices of the bins to which each value in input array belongs.

    =========  =============  ============================
    `right`    order of bins  returned index `i` satisfies
    =========  =============  ============================
    ``False``  increasing     ``bins[i-1] <= x < bins[i]``
    ``True``   increasing     ``bins[i-1] < x <= bins[i]``
    ``False``  decreasing     ``bins[i-1] > x >= bins[i]``
    ``True``   decreasing     ``bins[i-1] >= x > bins[i]``
    =========  =============  ============================

    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is
    returned as appropriate.

    Parameters
    ----------
    x : array_like
        Input array to be binned. Prior to NumPy 1.10.0, this array had to
        be 1-dimensional, but can now have any shape.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is open in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    indices : ndarray of ints
        Output array of indices, of same shape as `x`.

    Raises
    ------
    ValueError
        If `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    See Also
    --------
    bincount, histogram, unique, searchsorted

    Notes
    -----
    If values in `x` are such that they fall outside the bin range,
    attempting to index `bins` with the indices that `digitize` returns
    will result in an IndexError.

    .. versionadded:: 1.10.0

    `numpy.digitize` is  implemented in terms of `numpy.searchsorted`.
    This means that a binary search is used to bin the values, which scales
    much better for larger number of bins than the previous linear search.
    It also removes the requirement for the input array to be 1-dimensional.

    For monotonically *increasing* `bins`, the following are equivalent::

        np.digitize(x, bins, right=True)
        np.searchsorted(bins, x, side='left')

    Note that as the order of the arguments are reversed, the side must be too.
    The `searchsorted` call is marginally faster, as it does not do any
    monotonicity checks. Perhaps more importantly, it supports all dtypes.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = np.digitize(x, bins)
    >>> inds
    array([1, 4, 3, 2])
    >>> for n in range(x.size):
    ...   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])
    ...
    0.0 <= 0.2 < 1.0
    4.0 <= 6.4 < 10.0
    2.5 <= 3.0 < 4.0
    1.0 <= 1.6 < 2.5

    >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
    >>> bins = np.array([0, 5, 10, 15, 20])
    >>> np.digitize(x,bins,right=True)
    array([1, 2, 3, 4, 4])
    >>> np.digitize(x,bins,right=False)
    array([1, 3, 3, 4, 5])
    """
    x = _nx.asarray(x)
    bins = _nx.asarray(bins)

    # here for compatibility, searchsorted below is happy to take this
    if np.issubdtype(x.dtype, _nx.complexfloating):
        raise TypeError("x may not be complex")

    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")

    # this is backwards because the arguments below are swapped
    side = 'left' if right else 'right'
    if mono == -1:
        # reverse the bins, and invert the results
        return len(bins) - _nx.searchsorted(bins[::-1], x, side=side)
    else:
        return _nx.searchsorted(bins, x, side=side)

# === atelier-kyo-manager/app\utils\ai_background_remover.py ===
# ──────────────────────────────────────────────────────────────
#  existing_image_processor.py  ― 色ズレ・失敗検知・自動停止付き完全版
# ----------------------------------------------------------------
#  必要ライブラリ
#     pip install rembg pillow opencv-python numpy pyoxipng
# ----------------------------------------------------------------
import io, sys, zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

try:
    import pyoxipng           # Rust 圧縮ライブラリ
except ImportError:
    pyoxipng = None           # 無くても動く（Pillow 圧縮のみ）

# ── 設定 ────────────────────────────────────────────────────
BASE_DIR        = r"D:\catalog_images"   # 画像&ZIP ルート
MAX_W           = 1200                  # 横幅リサイズ上限
MAX_BYTES       = 5_000_000             # 5 MB 超は JPEG にフォールバック
JPEG_QUAL       = 90                    # JPEG 品質

MAX_CONSEC_FAIL = 5                     # 連続失敗で停止
MAX_TOTAL_FAIL  = 50                    # 累計失敗で停止
FAIL_SIZE_KB    = 15                    # 15 KB 未満を失敗とみなす
# ──────────────────────────────────────────────────────────


def cleanup_previous_results(base: Path):
    """旧 *_bg.* を全削除して再処理"""
    targets = [*base.rglob("*_bg.png"), *base.rglob("*_bg.jpg")]
    if targets:
        print(f"[CLEANUP] 旧結果 {len(targets)} ファイル削除")
        for f in targets:
            f.unlink(missing_ok=True)


def unzip_all(base: Path):
    """ZIP 自動展開"""
    for z in base.rglob("*.zip"):
        dst = z.with_suffix("")
        if dst.exists():
            continue
        try:
            print(f"[UNZIP] {z.relative_to(base)}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dst)
        except zipfile.BadZipFile:
            print(f"[WARN] 壊れた ZIP: {z}")


def optimize_png(rgba: np.ndarray, out_path: Path):
    """RGBA numpy → PNG-8 → (任意) oxipng 圧縮"""
    img = Image.fromarray(rgba, mode="RGBA")

    if img.width > MAX_W:
        h = int(img.height * MAX_W / img.width)
        img = img.resize((MAX_W, h), Image.LANCZOS)

    img8 = img.quantize(method=Image.Quantize.FASTOCTREE,
                        dither=Image.Dither.NONE)

    buf = io.BytesIO()
    img8.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()

    if pyoxipng:
        data = pyoxipng.optimize(data, level=4, strip=True)

    out_path.write_bytes(data)

    if out_path.stat().st_size > MAX_BYTES:
        raise ValueError("PNG > 5 MB")


def save_white_jpeg(rgba: np.ndarray, out_path: Path):
    """白背景 JPEG 保存（RGB → BGR 変換込み）"""
    rgb   = rgba[:, :, :3]
    alpha = rgba[:, :, 3:] / 255.0
    white = np.full_like(rgb, 255)
    merged = (alpha * rgb + (1 - alpha) * white).astype(np.uint8)
    bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL])


def is_failure(file: Path, rgba: np.ndarray) -> bool:
    """極端に小さい or アルファ全同値なら失敗"""
    if file.stat().st_size <= FAIL_SIZE_KB * 1024:
        return True
    a = rgba[:, :, 3]
    return a.min() == a.max()


def process_single(bgr: np.ndarray, png_out: Path, jpg_out: Path):
    """色空間を統一して 1 枚処理"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.array(remove(Image.fromarray(rgb)))  # RGB→RGBA
    try:
        optimize_png(rgba, png_out)
        return png_out, "PNG", rgba
    except Exception as e:
        print(f"[WARN] {e} → JPEG 保存")
        save_white_jpeg(rgba, jpg_out)
        return jpg_out, "JPG", rgba


def process_images(base_dir: str = BASE_DIR):
    base = Path(base_dir)

    print("[WARMUP] rembg モデル初期化（初回のみ数十秒）")
    new_session("u2net")
    print("[WARMUP] OK\n")

    cleanup_previous_results(base)
    unzip_all(base)

    files = [*base.rglob("*.jpg"), *base.rglob("*.jpeg")]
    print(f"[INFO] 処理対象 JPEG: {len(files)} 枚")

    consec_fail = total_fail = 0

    for idx, p in enumerate(files, 1):
        png_out = p.parent / f"{p.stem}_bg.png"
        jpg_out = p.parent / f"{p.stem}_bg.jpg"

        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"[SKIP] 読込失敗: {p}")
            continue

        try:
            out_file, kind, rgba = process_single(bgr, png_out, jpg_out)
        except Exception as e:
            print(f"[ERR] {p.name} → {e}")
            consec_fail += 1; total_fail += 1
            continue

        kb = out_file.stat().st_size // 1024
        print(f"[{idx}] {kind} {out_file.name} {kb} KB")

        if is_failure(out_file, rgba):
            print(f"[FAIL] {out_file.name} を異常検知")
            consec_fail += 1; total_fail += 1
        else:
            consec_fail = 0

        if consec_fail >= MAX_CONSEC_FAIL:
            print(f"[ABORT] 連続 {consec_fail} 回失敗 → 停止")
            sys.exit(1)
        if total_fail >= MAX_TOTAL_FAIL:
            print(f"[ABORT] 累計 {total_fail} 件失敗 → 停止")
            sys.exit(1)


if __name__ == "__main__":
    process_images()

# === atelier-kyo-manager/app\utils\utils\ai_background_remover.py ===
# ──────────────────────────────────────────────────────────────
#  existing_image_processor.py  ― 色ズレ・失敗検知・自動停止付き完全版
# ----------------------------------------------------------------
#  必要ライブラリ
#     pip install rembg pillow opencv-python numpy pyoxipng
# ----------------------------------------------------------------
import io, sys, zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

try:
    import pyoxipng           # Rust 圧縮ライブラリ
except ImportError:
    pyoxipng = None           # 無くても動く（Pillow 圧縮のみ）

# ── 設定 ────────────────────────────────────────────────────
BASE_DIR        = r"D:\catalog_images"   # 画像&ZIP ルート
MAX_W           = 1200                  # 横幅リサイズ上限
MAX_BYTES       = 5_000_000             # 5 MB 超は JPEG にフォールバック
JPEG_QUAL       = 90                    # JPEG 品質

MAX_CONSEC_FAIL = 5                     # 連続失敗で停止
MAX_TOTAL_FAIL  = 50                    # 累計失敗で停止
FAIL_SIZE_KB    = 15                    # 15 KB 未満を失敗とみなす
# ──────────────────────────────────────────────────────────


def cleanup_previous_results(base: Path):
    """旧 *_bg.* を全削除して再処理"""
    targets = [*base.rglob("*_bg.png"), *base.rglob("*_bg.jpg")]
    if targets:
        print(f"[CLEANUP] 旧結果 {len(targets)} ファイル削除")
        for f in targets:
            f.unlink(missing_ok=True)


def unzip_all(base: Path):
    """ZIP 自動展開"""
    for z in base.rglob("*.zip"):
        dst = z.with_suffix("")
        if dst.exists():
            continue
        try:
            print(f"[UNZIP] {z.relative_to(base)}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dst)
        except zipfile.BadZipFile:
            print(f"[WARN] 壊れた ZIP: {z}")


def optimize_png(rgba: np.ndarray, out_path: Path):
    """RGBA numpy → PNG-8 → (任意) oxipng 圧縮"""
    img = Image.fromarray(rgba, mode="RGBA")

    if img.width > MAX_W:
        h = int(img.height * MAX_W / img.width)
        img = img.resize((MAX_W, h), Image.LANCZOS)

    img8 = img.quantize(method=Image.Quantize.FASTOCTREE,
                        dither=Image.Dither.NONE)

    buf = io.BytesIO()
    img8.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()

    if pyoxipng:
        data = pyoxipng.optimize(data, level=4, strip=True)

    out_path.write_bytes(data)

    if out_path.stat().st_size > MAX_BYTES:
        raise ValueError("PNG > 5 MB")


def save_white_jpeg(rgba: np.ndarray, out_path: Path):
    """白背景 JPEG 保存（RGB → BGR 変換込み）"""
    rgb   = rgba[:, :, :3]
    alpha = rgba[:, :, 3:] / 255.0
    white = np.full_like(rgb, 255)
    merged = (alpha * rgb + (1 - alpha) * white).astype(np.uint8)
    bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUAL])


def is_failure(file: Path, rgba: np.ndarray) -> bool:
    """極端に小さい or アルファ全同値なら失敗"""
    if file.stat().st_size <= FAIL_SIZE_KB * 1024:
        return True
    a = rgba[:, :, 3]
    return a.min() == a.max()


def process_single(bgr: np.ndarray, png_out: Path, jpg_out: Path):
    """色空間を統一して 1 枚処理"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.array(remove(Image.fromarray(rgb)))  # RGB→RGBA
    try:
        optimize_png(rgba, png_out)
        return png_out, "PNG", rgba
    except Exception as e:
        print(f"[WARN] {e} → JPEG 保存")
        save_white_jpeg(rgba, jpg_out)
        return jpg_out, "JPG", rgba


def process_images(base_dir: str = BASE_DIR):
    base = Path(base_dir)

    print("[WARMUP] rembg モデル初期化（初回のみ数十秒）")
    new_session("u2net")
    print("[WARMUP] OK\n")

    cleanup_previous_results(base)
    unzip_all(base)

    files = [*base.rglob("*.jpg"), *base.rglob("*.jpeg")]
    print(f"[INFO] 処理対象 JPEG: {len(files)} 枚")

    consec_fail = total_fail = 0

    for idx, p in enumerate(files, 1):
        png_out = p.parent / f"{p.stem}_bg.png"
        jpg_out = p.parent / f"{p.stem}_bg.jpg"

        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"[SKIP] 読込失敗: {p}")
            continue

        try:
            out_file, kind, rgba = process_single(bgr, png_out, jpg_out)
        except Exception as e:
            print(f"[ERR] {p.name} → {e}")
            consec_fail += 1; total_fail += 1
            continue

        kb = out_file.stat().st_size // 1024
        print(f"[{idx}] {kind} {out_file.name} {kb} KB")

        if is_failure(out_file, rgba):
            print(f"[FAIL] {out_file.name} を異常検知")
            consec_fail += 1; total_fail += 1
        else:
            consec_fail = 0

        if consec_fail >= MAX_CONSEC_FAIL:
            print(f"[ABORT] 連続 {consec_fail} 回失敗 → 停止")
            sys.exit(1)
        if total_fail >= MAX_TOTAL_FAIL:
            print(f"[ABORT] 累計 {total_fail} 件失敗 → 停止")
            sys.exit(1)


if __name__ == "__main__":
    process_images()

# === atelier-kyo-manager/app\routes.py ===
from flask import Blueprint, render_template, redirect, url_for, request, flash, Response
from app import db
from app.models import Product
from app.forms import ProductForm
import csv
import io
from datetime import datetime

bp = Blueprint('main', __name__)

@bp.route('/manage', methods=['GET', 'POST'])
def manage_products():
    form = ProductForm()
    if form.validate_on_submit():
        product = Product(
            name=form.name.data,
            brand=form.brand.data,
            purchase_price=form.purchase_price.data,
            selling_price=form.selling_price.data,
            supplier_url=form.supplier_url.data,
            image_url=form.image_url.data,
            stock_status=form.stock_status.data,
            transaction_fee=form.transaction_fee.data,
            shipping_cost=form.shipping_cost.data,
            customs_duty=form.customs_duty.data,
            procurement_fee=form.procurement_fee.data
        )
        if hasattr(product, "calculate_profit"):
            product.profit = product.calculate_profit()
        db.session.add(product)
        db.session.commit()
        flash('商品を登録しました', 'success')
        return redirect(url_for('main.manage_products'))
    products = Product.query.all()
    return render_template('products/manage.html', form=form, products=products)

@bp.route('/products/<int:id>/delete', methods=['POST'])
def delete_product(id):
    product = Product.query.get_or_404(id)
    db.session.delete(product)
    db.session.commit()
    flash('商品を削除しました', 'success')
    return redirect(url_for('main.manage_products'))

@bp.route('/import_csv', methods=['POST'])
def import_csv():
    if 'file' not in request.files:
        flash('ファイルがアップロードされていません', 'error')
        return redirect(url_for('main.manage_products'))
    file = request.files['file']
    if file.filename == '':
        flash('ファイルが選択されていません', 'error')
        return redirect(url_for('main.manage_products'))
    if file and file.filename.endswith('.csv'):
        try:
            stream = io.StringIO(file.stream.read().decode('utf-8'))
            csv_reader = csv.DictReader(stream)
            count = 0
            for row in csv_reader:
                required_fields = ['name', 'purchase_price', 'selling_price']
                if not all(field in row for field in required_fields):
                    flash('CSVに必須項目が不足しています', 'error')
                    return redirect(url_for('main.manage_products'))
                product = Product(
                    name=row['name'],
                    brand=row.get('brand', ''),
                    purchase_price=float(row['purchase_price']),
                    selling_price=float(row['selling_price']),
                    supplier_url=row.get('supplier_url', ''),
                    image_url=row.get('image_url', ''),
                    stock_status=row.get('stock_status', 'true').lower() in ['true', '1', 'yes'],
                    transaction_fee=float(row.get('transaction_fee', 0)),
                    shipping_cost=float(row.get('shipping_cost', 0)),
                    customs_duty=float(row.get('customs_duty', 0)),
                    procurement_fee=float(row.get('procurement_fee', 0))
                )
                if hasattr(product, "calculate_profit"):
                    product.profit = product.calculate_profit()
                db.session.add(product)
                count += 1
            db.session.commit()
            flash(f'{count}件の商品をインポートしました', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'CSVインポート中にエラーが発生しました: {str(e)}', 'error')
    else:
        flash('CSVファイルのみアップロード可能です', 'error')
    return redirect(url_for('main.manage_products'))

@bp.route('/export_csv')
def export_csv():
    try:
        products = Product.query.all()
        output = io.StringIO()
        writer = csv.writer(output)

        # ヘッダー書き込み
        writer.writerow([
            'name', 'brand', 'purchase_price', 'selling_price',
            'transaction_fee', 'shipping_cost', 'customs_duty', 'procurement_fee',
            'supplier_url', 'image_url', 'stock_status'
        ])

        # データ書き込み
        for product in products:
            writer.writerow([
                product.name,
                product.brand,
                product.purchase_price,
                product.selling_price,
                product.transaction_fee,
                product.shipping_cost,
                product.customs_duty,
                product.procurement_fee,
                product.supplier_url,
                product.image_url,
                'true' if product.stock_status else 'false'
            ])

        output.seek(0)
        date_str = datetime.now().strftime('%Y%m%d%H%M%S')
        # 文字化け対策: BOM付きUTF-8
        bom = '\ufeff'
        csv_data = bom + output.getvalue()

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename=products_{date_str}.csv",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )

    except Exception as e:
        flash(f'CSVエクスポート中にエラーが発生しました: {str(e)}', 'error')
        return redirect(url_for('main.manage_products'))

@bp.route('/products/<int:product_id>/edit', methods=['GET', 'POST'])
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)
    form = ProductForm(obj=product)
    if form.validate_on_submit():
        form.populate_obj(product)
        if hasattr(product, "calculate_profit"):
            product.profit = product.calculate_profit()
        db.session.commit()
        flash('商品情報を更新しました', 'success')
        return redirect(url_for('main.manage_products'))
    return render_template('products/edit.html', form=form, product=product)

@bp.route('/')
def index():
    return redirect(url_for('main.manage_products'))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_function_base.py ===
import decimal
import math
import operator
import sys
import warnings
from fractions import Fraction
from functools import partial

import hypothesis
import hypothesis.strategies as st
import pytest
from hypothesis.extra.numpy import arrays

import numpy as np
import numpy.lib._function_base_impl as nfb
from numpy import (
    angle,
    average,
    bartlett,
    blackman,
    corrcoef,
    cov,
    delete,
    diff,
    digitize,
    extract,
    flipud,
    gradient,
    hamming,
    hanning,
    i0,
    insert,
    interp,
    kaiser,
    ma,
    meshgrid,
    piecewise,
    place,
    rot90,
    select,
    setxor1d,
    sinc,
    trapezoid,
    trim_zeros,
    unique,
    unwrap,
    vectorize,
)
from numpy._core.numeric import normalize_axis_tuple
from numpy.exceptions import AxisError
from numpy.random import rand
from numpy.testing import (
    HAS_REFCOUNT,
    IS_WASM,
    NOGIL_BUILD,
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
    suppress_warnings,
)


def get_mat(n):
    data = np.arange(n)
    data = np.add.outer(data, data)
    return data


def _make_complex(real, imag):
    """
    Like real + 1j * imag, but behaves as expected when imag contains non-finite
    values
    """
    ret = np.zeros(np.broadcast(real, imag).shape, np.complex128)
    ret.real = real
    ret.imag = imag
    return ret


class TestRot90:
    def test_basic(self):
        assert_raises(ValueError, rot90, np.ones(4))
        assert_raises(ValueError, rot90, np.ones((2, 2, 2)), axes=(0, 1, 2))
        assert_raises(ValueError, rot90, np.ones((2, 2)), axes=(0, 2))
        assert_raises(ValueError, rot90, np.ones((2, 2)), axes=(1, 1))
        assert_raises(ValueError, rot90, np.ones((2, 2, 2)), axes=(-2, 1))

        a = [[0, 1, 2],
             [3, 4, 5]]
        b1 = [[2, 5],
              [1, 4],
              [0, 3]]
        b2 = [[5, 4, 3],
              [2, 1, 0]]
        b3 = [[3, 0],
              [4, 1],
              [5, 2]]
        b4 = [[0, 1, 2],
              [3, 4, 5]]

        for k in range(-3, 13, 4):
            assert_equal(rot90(a, k=k), b1)
        for k in range(-2, 13, 4):
            assert_equal(rot90(a, k=k), b2)
        for k in range(-1, 13, 4):
            assert_equal(rot90(a, k=k), b3)
        for k in range(0, 13, 4):
            assert_equal(rot90(a, k=k), b4)

        assert_equal(rot90(rot90(a, axes=(0, 1)), axes=(1, 0)), a)
        assert_equal(rot90(a, k=1, axes=(1, 0)), rot90(a, k=-1, axes=(0, 1)))

    def test_axes(self):
        a = np.ones((50, 40, 3))
        assert_equal(rot90(a).shape, (40, 50, 3))
        assert_equal(rot90(a, axes=(0, 2)), rot90(a, axes=(0, -1)))
        assert_equal(rot90(a, axes=(1, 2)), rot90(a, axes=(-2, -1)))

    def test_rotation_axes(self):
        a = np.arange(8).reshape((2, 2, 2))

        a_rot90_01 = [[[2, 3],
                       [6, 7]],
                      [[0, 1],
                       [4, 5]]]
        a_rot90_12 = [[[1, 3],
                       [0, 2]],
                      [[5, 7],
                       [4, 6]]]
        a_rot90_20 = [[[4, 0],
                       [6, 2]],
                      [[5, 1],
                       [7, 3]]]
        a_rot90_10 = [[[4, 5],
                       [0, 1]],
                      [[6, 7],
                       [2, 3]]]

        assert_equal(rot90(a, axes=(0, 1)), a_rot90_01)
        assert_equal(rot90(a, axes=(1, 0)), a_rot90_10)
        assert_equal(rot90(a, axes=(1, 2)), a_rot90_12)

        for k in range(1, 5):
            assert_equal(rot90(a, k=k, axes=(2, 0)),
                         rot90(a_rot90_20, k=k - 1, axes=(2, 0)))


class TestFlip:

    def test_axes(self):
        assert_raises(AxisError, np.flip, np.ones(4), axis=1)
        assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=2)
        assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=-3)
        assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=(0, 3))

    def test_basic_lr(self):
        a = get_mat(4)
        b = a[:, ::-1]
        assert_equal(np.flip(a, 1), b)
        a = [[0, 1, 2],
             [3, 4, 5]]
        b = [[2, 1, 0],
             [5, 4, 3]]
        assert_equal(np.flip(a, 1), b)

    def test_basic_ud(self):
        a = get_mat(4)
        b = a[::-1, :]
        assert_equal(np.flip(a, 0), b)
        a = [[0, 1, 2],
             [3, 4, 5]]
        b = [[3, 4, 5],
             [0, 1, 2]]
        assert_equal(np.flip(a, 0), b)

    def test_3d_swap_axis0(self):
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])

        b = np.array([[[4, 5],
                       [6, 7]],
                      [[0, 1],
                       [2, 3]]])

        assert_equal(np.flip(a, 0), b)

    def test_3d_swap_axis1(self):
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])

        b = np.array([[[2, 3],
                       [0, 1]],
                      [[6, 7],
                       [4, 5]]])

        assert_equal(np.flip(a, 1), b)

    def test_3d_swap_axis2(self):
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])

        b = np.array([[[1, 0],
                       [3, 2]],
                      [[5, 4],
                       [7, 6]]])

        assert_equal(np.flip(a, 2), b)

    def test_4d(self):
        a = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        for i in range(a.ndim):
            assert_equal(np.flip(a, i),
                         np.flipud(a.swapaxes(0, i)).swapaxes(i, 0))

    def test_default_axis(self):
        a = np.array([[1, 2, 3],
                      [4, 5, 6]])
        b = np.array([[6, 5, 4],
                      [3, 2, 1]])
        assert_equal(np.flip(a), b)

    def test_multiple_axes(self):
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])

        assert_equal(np.flip(a, axis=()), a)

        b = np.array([[[5, 4],
                       [7, 6]],
                      [[1, 0],
                       [3, 2]]])

        assert_equal(np.flip(a, axis=(0, 2)), b)

        c = np.array([[[3, 2],
                       [1, 0]],
                      [[7, 6],
                       [5, 4]]])

        assert_equal(np.flip(a, axis=(1, 2)), c)


class TestAny:

    def test_basic(self):
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        assert_(np.any(y1))
        assert_(np.any(y3))
        assert_(not np.any(y2))

    def test_nd(self):
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        assert_(np.any(y1))
        assert_array_equal(np.any(y1, axis=0), [1, 1, 0])
        assert_array_equal(np.any(y1, axis=1), [0, 1, 1])


class TestAll:

    def test_basic(self):
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        assert_(not np.all(y1))
        assert_(np.all(y3))
        assert_(not np.all(y2))
        assert_(np.all(~np.array(y2)))

    def test_nd(self):
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        assert_(not np.all(y1))
        assert_array_equal(np.all(y1, axis=0), [0, 0, 1])
        assert_array_equal(np.all(y1, axis=1), [0, 0, 1])


@pytest.mark.parametrize("dtype", ["i8", "U10", "object", "datetime64[ms]"])
def test_any_and_all_result_dtype(dtype):
    arr = np.ones(3, dtype=dtype)
    assert np.any(arr).dtype == np.bool
    assert np.all(arr).dtype == np.bool


class TestCopy:

    def test_basic(self):
        a = np.array([[1, 2], [3, 4]])
        a_copy = np.copy(a)
        assert_array_equal(a, a_copy)
        a_copy[0, 0] = 10
        assert_equal(a[0, 0], 1)
        assert_equal(a_copy[0, 0], 10)

    def test_order(self):
        # It turns out that people rely on np.copy() preserving order by
        # default; changing this broke scikit-learn:
        # github.com/scikit-learn/scikit-learn/commit/7842748cf777412c506a8c0ed28090711d3a3783
        a = np.array([[1, 2], [3, 4]])
        assert_(a.flags.c_contiguous)
        assert_(not a.flags.f_contiguous)
        a_fort = np.array([[1, 2], [3, 4]], order="F")
        assert_(not a_fort.flags.c_contiguous)
        assert_(a_fort.flags.f_contiguous)
        a_copy = np.copy(a)
        assert_(a_copy.flags.c_contiguous)
        assert_(not a_copy.flags.f_contiguous)
        a_fort_copy = np.copy(a_fort)
        assert_(not a_fort_copy.flags.c_contiguous)
        assert_(a_fort_copy.flags.f_contiguous)

    def test_subok(self):
        mx = ma.ones(5)
        assert_(not ma.isMaskedArray(np.copy(mx, subok=False)))
        assert_(ma.isMaskedArray(np.copy(mx, subok=True)))
        # Default behavior
        assert_(not ma.isMaskedArray(np.copy(mx)))


class TestAverage:

    def test_basic(self):
        y1 = np.array([1, 2, 3])
        assert_(average(y1, axis=0) == 2.)
        y2 = np.array([1., 2., 3.])
        assert_(average(y2, axis=0) == 2.)
        y3 = [0., 0., 0.]
        assert_(average(y3, axis=0) == 0.)

        y4 = np.ones((4, 4))
        y4[0, 1] = 0
        y4[1, 0] = 2
        assert_almost_equal(y4.mean(0), average(y4, 0))
        assert_almost_equal(y4.mean(1), average(y4, 1))

        y5 = rand(5, 5)
        assert_almost_equal(y5.mean(0), average(y5, 0))
        assert_almost_equal(y5.mean(1), average(y5, 1))

    @pytest.mark.parametrize(
        'x, axis, expected_avg, weights, expected_wavg, expected_wsum',
        [([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]),
         ([[1, 2, 5], [1, 6, 11]], 0, [[1.0, 4.0, 8.0]],
          [1, 3], [[1.0, 5.0, 9.5]], [[4, 4, 4]])],
    )
    def test_basic_keepdims(self, x, axis, expected_avg,
                            weights, expected_wavg, expected_wsum):
        avg = np.average(x, axis=axis, keepdims=True)
        assert avg.shape == np.shape(expected_avg)
        assert_array_equal(avg, expected_avg)

        wavg = np.average(x, axis=axis, weights=weights, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)

        wavg, wsum = np.average(x, axis=axis, weights=weights, returned=True,
                                keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        assert wsum.shape == np.shape(expected_wsum)
        assert_array_equal(wsum, expected_wsum)

    def test_weights(self):
        y = np.arange(10)
        w = np.arange(10)
        actual = average(y, weights=w)
        desired = (np.arange(10) ** 2).sum() * 1. / np.arange(10).sum()
        assert_almost_equal(actual, desired)

        y1 = np.array([[1, 2, 3], [4, 5, 6]])
        w0 = [1, 2]
        actual = average(y1, weights=w0, axis=0)
        desired = np.array([3., 4., 5.])
        assert_almost_equal(actual, desired)

        w1 = [0, 0, 1]
        actual = average(y1, weights=w1, axis=1)
        desired = np.array([3., 6.])
        assert_almost_equal(actual, desired)

        # weights and input have different shapes but no axis is specified
        with pytest.raises(
                TypeError,
                match="Axis must be specified when shapes of a "
                      "and weights differ"):
            average(y1, weights=w1)

        # 2D Case
        w2 = [[0, 0, 1], [0, 0, 2]]
        desired = np.array([3., 6.])
        assert_array_equal(average(y1, weights=w2, axis=1), desired)
        assert_equal(average(y1, weights=w2), 5.)

        y3 = rand(5).astype(np.float32)
        w3 = rand(5).astype(np.float64)

        assert_(np.average(y3, weights=w3).dtype == np.result_type(y3, w3))

        # test weights with `keepdims=False` and `keepdims=True`
        x = np.array([2, 3, 4]).reshape(3, 1)
        w = np.array([4, 5, 6]).reshape(3, 1)

        actual = np.average(x, weights=w, axis=1, keepdims=False)
        desired = np.array([2., 3., 4.])
        assert_array_equal(actual, desired)

        actual = np.average(x, weights=w, axis=1, keepdims=True)
        desired = np.array([[2.], [3.], [4.]])
        assert_array_equal(actual, desired)

    def test_weight_and_input_dims_different(self):
        y = np.arange(12).reshape(2, 2, 3)
        w = np.array([0., 0., 1., .5, .5, 0., 0., .5, .5, 1., 0., 0.])\
            .reshape(2, 2, 3)

        subw0 = w[:, :, 0]
        actual = average(y, axis=(0, 1), weights=subw0)
        desired = np.array([7., 8., 9.])
        assert_almost_equal(actual, desired)

        subw1 = w[1, :, :]
        actual = average(y, axis=(1, 2), weights=subw1)
        desired = np.array([2.25, 8.25])
        assert_almost_equal(actual, desired)

        subw2 = w[:, 0, :]
        actual = average(y, axis=(0, 2), weights=subw2)
        desired = np.array([4.75, 7.75])
        assert_almost_equal(actual, desired)

        # here the weights have the wrong shape for the specified axes
        with pytest.raises(
                ValueError,
                match="Shape of weights must be consistent with "
                      "shape of a along specified axis"):
            average(y, axis=(0, 1, 2), weights=subw0)

        with pytest.raises(
                ValueError,
                match="Shape of weights must be consistent with "
                      "shape of a along specified axis"):
            average(y, axis=(0, 1), weights=subw1)

        # swapping the axes should be same as transposing weights
        actual = average(y, axis=(1, 0), weights=subw0)
        desired = average(y, axis=(0, 1), weights=subw0.T)
        assert_almost_equal(actual, desired)

        # if average over all axes, should have float output
        actual = average(y, axis=(0, 1, 2), weights=w)
        assert_(actual.ndim == 0)

    def test_returned(self):
        y = np.array([[1, 2, 3], [4, 5, 6]])

        # No weights
        avg, scl = average(y, returned=True)
        assert_equal(scl, 6.)

        avg, scl = average(y, 0, returned=True)
        assert_array_equal(scl, np.array([2., 2., 2.]))

        avg, scl = average(y, 1, returned=True)
        assert_array_equal(scl, np.array([3., 3.]))

        # With weights
        w0 = [1, 2]
        avg, scl = average(y, weights=w0, axis=0, returned=True)
        assert_array_equal(scl, np.array([3., 3., 3.]))

        w1 = [1, 2, 3]
        avg, scl = average(y, weights=w1, axis=1, returned=True)
        assert_array_equal(scl, np.array([6., 6.]))

        w2 = [[0, 0, 1], [1, 2, 3]]
        avg, scl = average(y, weights=w2, axis=1, returned=True)
        assert_array_equal(scl, np.array([1., 6.]))

    def test_subclasses(self):
        class subclass(np.ndarray):
            pass
        a = np.array([[1, 2], [3, 4]]).view(subclass)
        w = np.array([[1, 2], [3, 4]]).view(subclass)

        assert_equal(type(np.average(a)), subclass)
        assert_equal(type(np.average(a, weights=w)), subclass)

    def test_upcasting(self):
        typs = [('i4', 'i4', 'f8'), ('i4', 'f4', 'f8'), ('f4', 'i4', 'f8'),
                 ('f4', 'f4', 'f4'), ('f4', 'f8', 'f8')]
        for at, wt, rt in typs:
            a = np.array([[1, 2], [3, 4]], dtype=at)
            w = np.array([[1, 2], [3, 4]], dtype=wt)
            assert_equal(np.average(a, weights=w).dtype, np.dtype(rt))

    def test_object_dtype(self):
        a = np.array([decimal.Decimal(x) for x in range(10)])
        w = np.array([decimal.Decimal(1) for _ in range(10)])
        w /= w.sum()
        assert_almost_equal(a.mean(0), average(a, weights=w))

    def test_object_no_weights(self):
        a = np.array([decimal.Decimal(x) for x in range(10)])
        m = average(a)
        assert m == decimal.Decimal('4.5')

    def test_average_class_without_dtype(self):
        # see gh-21988
        a = np.array([Fraction(1, 5), Fraction(3, 5)])
        assert_equal(np.average(a), Fraction(2, 5))


class TestSelect:
    choices = [np.array([1, 2, 3]),
               np.array([4, 5, 6]),
               np.array([7, 8, 9])]
    conditions = [np.array([False, False, False]),
                  np.array([False, True, False]),
                  np.array([False, False, True])]

    def _select(self, cond, values, default=0):
        output = []
        for m in range(len(cond)):
            output += [V[m] for V, C in zip(values, cond) if C[m]] or [default]
        return output

    def test_basic(self):
        choices = self.choices
        conditions = self.conditions
        assert_array_equal(select(conditions, choices, default=15),
                           self._select(conditions, choices, default=15))

        assert_equal(len(choices), 3)
        assert_equal(len(conditions), 3)

    def test_broadcasting(self):
        conditions = [np.array(True), np.array([False, True, False])]
        choices = [1, np.arange(12).reshape(4, 3)]
        assert_array_equal(select(conditions, choices), np.ones((4, 3)))
        # default can broadcast too:
        assert_equal(select([True], [0], default=[0]).shape, (1,))

    def test_return_dtype(self):
        assert_equal(select(self.conditions, self.choices, 1j).dtype,
                     np.complex128)
        # But the conditions need to be stronger then the scalar default
        # if it is scalar.
        choices = [choice.astype(np.int8) for choice in self.choices]
        assert_equal(select(self.conditions, choices).dtype, np.int8)

        d = np.array([1, 2, 3, np.nan, 5, 7])
        m = np.isnan(d)
        assert_equal(select([m], [d]), [0, 0, 0, np.nan, 0, 0])

    def test_deprecated_empty(self):
        assert_raises(ValueError, select, [], [], 3j)
        assert_raises(ValueError, select, [], [])

    def test_non_bool_deprecation(self):
        choices = self.choices
        conditions = self.conditions[:]
        conditions[0] = conditions[0].astype(np.int_)
        assert_raises(TypeError, select, conditions, choices)
        conditions[0] = conditions[0].astype(np.uint8)
        assert_raises(TypeError, select, conditions, choices)
        assert_raises(TypeError, select, conditions, choices)

    def test_many_arguments(self):
        # This used to be limited by NPY_MAXARGS == 32
        conditions = [np.array([False])] * 100
        choices = [np.array([1])] * 100
        select(conditions, choices)


class TestInsert:

    def test_basic(self):
        a = [1, 2, 3]
        assert_equal(insert(a, 0, 1), [1, 1, 2, 3])
        assert_equal(insert(a, 3, 1), [1, 2, 3, 1])
        assert_equal(insert(a, [1, 1, 1], [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        assert_equal(insert(a, 1, [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        assert_equal(insert(a, [1, -1, 3], 9), [1, 9, 2, 9, 3, 9])
        assert_equal(insert(a, slice(-1, None, -1), 9), [9, 1, 9, 2, 9, 3])
        assert_equal(insert(a, [-1, 1, 3], [7, 8, 9]), [1, 8, 2, 7, 3, 9])
        b = np.array([0, 1], dtype=np.float64)
        assert_equal(insert(b, 0, b[0]), [0., 0., 1.])
        assert_equal(insert(b, [], []), b)
        assert_equal(insert(a, np.array([True] * 4), 9), [9, 1, 9, 2, 9, 3, 9])
        assert_equal(insert(a, np.array([True, False, True, False]), 9),
                     [9, 1, 2, 9, 3])

    def test_multidim(self):
        a = [[1, 1, 1]]
        r = [[2, 2, 2],
             [1, 1, 1]]
        assert_equal(insert(a, 0, [1]), [1, 1, 1, 1])
        assert_equal(insert(a, 0, [2, 2, 2], axis=0), r)
        assert_equal(insert(a, 0, 2, axis=0), r)
        assert_equal(insert(a, 2, 2, axis=1), [[1, 1, 2, 1]])

        a = np.array([[1, 1], [2, 2], [3, 3]])
        b = np.arange(1, 4).repeat(3).reshape(3, 3)
        c = np.concatenate(
            (a[:, 0:1], np.arange(1, 4).repeat(3).reshape(3, 3).T,
             a[:, 1:2]), axis=1)
        assert_equal(insert(a, [1], [[1], [2], [3]], axis=1), b)
        assert_equal(insert(a, [1], [1, 2, 3], axis=1), c)
        # scalars behave differently, in this case exactly opposite:
        assert_equal(insert(a, 1, [1, 2, 3], axis=1), b)
        assert_equal(insert(a, 1, [[1], [2], [3]], axis=1), c)

        a = np.arange(4).reshape(2, 2)
        assert_equal(insert(a[:, :1], 1, a[:, 1], axis=1), a)
        assert_equal(insert(a[:1, :], 1, a[1, :], axis=0), a)

        # negative axis value
        a = np.arange(24).reshape((2, 3, 4))
        assert_equal(insert(a, 1, a[:, :, 3], axis=-1),
                     insert(a, 1, a[:, :, 3], axis=2))
        assert_equal(insert(a, 1, a[:, 2, :], axis=-2),
                     insert(a, 1, a[:, 2, :], axis=1))

        # invalid axis value
        assert_raises(AxisError, insert, a, 1, a[:, 2, :], axis=3)
        assert_raises(AxisError, insert, a, 1, a[:, 2, :], axis=-4)

        # negative axis value
        a = np.arange(24).reshape((2, 3, 4))
        assert_equal(insert(a, 1, a[:, :, 3], axis=-1),
                     insert(a, 1, a[:, :, 3], axis=2))
        assert_equal(insert(a, 1, a[:, 2, :], axis=-2),
                     insert(a, 1, a[:, 2, :], axis=1))

    def test_0d(self):
        a = np.array(1)
        with pytest.raises(AxisError):
            insert(a, [], 2, axis=0)
        with pytest.raises(TypeError):
            insert(a, [], 2, axis="nonsense")

    def test_subclass(self):
        class SubClass(np.ndarray):
            pass
        a = np.arange(10).view(SubClass)
        assert_(isinstance(np.insert(a, 0, [0]), SubClass))
        assert_(isinstance(np.insert(a, [], []), SubClass))
        assert_(isinstance(np.insert(a, [0, 1], [1, 2]), SubClass))
        assert_(isinstance(np.insert(a, slice(1, 2), [1, 2]), SubClass))
        assert_(isinstance(np.insert(a, slice(1, -2, -1), []), SubClass))
        # This is an error in the future:
        a = np.array(1).view(SubClass)
        assert_(isinstance(np.insert(a, 0, [0]), SubClass))

    def test_index_array_copied(self):
        x = np.array([1, 1, 1])
        np.insert([0, 1, 2], x, [3, 4, 5])
        assert_equal(x, np.array([1, 1, 1]))

    def test_structured_array(self):
        a = np.array([(1, 'a'), (2, 'b'), (3, 'c')],
                     dtype=[('foo', 'i'), ('bar', 'S1')])
        val = (4, 'd')
        b = np.insert(a, 0, val)
        assert_array_equal(b[0], np.array(val, dtype=b.dtype))
        val = [(4, 'd')] * 2
        b = np.insert(a, [0, 2], val)
        assert_array_equal(b[[0, 3]], np.array(val, dtype=b.dtype))

    def test_index_floats(self):
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([1.0, 2.0]), [10, 20])
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([], dtype=float), [])

    @pytest.mark.parametrize('idx', [4, -4])
    def test_index_out_of_bounds(self, idx):
        with pytest.raises(IndexError, match='out of bounds'):
            np.insert([0, 1, 2], [idx], [3, 4])


class TestAmax:

    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amax(a), 10.0)
        b = [[3, 6.0, 9.0],
             [4, 10.0, 5.0],
             [8, 3.0, 2.0]]
        assert_equal(np.amax(b, axis=0), [8.0, 10.0, 9.0])
        assert_equal(np.amax(b, axis=1), [9.0, 10.0, 8.0])


class TestAmin:

    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amin(a), -5.0)
        b = [[3, 6.0, 9.0],
             [4, 10.0, 5.0],
             [8, 3.0, 2.0]]
        assert_equal(np.amin(b, axis=0), [3.0, 3.0, 2.0])
        assert_equal(np.amin(b, axis=1), [3.0, 4.0, 2.0])


class TestPtp:

    def test_basic(self):
        a = np.array([3, 4, 5, 10, -3, -5, 6.0])
        assert_equal(np.ptp(a, axis=0), 15.0)
        b = np.array([[3, 6.0, 9.0],
                      [4, 10.0, 5.0],
                      [8, 3.0, 2.0]])
        assert_equal(np.ptp(b, axis=0), [5.0, 7.0, 7.0])
        assert_equal(np.ptp(b, axis=-1), [6.0, 6.0, 6.0])

        assert_equal(np.ptp(b, axis=0, keepdims=True), [[5.0, 7.0, 7.0]])
        assert_equal(np.ptp(b, axis=(0, 1), keepdims=True), [[8.0]])


class TestCumsum:

    @pytest.mark.parametrize("cumsum", [np.cumsum, np.cumulative_sum])
    def test_basic(self, cumsum):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [np.int8, np.uint8, np.int16, np.uint16, np.int32,
                      np.uint32, np.float32, np.float64, np.complex64,
                      np.complex128]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)

            tgt = np.array([1, 3, 13, 24, 30, 35, 39], ctype)
            assert_array_equal(cumsum(a, axis=0), tgt)

            tgt = np.array(
                [[1, 2, 3, 4], [6, 8, 10, 13], [16, 11, 14, 18]], ctype)
            assert_array_equal(cumsum(a2, axis=0), tgt)

            tgt = np.array(
                [[1, 3, 6, 10], [5, 11, 18, 27], [10, 13, 17, 22]], ctype)
            assert_array_equal(cumsum(a2, axis=1), tgt)


class TestProd:

    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [np.int16, np.uint16, np.int32, np.uint32,
                      np.float32, np.float64, np.complex64, np.complex128]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            if ctype in ['1', 'b']:
                assert_raises(ArithmeticError, np.prod, a)
                assert_raises(ArithmeticError, np.prod, a2, 1)
            else:
                assert_equal(a.prod(axis=0), 26400)
                assert_array_equal(a2.prod(axis=0),
                                   np.array([50, 36, 84, 180], ctype))
                assert_array_equal(a2.prod(axis=-1),
                                   np.array([24, 1890, 600], ctype))


class TestCumprod:

    @pytest.mark.parametrize("cumprod", [np.cumprod, np.cumulative_prod])
    def test_basic(self, cumprod):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [np.int16, np.uint16, np.int32, np.uint32,
                      np.float32, np.float64, np.complex64, np.complex128]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            if ctype in ['1', 'b']:
                assert_raises(ArithmeticError, cumprod, a)
                assert_raises(ArithmeticError, cumprod, a2, 1)
                assert_raises(ArithmeticError, cumprod, a)
            else:
                assert_array_equal(cumprod(a, axis=-1),
                                   np.array([1, 2, 20, 220,
                                             1320, 6600, 26400], ctype))
                assert_array_equal(cumprod(a2, axis=0),
                                   np.array([[1, 2, 3, 4],
                                             [5, 12, 21, 36],
                                             [50, 36, 84, 180]], ctype))
                assert_array_equal(cumprod(a2, axis=-1),
                                   np.array([[1, 2, 6, 24],
                                             [5, 30, 210, 1890],
                                             [10, 30, 120, 600]], ctype))


def test_cumulative_include_initial():
    arr = np.arange(8).reshape((2, 2, 2))

    expected = np.array([
        [[0, 0], [0, 1], [2, 4]], [[0, 0], [4, 5], [10, 12]]
    ])
    assert_array_equal(
        np.cumulative_sum(arr, axis=1, include_initial=True), expected
    )

    expected = np.array([
        [[1, 0, 0], [1, 2, 6]], [[1, 4, 20], [1, 6, 42]]
    ])
    assert_array_equal(
        np.cumulative_prod(arr, axis=2, include_initial=True), expected
    )

    out = np.zeros((3, 2), dtype=np.float64)
    expected = np.array([[0, 0], [1, 2], [4, 6]], dtype=np.float64)
    arr = np.arange(1, 5).reshape((2, 2))
    np.cumulative_sum(arr, axis=0, out=out, include_initial=True)
    assert_array_equal(out, expected)

    expected = np.array([1, 2, 4])
    assert_array_equal(
        np.cumulative_prod(np.array([2, 2]), include_initial=True), expected
    )


class TestDiff:

    def test_basic(self):
        x = [1, 4, 6, 7, 12]
        out = np.array([3, 2, 1, 5])
        out2 = np.array([-1, -1, 4])
        out3 = np.array([0, 5])
        assert_array_equal(diff(x), out)
        assert_array_equal(diff(x, n=2), out2)
        assert_array_equal(diff(x, n=3), out3)

        x = [1.1, 2.2, 3.0, -0.2, -0.1]
        out = np.array([1.1, 0.8, -3.2, 0.1])
        assert_almost_equal(diff(x), out)

        x = [True, True, False, False]
        out = np.array([False, True, False])
        out2 = np.array([True, True])
        assert_array_equal(diff(x), out)
        assert_array_equal(diff(x, n=2), out2)

    def test_axis(self):
        x = np.zeros((10, 20, 30))
        x[:, 1::2, :] = 1
        exp = np.ones((10, 19, 30))
        exp[:, 1::2, :] = -1
        assert_array_equal(diff(x), np.zeros((10, 20, 29)))
        assert_array_equal(diff(x, axis=-1), np.zeros((10, 20, 29)))
        assert_array_equal(diff(x, axis=0), np.zeros((9, 20, 30)))
        assert_array_equal(diff(x, axis=1), exp)
        assert_array_equal(diff(x, axis=-2), exp)
        assert_raises(AxisError, diff, x, axis=3)
        assert_raises(AxisError, diff, x, axis=-4)

        x = np.array(1.11111111111, np.float64)
        assert_raises(ValueError, diff, x)

    def test_nd(self):
        x = 20 * rand(10, 20, 30)
        out1 = x[:, :, 1:] - x[:, :, :-1]
        out2 = out1[:, :, 1:] - out1[:, :, :-1]
        out3 = x[1:, :, :] - x[:-1, :, :]
        out4 = out3[1:, :, :] - out3[:-1, :, :]
        assert_array_equal(diff(x), out1)
        assert_array_equal(diff(x, n=2), out2)
        assert_array_equal(diff(x, axis=0), out3)
        assert_array_equal(diff(x, n=2, axis=0), out4)

    def test_n(self):
        x = list(range(3))
        assert_raises(ValueError, diff, x, n=-1)
        output = [diff(x, n=n) for n in range(1, 5)]
        expected = [[1, 1], [0], [], []]
        assert_(diff(x, n=0) is x)
        for n, (expected_n, output_n) in enumerate(zip(expected, output), start=1):
            assert_(type(output_n) is np.ndarray)
            assert_array_equal(output_n, expected_n)
            assert_equal(output_n.dtype, np.int_)
            assert_equal(len(output_n), max(0, len(x) - n))

    def test_times(self):
        x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)
        expected = [
            np.array([1, 1], dtype='timedelta64[D]'),
            np.array([0], dtype='timedelta64[D]'),
        ]
        expected.extend([np.array([], dtype='timedelta64[D]')] * 3)
        for n, exp in enumerate(expected, start=1):
            out = diff(x, n=n)
            assert_array_equal(out, exp)
            assert_equal(out.dtype, exp.dtype)

    def test_subclass(self):
        x = ma.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                     mask=[[False, False], [True, False],
                           [False, True], [True, True], [False, False]])
        out = diff(x)
        assert_array_equal(out.data, [[1], [1], [1], [1], [1]])
        assert_array_equal(out.mask, [[False], [True],
                                      [True], [True], [False]])
        assert_(type(out) is type(x))

        out3 = diff(x, n=3)
        assert_array_equal(out3.data, [[], [], [], [], []])
        assert_array_equal(out3.mask, [[], [], [], [], []])
        assert_(type(out3) is type(x))

    def test_prepend(self):
        x = np.arange(5) + 1
        assert_array_equal(diff(x, prepend=0), np.ones(5))
        assert_array_equal(diff(x, prepend=[0]), np.ones(5))
        assert_array_equal(np.cumsum(np.diff(x, prepend=0)), x)
        assert_array_equal(diff(x, prepend=[-1, 0]), np.ones(6))

        x = np.arange(4).reshape(2, 2)
        result = np.diff(x, axis=1, prepend=0)
        expected = [[0, 1], [2, 1]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=1, prepend=[[0], [0]])
        assert_array_equal(result, expected)

        result = np.diff(x, axis=0, prepend=0)
        expected = [[0, 1], [2, 2]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=0, prepend=[[0, 0]])
        assert_array_equal(result, expected)

        assert_raises(ValueError, np.diff, x, prepend=np.zeros((3, 3)))

        assert_raises(AxisError, diff, x, prepend=0, axis=3)

    def test_append(self):
        x = np.arange(5)
        result = diff(x, append=0)
        expected = [1, 1, 1, 1, -4]
        assert_array_equal(result, expected)
        result = diff(x, append=[0])
        assert_array_equal(result, expected)
        result = diff(x, append=[0, 2])
        expected = expected + [2]
        assert_array_equal(result, expected)

        x = np.arange(4).reshape(2, 2)
        result = np.diff(x, axis=1, append=0)
        expected = [[1, -1], [1, -3]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=1, append=[[0], [0]])
        assert_array_equal(result, expected)

        result = np.diff(x, axis=0, append=0)
        expected = [[2, 2], [-2, -3]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=0, append=[[0, 0]])
        assert_array_equal(result, expected)

        assert_raises(ValueError, np.diff, x, append=np.zeros((3, 3)))

        assert_raises(AxisError, diff, x, append=0, axis=3)


class TestDelete:

    def setup_method(self):
        self.a = np.arange(5)
        self.nd_a = np.arange(5).repeat(2).reshape(1, 5, 2)

    def _check_inverse_of_slicing(self, indices):
        a_del = delete(self.a, indices)
        nd_a_del = delete(self.nd_a, indices, axis=1)
        msg = f'Delete failed for obj: {indices!r}'
        assert_array_equal(setxor1d(a_del, self.a[indices, ]), self.a,
                           err_msg=msg)
        xor = setxor1d(nd_a_del[0, :, 0], self.nd_a[0, indices, 0])
        assert_array_equal(xor, self.nd_a[0, :, 0], err_msg=msg)

    def test_slices(self):
        lims = [-6, -2, 0, 1, 2, 4, 5]
        steps = [-3, -1, 1, 3]
        for start in lims:
            for stop in lims:
                for step in steps:
                    s = slice(start, stop, step)
                    self._check_inverse_of_slicing(s)

    def test_fancy(self):
        self._check_inverse_of_slicing(np.array([[0, 1], [2, 1]]))
        with pytest.raises(IndexError):
            delete(self.a, [100])
        with pytest.raises(IndexError):
            delete(self.a, [-100])

        self._check_inverse_of_slicing([0, -1, 2, 2])

        self._check_inverse_of_slicing([True, False, False, True, False])

        # not legal, indexing with these would change the dimension
        with pytest.raises(ValueError):
            delete(self.a, True)
        with pytest.raises(ValueError):
            delete(self.a, False)

        # not enough items
        with pytest.raises(ValueError):
            delete(self.a, [False] * 4)

    def test_single(self):
        self._check_inverse_of_slicing(0)
        self._check_inverse_of_slicing(-4)

    def test_0d(self):
        a = np.array(1)
        with pytest.raises(AxisError):
            delete(a, [], axis=0)
        with pytest.raises(TypeError):
            delete(a, [], axis="nonsense")

    def test_subclass(self):
        class SubClass(np.ndarray):
            pass
        a = self.a.view(SubClass)
        assert_(isinstance(delete(a, 0), SubClass))
        assert_(isinstance(delete(a, []), SubClass))
        assert_(isinstance(delete(a, [0, 1]), SubClass))
        assert_(isinstance(delete(a, slice(1, 2)), SubClass))
        assert_(isinstance(delete(a, slice(1, -2)), SubClass))

    def test_array_order_preserve(self):
        # See gh-7113
        k = np.arange(10).reshape(2, 5, order='F')
        m = delete(k, slice(60, None), axis=1)

        # 'k' is Fortran ordered, and 'm' should have the
        # same ordering as 'k' and NOT become C ordered
        assert_equal(m.flags.c_contiguous, k.flags.c_contiguous)
        assert_equal(m.flags.f_contiguous, k.flags.f_contiguous)

    def test_index_floats(self):
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([], dtype=float))

    @pytest.mark.parametrize("indexer", [np.array([1]), [1]])
    def test_single_item_array(self, indexer):
        a_del_int = delete(self.a, 1)
        a_del = delete(self.a, indexer)
        assert_equal(a_del_int, a_del)

        nd_a_del_int = delete(self.nd_a, 1, axis=1)
        nd_a_del = delete(self.nd_a, np.array([1]), axis=1)
        assert_equal(nd_a_del_int, nd_a_del)

    def test_single_item_array_non_int(self):
        # Special handling for integer arrays must not affect non-integer ones.
        # If `False` was cast to `0` it would delete the element:
        res = delete(np.ones(1), np.array([False]))
        assert_array_equal(res, np.ones(1))

        # Test the more complicated (with axis) case from gh-21840
        x = np.ones((3, 1))
        false_mask = np.array([False], dtype=bool)
        true_mask = np.array([True], dtype=bool)

        res = delete(x, false_mask, axis=-1)
        assert_array_equal(res, x)
        res = delete(x, true_mask, axis=-1)
        assert_array_equal(res, x[:, :0])

        # Object or e.g. timedeltas should *not* be allowed
        with pytest.raises(IndexError):
            delete(np.ones(2), np.array([0], dtype=object))

        with pytest.raises(IndexError):
            # timedeltas are sometimes "integral, but clearly not allowed:
            delete(np.ones(2), np.array([0], dtype="m8[ns]"))


class TestGradient:

    def test_basic(self):
        v = [[1, 1], [3, 4]]
        x = np.array(v)
        dx = [np.array([[2., 3.], [2., 3.]]),
              np.array([[0., 0.], [1., 1.]])]
        assert_array_equal(gradient(x), dx)
        assert_array_equal(gradient(v), dx)

    def test_args(self):
        dx = np.cumsum(np.ones(5))
        dx_uneven = [1., 2., 5., 9., 11.]
        f_2d = np.arange(25).reshape(5, 5)

        # distances must be scalars or have size equal to gradient[axis]
        gradient(np.arange(5), 3.)
        gradient(np.arange(5), np.array(3.))
        gradient(np.arange(5), dx)
        # dy is set equal to dx because scalar
        gradient(f_2d, 1.5)
        gradient(f_2d, np.array(1.5))

        gradient(f_2d, dx_uneven, dx_uneven)
        # mix between even and uneven spaces and
        # mix between scalar and vector
        gradient(f_2d, dx, 2)

        # 2D but axis specified
        gradient(f_2d, dx, axis=1)

        # 2d coordinate arguments are not yet allowed
        assert_raises_regex(ValueError, '.*scalars or 1d',
            gradient, f_2d, np.stack([dx] * 2, axis=-1), 1)

    def test_badargs(self):
        f_2d = np.arange(25).reshape(5, 5)
        x = np.cumsum(np.ones(5))

        # wrong sizes
        assert_raises(ValueError, gradient, f_2d, x, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, 1, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, np.ones(2), np.ones(2))
        # wrong number of arguments
        assert_raises(TypeError, gradient, f_2d, x)
        assert_raises(TypeError, gradient, f_2d, x, axis=(0, 1))
        assert_raises(TypeError, gradient, f_2d, x, x, x)
        assert_raises(TypeError, gradient, f_2d, 1, 1, 1)
        assert_raises(TypeError, gradient, f_2d, x, x, axis=1)
        assert_raises(TypeError, gradient, f_2d, 1, 1, axis=1)

    def test_datetime64(self):
        # Make sure gradient() can handle special types like datetime64
        x = np.array(
            ['1910-08-16', '1910-08-11', '1910-08-10', '1910-08-12',
             '1910-10-12', '1910-12-12', '1912-12-12'],
            dtype='datetime64[D]')
        dx = np.array(
            [-5, -3, 0, 31, 61, 396, 731],
            dtype='timedelta64[D]')
        assert_array_equal(gradient(x), dx)
        assert_(dx.dtype == np.dtype('timedelta64[D]'))

    def test_masked(self):
        # Make sure that gradient supports subclasses like masked arrays
        x = np.ma.array([[1, 1], [3, 4]],
                        mask=[[False, False], [False, False]])
        out = gradient(x)[0]
        assert_equal(type(out), type(x))
        # And make sure that the output and input don't have aliased mask
        # arrays
        assert_(x._mask is not out._mask)
        # Also check that edge_order=2 doesn't alter the original mask
        x2 = np.ma.arange(5)
        x2[2] = np.ma.masked
        np.gradient(x2, edge_order=2)
        assert_array_equal(x2.mask, [False, False, True, False, False])

    def test_second_order_accurate(self):
        # Testing that the relative numerical error is less that 3% for
        # this example problem. This corresponds to second order
        # accurate finite differences for all interior and boundary
        # points.
        x = np.linspace(0, 1, 10)
        dx = x[1] - x[0]
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        analytical = 6 * x ** 2 + 8 * x + 2
        num_error = np.abs((np.gradient(y, dx, edge_order=2) / analytical) - 1)
        assert_(np.all(num_error < 0.03) == True)

        # test with unevenly spaced
        np.random.seed(0)
        x = np.sort(np.random.random(10))
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        analytical = 6 * x ** 2 + 8 * x + 2
        num_error = np.abs((np.gradient(y, x, edge_order=2) / analytical) - 1)
        assert_(np.all(num_error < 0.03) == True)

    def test_spacing(self):
        f = np.array([0, 2., 3., 4., 5., 5.])
        f = np.tile(f, (6, 1)) + f.reshape(-1, 1)
        x_uneven = np.array([0., 0.5, 1., 3., 5., 7.])
        x_even = np.arange(6.)

        fdx_even_ord1 = np.tile([2., 1.5, 1., 1., 0.5, 0.], (6, 1))
        fdx_even_ord2 = np.tile([2.5, 1.5, 1., 1., 0.5, -0.5], (6, 1))
        fdx_uneven_ord1 = np.tile([4., 3., 1.7, 0.5, 0.25, 0.], (6, 1))
        fdx_uneven_ord2 = np.tile([5., 3., 1.7, 0.5, 0.25, -0.25], (6, 1))

        # evenly spaced
        for edge_order, exp_res in [(1, fdx_even_ord1), (2, fdx_even_ord2)]:
            res1 = gradient(f, 1., axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_even, x_even,
                            axis=(0, 1), edge_order=edge_order)
            res3 = gradient(f, x_even, x_even,
                            axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_array_equal(res2, res3)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            res1 = gradient(f, 1., axis=0, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=0, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_almost_equal(res2, exp_res.T)

            res1 = gradient(f, 1., axis=1, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=1, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_array_equal(res2, exp_res)

        # unevenly spaced
        for edge_order, exp_res in [(1, fdx_uneven_ord1), (2, fdx_uneven_ord2)]:
            res1 = gradient(f, x_uneven, x_uneven,
                            axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_uneven, x_uneven,
                            axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            res1 = gradient(f, x_uneven, axis=0, edge_order=edge_order)
            assert_almost_equal(res1, exp_res.T)

            res1 = gradient(f, x_uneven, axis=1, edge_order=edge_order)
            assert_almost_equal(res1, exp_res)

        # mixed
        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=1)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=1)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord1.T)
        assert_almost_equal(res1[1], fdx_uneven_ord1)

        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=2)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=2)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord2.T)
        assert_almost_equal(res1[1], fdx_uneven_ord2)

    def test_specific_axes(self):
        # Testing that gradient can work on a given axis only
        v = [[1, 1], [3, 4]]
        x = np.array(v)
        dx = [np.array([[2., 3.], [2., 3.]]),
              np.array([[0., 0.], [1., 1.]])]
        assert_array_equal(gradient(x, axis=0), dx[0])
        assert_array_equal(gradient(x, axis=1), dx[1])
        assert_array_equal(gradient(x, axis=-1), dx[1])
        assert_array_equal(gradient(x, axis=(1, 0)), [dx[1], dx[0]])

        # test axis=None which means all axes
        assert_almost_equal(gradient(x, axis=None), [dx[0], dx[1]])
        # and is the same as no axis keyword given
        assert_almost_equal(gradient(x, axis=None), gradient(x))

        # test vararg order
        assert_array_equal(gradient(x, 2, 3, axis=(1, 0)),
                           [dx[1] / 2.0, dx[0] / 3.0])
        # test maximal number of varargs
        assert_raises(TypeError, gradient, x, 1, 2, axis=1)

        assert_raises(AxisError, gradient, x, axis=3)
        assert_raises(AxisError, gradient, x, axis=-3)
        # assert_raises(TypeError, gradient, x, axis=[1,])

    def test_timedelta64(self):
        # Make sure gradient() can handle special types like timedelta64
        x = np.array(
            [-5, -3, 10, 12, 61, 321, 300],
            dtype='timedelta64[D]')
        dx = np.array(
            [2, 7, 7, 25, 154, 119, -21],
            dtype='timedelta64[D]')
        assert_array_equal(gradient(x), dx)
        assert_(dx.dtype == np.dtype('timedelta64[D]'))

    def test_inexact_dtypes(self):
        for dt in [np.float16, np.float32, np.float64]:
            # dtypes should not be promoted in a different way to what diff does
            x = np.array([1, 2, 3], dtype=dt)
            assert_equal(gradient(x).dtype, np.diff(x).dtype)

    def test_values(self):
        # needs at least 2 points for edge_order ==1
        gradient(np.arange(2), edge_order=1)
        # needs at least 3 points for edge_order ==1
        gradient(np.arange(3), edge_order=2)

        assert_raises(ValueError, gradient, np.arange(0), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(0), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(2), edge_order=2)

    @pytest.mark.parametrize('f_dtype', [np.uint8, np.uint16,
                                         np.uint32, np.uint64])
    def test_f_decreasing_unsigned_int(self, f_dtype):
        f = np.array([5, 4, 3, 2, 1], dtype=f_dtype)
        g = gradient(f)
        assert_array_equal(g, [-1] * len(f))

    @pytest.mark.parametrize('f_dtype', [np.int8, np.int16,
                                         np.int32, np.int64])
    def test_f_signed_int_big_jump(self, f_dtype):
        maxint = np.iinfo(f_dtype).max
        x = np.array([1, 3])
        f = np.array([-1, maxint], dtype=f_dtype)
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [(maxint + 1) // 2] * 2)

    @pytest.mark.parametrize('x_dtype', [np.uint8, np.uint16,
                                         np.uint32, np.uint64])
    def test_x_decreasing_unsigned(self, x_dtype):
        x = np.array([3, 2, 1], dtype=x_dtype)
        f = np.array([0, 2, 4])
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [-2] * len(x))

    @pytest.mark.parametrize('x_dtype', [np.int8, np.int16,
                                         np.int32, np.int64])
    def test_x_signed_int_big_jump(self, x_dtype):
        minint = np.iinfo(x_dtype).min
        maxint = np.iinfo(x_dtype).max
        x = np.array([-1, maxint], dtype=x_dtype)
        f = np.array([minint // 2, 0])
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [0.5, 0.5])

    def test_return_type(self):
        res = np.gradient(([1, 2], [2, 3]))
        assert type(res) is tuple


class TestAngle:

    def test_basic(self):
        x = [1 + 3j, np.sqrt(2) / 2.0 + 1j * np.sqrt(2) / 2,
             1, 1j, -1, -1j, 1 - 3j, -1 + 3j]
        y = angle(x)
        yo = [
            np.arctan(3.0 / 1.0),
            np.arctan(1.0), 0, np.pi / 2, np.pi, -np.pi / 2.0,
            -np.arctan(3.0 / 1.0), np.pi - np.arctan(3.0 / 1.0)]
        z = angle(x, deg=True)
        zo = np.array(yo) * 180 / np.pi
        assert_array_almost_equal(y, yo, 11)
        assert_array_almost_equal(z, zo, 11)

    def test_subclass(self):
        x = np.ma.array([1 + 3j, 1, np.sqrt(2) / 2 * (1 + 1j)])
        x[1] = np.ma.masked
        expected = np.ma.array([np.arctan(3.0 / 1.0), 0, np.arctan(1.0)])
        expected[1] = np.ma.masked
        actual = angle(x)
        assert_equal(type(actual), type(expected))
        assert_equal(actual.mask, expected.mask)
        assert_equal(actual, expected)


class TestTrimZeros:

    a = np.array([0, 0, 1, 0, 2, 3, 4, 0])
    b = a.astype(float)
    c = a.astype(complex)
    d = a.astype(object)

    def values(self):
        attr_names = ('a', 'b', 'c', 'd')
        return (getattr(self, name) for name in attr_names)

    def test_basic(self):
        slc = np.s_[2:-1]
        for arr in self.values():
            res = trim_zeros(arr)
            assert_array_equal(res, arr[slc])

    def test_leading_skip(self):
        slc = np.s_[:-1]
        for arr in self.values():
            res = trim_zeros(arr, trim='b')
            assert_array_equal(res, arr[slc])

    def test_trailing_skip(self):
        slc = np.s_[2:]
        for arr in self.values():
            res = trim_zeros(arr, trim='F')
            assert_array_equal(res, arr[slc])

    def test_all_zero(self):
        for _arr in self.values():
            arr = np.zeros_like(_arr, dtype=_arr.dtype)

            res1 = trim_zeros(arr, trim='B')
            assert len(res1) == 0

            res2 = trim_zeros(arr, trim='f')
            assert len(res2) == 0

    def test_size_zero(self):
        arr = np.zeros(0)
        res = trim_zeros(arr)
        assert_array_equal(arr, res)

    @pytest.mark.parametrize(
        'arr',
        [np.array([0, 2**62, 0]),
         np.array([0, 2**63, 0]),
         np.array([0, 2**64, 0])]
    )
    def test_overflow(self, arr):
        slc = np.s_[1:2]
        res = trim_zeros(arr)
        assert_array_equal(res, arr[slc])

    def test_no_trim(self):
        arr = np.array([None, 1, None])
        res = trim_zeros(arr)
        assert_array_equal(arr, res)

    def test_list_to_list(self):
        res = trim_zeros(self.a.tolist())
        assert isinstance(res, list)

    @pytest.mark.parametrize("ndim", (0, 1, 2, 3, 10))
    def test_nd_basic(self, ndim):
        a = np.ones((2,) * ndim)
        b = np.pad(a, (2, 1), mode="constant", constant_values=0)
        res = trim_zeros(b, axis=None)
        assert_array_equal(a, res)

    @pytest.mark.parametrize("ndim", (0, 1, 2, 3))
    def test_allzero(self, ndim):
        a = np.zeros((3,) * ndim)
        res = trim_zeros(a, axis=None)
        assert_array_equal(res, np.zeros((0,) * ndim))

    def test_trim_arg(self):
        a = np.array([0, 1, 2, 0])

        res = trim_zeros(a, trim='f')
        assert_array_equal(res, [1, 2, 0])

        res = trim_zeros(a, trim='b')
        assert_array_equal(res, [0, 1, 2])

    @pytest.mark.parametrize("trim", ("front", ""))
    def test_unexpected_trim_value(self, trim):
        arr = self.a
        with pytest.raises(ValueError, match=r"unexpected character\(s\) in `trim`"):
            trim_zeros(arr, trim=trim)


class TestExtins:

    def test_basic(self):
        a = np.array([1, 3, 2, 1, 2, 3, 3])
        b = extract(a > 1, a)
        assert_array_equal(b, [3, 2, 2, 3, 3])

    def test_place(self):
        # Make sure that non-np.ndarray objects
        # raise an error instead of doing nothing
        assert_raises(TypeError, place, [1, 2, 3], [True, False], [0, 1])

        a = np.array([1, 4, 3, 2, 5, 8, 7])
        place(a, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
        assert_array_equal(a, [1, 2, 3, 4, 5, 6, 7])

        place(a, np.zeros(7), [])
        assert_array_equal(a, np.arange(1, 8))

        place(a, [1, 0, 1, 0, 1, 0, 1], [8, 9])
        assert_array_equal(a, [8, 2, 9, 4, 8, 6, 9])
        assert_raises_regex(ValueError, "Cannot insert from an empty array",
                            lambda: place(a, [0, 0, 0, 0, 0, 1, 0], []))

        # See Issue #6974
        a = np.array(['12', '34'])
        place(a, [0, 1], '9')
        assert_array_equal(a, ['12', '9'])

    def test_both(self):
        a = rand(10)
        mask = a > 0.5
        ac = a.copy()
        c = extract(mask, a)
        place(a, mask, 0)
        place(a, mask, c)
        assert_array_equal(a, ac)


# _foo1 and _foo2 are used in some tests in TestVectorize.

def _foo1(x, y=1.0):
    return y * math.floor(x)


def _foo2(x, y=1.0, z=0.0):
    return y * math.floor(x) + z


class TestVectorize:

    def test_simple(self):
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_scalar(self):
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], 5)
        assert_array_equal(r, [5, 8, 1, 4])

    def test_large(self):
        x = np.linspace(-3, 2, 10000)
        f = vectorize(lambda x: x)
        y = f(x)
        assert_array_equal(y, x)

    def test_ufunc(self):
        f = vectorize(math.cos)
        args = np.array([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
        r1 = f(args)
        r2 = np.cos(args)
        assert_array_almost_equal(r1, r2)

    def test_keywords(self):

        def foo(a, b=1):
            return a + b

        f = vectorize(foo)
        args = np.array([1, 2, 3])
        r1 = f(args)
        r2 = np.array([2, 3, 4])
        assert_array_equal(r1, r2)
        r1 = f(args, 2)
        r2 = np.array([3, 4, 5])
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order1(self):
        # gh-1620: The second call of f would crash with
        # `ValueError: invalid number of arguments`.
        f = vectorize(_foo1, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(np.arange(3.0), 1.0)
        r2 = f(np.arange(3.0))
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order2(self):
        # gh-1620: The second call of f would crash with
        # `ValueError: non-broadcastable output operand with shape ()
        # doesn't match the broadcast shape (3,)`.
        f = vectorize(_foo1, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(np.arange(3.0))
        r2 = f(np.arange(3.0), 1.0)
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order3(self):
        # gh-1620: The third call of f would crash with
        # `ValueError: invalid number of arguments`.
        f = vectorize(_foo1, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(np.arange(3.0))
        r2 = f(np.arange(3.0), y=1.0)
        r3 = f(np.arange(3.0))
        assert_array_equal(r1, r2)
        assert_array_equal(r1, r3)

    def test_keywords_with_otypes_several_kwd_args1(self):
        # gh-1620 Make sure different uses of keyword arguments
        # don't break the vectorized function.
        f = vectorize(_foo2, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(10.4, z=100)
        r2 = f(10.4, y=-1)
        r3 = f(10.4)
        assert_equal(r1, _foo2(10.4, z=100))
        assert_equal(r2, _foo2(10.4, y=-1))
        assert_equal(r3, _foo2(10.4))

    def test_keywords_with_otypes_several_kwd_args2(self):
        # gh-1620 Make sure different uses of keyword arguments
        # don't break the vectorized function.
        f = vectorize(_foo2, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(z=100, x=10.4, y=-1)
        r2 = f(1, 2, 3)
        assert_equal(r1, _foo2(z=100, x=10.4, y=-1))
        assert_equal(r2, _foo2(1, 2, 3))

    def test_keywords_no_func_code(self):
        # This needs to test a function that has keywords but
        # no func_code attribute, since otherwise vectorize will
        # inspect the func_code.
        import random
        try:
            vectorize(random.randrange)  # Should succeed
        except Exception:
            raise AssertionError

    def test_keywords2_ticket_2100(self):
        # Test kwarg support: enhancement ticket 2100

        def foo(a, b=1):
            return a + b

        f = vectorize(foo)
        args = np.array([1, 2, 3])
        r1 = f(a=args)
        r2 = np.array([2, 3, 4])
        assert_array_equal(r1, r2)
        r1 = f(b=1, a=args)
        assert_array_equal(r1, r2)
        r1 = f(args, b=2)
        r2 = np.array([3, 4, 5])
        assert_array_equal(r1, r2)

    def test_keywords3_ticket_2100(self):
        # Test excluded with mixed positional and kwargs: ticket 2100
        def mypolyval(x, p):
            _p = list(p)
            res = _p.pop(0)
            while _p:
                res = res * x + _p.pop(0)
            return res

        vpolyval = np.vectorize(mypolyval, excluded=['p', 1])
        ans = [3, 6]
        assert_array_equal(ans, vpolyval(x=[0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], [1, 2, 3]))

    def test_keywords4_ticket_2100(self):
        # Test vectorizing function with no positional args.
        @vectorize
        def f(**kw):
            res = 1.0
            for _k in kw:
                res *= kw[_k]
            return res

        assert_array_equal(f(a=[1, 2], b=[3, 4]), [3, 8])

    def test_keywords5_ticket_2100(self):
        # Test vectorizing function with no kwargs args.
        @vectorize
        def f(*v):
            return np.prod(v)

        assert_array_equal(f([1, 2], [3, 4]), [3, 8])

    def test_coverage1_ticket_2100(self):
        def foo():
            return 1

        f = vectorize(foo)
        assert_array_equal(f(), 1)

    def test_assigning_docstring(self):
        def foo(x):
            """Original documentation"""
            return x

        f = vectorize(foo)
        assert_equal(f.__doc__, foo.__doc__)

        doc = "Provided documentation"
        f = vectorize(foo, doc=doc)
        assert_equal(f.__doc__, doc)

    def test_UnboundMethod_ticket_1156(self):
        # Regression test for issue 1156
        class Foo:
            b = 2

            def bar(self, a):
                return a ** self.b

        assert_array_equal(vectorize(Foo().bar)(np.arange(9)),
                           np.arange(9) ** 2)
        assert_array_equal(vectorize(Foo.bar)(Foo(), np.arange(9)),
                           np.arange(9) ** 2)

    def test_execution_order_ticket_1487(self):
        # Regression test for dependence on execution order: issue 1487
        f1 = vectorize(lambda x: x)
        res1a = f1(np.arange(3))
        res1b = f1(np.arange(0.1, 3))
        f2 = vectorize(lambda x: x)
        res2b = f2(np.arange(0.1, 3))
        res2a = f2(np.arange(3))
        assert_equal(res1a, res2a)
        assert_equal(res1b, res2b)

    def test_string_ticket_1892(self):
        # Test vectorization over strings: issue 1892.
        f = np.vectorize(lambda x: x)
        s = '0123456789' * 10
        assert_equal(s, f(s))

    def test_cache(self):
        # Ensure that vectorized func called exactly once per argument.
        _calls = [0]

        @vectorize
        def f(x):
            _calls[0] += 1
            return x ** 2

        f.cache = True
        x = np.arange(5)
        assert_array_equal(f(x), x * x)
        assert_equal(_calls[0], len(x))

    def test_otypes(self):
        f = np.vectorize(lambda x: x)
        f.otypes = 'i'
        x = np.arange(5)
        assert_array_equal(f(x), x)

    def test_otypes_object_28624(self):
        # with object otype, the vectorized function should return y
        # wrapped into an object array
        y = np.arange(3)
        f = vectorize(lambda x: y, otypes=[object])

        assert f(None).item() is y
        assert f([None]).item() is y

        y = [1, 2, 3]
        f = vectorize(lambda x: y, otypes=[object])

        assert f(None).item() is y
        assert f([None]).item() is y

    def test_parse_gufunc_signature(self):
        assert_equal(nfb._parse_gufunc_signature('(x)->()'), ([('x',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x,y)->()'),
                     ([('x', 'y')], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x),(y)->()'),
                     ([('x',), ('y',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x)->(y)'),
                     ([('x',)], [('y',)]))
        assert_equal(nfb._parse_gufunc_signature('(x)->(y),()'),
                     ([('x',)], [('y',), ()]))
        assert_equal(nfb._parse_gufunc_signature('(),(a,b,c),(d)->(d,e)'),
                     ([(), ('a', 'b', 'c'), ('d',)], [('d', 'e')]))

        # Tests to check if whitespaces are ignored
        assert_equal(nfb._parse_gufunc_signature('(x )->()'), ([('x',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('( x , y )->(  )'),
                     ([('x', 'y')], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x),( y) ->()'),
                     ([('x',), ('y',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('(  x)-> (y )  '),
                     ([('x',)], [('y',)]))
        assert_equal(nfb._parse_gufunc_signature(' (x)->( y),( )'),
                     ([('x',)], [('y',), ()]))
        assert_equal(nfb._parse_gufunc_signature(
                     '(  ), ( a,  b,c )  ,(  d)   ->   (d  ,  e)'),
                     ([(), ('a', 'b', 'c'), ('d',)], [('d', 'e')]))

        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('(x)(y)->()')
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('(x),(y)->')
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('((x))->(x)')

    def test_signature_simple(self):
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        f = vectorize(addsubtract, signature='(),()->()')
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_signature_mean_last(self):
        def mean(a):
            return a.mean()

        f = vectorize(mean, signature='(n)->()')
        r = f([[1, 3], [2, 4]])
        assert_array_equal(r, [2, 3])

    def test_signature_center(self):
        def center(a):
            return a - a.mean()

        f = vectorize(center, signature='(n)->(n)')
        r = f([[1, 3], [2, 4]])
        assert_array_equal(r, [[-1, 1], [-1, 1]])

    def test_signature_two_outputs(self):
        f = vectorize(lambda x: (x, x), signature='()->(),()')
        r = f([1, 2, 3])
        assert_(isinstance(r, tuple) and len(r) == 2)
        assert_array_equal(r[0], [1, 2, 3])
        assert_array_equal(r[1], [1, 2, 3])

    def test_signature_outer(self):
        f = vectorize(np.outer, signature='(a),(b)->(a,b)')
        r = f([1, 2], [1, 2, 3])
        assert_array_equal(r, [[1, 2, 3], [2, 4, 6]])

        r = f([[[1, 2]]], [1, 2, 3])
        assert_array_equal(r, [[[[1, 2, 3], [2, 4, 6]]]])

        r = f([[1, 0], [2, 0]], [1, 2, 3])
        assert_array_equal(r, [[[1, 2, 3], [0, 0, 0]],
                               [[2, 4, 6], [0, 0, 0]]])

        r = f([1, 2], [[1, 2, 3], [0, 0, 0]])
        assert_array_equal(r, [[[1, 2, 3], [2, 4, 6]],
                               [[0, 0, 0], [0, 0, 0]]])

    def test_signature_computed_size(self):
        f = vectorize(lambda x: x[:-1], signature='(n)->(m)')
        r = f([1, 2, 3])
        assert_array_equal(r, [1, 2])

        r = f([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(r, [[1, 2], [2, 3]])

    def test_signature_excluded(self):

        def foo(a, b=1):
            return a + b

        f = vectorize(foo, signature='()->()', excluded={'b'})
        assert_array_equal(f([1, 2, 3]), [2, 3, 4])
        assert_array_equal(f([1, 2, 3], b=0), [1, 2, 3])

    def test_signature_otypes(self):
        f = vectorize(lambda x: x, signature='(n)->(n)', otypes=['float64'])
        r = f([1, 2, 3])
        assert_equal(r.dtype, np.dtype('float64'))
        assert_array_equal(r, [1, 2, 3])

    def test_signature_invalid_inputs(self):
        f = vectorize(operator.add, signature='(n),(n)->(n)')
        with assert_raises_regex(TypeError, 'wrong number of positional'):
            f([1, 2])
        with assert_raises_regex(
                ValueError, 'does not have enough dimensions'):
            f(1, 2)
        with assert_raises_regex(
                ValueError, 'inconsistent size for core dimension'):
            f([1, 2], [1, 2, 3])

        f = vectorize(operator.add, signature='()->()')
        with assert_raises_regex(TypeError, 'wrong number of positional'):
            f(1, 2)

    def test_signature_invalid_outputs(self):

        f = vectorize(lambda x: x[:-1], signature='(n)->(n)')
        with assert_raises_regex(
                ValueError, 'inconsistent size for core dimension'):
            f([1, 2, 3])

        f = vectorize(lambda x: x, signature='()->(),()')
        with assert_raises_regex(ValueError, 'wrong number of outputs'):
            f(1)

        f = vectorize(lambda x: (x, x), signature='()->()')
        with assert_raises_regex(ValueError, 'wrong number of outputs'):
            f([1, 2])

    def test_size_zero_output(self):
        # see issue 5868
        f = np.vectorize(lambda x: x)
        x = np.zeros([0, 5], dtype=int)
        with assert_raises_regex(ValueError, 'otypes'):
            f(x)

        f.otypes = 'i'
        assert_array_equal(f(x), x)

        f = np.vectorize(lambda x: x, signature='()->()')
        with assert_raises_regex(ValueError, 'otypes'):
            f(x)

        f = np.vectorize(lambda x: x, signature='()->()', otypes='i')
        assert_array_equal(f(x), x)

        f = np.vectorize(lambda x: x, signature='(n)->(n)', otypes='i')
        assert_array_equal(f(x), x)

        f = np.vectorize(lambda x: x, signature='(n)->(n)')
        assert_array_equal(f(x.T), x.T)

        f = np.vectorize(lambda x: [x], signature='()->(n)', otypes='i')
        with assert_raises_regex(ValueError, 'new output dimensions'):
            f(x)

    def test_subclasses(self):
        class subclass(np.ndarray):
            pass

        m = np.array([[1., 0., 0.],
                      [0., 0., 1.],
                      [0., 1., 0.]]).view(subclass)
        v = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).view(subclass)
        # generalized (gufunc)
        matvec = np.vectorize(np.matmul, signature='(m,m),(m)->(m)')
        r = matvec(m, v)
        assert_equal(type(r), subclass)
        assert_equal(r, [[1., 3., 2.], [4., 6., 5.], [7., 9., 8.]])

        # element-wise (ufunc)
        mult = np.vectorize(lambda x, y: x * y)
        r = mult(m, v)
        assert_equal(type(r), subclass)
        assert_equal(r, m * v)

    def test_name(self):
        # gh-23021
        @np.vectorize
        def f2(a, b):
            return a + b

        assert f2.__name__ == 'f2'

    def test_decorator(self):
        @vectorize
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        r = addsubtract([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_docstring(self):
        @vectorize
        def f(x):
            """Docstring"""
            return x

        if sys.flags.optimize < 2:
            assert f.__doc__ == "Docstring"

    def test_partial(self):
        def foo(x, y):
            return x + y

        bar = partial(foo, 3)
        vbar = np.vectorize(bar)
        assert vbar(1) == 4

    def test_signature_otypes_decorator(self):
        @vectorize(signature='(n)->(n)', otypes=['float64'])
        def f(x):
            return x

        r = f([1, 2, 3])
        assert_equal(r.dtype, np.dtype('float64'))
        assert_array_equal(r, [1, 2, 3])
        assert f.__name__ == 'f'

    def test_bad_input(self):
        with assert_raises(TypeError):
            A = np.vectorize(pyfunc=3)

    def test_no_keywords(self):
        with assert_raises(TypeError):
            @np.vectorize("string")
            def foo():
                return "bar"

    def test_positional_regression_9477(self):
        # This supplies the first keyword argument as a positional,
        # to ensure that they are still properly forwarded after the
        # enhancement for #9477
        f = vectorize((lambda x: x), ['float64'])
        r = f([2])
        assert_equal(r.dtype, np.dtype('float64'))

    def test_datetime_conversion(self):
        otype = "datetime64[ns]"
        arr = np.array(['2024-01-01', '2024-01-02', '2024-01-03'],
                       dtype='datetime64[ns]')
        assert_array_equal(np.vectorize(lambda x: x, signature="(i)->(j)",
                                        otypes=[otype])(arr), arr)


class TestLeaks:
    class A:
        iters = 20

        def bound(self, *args):
            return 0

        @staticmethod
        def unbound(*args):
            return 0

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    @pytest.mark.skipif(NOGIL_BUILD,
                        reason=("Functions are immortalized if a thread is "
                                "launched, making this test flaky"))
    @pytest.mark.parametrize('name, incr', [
            ('bound', A.iters),
            ('unbound', 0),
            ])
    def test_frompyfunc_leaks(self, name, incr):
        # exposed in gh-11867 as np.vectorized, but the problem stems from
        # frompyfunc.
        # class.attribute = np.frompyfunc(<method>) creates a
        # reference cycle if <method> is a bound class method.
        # It requires a gc collection cycle to break the cycle.
        import gc
        A_func = getattr(self.A, name)
        gc.disable()
        try:
            refcount = sys.getrefcount(A_func)
            for i in range(self.A.iters):
                a = self.A()
                a.f = np.frompyfunc(getattr(a, name), 1, 1)
                out = a.f(np.arange(10))
            a = None
            # A.func is part of a reference cycle if incr is non-zero
            assert_equal(sys.getrefcount(A_func), refcount + incr)
            for i in range(5):
                gc.collect()
            assert_equal(sys.getrefcount(A_func), refcount)
        finally:
            gc.enable()


class TestDigitize:

    def test_forward(self):
        x = np.arange(-6, 5)
        bins = np.arange(-5, 5)
        assert_array_equal(digitize(x, bins), np.arange(11))

    def test_reverse(self):
        x = np.arange(5, -6, -1)
        bins = np.arange(5, -5, -1)
        assert_array_equal(digitize(x, bins), np.arange(11))

    def test_random(self):
        x = rand(10)
        bin = np.linspace(x.min(), x.max(), 10)
        assert_(np.all(digitize(x, bin) != 0))

    def test_right_basic(self):
        x = [1, 5, 4, 10, 8, 11, 0]
        bins = [1, 5, 10]
        default_answer = [1, 2, 1, 3, 2, 3, 0]
        assert_array_equal(digitize(x, bins), default_answer)
        right_answer = [0, 1, 1, 2, 2, 3, 0]
        assert_array_equal(digitize(x, bins, True), right_answer)

    def test_right_open(self):
        x = np.arange(-6, 5)
        bins = np.arange(-6, 4)
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    def test_right_open_reverse(self):
        x = np.arange(5, -6, -1)
        bins = np.arange(4, -6, -1)
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    def test_right_open_random(self):
        x = rand(10)
        bins = np.linspace(x.min(), x.max(), 10)
        assert_(np.all(digitize(x, bins, True) != 10))

    def test_monotonic(self):
        x = [-1, 0, 1, 2]
        bins = [0, 0, 1]
        assert_array_equal(digitize(x, bins, False), [0, 2, 3, 3])
        assert_array_equal(digitize(x, bins, True), [0, 0, 2, 3])
        bins = [1, 1, 0]
        assert_array_equal(digitize(x, bins, False), [3, 2, 0, 0])
        assert_array_equal(digitize(x, bins, True), [3, 3, 2, 0])
        bins = [1, 1, 1, 1]
        assert_array_equal(digitize(x, bins, False), [0, 0, 4, 4])
        assert_array_equal(digitize(x, bins, True), [0, 0, 0, 4])
        bins = [0, 0, 1, 0]
        assert_raises(ValueError, digitize, x, bins)
        bins = [1, 1, 0, 1]
        assert_raises(ValueError, digitize, x, bins)

    def test_casting_error(self):
        x = [1, 2, 3 + 1.j]
        bins = [1, 2, 3]
        assert_raises(TypeError, digitize, x, bins)
        x, bins = bins, x
        assert_raises(TypeError, digitize, x, bins)

    def test_return_type(self):
        # Functions returning indices should always return base ndarrays
        class A(np.ndarray):
            pass
        a = np.arange(5).view(A)
        b = np.arange(1, 3).view(A)
        assert_(not isinstance(digitize(b, a, False), A))
        assert_(not isinstance(digitize(b, a, True), A))

    def test_large_integers_increasing(self):
        # gh-11022
        x = 2**54  # loses precision in a float
        assert_equal(np.digitize(x, [x - 1, x + 1]), 1)

    @pytest.mark.xfail(
        reason="gh-11022: np._core.multiarray._monoticity loses precision")
    def test_large_integers_decreasing(self):
        # gh-11022
        x = 2**54  # loses precision in a float
        assert_equal(np.digitize(x, [x + 1, x - 1]), 1)


class TestUnwrap:

    def test_simple(self):
        # check that unwrap removes jumps greater that 2*pi
        assert_array_equal(unwrap([1, 1 + 2 * np.pi]), [1, 1])
        # check that unwrap maintains continuity
        assert_(np.all(diff(unwrap(rand(10) * 100)) < np.pi))

    def test_period(self):
        # check that unwrap removes jumps greater that 255
        assert_array_equal(unwrap([1, 1 + 256], period=255), [1, 2])
        # check that unwrap maintains continuity
        assert_(np.all(diff(unwrap(rand(10) * 1000, period=255)) < 255))
        # check simple case
        simple_seq = np.array([0, 75, 150, 225, 300])
        wrap_seq = np.mod(simple_seq, 255)
        assert_array_equal(unwrap(wrap_seq, period=255), simple_seq)
        # check custom discont value
        uneven_seq = np.array([0, 75, 150, 225, 300, 430])
        wrap_uneven = np.mod(uneven_seq, 250)
        no_discont = unwrap(wrap_uneven, period=250)
        assert_array_equal(no_discont, [0, 75, 150, 225, 300, 180])
        sm_discont = unwrap(wrap_uneven, period=250, discont=140)
        assert_array_equal(sm_discont, [0, 75, 150, 225, 300, 430])
        assert sm_discont.dtype == wrap_uneven.dtype


@pytest.mark.parametrize(
    "dtype", "O" + np.typecodes["AllInteger"] + np.typecodes["Float"]
)
@pytest.mark.parametrize("M", [0, 1, 10])
class TestFilterwindows:

    def test_hanning(self, dtype: str, M: int) -> None:
        scalar = np.array(M, dtype=dtype)[()]

        w = hanning(scalar)
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_equal(w, flipud(w))

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.500, 4)

    def test_hamming(self, dtype: str, M: int) -> None:
        scalar = np.array(M, dtype=dtype)[()]

        w = hamming(scalar)
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_equal(w, flipud(w))

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.9400, 4)

    def test_bartlett(self, dtype: str, M: int) -> None:
        scalar = np.array(M, dtype=dtype)[()]

        w = bartlett(scalar)
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_equal(w, flipud(w))

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.4444, 4)

    def test_blackman(self, dtype: str, M: int) -> None:
        scalar = np.array(M, dtype=dtype)[()]

        w = blackman(scalar)
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_equal(w, flipud(w))

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 3.7800, 4)

    def test_kaiser(self, dtype: str, M: int) -> None:
        scalar = np.array(M, dtype=dtype)[()]

        w = kaiser(scalar, 0)
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_equal(w, flipud(w))

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 10, 15)


class TestTrapezoid:

    def test_simple(self):
        x = np.arange(-10, 10, .1)
        r = trapezoid(np.exp(-.5 * x ** 2) / np.sqrt(2 * np.pi), dx=0.1)
        # check integral of normal equals 1
        assert_almost_equal(r, 1, 7)

    def test_ndim(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)

        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        q = x[:, None, None] + y[None, :, None] + z[None, None, :]

        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # n-d `x`
        r = trapezoid(q, x=x[:, None, None], axis=0)
        assert_almost_equal(r, qx)
        r = trapezoid(q, x=y[None, :, None], axis=1)
        assert_almost_equal(r, qy)
        r = trapezoid(q, x=z[None, None, :], axis=2)
        assert_almost_equal(r, qz)

        # 1-d `x`
        r = trapezoid(q, x=x, axis=0)
        assert_almost_equal(r, qx)
        r = trapezoid(q, x=y, axis=1)
        assert_almost_equal(r, qy)
        r = trapezoid(q, x=z, axis=2)
        assert_almost_equal(r, qz)

    def test_masked(self):
        # Testing that masked arrays behave as if the function is 0 where
        # masked
        x = np.arange(5)
        y = x * x
        mask = x == 2
        ym = np.ma.array(y, mask=mask)
        r = 13.0  # sum(0.5 * (0 + 1) * 1.0 + 0.5 * (9 + 16))
        assert_almost_equal(trapezoid(ym, x), r)

        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapezoid(ym, xm), r)

        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapezoid(y, xm), r)


class TestSinc:

    def test_simple(self):
        assert_(sinc(0) == 1)
        w = sinc(np.linspace(-1, 1, 100))
        # check symmetry
        assert_array_almost_equal(w, flipud(w), 7)

    def test_array_like(self):
        x = [0, 0.5]
        y1 = sinc(np.array(x))
        y2 = sinc(list(x))
        y3 = sinc(tuple(x))
        assert_array_equal(y1, y2)
        assert_array_equal(y1, y3)

    def test_bool_dtype(self):
        x = (np.arange(4, dtype=np.uint8) % 2 == 1)
        actual = sinc(x)
        expected = sinc(x.astype(np.float64))
        assert_allclose(actual, expected)
        assert actual.dtype == np.float64

    @pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.uint64])
    def test_int_dtypes(self, dtype):
        x = np.arange(4, dtype=dtype)
        actual = sinc(x)
        expected = sinc(x.astype(np.float64))
        assert_allclose(actual, expected)
        assert actual.dtype == np.float64

    @pytest.mark.parametrize(
            'dtype',
            [np.float16, np.float32, np.longdouble, np.complex64, np.complex128]
    )
    def test_float_dtypes(self, dtype):
        x = np.arange(4, dtype=dtype)
        assert sinc(x).dtype == x.dtype

    def test_float16_underflow(self):
        x = np.float16(0)
        # before gh-27784, fill value for 0 in input would underflow float16,
        # resulting in nan
        assert_array_equal(sinc(x), np.asarray(1.0))

class TestUnique:

    def test_simple(self):
        x = np.array([4, 3, 2, 1, 1, 2, 3, 4, 0])
        assert_(np.all(unique(x) == [0, 1, 2, 3, 4]))
        assert_(unique(np.array([1, 1, 1, 1, 1])) == np.array([1]))
        x = ['widget', 'ham', 'foo', 'bar', 'foo', 'ham']
        assert_(np.all(unique(x) == ['bar', 'foo', 'ham', 'widget']))
        x = np.array([5 + 6j, 1 + 1j, 1 + 10j, 10, 5 + 6j])
        assert_(np.all(unique(x) == [1 + 1j, 1 + 10j, 5 + 6j, 10]))


class TestCheckFinite:

    def test_simple(self):
        a = [1, 2, 3]
        b = [1, 2, np.inf]
        c = [1, 2, np.nan]
        np.asarray_chkfinite(a)
        assert_raises(ValueError, np.asarray_chkfinite, b)
        assert_raises(ValueError, np.asarray_chkfinite, c)

    def test_dtype_order(self):
        # Regression test for missing dtype and order arguments
        a = [1, 2, 3]
        a = np.asarray_chkfinite(a, order='F', dtype=np.float64)
        assert_(a.dtype == np.float64)


class TestCorrCoef:
    A = np.array(
        [[0.15391142, 0.18045767, 0.14197213],
         [0.70461506, 0.96474128, 0.27906989],
         [0.9297531, 0.32296769, 0.19267156]])
    B = np.array(
        [[0.10377691, 0.5417086, 0.49807457],
         [0.82872117, 0.77801674, 0.39226705],
         [0.9314666, 0.66800209, 0.03538394]])
    res1 = np.array(
        [[1., 0.9379533, -0.04931983],
         [0.9379533, 1., 0.30007991],
         [-0.04931983, 0.30007991, 1.]])
    res2 = np.array(
        [[1., 0.9379533, -0.04931983, 0.30151751, 0.66318558, 0.51532523],
         [0.9379533, 1., 0.30007991, -0.04781421, 0.88157256, 0.78052386],
         [-0.04931983, 0.30007991, 1., -0.96717111, 0.71483595, 0.83053601],
         [0.30151751, -0.04781421, -0.96717111, 1., -0.51366032, -0.66173113],
         [0.66318558, 0.88157256, 0.71483595, -0.51366032, 1., 0.98317823],
         [0.51532523, 0.78052386, 0.83053601, -0.66173113, 0.98317823, 1.]])

    def test_non_array(self):
        assert_almost_equal(np.corrcoef([0, 1, 0], [1, 0, 1]),
                            [[1., -1.], [-1., 1.]])

    def test_simple(self):
        tgt1 = corrcoef(self.A)
        assert_almost_equal(tgt1, self.res1)
        assert_(np.all(np.abs(tgt1) <= 1.0))

        tgt2 = corrcoef(self.A, self.B)
        assert_almost_equal(tgt2, self.res2)
        assert_(np.all(np.abs(tgt2) <= 1.0))

    def test_ddof(self):
        # ddof raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, ddof=-1)
            sup.filter(DeprecationWarning)
            # ddof has no or negligible effect on the function
            assert_almost_equal(corrcoef(self.A, ddof=-1), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=-1), self.res2)
            assert_almost_equal(corrcoef(self.A, ddof=3), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=3), self.res2)

    def test_bias(self):
        # bias raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, self.B, 1, 0)
            assert_warns(DeprecationWarning, corrcoef, self.A, bias=0)
            sup.filter(DeprecationWarning)
            # bias has no or negligible effect on the function
            assert_almost_equal(corrcoef(self.A, bias=1), self.res1)

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = corrcoef(x)
        tgt = np.array([[1., -1.j], [1.j, 1.]])
        assert_allclose(res, tgt)
        assert_(np.all(np.abs(res) <= 1.0))

    def test_xy(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(np.corrcoef(x, y), np.array([[1., -1.j], [1.j, 1.]]))

    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', RuntimeWarning)
            assert_array_equal(corrcoef(np.array([])), np.nan)
            assert_array_equal(corrcoef(np.array([]).reshape(0, 2)),
                               np.array([]).reshape(0, 0))
            assert_array_equal(corrcoef(np.array([]).reshape(2, 0)),
                               np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    def test_extreme(self):
        x = [[1e-100, 1e100], [1e100, 1e-100]]
        with np.errstate(all='raise'):
            c = corrcoef(x)
        assert_array_almost_equal(c, np.array([[1., -1.], [-1., 1.]]))
        assert_(np.all(np.abs(c) <= 1.0))

    @pytest.mark.parametrize("test_type", [np.half, np.single, np.double, np.longdouble])
    def test_corrcoef_dtype(self, test_type):
        cast_A = self.A.astype(test_type)
        res = corrcoef(cast_A, dtype=test_type)
        assert test_type == res.dtype


class TestCov:
    x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
    res1 = np.array([[1., -1.], [-1., 1.]])
    x2 = np.array([0.0, 1.0, 2.0], ndmin=2)
    frequencies = np.array([1, 4, 1])
    x2_repeats = np.array([[0.0], [1.0], [1.0], [1.0], [1.0], [2.0]]).T
    res2 = np.array([[0.4, -0.4], [-0.4, 0.4]])
    unit_frequencies = np.ones(3, dtype=np.int_)
    weights = np.array([1.0, 4.0, 1.0])
    res3 = np.array([[2. / 3., -2. / 3.], [-2. / 3., 2. / 3.]])
    unit_weights = np.ones(3)
    x3 = np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])

    def test_basic(self):
        assert_allclose(cov(self.x1), self.res1)

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = np.array([[1., -1.j], [1.j, 1.]])
        assert_allclose(cov(x), res)
        assert_allclose(cov(x, aweights=np.ones(3)), res)

    def test_xy(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(cov(x, y), np.array([[1., -1.j], [1.j, 1.]]))

    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', RuntimeWarning)
            assert_array_equal(cov(np.array([])), np.nan)
            assert_array_equal(cov(np.array([]).reshape(0, 2)),
                               np.array([]).reshape(0, 0))
            assert_array_equal(cov(np.array([]).reshape(2, 0)),
                               np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    def test_wrong_ddof(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', RuntimeWarning)
            assert_array_equal(cov(self.x1, ddof=5),
                               np.array([[np.inf, -np.inf],
                                         [-np.inf, np.inf]]))

    def test_1D_rowvar(self):
        assert_allclose(cov(self.x3), cov(self.x3, rowvar=False))
        y = np.array([0.0780, 0.3107, 0.2111, 0.0334, 0.8501])
        assert_allclose(cov(self.x3, y), cov(self.x3, y, rowvar=False))

    def test_1D_variance(self):
        assert_allclose(cov(self.x3, ddof=1), np.var(self.x3, ddof=1))

    def test_fweights(self):
        assert_allclose(cov(self.x2, fweights=self.frequencies),
                        cov(self.x2_repeats))
        assert_allclose(cov(self.x1, fweights=self.frequencies),
                        self.res2)
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies),
                        self.res1)
        nonint = self.frequencies + 0.5
        assert_raises(TypeError, cov, self.x1, fweights=nonint)
        f = np.ones((2, 3), dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = np.ones(2, dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = -1 * np.ones(3, dtype=np.int_)
        assert_raises(ValueError, cov, self.x1, fweights=f)

    def test_aweights(self):
        assert_allclose(cov(self.x1, aweights=self.weights), self.res3)
        assert_allclose(cov(self.x1, aweights=3.0 * self.weights),
                        cov(self.x1, aweights=self.weights))
        assert_allclose(cov(self.x1, aweights=self.unit_weights), self.res1)
        w = np.ones((2, 3))
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        w = np.ones(2)
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        w = -1.0 * np.ones(3)
        assert_raises(ValueError, cov, self.x1, aweights=w)

    def test_unit_fweights_and_aweights(self):
        assert_allclose(cov(self.x2, fweights=self.frequencies,
                            aweights=self.unit_weights),
                        cov(self.x2_repeats))
        assert_allclose(cov(self.x1, fweights=self.frequencies,
                            aweights=self.unit_weights),
                        self.res2)
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=self.unit_weights),
                        self.res1)
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=self.weights),
                        self.res3)
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=3.0 * self.weights),
                        cov(self.x1, aweights=self.weights))
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=self.unit_weights),
                        self.res1)

    @pytest.mark.parametrize("test_type", [np.half, np.single, np.double, np.longdouble])
    def test_cov_dtype(self, test_type):
        cast_x1 = self.x1.astype(test_type)
        res = cov(cast_x1, dtype=test_type)
        assert test_type == res.dtype

    def test_gh_27658(self):
        x = np.ones((3, 1))
        expected = np.cov(x, ddof=0, rowvar=True)
        actual = np.cov(x.T, ddof=0, rowvar=False)
        assert_allclose(actual, expected, strict=True)


class Test_I0:

    def test_simple(self):
        assert_almost_equal(
            i0(0.5),
            np.array(1.0634833707413234))

        # need at least one test above 8, as the implementation is piecewise
        A = np.array([0.49842636, 0.6969809, 0.22011976, 0.0155549, 10.0])
        expected = np.array([1.06307822, 1.12518299, 1.01214991, 1.00006049, 2815.71662847])
        assert_almost_equal(i0(A), expected)
        assert_almost_equal(i0(-A), expected)

        B = np.array([[0.827002, 0.99959078],
                      [0.89694769, 0.39298162],
                      [0.37954418, 0.05206293],
                      [0.36465447, 0.72446427],
                      [0.48164949, 0.50324519]])
        assert_almost_equal(
            i0(B),
            np.array([[1.17843223, 1.26583466],
                      [1.21147086, 1.03898290],
                      [1.03633899, 1.00067775],
                      [1.03352052, 1.13557954],
                      [1.05884290, 1.06432317]]))
        # Regression test for gh-11205
        i0_0 = np.i0([0.])
        assert_equal(i0_0.shape, (1,))
        assert_array_equal(np.i0([0.]), np.array([1.]))

    def test_non_array(self):
        a = np.arange(4)

        class array_like:
            __array_interface__ = a.__array_interface__

            def __array_wrap__(self, arr, context, return_scalar):
                return self

        # E.g. pandas series survive ufunc calls through array-wrap:
        assert isinstance(np.abs(array_like()), array_like)
        exp = np.i0(a)
        res = np.i0(array_like())

        assert_array_equal(exp, res)

    def test_complex(self):
        a = np.array([0, 1 + 2j])
        with pytest.raises(TypeError, match="i0 not supported for complex values"):
            res = i0(a)


class TestKaiser:

    def test_simple(self):
        assert_(np.isfinite(kaiser(1, 1.0)))
        assert_almost_equal(kaiser(0, 1.0),
                            np.array([]))
        assert_almost_equal(kaiser(2, 1.0),
                            np.array([0.78984831, 0.78984831]))
        assert_almost_equal(kaiser(5, 1.0),
                            np.array([0.78984831, 0.94503323, 1.,
                                      0.94503323, 0.78984831]))
        assert_almost_equal(kaiser(5, 1.56789),
                            np.array([0.58285404, 0.88409679, 1.,
                                      0.88409679, 0.58285404]))

    def test_int_beta(self):
        kaiser(3, 4)


class TestMeshgrid:

    def test_simple(self):
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7])
        assert_array_equal(X, np.array([[1, 2, 3],
                                        [1, 2, 3],
                                        [1, 2, 3],
                                        [1, 2, 3]]))
        assert_array_equal(Y, np.array([[4, 4, 4],
                                        [5, 5, 5],
                                        [6, 6, 6],
                                        [7, 7, 7]]))

    def test_single_input(self):
        [X] = meshgrid([1, 2, 3, 4])
        assert_array_equal(X, np.array([1, 2, 3, 4]))

    def test_no_input(self):
        args = []
        assert_array_equal([], meshgrid(*args))
        assert_array_equal([], meshgrid(*args, copy=False))

    def test_indexing(self):
        x = [1, 2, 3]
        y = [4, 5, 6, 7]
        [X, Y] = meshgrid(x, y, indexing='ij')
        assert_array_equal(X, np.array([[1, 1, 1, 1],
                                        [2, 2, 2, 2],
                                        [3, 3, 3, 3]]))
        assert_array_equal(Y, np.array([[4, 5, 6, 7],
                                        [4, 5, 6, 7],
                                        [4, 5, 6, 7]]))

        # Test expected shapes:
        z = [8, 9]
        assert_(meshgrid(x, y)[0].shape == (4, 3))
        assert_(meshgrid(x, y, indexing='ij')[0].shape == (3, 4))
        assert_(meshgrid(x, y, z)[0].shape == (4, 3, 2))
        assert_(meshgrid(x, y, z, indexing='ij')[0].shape == (3, 4, 2))

        assert_raises(ValueError, meshgrid, x, y, indexing='notvalid')

    def test_sparse(self):
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7], sparse=True)
        assert_array_equal(X, np.array([[1, 2, 3]]))
        assert_array_equal(Y, np.array([[4], [5], [6], [7]]))

    def test_invalid_arguments(self):
        # Test that meshgrid complains about invalid arguments
        # Regression test for issue #4755:
        # https://github.com/numpy/numpy/issues/4755
        assert_raises(TypeError, meshgrid,
                      [1, 2, 3], [4, 5, 6, 7], indices='ij')

    def test_return_type(self):
        # Test for appropriate dtype in returned arrays.
        # Regression test for issue #5297
        # https://github.com/numpy/numpy/issues/5297
        x = np.arange(0, 10, dtype=np.float32)
        y = np.arange(10, 20, dtype=np.float64)

        X, Y = np.meshgrid(x, y)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # copy
        X, Y = np.meshgrid(x, y, copy=True)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # sparse
        X, Y = np.meshgrid(x, y, sparse=True)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

    def test_writeback(self):
        # Issue 8561
        X = np.array([1.1, 2.2])
        Y = np.array([3.3, 4.4])
        x, y = np.meshgrid(X, Y, sparse=False, copy=True)

        x[0, :] = 0
        assert_equal(x[0, :], 0)
        assert_equal(x[1, :], X)

    def test_nd_shape(self):
        a, b, c, d, e = np.meshgrid(*([0] * i for i in range(1, 6)))
        expected_shape = (2, 1, 3, 4, 5)
        assert_equal(a.shape, expected_shape)
        assert_equal(b.shape, expected_shape)
        assert_equal(c.shape, expected_shape)
        assert_equal(d.shape, expected_shape)
        assert_equal(e.shape, expected_shape)

    def test_nd_values(self):
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5])
        assert_equal(a, [[[0, 0, 0]], [[0, 0, 0]]])
        assert_equal(b, [[[1, 1, 1]], [[2, 2, 2]]])
        assert_equal(c, [[[3, 4, 5]], [[3, 4, 5]]])

    def test_nd_indexing(self):
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5], indexing='ij')
        assert_equal(a, [[[0, 0, 0], [0, 0, 0]]])
        assert_equal(b, [[[1, 1, 1], [2, 2, 2]]])
        assert_equal(c, [[[3, 4, 5], [3, 4, 5]]])


class TestPiecewise:

    def test_simple(self):
        # Condition is single bool list
        x = piecewise([0, 0], [True, False], [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: single bool list
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])

        # Conditions is single bool array
        x = piecewise([0, 0], np.array([True, False]), [1])
        assert_array_equal(x, [1, 0])

        # Condition is single int array
        x = piecewise([0, 0], np.array([1, 0]), [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: int array
        x = piecewise([0, 0], [np.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

        x = piecewise([0, 0], [[False, True]], [lambda x:-1])
        assert_array_equal(x, [0, -1])

        assert_raises_regex(ValueError, '1 or 2 functions are expected',
            piecewise, [0, 0], [[False, True]], [])
        assert_raises_regex(ValueError, '1 or 2 functions are expected',
            piecewise, [0, 0], [[False, True]], [1, 2, 3])

    def test_two_conditions(self):
        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_scalar_domains_three_conditions(self):
        x = piecewise(3, [True, False, False], [4, 2, 0])
        assert_equal(x, 4)

    def test_default(self):
        # No value specified for x[1], should be 0
        x = piecewise([1, 2], [True, False], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = piecewise([1, 2], [True, False], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = np.array(3)
        y = piecewise(x, x > 3, [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)

        x = 5
        y = piecewise(x, [True, False], [1, 0])
        assert_(y.ndim == 0)
        assert_(y == 1)

        # With 3 ranges (It was failing, before)
        y = piecewise(x, [False, False, True], [1, 2, 3])
        assert_array_equal(y, 3)

    def test_0d_comparison(self):
        x = 3
        y = piecewise(x, [x <= 3, x > 3], [4, 0])  # Should succeed.
        assert_equal(y, 4)

        # With 3 ranges (It was failing, before)
        x = 4
        y = piecewise(x, [x <= 3, (x > 3) * (x <= 5), x > 5], [1, 2, 3])
        assert_array_equal(y, 2)

        assert_raises_regex(ValueError, '2 or 3 functions are expected',
            piecewise, x, [x <= 3, x > 3], [1])
        assert_raises_regex(ValueError, '2 or 3 functions are expected',
            piecewise, x, [x <= 3, x > 3], [1, 1, 1, 1])

    def test_0d_0d_condition(self):
        x = np.array(3)
        c = np.array(x > 3)
        y = piecewise(x, [c], [1, 2])
        assert_equal(y, 2)

    def test_multidimensional_extrafunc(self):
        x = np.array([[-2.5, -1.5, -0.5],
                      [0.5, 1.5, 2.5]])
        y = piecewise(x, [x < 0, x >= 2], [-1, 1, 3])
        assert_array_equal(y, np.array([[-1., -1., -1.],
                                        [3., 3., 1.]]))

    def test_subclasses(self):
        class subclass(np.ndarray):
            pass
        x = np.arange(5.).view(subclass)
        r = piecewise(x, [x < 2., x >= 4], [-1., 1., 0.])
        assert_equal(type(r), subclass)
        assert_equal(r, [-1., -1., 0., 0., 1.])


class TestBincount:

    def test_simple(self):
        y = np.bincount(np.arange(4))
        assert_array_equal(y, np.ones(4))

    def test_simple2(self):
        y = np.bincount(np.array([1, 5, 2, 4, 1]))
        assert_array_equal(y, np.array([0, 2, 1, 0, 1, 1]))

    def test_simple_weight(self):
        x = np.arange(4)
        w = np.array([0.2, 0.3, 0.5, 0.1])
        y = np.bincount(x, w)
        assert_array_equal(y, w)

    def test_simple_weight2(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1]))

    def test_with_minlength(self):
        x = np.array([0, 1, 0, 1, 1])
        y = np.bincount(x, minlength=3)
        assert_array_equal(y, np.array([2, 3, 0]))
        x = []
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([]))

    def test_with_minlength_smaller_than_maxvalue(self):
        x = np.array([0, 1, 1, 2, 2, 3, 3])
        y = np.bincount(x, minlength=2)
        assert_array_equal(y, np.array([1, 2, 2, 2]))
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([1, 2, 2, 2]))

    def test_with_minlength_and_weights(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w, 8)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0]))

    def test_empty(self):
        x = np.array([], dtype=int)
        y = np.bincount(x)
        assert_array_equal(x, y)

    def test_empty_with_minlength(self):
        x = np.array([], dtype=int)
        y = np.bincount(x, minlength=5)
        assert_array_equal(y, np.zeros(5, dtype=int))

    @pytest.mark.parametrize('minlength', [0, 3])
    def test_empty_list(self, minlength):
        assert_array_equal(np.bincount([], minlength=minlength),
                           np.zeros(minlength, dtype=int))

    def test_with_incorrect_minlength(self):
        x = np.array([], dtype=int)
        assert_raises_regex(TypeError,
                            "'str' object cannot be interpreted",
                            lambda: np.bincount(x, minlength="foobar"))
        assert_raises_regex(ValueError,
                            "must not be negative",
                            lambda: np.bincount(x, minlength=-1))

        x = np.arange(5)
        assert_raises_regex(TypeError,
                            "'str' object cannot be interpreted",
                            lambda: np.bincount(x, minlength="foobar"))
        assert_raises_regex(ValueError,
                            "must not be negative",
                            lambda: np.bincount(x, minlength=-1))

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_dtype_reference_leaks(self):
        # gh-6805
        intp_refcount = sys.getrefcount(np.dtype(np.intp))
        double_refcount = sys.getrefcount(np.dtype(np.double))

        for j in range(10):
            np.bincount([1, 2, 3])
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

        for j in range(10):
            np.bincount([1, 2, 3], [4, 5, 6])
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

    @pytest.mark.parametrize("vals", [[[2, 2]], 2])
    def test_error_not_1d(self, vals):
        # Test that values has to be 1-D (both as array and nested list)
        vals_arr = np.asarray(vals)
        with assert_raises(ValueError):
            np.bincount(vals_arr)
        with assert_raises(ValueError):
            np.bincount(vals)

    @pytest.mark.parametrize("dt", np.typecodes["AllInteger"])
    def test_gh_28354(self, dt):
        a = np.array([0, 1, 1, 3, 2, 1, 7], dtype=dt)
        actual = np.bincount(a)
        expected = [1, 3, 1, 1, 0, 0, 0, 1]
        assert_array_equal(actual, expected)

    def test_contiguous_handling(self):
        # check for absence of hard crash
        np.bincount(np.arange(10000)[::2])

    def test_gh_28354_array_like(self):
        class A:
            def __array__(self):
                return np.array([0, 1, 1, 3, 2, 1, 7], dtype=np.uint64)

        a = A()
        actual = np.bincount(a)
        expected = [1, 3, 1, 1, 0, 0, 0, 1]
        assert_array_equal(actual, expected)


class TestInterp:

    def test_exceptions(self):
        assert_raises(ValueError, interp, 0, [], [])
        assert_raises(ValueError, interp, 0, [0], [1, 2])
        assert_raises(ValueError, interp, 0, [0, 1], [1, 2], period=0)
        assert_raises(ValueError, interp, 0, [], [], period=360)
        assert_raises(ValueError, interp, 0, [0], [1, 2], period=360)

    def test_basic(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.linspace(0, 1, 50)
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_right_left_behavior(self):
        # Needs range of sizes to test different code paths.
        # size ==1 is special cased, 1 < size < 5 is linear search, and
        # size >= 5 goes through local search and possibly binary search.
        for size in range(1, 10):
            xp = np.arange(size, dtype=np.double)
            yp = np.ones(size, dtype=np.double)
            incpts = np.array([-1, 0, size - 1, size], dtype=np.double)
            decpts = incpts[::-1]

            incres = interp(incpts, xp, yp)
            decres = interp(decpts, xp, yp)
            inctgt = np.array([1, 1, 1, 1], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            incres = interp(incpts, xp, yp, left=0)
            decres = interp(decpts, xp, yp, left=0)
            inctgt = np.array([0, 1, 1, 1], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            incres = interp(incpts, xp, yp, right=2)
            decres = interp(decpts, xp, yp, right=2)
            inctgt = np.array([1, 1, 1, 2], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            incres = interp(incpts, xp, yp, left=0, right=2)
            decres = interp(decpts, xp, yp, left=0, right=2)
            inctgt = np.array([0, 1, 1, 2], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

    def test_scalar_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = 0
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = .3
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.float32(.3)
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.float64(.3)
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.nan
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_non_finite_behavior_exact_x(self):
        x = [1, 2, 2.5, 3, 4]
        xp = [1, 2, 3, 4]
        fp = [1, 2, np.inf, 4]
        assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.inf, np.inf, 4])
        fp = [1, 2, np.nan, 4]
        assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.nan, np.nan, 4])

    @pytest.fixture(params=[
        np.float64,
        lambda x: _make_complex(x, 0),
        lambda x: _make_complex(0, x),
        lambda x: _make_complex(x, np.multiply(x, -2))
    ], ids=[
        'real',
        'complex-real',
        'complex-imag',
        'complex-both'
    ])
    def sc(self, request):
        """ scale function used by the below tests """
        return request.param

    def test_non_finite_any_nan(self, sc):
        """ test that nans are propagated """
        assert_equal(np.interp(0.5, [np.nan,      1], sc([     0,     10])), sc(np.nan))
        assert_equal(np.interp(0.5, [     0, np.nan], sc([     0,     10])), sc(np.nan))
        assert_equal(np.interp(0.5, [     0,      1], sc([np.nan,     10])), sc(np.nan))
        assert_equal(np.interp(0.5, [     0,      1], sc([     0, np.nan])), sc(np.nan))

    def test_non_finite_inf(self, sc):
        """ Test that interp between opposite infs gives nan """
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([      0,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0,       1], sc([-np.inf, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0,       1], sc([+np.inf, -np.inf])), sc(np.nan))

        # unless the y values are equal
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([     10,      10])), sc(10))

    def test_non_finite_half_inf_xf(self, sc):
        """ Test that interp where both axes have a bound at inf gives nan """
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([-np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([+np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([      0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([      0, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([-np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([+np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([      0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([      0, +np.inf])), sc(np.nan))

    def test_non_finite_half_inf_x(self, sc):
        """ Test interp where the x axis has a bound at inf """
        assert_equal(np.interp(0.5, [-np.inf, -np.inf], sc([0, 10])), sc(10))
        assert_equal(np.interp(0.5, [-np.inf, 1      ], sc([0, 10])), sc(10))  # noqa: E202
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([0, 10])), sc(0))
        assert_equal(np.interp(0.5, [+np.inf, +np.inf], sc([0, 10])), sc(0))

    def test_non_finite_half_inf_f(self, sc):
        """ Test interp where the f axis has a bound at inf """
        assert_equal(np.interp(0.5, [0, 1], sc([      0, -np.inf])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([      0, +np.inf])), sc(+np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf,      10])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf,      10])), sc(+np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, -np.inf])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, +np.inf])), sc(+np.inf))

    def test_complex_interp(self):
        # test complex interpolation
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5) + (1 + np.linspace(0, 1, 5)) * 1.0j
        x0 = 0.3
        y0 = x0 + (1 + x0) * 1.0j
        assert_almost_equal(np.interp(x0, x, y), y0)
        # test complex left and right
        x0 = -1
        left = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, left=left), left)
        x0 = 2.0
        right = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, right=right), right)
        # test complex non finite
        x = [1, 2, 2.5, 3, 4]
        xp = [1, 2, 3, 4]
        fp = [1, 2 + 1j, np.inf, 4]
        y = [1, 2 + 1j, np.inf + 0.5j, np.inf, 4]
        assert_almost_equal(np.interp(x, xp, fp), y)
        # test complex periodic
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5 + 1.0j, 10 + 2j, 3 + 3j, 4 + 4j]
        y = [7.5 + 1.5j, 5. + 1.0j, 8.75 + 1.75j, 6.25 + 1.25j, 3. + 3j, 3.25 + 3.25j,
             3.5 + 3.5j, 3.75 + 3.75j]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)

    def test_zero_dimensional_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.array(.3)
        assert_almost_equal(np.interp(x0, x, y), x0)

        xp = np.array([0, 2, 4])
        fp = np.array([1, -1, 1])

        actual = np.interp(np.array(1), xp, fp)
        assert_equal(actual, 0)
        assert_(isinstance(actual, np.float64))

        actual = np.interp(np.array(4.5), xp, fp, period=4)
        assert_equal(actual, 0.5)
        assert_(isinstance(actual, np.float64))

    def test_if_len_x_is_small(self):
        xp = np.arange(0, 10, 0.0001)
        fp = np.sin(xp)
        assert_almost_equal(np.interp(np.pi, xp, fp), 0.0)

    def test_period(self):
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5, 10, 3, 4]
        y = [7.5, 5., 8.75, 6.25, 3., 3.25, 3.5, 3.75]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)
        x = np.array(x, order='F').reshape(2, -1)
        y = np.array(y, order='C').reshape(2, -1)
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)


class TestPercentile:

    def test_basic(self):
        x = np.arange(8) * 0.5
        assert_equal(np.percentile(x, 0), 0.)
        assert_equal(np.percentile(x, 100), 3.5)
        assert_equal(np.percentile(x, 50), 1.75)
        x[1] = np.nan
        assert_equal(np.percentile(x, 0), np.nan)
        assert_equal(np.percentile(x, 0, method='nearest'), np.nan)
        assert_equal(np.percentile(x, 0, method='inverted_cdf'), np.nan)
        assert_equal(
            np.percentile(x, 0, method='inverted_cdf',
                          weights=np.ones_like(x)),
            np.nan,
        )

    def test_fraction(self):
        x = [Fraction(i, 2) for i in range(8)]

        p = np.percentile(x, Fraction(0))
        assert_equal(p, Fraction(0))
        assert_equal(type(p), Fraction)

        p = np.percentile(x, Fraction(100))
        assert_equal(p, Fraction(7, 2))
        assert_equal(type(p), Fraction)

        p = np.percentile(x, Fraction(50))
        assert_equal(p, Fraction(7, 4))
        assert_equal(type(p), Fraction)

        p = np.percentile(x, [Fraction(50)])
        assert_equal(p, np.array([Fraction(7, 4)]))
        assert_equal(type(p), np.ndarray)

    def test_api(self):
        d = np.ones(5)
        np.percentile(d, 5, None, None, False)
        np.percentile(d, 5, None, None, False, 'linear')
        o = np.ones((1,))
        np.percentile(d, 5, None, o, False, 'linear')

    def test_complex(self):
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='G')
        assert_raises(TypeError, np.percentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='D')
        assert_raises(TypeError, np.percentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='F')
        assert_raises(TypeError, np.percentile, arr_c, 0.5)

    def test_2D(self):
        x = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [4, 4, 3],
                      [1, 1, 1],
                      [1, 1, 1]])
        assert_array_equal(np.percentile(x, 50, axis=0), [1, 1, 1])

    @pytest.mark.parametrize("dtype", np.typecodes["Float"])
    def test_linear_nan_1D(self, dtype):
        # METHOD 1 of H&F
        arr = np.asarray([15.0, np.nan, 35.0, 40.0, 50.0], dtype=dtype)
        res = np.percentile(
            arr,
            40.0,
            method="linear")
        np.testing.assert_equal(res, np.nan)
        np.testing.assert_equal(res.dtype, arr.dtype)

    H_F_TYPE_CODES = [(int_type, np.float64)
                      for int_type in np.typecodes["AllInteger"]
                      ] + [(np.float16, np.float16),
                           (np.float32, np.float32),
                           (np.float64, np.float64),
                           (np.longdouble, np.longdouble),
                           (np.dtype("O"), np.float64)]

    @pytest.mark.parametrize(["function", "quantile"],
                             [(np.quantile, 0.4),
                              (np.percentile, 40.0)])
    @pytest.mark.parametrize(["input_dtype", "expected_dtype"], H_F_TYPE_CODES)
    @pytest.mark.parametrize(["method", "weighted", "expected"],
                              [("inverted_cdf", False, 20),
                              ("inverted_cdf", True, 20),
                              ("averaged_inverted_cdf", False, 27.5),
                              ("closest_observation", False, 20),
                              ("interpolated_inverted_cdf", False, 20),
                              ("hazen", False, 27.5),
                              ("weibull", False, 26),
                              ("linear", False, 29),
                              ("median_unbiased", False, 27),
                              ("normal_unbiased", False, 27.125),
                               ])
    def test_linear_interpolation(self,
                                  function,
                                  quantile,
                                  method,
                                  weighted,
                                  expected,
                                  input_dtype,
                                  expected_dtype):
        expected_dtype = np.dtype(expected_dtype)

        arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0], dtype=input_dtype)
        weights = np.ones_like(arr) if weighted else None
        if input_dtype is np.longdouble:
            if function is np.quantile:
                # 0.4 is not exactly representable and it matters
                # for "averaged_inverted_cdf", so we need to cheat.
                quantile = input_dtype("0.4")
            # We want to use nulp, but that does not work for longdouble
            test_function = np.testing.assert_almost_equal
        else:
            test_function = np.testing.assert_array_almost_equal_nulp

        actual = function(arr, quantile, method=method, weights=weights)

        test_function(actual, expected_dtype.type(expected))

        if method in ["inverted_cdf", "closest_observation"]:
            if input_dtype == "O":
                np.testing.assert_equal(np.asarray(actual).dtype, np.float64)
            else:
                np.testing.assert_equal(np.asarray(actual).dtype,
                                        np.dtype(input_dtype))
        else:
            np.testing.assert_equal(np.asarray(actual).dtype,
                                    np.dtype(expected_dtype))

    TYPE_CODES = np.typecodes["AllInteger"] + np.typecodes["Float"] + "O"

    @pytest.mark.parametrize("dtype", TYPE_CODES)
    def test_lower_higher(self, dtype):
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 50,
                                   method='lower'), 4)
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 50,
                                   method='higher'), 5)

    @pytest.mark.parametrize("dtype", TYPE_CODES)
    def test_midpoint(self, dtype):
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 51,
                                   method='midpoint'), 4.5)
        assert_equal(np.percentile(np.arange(9, dtype=dtype) + 1, 50,
                                   method='midpoint'), 5)
        assert_equal(np.percentile(np.arange(11, dtype=dtype), 51,
                                   method='midpoint'), 5.5)
        assert_equal(np.percentile(np.arange(11, dtype=dtype), 50,
                                   method='midpoint'), 5)

    @pytest.mark.parametrize("dtype", TYPE_CODES)
    def test_nearest(self, dtype):
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 51,
                                   method='nearest'), 5)
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 49,
                                   method='nearest'), 4)

    def test_linear_interpolation_extrapolation(self):
        arr = np.random.rand(5)

        actual = np.percentile(arr, 100)
        np.testing.assert_equal(actual, arr.max())

        actual = np.percentile(arr, 0)
        np.testing.assert_equal(actual, arr.min())

    def test_sequence(self):
        x = np.arange(8) * 0.5
        assert_equal(np.percentile(x, [0, 100, 50]), [0, 3.5, 1.75])

    def test_axis(self):
        x = np.arange(12).reshape(3, 4)

        assert_equal(np.percentile(x, (25, 50, 100)), [2.75, 5.5, 11.0])

        r0 = [[2, 3, 4, 5], [4, 5, 6, 7], [8, 9, 10, 11]]
        assert_equal(np.percentile(x, (25, 50, 100), axis=0), r0)

        r1 = [[0.75, 1.5, 3], [4.75, 5.5, 7], [8.75, 9.5, 11]]
        assert_equal(np.percentile(x, (25, 50, 100), axis=1), np.array(r1).T)

        # ensure qth axis is always first as with np.array(old_percentile(..))
        x = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        assert_equal(np.percentile(x, (25, 50)).shape, (2,))
        assert_equal(np.percentile(x, (25, 50, 75)).shape, (3,))
        assert_equal(np.percentile(x, (25, 50), axis=0).shape, (2, 4, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=1).shape, (2, 3, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=2).shape, (2, 3, 4, 6))
        assert_equal(np.percentile(x, (25, 50), axis=3).shape, (2, 3, 4, 5))
        assert_equal(
            np.percentile(x, (25, 50, 75), axis=1).shape, (3, 3, 5, 6))
        assert_equal(np.percentile(x, (25, 50),
                                   method="higher").shape, (2,))
        assert_equal(np.percentile(x, (25, 50, 75),
                                   method="higher").shape, (3,))
        assert_equal(np.percentile(x, (25, 50), axis=0,
                                   method="higher").shape, (2, 4, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=1,
                                   method="higher").shape, (2, 3, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=2,
                                   method="higher").shape, (2, 3, 4, 6))
        assert_equal(np.percentile(x, (25, 50), axis=3,
                                   method="higher").shape, (2, 3, 4, 5))
        assert_equal(np.percentile(x, (25, 50, 75), axis=1,
                                   method="higher").shape, (3, 3, 5, 6))

    def test_scalar_q(self):
        # test for no empty dimensions for compatibility with old percentile
        x = np.arange(12).reshape(3, 4)
        assert_equal(np.percentile(x, 50), 5.5)
        assert_(np.isscalar(np.percentile(x, 50)))
        r0 = np.array([4., 5., 6., 7.])
        assert_equal(np.percentile(x, 50, axis=0), r0)
        assert_equal(np.percentile(x, 50, axis=0).shape, r0.shape)
        r1 = np.array([1.5, 5.5, 9.5])
        assert_almost_equal(np.percentile(x, 50, axis=1), r1)
        assert_equal(np.percentile(x, 50, axis=1).shape, r1.shape)

        out = np.empty(1)
        assert_equal(np.percentile(x, 50, out=out), 5.5)
        assert_equal(out, 5.5)
        out = np.empty(4)
        assert_equal(np.percentile(x, 50, axis=0, out=out), r0)
        assert_equal(out, r0)
        out = np.empty(3)
        assert_equal(np.percentile(x, 50, axis=1, out=out), r1)
        assert_equal(out, r1)

        # test for no empty dimensions for compatibility with old percentile
        x = np.arange(12).reshape(3, 4)
        assert_equal(np.percentile(x, 50, method='lower'), 5.)
        assert_(np.isscalar(np.percentile(x, 50)))
        r0 = np.array([4., 5., 6., 7.])
        c0 = np.percentile(x, 50, method='lower', axis=0)
        assert_equal(c0, r0)
        assert_equal(c0.shape, r0.shape)
        r1 = np.array([1., 5., 9.])
        c1 = np.percentile(x, 50, method='lower', axis=1)
        assert_almost_equal(c1, r1)
        assert_equal(c1.shape, r1.shape)

        out = np.empty((), dtype=x.dtype)
        c = np.percentile(x, 50, method='lower', out=out)
        assert_equal(c, 5)
        assert_equal(out, 5)
        out = np.empty(4, dtype=x.dtype)
        c = np.percentile(x, 50, method='lower', axis=0, out=out)
        assert_equal(c, r0)
        assert_equal(out, r0)
        out = np.empty(3, dtype=x.dtype)
        c = np.percentile(x, 50, method='lower', axis=1, out=out)
        assert_equal(c, r1)
        assert_equal(out, r1)

    def test_exception(self):
        assert_raises(ValueError, np.percentile, [1, 2], 56,
                      method='foobar')
        assert_raises(ValueError, np.percentile, [1], 101)
        assert_raises(ValueError, np.percentile, [1], -1)
        assert_raises(ValueError, np.percentile, [1], list(range(50)) + [101])
        assert_raises(ValueError, np.percentile, [1], list(range(50)) + [-0.1])

    def test_percentile_list(self):
        assert_equal(np.percentile([1, 2, 3], 0), 1)

    @pytest.mark.parametrize(
        "percentile, with_weights",
        [
            (np.percentile, False),
            (partial(np.percentile, method="inverted_cdf"), True),
        ]
    )
    def test_percentile_out(self, percentile, with_weights):
        out_dtype = int if with_weights else float
        x = np.array([1, 2, 3])
        y = np.zeros((3,), dtype=out_dtype)
        p = (1, 2, 3)
        weights = np.ones_like(x) if with_weights else None
        r = percentile(x, p, out=y, weights=weights)
        assert r is y
        assert_equal(percentile(x, p, weights=weights), y)

        x = np.array([[1, 2, 3],
                      [4, 5, 6]])
        y = np.zeros((3, 3), dtype=out_dtype)
        weights = np.ones_like(x) if with_weights else None
        r = percentile(x, p, axis=0, out=y, weights=weights)
        assert r is y
        assert_equal(percentile(x, p, weights=weights, axis=0), y)

        y = np.zeros((3, 2), dtype=out_dtype)
        percentile(x, p, axis=1, out=y, weights=weights)
        assert_equal(percentile(x, p, weights=weights, axis=1), y)

        x = np.arange(12).reshape(3, 4)
        # q.dim > 1, float
        if with_weights:
            r0 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        else:
            r0 = np.array([[2., 3., 4., 5.], [4., 5., 6., 7.]])
        out = np.empty((2, 4), dtype=out_dtype)
        weights = np.ones_like(x) if with_weights else None
        assert_equal(
            percentile(x, (25, 50), axis=0, out=out, weights=weights), r0
        )
        assert_equal(out, r0)
        r1 = np.array([[0.75, 4.75, 8.75], [1.5, 5.5, 9.5]])
        out = np.empty((2, 3))
        assert_equal(np.percentile(x, (25, 50), axis=1, out=out), r1)
        assert_equal(out, r1)

        # q.dim > 1, int
        r0 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        out = np.empty((2, 4), dtype=x.dtype)
        c = np.percentile(x, (25, 50), method='lower', axis=0, out=out)
        assert_equal(c, r0)
        assert_equal(out, r0)
        r1 = np.array([[0, 4, 8], [1, 5, 9]])
        out = np.empty((2, 3), dtype=x.dtype)
        c = np.percentile(x, (25, 50), method='lower', axis=1, out=out)
        assert_equal(c, r1)
        assert_equal(out, r1)

    def test_percentile_empty_dim(self):
        # empty dims are preserved
        d = np.arange(11 * 2).reshape(11, 1, 2, 1)
        assert_array_equal(np.percentile(d, 50, axis=0).shape, (1, 2, 1))
        assert_array_equal(np.percentile(d, 50, axis=1).shape, (11, 2, 1))
        assert_array_equal(np.percentile(d, 50, axis=2).shape, (11, 1, 1))
        assert_array_equal(np.percentile(d, 50, axis=3).shape, (11, 1, 2))
        assert_array_equal(np.percentile(d, 50, axis=-1).shape, (11, 1, 2))
        assert_array_equal(np.percentile(d, 50, axis=-2).shape, (11, 1, 1))
        assert_array_equal(np.percentile(d, 50, axis=-3).shape, (11, 2, 1))
        assert_array_equal(np.percentile(d, 50, axis=-4).shape, (1, 2, 1))

        assert_array_equal(np.percentile(d, 50, axis=2,
                                         method='midpoint').shape,
                           (11, 1, 1))
        assert_array_equal(np.percentile(d, 50, axis=-2,
                                         method='midpoint').shape,
                           (11, 1, 1))

        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=0)).shape,
                           (2, 1, 2, 1))
        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=1)).shape,
                           (2, 11, 2, 1))
        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=2)).shape,
                           (2, 11, 1, 1))
        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=3)).shape,
                           (2, 11, 1, 2))

    def test_percentile_no_overwrite(self):
        a = np.array([2, 3, 4, 1])
        np.percentile(a, [50], overwrite_input=False)
        assert_equal(a, np.array([2, 3, 4, 1]))

        a = np.array([2, 3, 4, 1])
        np.percentile(a, [50])
        assert_equal(a, np.array([2, 3, 4, 1]))

    def test_no_p_overwrite(self):
        p = np.linspace(0., 100., num=5)
        np.percentile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, np.linspace(0., 100., num=5))
        p = np.linspace(0., 100., num=5).tolist()
        np.percentile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, np.linspace(0., 100., num=5).tolist())

    def test_percentile_overwrite(self):
        a = np.array([2, 3, 4, 1])
        b = np.percentile(a, [50], overwrite_input=True)
        assert_equal(b, np.array([2.5]))

        b = np.percentile([2, 3, 4, 1], [50], overwrite_input=True)
        assert_equal(b, np.array([2.5]))

    def test_extended_axis(self):
        o = np.random.normal(size=(71, 23))
        x = np.dstack([o] * 10)
        assert_equal(np.percentile(x, 30, axis=(0, 1)), np.percentile(o, 30))
        x = np.moveaxis(x, -1, 0)
        assert_equal(np.percentile(x, 30, axis=(-2, -1)), np.percentile(o, 30))
        x = x.swapaxes(0, 1).copy()
        assert_equal(np.percentile(x, 30, axis=(0, -1)), np.percentile(o, 30))
        x = x.swapaxes(0, 1).copy()

        assert_equal(np.percentile(x, [25, 60], axis=(0, 1, 2)),
                     np.percentile(x, [25, 60], axis=None))
        assert_equal(np.percentile(x, [25, 60], axis=(0,)),
                     np.percentile(x, [25, 60], axis=0))

        d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
        np.random.shuffle(d.ravel())
        assert_equal(np.percentile(d, 25, axis=(0, 1, 2))[0],
                     np.percentile(d[:, :, :, 0].flatten(), 25))
        assert_equal(np.percentile(d, [10, 90], axis=(0, 1, 3))[:, 1],
                     np.percentile(d[:, :, 1, :].flatten(), [10, 90]))
        assert_equal(np.percentile(d, 25, axis=(3, 1, -4))[2],
                     np.percentile(d[:, :, 2, :].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(3, 1, 2))[2],
                     np.percentile(d[2, :, :, :].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(3, 2))[2, 1],
                     np.percentile(d[2, 1, :, :].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(1, -2))[2, 1],
                     np.percentile(d[2, :, :, 1].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(1, 3))[2, 2],
                     np.percentile(d[2, :, 2, :].flatten(), 25))

    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        assert_raises(AxisError, np.percentile, d, axis=-5, q=25)
        assert_raises(AxisError, np.percentile, d, axis=(0, -5), q=25)
        assert_raises(AxisError, np.percentile, d, axis=4, q=25)
        assert_raises(AxisError, np.percentile, d, axis=(0, 4), q=25)
        # each of these refers to the same axis twice
        assert_raises(ValueError, np.percentile, d, axis=(1, 1), q=25)
        assert_raises(ValueError, np.percentile, d, axis=(-1, -1), q=25)
        assert_raises(ValueError, np.percentile, d, axis=(3, -1), q=25)

    def test_keepdims(self):
        d = np.ones((3, 5, 7, 11))
        assert_equal(np.percentile(d, 7, axis=None, keepdims=True).shape,
                     (1, 1, 1, 1))
        assert_equal(np.percentile(d, 7, axis=(0, 1), keepdims=True).shape,
                     (1, 1, 7, 11))
        assert_equal(np.percentile(d, 7, axis=(0, 3), keepdims=True).shape,
                     (1, 5, 7, 1))
        assert_equal(np.percentile(d, 7, axis=(1,), keepdims=True).shape,
                     (3, 1, 7, 11))
        assert_equal(np.percentile(d, 7, (0, 1, 2, 3), keepdims=True).shape,
                     (1, 1, 1, 1))
        assert_equal(np.percentile(d, 7, axis=(0, 1, 3), keepdims=True).shape,
                     (1, 1, 7, 1))

        assert_equal(np.percentile(d, [1, 7], axis=(0, 1, 3),
                                   keepdims=True).shape, (2, 1, 1, 7, 1))
        assert_equal(np.percentile(d, [1, 7], axis=(0, 3),
                                   keepdims=True).shape, (2, 1, 5, 7, 1))

    @pytest.mark.parametrize('q', [7, [1, 7]])
    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1,),
            (0, 1),
            (-3, -1),
        ]
    )
    def test_keepdims_out(self, q, axis):
        d = np.ones((3, 5, 7, 11))
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        shape_out = np.shape(q) + shape_out

        out = np.empty(shape_out)
        result = np.percentile(d, q, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_out(self):
        o = np.zeros((4,))
        d = np.ones((3, 4))
        assert_equal(np.percentile(d, 0, 0, out=o), o)
        assert_equal(np.percentile(d, 0, 0, method='nearest', out=o), o)
        o = np.zeros((3,))
        assert_equal(np.percentile(d, 1, 1, out=o), o)
        assert_equal(np.percentile(d, 1, 1, method='nearest', out=o), o)

        o = np.zeros(())
        assert_equal(np.percentile(d, 2, out=o), o)
        assert_equal(np.percentile(d, 2, method='nearest', out=o), o)

    @pytest.mark.parametrize("method, weighted", [
        ("linear", False),
        ("nearest", False),
        ("inverted_cdf", False),
        ("inverted_cdf", True),
    ])
    def test_out_nan(self, method, weighted):
        if weighted:
            kwargs = {"weights": np.ones((3, 4)), "method": method}
        else:
            kwargs = {"method": method}
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always', '', RuntimeWarning)
            o = np.zeros((4,))
            d = np.ones((3, 4))
            d[2, 1] = np.nan
            assert_equal(np.percentile(d, 0, 0, out=o, **kwargs), o)

            o = np.zeros((3,))
            assert_equal(np.percentile(d, 1, 1, out=o, **kwargs), o)

            o = np.zeros(())
            assert_equal(np.percentile(d, 1, out=o, **kwargs), o)

    def test_nan_behavior(self):
        a = np.arange(24, dtype=float)
        a[2] = np.nan
        assert_equal(np.percentile(a, 0.3), np.nan)
        assert_equal(np.percentile(a, 0.3, axis=0), np.nan)
        assert_equal(np.percentile(a, [0.3, 0.6], axis=0),
                     np.array([np.nan] * 2))

        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # no axis
        assert_equal(np.percentile(a, 0.3), np.nan)
        assert_equal(np.percentile(a, 0.3).ndim, 0)

        # axis0 zerod
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, 0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.percentile(a, 0.3, 0), b)

        # axis0 not zerod
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4),
                          [0.3, 0.6], 0)
        b[:, 2, 3] = np.nan
        b[:, 1, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], 0), b)

        # axis1 zerod
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, 1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.percentile(a, 0.3, 1), b)
        # axis1 not zerod
        b = np.percentile(
            np.arange(24, dtype=float).reshape(2, 3, 4), [0.3, 0.6], 1)
        b[:, 1, 3] = np.nan
        b[:, 1, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], 1), b)

        # axis02 zerod
        b = np.percentile(
            np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, (0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.percentile(a, 0.3, (0, 2)), b)
        # axis02 not zerod
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4),
                          [0.3, 0.6], (0, 2))
        b[:, 1] = np.nan
        b[:, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], (0, 2)), b)
        # axis02 not zerod with method='nearest'
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4),
                          [0.3, 0.6], (0, 2), method='nearest')
        b[:, 1] = np.nan
        b[:, 2] = np.nan
        assert_equal(np.percentile(
            a, [0.3, 0.6], (0, 2), method='nearest'), b)

    def test_nan_q(self):
        # GH18830
        with pytest.raises(ValueError, match="Percentiles must be in"):
            np.percentile([1, 2, 3, 4.0], np.nan)
        with pytest.raises(ValueError, match="Percentiles must be in"):
            np.percentile([1, 2, 3, 4.0], [np.nan])
        q = np.linspace(1.0, 99.0, 16)
        q[0] = np.nan
        with pytest.raises(ValueError, match="Percentiles must be in"):
            np.percentile([1, 2, 3, 4.0], q)

    @pytest.mark.parametrize("dtype", ["m8[D]", "M8[s]"])
    @pytest.mark.parametrize("pos", [0, 23, 10])
    def test_nat_basic(self, dtype, pos):
        # TODO: Note that times have dubious rounding as of fixing NaTs!
        # NaT and NaN should behave the same, do basic tests for NaT:
        a = np.arange(0, 24, dtype=dtype)
        a[pos] = "NaT"
        res = np.percentile(a, 30)
        assert res.dtype == dtype
        assert np.isnat(res)
        res = np.percentile(a, [30, 60])
        assert res.dtype == dtype
        assert np.isnat(res).all()

        a = np.arange(0, 24 * 3, dtype=dtype).reshape(-1, 3)
        a[pos, 1] = "NaT"
        res = np.percentile(a, 30, axis=0)
        assert_array_equal(np.isnat(res), [False, True, False])


quantile_methods = [
    'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation',
    'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear',
    'median_unbiased', 'normal_unbiased', 'nearest', 'lower', 'higher',
    'midpoint']


methods_supporting_weights = ["inverted_cdf"]


class TestQuantile:
    # most of this is already tested by TestPercentile

    def V(self, x, y, alpha):
        # Identification function used in several tests.
        return (x >= y) - alpha

    def test_max_ulp(self):
        x = [0.0, 0.2, 0.4]
        a = np.quantile(x, 0.45)
        # The default linear method would result in 0 + 0.2 * (0.45/2) = 0.18.
        # 0.18 is not exactly representable and the formula leads to a 1 ULP
        # different result. Ensure it is this exact within 1 ULP, see gh-20331.
        np.testing.assert_array_max_ulp(a, 0.18, maxulp=1)

    def test_basic(self):
        x = np.arange(8) * 0.5
        assert_equal(np.quantile(x, 0), 0.)
        assert_equal(np.quantile(x, 1), 3.5)
        assert_equal(np.quantile(x, 0.5), 1.75)

    def test_correct_quantile_value(self):
        a = np.array([True])
        tf_quant = np.quantile(True, False)
        assert_equal(tf_quant, a[0])
        assert_equal(type(tf_quant), a.dtype)
        a = np.array([False, True, True])
        quant_res = np.quantile(a, a)
        assert_array_equal(quant_res, a)
        assert_equal(quant_res.dtype, a.dtype)

    def test_fraction(self):
        # fractional input, integral quantile
        x = [Fraction(i, 2) for i in range(8)]
        q = np.quantile(x, 0)
        assert_equal(q, 0)
        assert_equal(type(q), Fraction)

        q = np.quantile(x, 1)
        assert_equal(q, Fraction(7, 2))
        assert_equal(type(q), Fraction)

        q = np.quantile(x, .5)
        assert_equal(q, 1.75)
        assert_equal(type(q), np.float64)

        q = np.quantile(x, Fraction(1, 2))
        assert_equal(q, Fraction(7, 4))
        assert_equal(type(q), Fraction)

        q = np.quantile(x, [Fraction(1, 2)])
        assert_equal(q, np.array([Fraction(7, 4)]))
        assert_equal(type(q), np.ndarray)

        q = np.quantile(x, [[Fraction(1, 2)]])
        assert_equal(q, np.array([[Fraction(7, 4)]]))
        assert_equal(type(q), np.ndarray)

        # repeat with integral input but fractional quantile
        x = np.arange(8)
        assert_equal(np.quantile(x, Fraction(1, 2)), Fraction(7, 2))

    def test_complex(self):
        # gh-22652
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='G')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='D')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='F')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)

    def test_no_p_overwrite(self):
        # this is worth retesting, because quantile does not make a copy
        p0 = np.array([0, 0.75, 0.25, 0.5, 1.0])
        p = p0.copy()
        np.quantile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, p0)

        p0 = p0.tolist()
        p = p.tolist()
        np.quantile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, p0)

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_quantile_preserve_int_type(self, dtype):
        res = np.quantile(np.array([1, 2], dtype=dtype), [0.5],
                          method="nearest")
        assert res.dtype == dtype

    @pytest.mark.parametrize("method", quantile_methods)
    def test_q_zero_one(self, method):
        # gh-24710
        arr = [10, 11, 12]
        quantile = np.quantile(arr, q=[0, 1], method=method)
        assert_equal(quantile, np.array([10, 12]))

    @pytest.mark.parametrize("method", quantile_methods)
    def test_quantile_monotonic(self, method):
        # GH 14685
        # test that the return value of quantile is monotonic if p0 is ordered
        # Also tests that the boundary values are not mishandled.
        p0 = np.linspace(0, 1, 101)
        quantile = np.quantile(np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9,
                                         8, 8, 7]) * 0.1, p0, method=method)
        assert_equal(np.sort(quantile), quantile)

        # Also test one where the number of data points is clearly divisible:
        quantile = np.quantile([0., 1., 2., 3.], p0, method=method)
        assert_equal(np.sort(quantile), quantile)

    @hypothesis.given(
            arr=arrays(dtype=np.float64,
                       shape=st.integers(min_value=3, max_value=1000),
                       elements=st.floats(allow_infinity=False, allow_nan=False,
                                          min_value=-1e300, max_value=1e300)))
    def test_quantile_monotonic_hypo(self, arr):
        p0 = np.arange(0, 1, 0.01)
        quantile = np.quantile(arr, p0)
        assert_equal(np.sort(quantile), quantile)

    def test_quantile_scalar_nan(self):
        a = np.array([[10., 7., 4.], [3., 2., 1.]])
        a[0][1] = np.nan
        actual = np.quantile(a, 0.5)
        assert np.isscalar(actual)
        assert_equal(np.quantile(a, 0.5), np.nan)

    @pytest.mark.parametrize("weights", [False, True])
    @pytest.mark.parametrize("method", quantile_methods)
    @pytest.mark.parametrize("alpha", [0.2, 0.5, 0.9])
    def test_quantile_identification_equation(self, weights, method, alpha):
        # Test that the identification equation holds for the empirical
        # CDF:
        #   E[V(x, Y)] = 0  <=>  x is quantile
        # with Y the random variable for which we have observed values and
        # V(x, y) the canonical identification function for the quantile (at
        # level alpha), see
        # https://doi.org/10.48550/arXiv.0912.0902
        if weights and method not in methods_supporting_weights:
            pytest.skip("Weights not supported by method.")
        rng = np.random.default_rng(4321)
        # We choose n and alpha such that we cover 3 cases:
        #  - n * alpha is an integer
        #  - n * alpha is a float that gets rounded down
        #  - n * alpha is a float that gest rounded up
        n = 102  # n * alpha = 20.4, 51. , 91.8
        y = rng.random(n)
        w = rng.integers(low=0, high=10, size=n) if weights else None
        x = np.quantile(y, alpha, method=method, weights=w)

        if method in ("higher",):
            # These methods do not fulfill the identification equation.
            assert np.abs(np.mean(self.V(x, y, alpha))) > 0.1 / n
        elif int(n * alpha) == n * alpha and not weights:
            # We can expect exact results, up to machine precision.
            assert_allclose(
                np.average(self.V(x, y, alpha), weights=w), 0, atol=1e-14,
            )
        else:
            # V = (x >= y) - alpha cannot sum to zero exactly but within
            # "sample precision".
            assert_allclose(np.average(self.V(x, y, alpha), weights=w), 0,
                atol=1 / n / np.amin([alpha, 1 - alpha]))

    @pytest.mark.parametrize("weights", [False, True])
    @pytest.mark.parametrize("method", quantile_methods)
    @pytest.mark.parametrize("alpha", [0.2, 0.5, 0.9])
    def test_quantile_add_and_multiply_constant(self, weights, method, alpha):
        # Test that
        #  1. quantile(c + x) = c + quantile(x)
        #  2. quantile(c * x) = c * quantile(x)
        #  3. quantile(-x) = -quantile(x, 1 - alpha)
        #     On empirical quantiles, this equation does not hold exactly.
        # Koenker (2005) "Quantile Regression" Chapter 2.2.3 calls these
        # properties equivariance.
        if weights and method not in methods_supporting_weights:
            pytest.skip("Weights not supported by method.")
        rng = np.random.default_rng(4321)
        # We choose n and alpha such that we have cases for
        #  - n * alpha is an integer
        #  - n * alpha is a float that gets rounded down
        #  - n * alpha is a float that gest rounded up
        n = 102  # n * alpha = 20.4, 51. , 91.8
        y = rng.random(n)
        w = rng.integers(low=0, high=10, size=n) if weights else None
        q = np.quantile(y, alpha, method=method, weights=w)
        c = 13.5

        # 1
        assert_allclose(np.quantile(c + y, alpha, method=method, weights=w),
                        c + q)
        # 2
        assert_allclose(np.quantile(c * y, alpha, method=method, weights=w),
                        c * q)
        # 3
        if weights:
            # From here on, we would need more methods to support weights.
            return
        q = -np.quantile(-y, 1 - alpha, method=method)
        if method == "inverted_cdf":
            if (
                n * alpha == int(n * alpha)
                or np.round(n * alpha) == int(n * alpha) + 1
            ):
                assert_allclose(q, np.quantile(y, alpha, method="higher"))
            else:
                assert_allclose(q, np.quantile(y, alpha, method="lower"))
        elif method == "closest_observation":
            if n * alpha == int(n * alpha):
                assert_allclose(q, np.quantile(y, alpha, method="higher"))
            elif np.round(n * alpha) == int(n * alpha) + 1:
                assert_allclose(
                    q, np.quantile(y, alpha + 1 / n, method="higher"))
            else:
                assert_allclose(q, np.quantile(y, alpha, method="lower"))
        elif method == "interpolated_inverted_cdf":
            assert_allclose(q, np.quantile(y, alpha + 1 / n, method=method))
        elif method == "nearest":
            if n * alpha == int(n * alpha):
                assert_allclose(q, np.quantile(y, alpha + 1 / n, method=method))
            else:
                assert_allclose(q, np.quantile(y, alpha, method=method))
        elif method == "lower":
            assert_allclose(q, np.quantile(y, alpha, method="higher"))
        elif method == "higher":
            assert_allclose(q, np.quantile(y, alpha, method="lower"))
        else:
            # "averaged_inverted_cdf", "hazen", "weibull", "linear",
            # "median_unbiased", "normal_unbiased", "midpoint"
            assert_allclose(q, np.quantile(y, alpha, method=method))

    @pytest.mark.parametrize("method", methods_supporting_weights)
    @pytest.mark.parametrize("alpha", [0.2, 0.5, 0.9])
    def test_quantile_constant_weights(self, method, alpha):
        rng = np.random.default_rng(4321)
        # We choose n and alpha such that we have cases for
        #  - n * alpha is an integer
        #  - n * alpha is a float that gets rounded down
        #  - n * alpha is a float that gest rounded up
        n = 102  # n * alpha = 20.4, 51. , 91.8
        y = rng.random(n)
        q = np.quantile(y, alpha, method=method)

        w = np.ones_like(y)
        qw = np.quantile(y, alpha, method=method, weights=w)
        assert_allclose(qw, q)

        w = 8.125 * np.ones_like(y)
        qw = np.quantile(y, alpha, method=method, weights=w)
        assert_allclose(qw, q)

    @pytest.mark.parametrize("method", methods_supporting_weights)
    @pytest.mark.parametrize("alpha", [0, 0.2, 0.5, 0.9, 1])
    def test_quantile_with_integer_weights(self, method, alpha):
        # Integer weights can be interpreted as repeated observations.
        rng = np.random.default_rng(4321)
        # We choose n and alpha such that we have cases for
        #  - n * alpha is an integer
        #  - n * alpha is a float that gets rounded down
        #  - n * alpha is a float that gest rounded up
        n = 102  # n * alpha = 20.4, 51. , 91.8
        y = rng.random(n)
        w = rng.integers(low=0, high=10, size=n, dtype=np.int32)

        qw = np.quantile(y, alpha, method=method, weights=w)
        q = np.quantile(np.repeat(y, w), alpha, method=method)
        assert_allclose(qw, q)

    @pytest.mark.parametrize("method", methods_supporting_weights)
    def test_quantile_with_weights_and_axis(self, method):
        rng = np.random.default_rng(4321)

        # 1d weight and single alpha
        y = rng.random((2, 10, 3))
        w = np.abs(rng.random(10))
        alpha = 0.5
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)
        q_res = np.zeros(shape=(2, 3))
        for i in range(2):
            for j in range(3):
                q_res[i, j] = np.quantile(
                    y[i, :, j], alpha, method=method, weights=w
                )
        assert_allclose(q, q_res)

        # 1d weight and 1d alpha
        alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]  # shape (6,)
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)
        q_res = np.zeros(shape=(6, 2, 3))
        for i in range(2):
            for j in range(3):
                q_res[:, i, j] = np.quantile(
                    y[i, :, j], alpha, method=method, weights=w
                )
        assert_allclose(q, q_res)

        # 1d weight and 2d alpha
        alpha = [[0, 0.2], [0.4, 0.6], [0.8, 1]]  # shape (3, 2)
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)
        q_res = q_res.reshape((3, 2, 2, 3))
        assert_allclose(q, q_res)

        # shape of weights equals shape of y
        w = np.abs(rng.random((2, 10, 3)))
        alpha = 0.5
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)
        q_res = np.zeros(shape=(2, 3))
        for i in range(2):
            for j in range(3):
                q_res[i, j] = np.quantile(
                    y[i, :, j], alpha, method=method, weights=w[i, :, j]
                )
        assert_allclose(q, q_res)

    @pytest.mark.parametrize("method", methods_supporting_weights)
    def test_quantile_weights_min_max(self, method):
        # Test weighted quantile at 0 and 1 with leading and trailing zero
        # weights.
        w = [0, 0, 1, 2, 3, 0]
        y = np.arange(6)
        y_min = np.quantile(y, 0, weights=w, method="inverted_cdf")
        y_max = np.quantile(y, 1, weights=w, method="inverted_cdf")
        assert y_min == y[2]  # == 2
        assert y_max == y[4]  # == 4

    def test_quantile_weights_raises_negative_weights(self):
        y = [1, 2]
        w = [-0.5, 1]
        with pytest.raises(ValueError, match="Weights must be non-negative"):
            np.quantile(y, 0.5, weights=w, method="inverted_cdf")

    @pytest.mark.parametrize(
            "method",
            sorted(set(quantile_methods) - set(methods_supporting_weights)),
    )
    def test_quantile_weights_raises_unsupported_methods(self, method):
        y = [1, 2]
        w = [0.5, 1]
        msg = "Only method 'inverted_cdf' supports weights"
        with pytest.raises(ValueError, match=msg):
            np.quantile(y, 0.5, weights=w, method=method)

    def test_weibull_fraction(self):
        arr = [Fraction(0, 1), Fraction(1, 10)]
        quantile = np.quantile(arr, [0, ], method='weibull')
        assert_equal(quantile, np.array(Fraction(0, 1)))
        quantile = np.quantile(arr, [Fraction(1, 2)], method='weibull')
        assert_equal(quantile, np.array(Fraction(1, 20)))

    def test_closest_observation(self):
        # Round ties to nearest even order statistic (see #26656)
        m = 'closest_observation'
        q = 0.5
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert_equal(2, np.quantile(arr[0:3], q, method=m))
        assert_equal(2, np.quantile(arr[0:4], q, method=m))
        assert_equal(2, np.quantile(arr[0:5], q, method=m))
        assert_equal(3, np.quantile(arr[0:6], q, method=m))
        assert_equal(4, np.quantile(arr[0:7], q, method=m))
        assert_equal(4, np.quantile(arr[0:8], q, method=m))
        assert_equal(4, np.quantile(arr[0:9], q, method=m))
        assert_equal(5, np.quantile(arr, q, method=m))


class TestLerp:
    @hypothesis.given(t0=st.floats(allow_nan=False, allow_infinity=False,
                                   min_value=0, max_value=1),
                      t1=st.floats(allow_nan=False, allow_infinity=False,
                                   min_value=0, max_value=1),
                      a=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=-1e300, max_value=1e300),
                      b=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=-1e300, max_value=1e300))
    def test_linear_interpolation_formula_monotonic(self, t0, t1, a, b):
        l0 = nfb._lerp(a, b, t0)
        l1 = nfb._lerp(a, b, t1)
        if t0 == t1 or a == b:
            assert l0 == l1  # uninteresting
        elif (t0 < t1) == (a < b):
            assert l0 <= l1
        else:
            assert l0 >= l1

    @hypothesis.given(t=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=0, max_value=1),
                      a=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=-1e300, max_value=1e300),
                      b=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=-1e300, max_value=1e300))
    def test_linear_interpolation_formula_bounded(self, t, a, b):
        if a <= b:
            assert a <= nfb._lerp(a, b, t) <= b
        else:
            assert b <= nfb._lerp(a, b, t) <= a

    @hypothesis.given(t=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=0, max_value=1),
                      a=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=-1e300, max_value=1e300),
                      b=st.floats(allow_nan=False, allow_infinity=False,
                                  min_value=-1e300, max_value=1e300))
    def test_linear_interpolation_formula_symmetric(self, t, a, b):
        # double subtraction is needed to remove the extra precision of t < 0.5
        left = nfb._lerp(a, b, 1 - (1 - t))
        right = nfb._lerp(b, a, 1 - t)
        assert_allclose(left, right)

    def test_linear_interpolation_formula_0d_inputs(self):
        a = np.array(2)
        b = np.array(5)
        t = np.array(0.2)
        assert nfb._lerp(a, b, t) == 2.6


class TestMedian:

    def test_basic(self):
        a0 = np.array(1)
        a1 = np.arange(2)
        a2 = np.arange(6).reshape(2, 3)
        assert_equal(np.median(a0), 1)
        assert_allclose(np.median(a1), 0.5)
        assert_allclose(np.median(a2), 2.5)
        assert_allclose(np.median(a2, axis=0), [1.5, 2.5, 3.5])
        assert_equal(np.median(a2, axis=1), [1, 4])
        assert_allclose(np.median(a2, axis=None), 2.5)

        a = np.array([0.0444502, 0.0463301, 0.141249, 0.0606775])
        assert_almost_equal((a[1] + a[3]) / 2., np.median(a))
        a = np.array([0.0463301, 0.0444502, 0.141249])
        assert_equal(a[0], np.median(a))
        a = np.array([0.0444502, 0.141249, 0.0463301])
        assert_equal(a[-1], np.median(a))
        # check array scalar result
        assert_equal(np.median(a).ndim, 0)
        a[1] = np.nan
        assert_equal(np.median(a).ndim, 0)

    def test_axis_keyword(self):
        a3 = np.array([[2, 3],
                       [0, 1],
                       [6, 7],
                       [4, 5]])
        for a in [a3, np.random.randint(0, 100, size=(2, 3, 4))]:
            orig = a.copy()
            np.median(a, axis=None)
            for ax in range(a.ndim):
                np.median(a, axis=ax)
            assert_array_equal(a, orig)

        assert_allclose(np.median(a3, axis=0), [3, 4])
        assert_allclose(np.median(a3.T, axis=1), [3, 4])
        assert_allclose(np.median(a3), 3.5)
        assert_allclose(np.median(a3, axis=None), 3.5)
        assert_allclose(np.median(a3.T), 3.5)

    def test_overwrite_keyword(self):
        a3 = np.array([[2, 3],
                       [0, 1],
                       [6, 7],
                       [4, 5]])
        a0 = np.array(1)
        a1 = np.arange(2)
        a2 = np.arange(6).reshape(2, 3)
        assert_allclose(np.median(a0.copy(), overwrite_input=True), 1)
        assert_allclose(np.median(a1.copy(), overwrite_input=True), 0.5)
        assert_allclose(np.median(a2.copy(), overwrite_input=True), 2.5)
        assert_allclose(
            np.median(a2.copy(), overwrite_input=True, axis=0), [1.5, 2.5, 3.5])
        assert_allclose(
            np.median(a2.copy(), overwrite_input=True, axis=1), [1, 4])
        assert_allclose(
            np.median(a2.copy(), overwrite_input=True, axis=None), 2.5)
        assert_allclose(
            np.median(a3.copy(), overwrite_input=True, axis=0), [3, 4])
        assert_allclose(
            np.median(a3.T.copy(), overwrite_input=True, axis=1), [3, 4])

        a4 = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
        np.random.shuffle(a4.ravel())
        assert_allclose(np.median(a4, axis=None),
                        np.median(a4.copy(), axis=None, overwrite_input=True))
        assert_allclose(np.median(a4, axis=0),
                        np.median(a4.copy(), axis=0, overwrite_input=True))
        assert_allclose(np.median(a4, axis=1),
                        np.median(a4.copy(), axis=1, overwrite_input=True))
        assert_allclose(np.median(a4, axis=2),
                        np.median(a4.copy(), axis=2, overwrite_input=True))

    def test_array_like(self):
        x = [1, 2, 3]
        assert_almost_equal(np.median(x), 2)
        x2 = [x]
        assert_almost_equal(np.median(x2), 2)
        assert_allclose(np.median(x2, axis=0), x)

    def test_subclass(self):
        # gh-3846
        class MySubClass(np.ndarray):

            def __new__(cls, input_array, info=None):
                obj = np.asarray(input_array).view(cls)
                obj.info = info
                return obj

            def mean(self, axis=None, dtype=None, out=None):
                return -7

        a = MySubClass([1, 2, 3])
        assert_equal(np.median(a), -7)

    @pytest.mark.parametrize('arr',
                             ([1., 2., 3.], [1., np.nan, 3.], np.nan, 0.))
    def test_subclass2(self, arr):
        """Check that we return subclasses, even if a NaN scalar."""
        class MySubclass(np.ndarray):
            pass

        m = np.median(np.array(arr).view(MySubclass))
        assert isinstance(m, MySubclass)

    def test_out(self):
        o = np.zeros((4,))
        d = np.ones((3, 4))
        assert_equal(np.median(d, 0, out=o), o)
        o = np.zeros((3,))
        assert_equal(np.median(d, 1, out=o), o)
        o = np.zeros(())
        assert_equal(np.median(d, out=o), o)

    def test_out_nan(self):
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always', '', RuntimeWarning)
            o = np.zeros((4,))
            d = np.ones((3, 4))
            d[2, 1] = np.nan
            assert_equal(np.median(d, 0, out=o), o)
            o = np.zeros((3,))
            assert_equal(np.median(d, 1, out=o), o)
            o = np.zeros(())
            assert_equal(np.median(d, out=o), o)

    def test_nan_behavior(self):
        a = np.arange(24, dtype=float)
        a[2] = np.nan
        assert_equal(np.median(a), np.nan)
        assert_equal(np.median(a, axis=0), np.nan)

        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # no axis
        assert_equal(np.median(a), np.nan)
        assert_equal(np.median(a).ndim, 0)

        # axis0
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.median(a, 0), b)

        # axis1
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.median(a, 1), b)

        # axis02
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), (0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.median(a, (0, 2)), b)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work correctly")
    def test_empty(self):
        # mean(empty array) emits two warnings: empty slice and divide by 0
        a = np.array([], dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.median(a), np.nan)
            assert_(w[0].category is RuntimeWarning)
            assert_equal(len(w), 2)

        # multiple dimensions
        a = np.array([], dtype=float, ndmin=3)
        # no axis
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.median(a), np.nan)
            assert_(w[0].category is RuntimeWarning)

        # axis 0 and 1
        b = np.array([], dtype=float, ndmin=2)
        assert_equal(np.median(a, axis=0), b)
        assert_equal(np.median(a, axis=1), b)

        # axis 2
        b = np.array(np.nan, dtype=float, ndmin=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.median(a, axis=2), b)
            assert_(w[0].category is RuntimeWarning)

    def test_object(self):
        o = np.arange(7.)
        assert_(type(np.median(o.astype(object))), float)
        o[2] = np.nan
        assert_(type(np.median(o.astype(object))), float)

    def test_extended_axis(self):
        o = np.random.normal(size=(71, 23))
        x = np.dstack([o] * 10)
        assert_equal(np.median(x, axis=(0, 1)), np.median(o))
        x = np.moveaxis(x, -1, 0)
        assert_equal(np.median(x, axis=(-2, -1)), np.median(o))
        x = x.swapaxes(0, 1).copy()
        assert_equal(np.median(x, axis=(0, -1)), np.median(o))

        assert_equal(np.median(x, axis=(0, 1, 2)), np.median(x, axis=None))
        assert_equal(np.median(x, axis=(0, )), np.median(x, axis=0))
        assert_equal(np.median(x, axis=(-1, )), np.median(x, axis=-1))

        d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
        np.random.shuffle(d.ravel())
        assert_equal(np.median(d, axis=(0, 1, 2))[0],
                     np.median(d[:, :, :, 0].flatten()))
        assert_equal(np.median(d, axis=(0, 1, 3))[1],
                     np.median(d[:, :, 1, :].flatten()))
        assert_equal(np.median(d, axis=(3, 1, -4))[2],
                     np.median(d[:, :, 2, :].flatten()))
        assert_equal(np.median(d, axis=(3, 1, 2))[2],
                     np.median(d[2, :, :, :].flatten()))
        assert_equal(np.median(d, axis=(3, 2))[2, 1],
                     np.median(d[2, 1, :, :].flatten()))
        assert_equal(np.median(d, axis=(1, -2))[2, 1],
                     np.median(d[2, :, :, 1].flatten()))
        assert_equal(np.median(d, axis=(1, 3))[2, 2],
                     np.median(d[2, :, 2, :].flatten()))

    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        assert_raises(AxisError, np.median, d, axis=-5)
        assert_raises(AxisError, np.median, d, axis=(0, -5))
        assert_raises(AxisError, np.median, d, axis=4)
        assert_raises(AxisError, np.median, d, axis=(0, 4))
        assert_raises(ValueError, np.median, d, axis=(1, 1))

    def test_keepdims(self):
        d = np.ones((3, 5, 7, 11))
        assert_equal(np.median(d, axis=None, keepdims=True).shape,
                     (1, 1, 1, 1))
        assert_equal(np.median(d, axis=(0, 1), keepdims=True).shape,
                     (1, 1, 7, 11))
        assert_equal(np.median(d, axis=(0, 3), keepdims=True).shape,
                     (1, 5, 7, 1))
        assert_equal(np.median(d, axis=(1,), keepdims=True).shape,
                     (3, 1, 7, 11))
        assert_equal(np.median(d, axis=(0, 1, 2, 3), keepdims=True).shape,
                     (1, 1, 1, 1))
        assert_equal(np.median(d, axis=(0, 1, 3), keepdims=True).shape,
                     (1, 1, 7, 1))

    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1, ),
            (0, 1),
            (-3, -1),
        ]
    )
    def test_keepdims_out(self, axis):
        d = np.ones((3, 5, 7, 11))
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        out = np.empty(shape_out)
        result = np.median(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    @pytest.mark.parametrize("dtype", ["m8[s]"])
    @pytest.mark.parametrize("pos", [0, 23, 10])
    def test_nat_behavior(self, dtype, pos):
        # TODO: Median does not support Datetime, due to `mean`.
        # NaT and NaN should behave the same, do basic tests for NaT.
        a = np.arange(0, 24, dtype=dtype)
        a[pos] = "NaT"
        res = np.median(a)
        assert res.dtype == dtype
        assert np.isnat(res)
        res = np.percentile(a, [30, 60])
        assert res.dtype == dtype
        assert np.isnat(res).all()

        a = np.arange(0, 24 * 3, dtype=dtype).reshape(-1, 3)
        a[pos, 1] = "NaT"
        res = np.median(a, axis=0)
        assert_array_equal(np.isnat(res), [False, True, False])


class TestSortComplex:

    @pytest.mark.parametrize("type_in, type_out", [
        ('l', 'D'),
        ('h', 'F'),
        ('H', 'F'),
        ('b', 'F'),
        ('B', 'F'),
        ('g', 'G'),
        ])
    def test_sort_real(self, type_in, type_out):
        # sort_complex() type casting for real input types
        a = np.array([5, 3, 6, 2, 1], dtype=type_in)
        actual = np.sort_complex(a)
        expected = np.sort(a).astype(type_out)
        assert_equal(actual, expected)
        assert_equal(actual.dtype, expected.dtype)

    def test_sort_complex(self):
        # sort_complex() handling of complex input
        a = np.array([2 + 3j, 1 - 2j, 1 - 3j, 2 + 1j], dtype='D')
        expected = np.array([1 - 3j, 1 - 2j, 2 + 1j, 2 + 3j], dtype='D')
        actual = np.sort_complex(a)
        assert_equal(actual, expected)
        assert_equal(actual.dtype, expected.dtype)