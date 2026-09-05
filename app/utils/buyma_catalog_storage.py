"""
BUYMA カタログ ストレージ — ダウンロード・ZIP解凍・CSV・Google Sheets 連携

buyma_catalog_manager.py から分離。
"""
from __future__ import annotations

import csv
import hashlib
import os
import zipfile
from datetime import datetime

import requests

CONFIG = {
    "profile_path": r"C:/Users/USER/AppData/Local/Google/Chrome/SeleniumProfile",
    "base_dir": "D:/catalog_images",
    "screenshot_dir": "D:/screenshots",
    "csv_path": "D:/catalog_data.csv",
    "extracted_images_dir": "D:/extracted_images",
    "google_credentials": "D:/credentials.json",
    "spreadsheet_id": "1z9_lczAbnbsMYpAEslamfekEMrPQVIM1rfHqNbzze_Y",
    "worksheet_name": "catalog_data",
    "safety": {
        "max_daily_requests": 500,
        "request_interval": (5, 10),
        "error_threshold": 10,
        "response_time_threshold": 8.0,
    },
}


class CatalogStorage:
    """カタログ画像のダウンロード・保存・外部連携を管理"""

    def __init__(self) -> None:
        self.downloaded_hashes: set[str] = set()
        self.downloaded_catalog_ids: set[str] = set()
        self.csv_records: list[dict] = []
        self._setup_directories()
        self._init_google_sheets()

    def _setup_directories(self) -> None:
        for directory in [CONFIG["screenshot_dir"], CONFIG["base_dir"], CONFIG["extracted_images_dir"]]:
            os.makedirs(directory, exist_ok=True)

    def _init_google_sheets(self) -> None:
        # gspread/google-auth は requirements 未宣言の重い実行時依存のため
        # ここで遅延 import（未インストール環境では既存の try/except で fail-open）
        import gspread
        from google.oauth2.service_account import Credentials

        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_file(CONFIG["google_credentials"], scopes=scope)
            self.gc = gspread.authorize(creds)
            self.worksheet = self.gc.open_by_key(CONFIG["spreadsheet_id"]).worksheet(CONFIG["worksheet_name"])
            print("Googleスプレッドシート接続成功")
        except Exception as e:
            print(f"Googleスプレッドシート接続エラー: {e}")
            self.gc = None
            self.worksheet = None

    def extract_images_from_zip(self, zip_path: str, brand_name: str, catalog_id: str) -> tuple[int, list[str]]:
        try:
            extract_dir = os.path.join(CONFIG["extracted_images_dir"], brand_name, catalog_id)
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            image_files = []
            for root, _dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                        old_path = os.path.join(root, file)
                        new_filename = f"{brand_name}_{catalog_id}_{len(image_files) + 1}_{file}"
                        new_path = os.path.join(extract_dir, new_filename)
                        os.rename(old_path, new_path)
                        image_files.append(new_path)

            return len(image_files), image_files
        except Exception as e:
            print(f"ZIP解凍エラー: {e}")
            return 0, []

    def download_file(
        self, url: str, brand_name: str, catalog_id: str, cookies: list[dict], referer: str, user_agent: str
    ) -> tuple[bool, dict | None]:
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie["name"], cookie["value"])
        headers = {"Referer": referer, "User-Agent": user_agent}

        response = session.get(url, headers=headers)
        if response.status_code != 200:
            return False, None

        file_hash = hashlib.md5(response.content).hexdigest()
        if catalog_id in self.downloaded_catalog_ids or file_hash in self.downloaded_hashes:
            return False, None

        save_dir = os.path.join(CONFIG["base_dir"], brand_name, catalog_id)
        os.makedirs(save_dir, exist_ok=True)
        zip_path = os.path.join(save_dir, f"catalog_{catalog_id}.zip")

        with open(zip_path, "wb") as f:
            f.write(response.content)

        image_count, image_files = self.extract_images_from_zip(zip_path, brand_name, catalog_id)

        record = {
            "brand": brand_name,
            "catalog_id": catalog_id,
            "zip_path": zip_path,
            "extracted_dir": os.path.join(CONFIG["extracted_images_dir"], brand_name, catalog_id),
            "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_count": image_count,
            "file_size": os.path.getsize(zip_path),
            "first_image_path": image_files[0] if image_files else "",
            "all_image_paths": "|".join(image_files),
            "status": "success",
        }

        self.csv_records.append(record)
        self.downloaded_hashes.add(file_hash)
        self.downloaded_catalog_ids.add(catalog_id)
        self.add_to_google_sheet(record)

        return True, record

    def add_to_google_sheet(self, record: dict) -> None:
        if not self.worksheet:
            return
        try:
            row_data = [
                record["brand"],
                record["catalog_id"],
                record["zip_path"],
                record["extracted_dir"],
                record["download_date"],
                record["image_count"],
                record["file_size"],
                record["first_image_path"],
                record["all_image_paths"],
                record["status"],
            ]
            self.worksheet.append_row(row_data)
            print(f"Googleスプレッドシートに追加: {record['brand']} {record['catalog_id']}")
        except Exception as e:
            print(f"Googleスプレッドシート追加エラー: {e}")

    def save_csv_summary(self) -> None:
        if not self.csv_records:
            return

        fieldnames = [
            "brand",
            "catalog_id",
            "zip_path",
            "extracted_dir",
            "download_date",
            "image_count",
            "file_size",
            "first_image_path",
            "all_image_paths",
            "status",
        ]

        with open(CONFIG["csv_path"], "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_records)

        print(f"詳細レポート保存: {CONFIG['csv_path']}")

        total_downloads = len(self.csv_records)
        total_images = sum(record["image_count"] for record in self.csv_records)
        total_size_mb = sum(record["file_size"] for record in self.csv_records) / (1024 * 1024)

        summary = f"""
=== ダウンロード完了サマリー ===
総ダウンロード数: {total_downloads}件
総画像数: {total_images}枚
総ファイルサイズ: {total_size_mb:.2f}MB
保存先: {CONFIG["base_dir"]}
解凍画像: {CONFIG["extracted_images_dir"]}
CSV詳細: {CONFIG["csv_path"]}
Googleスプレッドシート: {"連携済み" if self.worksheet else "未接続"}
        """
        print(summary)
