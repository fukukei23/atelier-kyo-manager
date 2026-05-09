# メモ:
# ファイル名: run_test.py
# 格納場所: /tests/
# 役割: リジリエント版 ai_image_crawler.py の動作をテストするための実行スクリプト。

import logging
import os
import pprint
import sys

from flask import Flask

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils.ai_image_crawler import CrawlerService

# --- テスト設定 (ここを編集) ---
TEST_TARGET_URL = "https://www.farfetch.com/jp/shopping/women/off-white--item-22356940.aspx"
TEST_MASTER_IMAGE_URL = "https://cdn-images.farfetch-contents.com/22/35/69/40/22356940_52495907_1000.jpg"

app = Flask(__name__)

# --- CONFIG ---
# ◆◆◆ 重要: 以下のAPIキーをご自身のものに置き換えてください ◆◆◆
app.config["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
app.config["DEEPSEEK_API_KEY"] = "YOUR_DEEPSEEK_API_KEY_HERE"

# --- クローラー設定 ---
app.config["CRAWLER_HEADLESS"] = True
app.config["CRAWLER_DEFAULT_WAIT_TIME"] = 60
app.config["CRAWLER_DISABLE_CLIP"] = False

# --- ターゲットサイト設定 ---
app.config["CRAWLER_TARGET_SITES"] = {
    "farfetch.com": {
        "search_url_template": "https://www.farfetch.com/jp/search?q={query}",
        "cookie_accept_selector": "#onetrust-accept-btn-handler",
        # ---【戦術1】セレクタの冗長化 ---
        "search_result_link_selector": "a[data-testid='productCard-link'], a[data-component*='ProductCard'], a[data-test*='productCard'], a[href*='-item-']",
        "structured_data_selector": 'script[type="application/ld+json"]',
    }
}


def run_test():
    """テストを実行し、結果を出力します。"""
    service = None
    with app.app_context():
        try:
            print("--- テスト開始 ---")
            print(f"ターゲットURL: {TEST_TARGET_URL}")

            service = CrawlerService()

            result = service.get_product_info(url=TEST_TARGET_URL, master_image_url=TEST_MASTER_IMAGE_URL)

            print("\n--- テスト結果 ---")
            pprint.pprint(result)

        except Exception:
            logging.error("テスト中に予期せぬエラーが発生しました。", exc_info=True)
        finally:
            if service:
                print("\n--- クリーンアップ処理 ---")
                service.quit()
            print("--- テスト完了 ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_test()
