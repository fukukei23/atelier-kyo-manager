# -*- coding: utf-8 -*-
"""
Bright Data Scraper API 統合サービス
SSENSE等のECサイトから商品データを取得し、利益計算を行う

API仕様（テスト結果より）:
- ジョブ開始: POST /dca/trigger_immediate?collector=XXX&queue_next=1
- 結果取得: GET /dca/get_result?response_id=XXX
- 認証: Bearer token
"""

import os
import sys
import time
import logging
import requests

# ルートディレクトリをsys.pathに追加してpricing_calculatorをインポート
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pricing_calculator import calculate_profit

logger = logging.getLogger(__name__)

COLLECTOR_ID = "c_mnub8vs31ch29pinx1"
API_BASE = "https://api.brightdata.com/dca"


class BrightDataScraper:
    """Bright Data Scraper APIを利用してECサイトの商品データを取得・分析する"""

    def __init__(self, collector_id: str = COLLECTOR_ID):
        self.api_token = os.getenv("BRIGHTDATA_API_TOKEN")
        if not self.api_token:
            raise ValueError("BRIGHTDATA_API_TOKEN is not set in environment variables.")

        self.collector_id = collector_id
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        self.trigger_url = f"{API_BASE}/trigger_immediate?collector={collector_id}&queue_next=1"
        self.result_url = f"{API_BASE}/get_result"

    def _trigger_job(self, url: str) -> str:
        """スクレイピングジョブを開始し、response_idを返却する"""
        logger.info(f"Triggering Bright Data job for: {url}")

        payload = [{"url": url}]

        try:
            response = requests.post(
                self.trigger_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            response_id = data.get("response_id") or data.get("id")

            if not response_id:
                logger.error(f"No response_id in trigger response: {data}")
                raise ValueError("API response does not contain response_id")

            logger.info(f"Job triggered. response_id: {response_id}")
            return response_id

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger job: {e}")
            raise ConnectionError(f"Bright Data API trigger failed: {e}")

    def _poll_result(self, response_id: str, timeout: int = 120, interval: int = 5) -> list:
        """ジョブ結果をポーリングして取得する"""
        logger.info(f"Polling result for response_id: {response_id} (timeout={timeout}s)")

        start_time = time.time()
        url = f"{self.result_url}?response_id={response_id}"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()

                data = response.json()

                # 結果が配列で返ってきたら完了
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"Result retrieved: {len(data)} products")
                    return data

                # まだ処理中
                logger.debug(f"Job still running. Waiting {interval}s...")
                time.sleep(interval)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Polling error: {e}. Retrying...")
                time.sleep(interval)

        raise TimeoutError(f"Polling timed out after {timeout}s for response_id: {response_id}")

    def fetch_products(self, url: str) -> list:
        """指定URLから商品リストを取得する"""
        response_id = self._trigger_job(url)
        products = self._poll_result(response_id)

        if not products:
            raise ValueError("No products returned from scraper.")

        logger.info(f"Fetched {len(products)} products from {url}")
        return products

    def calculate_profits(self, products: list, buyma_markup_rate: float = 1.5, shipping_cost_usd: float = 30.0) -> list:
        """
        各商品の利益計算を行う

        Args:
            products: fetch_products()で取得した商品データリスト
            buyma_markup_rate: BUYMA価格 = source_price × 為替 × この倍率
            shipping_cost_usd: 送料（USD）
        """
        calculated = []

        for product in products:
            try:
                price_info = product.get("price", {})
                source_price = price_info.get("value", 0.0)
                source_currency = price_info.get("currency", "USD")

                if source_price <= 0:
                    continue

                # BUYMA販売価格 = 仕入れ価格(USD) × 為替 × マークアップ率
                # pricing_calculator内部で為替取得→JPY換算されるので、
                # buyma_priceは source_price(JPY換算) × markup として計算
                # ※pricing_calculatorが為替取得するので、ここでは単純に金額を渡す
                buyma_price = source_price * 150.0 * buyma_markup_rate  # 概算JPY

                profit_info = calculate_profit(
                    buyma_price=buyma_price,
                    source_price=source_price,
                    shipping_cost=shipping_cost_usd,
                    customs_duty_rate=0.1,
                    buyma_commission_rate=0.073,
                    buyma_system_fee=200,
                    source_currency=source_currency
                )

                enriched = {
                    **product,
                    "calculated": {
                        "buyma_price": round(buyma_price, 2),
                        "shipping_cost_usd": shipping_cost_usd,
                        "profit_estimate_jpy": profit_info.get("profit_estimate", 0),
                        "profit_rate_pct": profit_info.get("profit_rate", 0)
                    }
                }
                calculated.append(enriched)

            except Exception as e:
                logger.error(f"Profit calc failed for {product.get('product_name', '?')}: {e}")
                continue

        logger.info(f"Calculated profits for {len(calculated)}/{len(products)} products")
        return calculated

    def find_profitable_products(self, url: str, min_profit_rate: float = 20.0, buyma_markup_rate: float = 1.5) -> list:
        """利益率が基準以上の商品を抽出する"""
        logger.info(f"Finding products with profit_rate >= {min_profit_rate}%")

        products = self.fetch_products(url)
        calculated = self.calculate_profits(products, buyma_markup_rate=buyma_markup_rate)

        profitable = [
            p for p in calculated
            if p.get("calculated", {}).get("profit_rate_pct", 0) >= min_profit_rate
        ]

        logger.info(f"Found {len(profitable)}/{len(calculated)} profitable products")
        return profitable


# CLI テスト用
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_project_root, '.env'))

    logging.basicConfig(level=logging.INFO)

    scraper = BrightDataScraper()

    # Step 1: 商品取得テスト
    print("=== Fetching SSENSE products ===")
    products = scraper.fetch_products("https://www.ssense.com/en-us/men/clothing")
    print(f"Got {len(products)} products")
    print(f"Sample: {products[0]['brand']} - {products[0]['product_name']} - ${products[0]['price']['value']}")

    # Step 2: 利益計算テスト
    print("\n=== Calculating profits ===")
    calculated = scraper.calculate_profits(products[:5])
    for p in calculated:
        c = p["calculated"]
        print(f"{p['brand']}: ${p['price']['value']} → ¥{c['buyma_price']:,} → 利益¥{c['profit_estimate_jpy']:,} ({c['profit_rate_pct']:.1f}%)")

    # Step 3: 利益率20%以上の商品
    print("\n=== Profitable products (>= 20%) ===")
    profitable = scraper.find_profitable_products("https://www.ssense.com/en-us/men/clothing")
    print(f"Found {len(profitable)} products")
