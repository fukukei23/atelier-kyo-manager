import json
from typing import List, Dict, Any

# pricing_calculator.py から利益計算関数をインポート
from pricing_calculator import calculate_profit, get_exchange_rate # get_exchange_rateもインポート

class BuymaResearchTool:
    def __init__(self):
        # 今後の機能拡張のために初期化
        pass

    def analyze_product_info(self, product_query: str) -> Dict[str, str]:
        """
        商品名または型番から商品情報を解析し、カテゴリなどを推定します。
        現時点ではシンプルなルールベースの推定を行います。
        将来的にはGPT連携を考慮します。
        """
        brand = "不明"
        category = "未分類"
        description = f"'{product_query}'に関する一般的な商品。"

        # 例: シンプルなキーワードマッチング
        if "Nike Air Max" in product_query:
            brand = "Nike"
            category = "メンズ > 靴 > スニーカー"
            description = "軽量で快適なAir Maxモデル。"
        elif "Louis Vuitton" in product_query or "LV" in product_query:
            brand = "Louis Vuitton"
            category = "レディース > バッグ・カバン > トートバッグ" # 例
            description = "高級感あふれるルイ・ヴィトンのアイテム。"
        # 他のルールを追加...

        return {
            "brand": brand,
            "category": category,
            "description": description
        }

    def search_suppliers(self, product_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        海外ECサイトから仕入れ先情報を検索します。
        この部分はWebスクレイピングやAPI連携が必要なため、現在はモックデータです。
        """
        print(f"DEBUG: Searching suppliers for {product_info['brand']} {product_info['description']}")
        # 実際のスクレイピング/API呼び出しの代わりにモックデータを返します
        mock_suppliers = [
            {
                "source_name": "SSENSE",
                "source_url": "https://www.ssense.com/mock_product_url",
                "source_price": 120, # USD
                "shipping_cost": 20, # USD
                "currency": "USD",
                "in_stock": True,
                "variants": {"27cm": "在庫あり", "28cm": "在庫切れ"}
            },
            {
                "source_name": "Farfetch",
                "source_url": "https://www.farfetch.com/mock_product_url",
                "source_price": 130, # USD
                "shipping_cost": 15, # USD
                "currency": "USD",
                "in_stock": False,
                "variants": {"27cm": "在庫切れ", "28cm": "在庫切れ"}
            }
        ]
        return mock_suppliers

    def get_buyma_sales_status(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        BUYMAでの販売状況を取得します。
        この部分はSeleniumなどによるクロールが必要なため、現在はモックデータです。
        """
        print(f"DEBUG: Getting BUYMA sales status for {product_info['brand']} {product_info['description']}")
        # 実際のクロールの代わりにモックデータを返します
        return {
            "buyma_lowest_price": 25300,
            "buyma_competitors": 7,
            "buyma_sales_recent": "30日以内に2件販売"
        }

    def calculate_and_rank_profit(
        self,
        buyma_target_price: float,
        suppliers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        各仕入れ先からの利益を計算し、利益率に基づいてランキングします。
        """
        profitable_options = []
        for supplier in suppliers:
            if supplier["in_stock"]:
                profit_data = calculate_profit(
                    buyma_price=buyma_target_price,
                    source_price=supplier["source_price"],
                    shipping_cost=supplier["shipping_cost"],
                    source_currency=supplier["currency"] # 仕入れ元の通貨を渡す
                )
                profit_estimate = profit_data["profit_estimate"]
                profit_rate = profit_data["profit_rate"]

                if profit_rate >= 10: # 利益率10%以上を候補とする
                    supplier_with_profit = supplier.copy()
                    supplier_with_profit["profit_estimate"] = profit_estimate
                    supplier_with_profit["profit_rate"] = f"{profit_rate}%"
                    supplier_with_profit["is_profitable"] = True
                    # 優先度スコアは仮で利益率をベースに設定
                    supplier_with_profit["priority_score"] = int(profit_rate)
                    profitable_options.append(supplier_with_profit)
            else:
                # 在庫がない場合は利益計算対象外
                supplier_with_profit = supplier.copy()
                supplier_with_profit["profit_estimate"] = 0
                supplier_with_profit["profit_rate"] = "0%"
                supplier_with_profit["is_profitable"] = False
                supplier_with_profit["priority_score"] = 0
                profitable_options.append(supplier_with_profit)


        # 利益率の高い順にソート
        profitable_options.sort(key=lambda x: float(x.get("profit_rate", "0%").replace('%', '')), reverse=True)
        return profitable_options

    def run_research(self, product_query: str) -> List[Dict[str, Any]]:
        """
        一連のリサーチプロセスを実行し、結果を返します。
        """
        print(f"STEP 1: Analyzing product information for '{product_query}'...")
        product_info = self.analyze_product_info(product_query)
        print(f"  - Brand: {product_info['brand']}, Category: {product_info['category']}")

        print("STEP 2: Searching for suppliers on cross-border EC sites...")
        suppliers = self.search_suppliers(product_info)
        print(f"  - Found {len(suppliers)} potential suppliers (mock data).")

        print("STEP 3: Getting BUYMA sales status...")
        buyma_status = self.get_buyma_sales_status(product_info)
        print(f"  - BUYMA Lowest Price: {buyma_status['buyma_lowest_price']}円, Competitors: {buyma_status['buyma_competitors']}")

        # BUYMAの目標販売価格を仮で設定（最低販売価格を参考に）
        buyma_target_price = buyma_status["buyma_lowest_price"]

        print("STEP 4: Calculating profit and ranking options...")
        ranked_options = self.calculate_and_rank_profit(buyma_target_price, suppliers)

        final_results = []
        for option in ranked_options:
            # source_total_price を動的に計算
            exchange_rate_to_jpy = get_exchange_rate(option["currency"], "JPY")
            source_total_price_jpy = (option["source_price"] + option["shipping_cost"]) * exchange_rate_to_jpy

            result = {
                "title": f"{product_info['brand']} {product_info['description']}",
                "brand": product_info["brand"],
                "category": product_info["category"],
                "buyma_price": buyma_target_price,
                "description": product_info["description"],
                "variants": option.get("variants", {}),
                "source_url": option["source_url"],
                "source_name": option["source_name"],
                "source_total_price": round(source_total_price_jpy, 2), # 動的に計算された合計仕入れ価格
                "profit_estimate": option["profit_estimate"],
                "profit_rate": option["profit_rate"],
                "buyma_competitors": buyma_status["buyma_competitors"],
                "buyma_lowest_price": buyma_status["buyma_lowest_price"],
                "buyma_sales_recent": buyma_status["buyma_sales_recent"],
                "is_profitable": option["is_profitable"],
                "priority_score": option["priority_score"]
            }
            final_results.append(result)

        print("Research complete. Generating output.")
        return final_results

# テスト用の使用例
if __name__ == "__main__":
    tool = BuymaResearchTool()
    query = "Nike Air Max 97 White Metallic Silver"
    results = tool.run_research(query)

    print("\n--- Research Results ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    query_lv = "Louis Vuitton Neverfull MM"
    results_lv = tool.run_research(query_lv)

    print("\n--- Research Results (Louis Vuitton) ---")
    print(json.dumps(results_lv, indent=2, ensure_ascii=False))
