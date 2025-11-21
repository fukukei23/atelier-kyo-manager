# pricing_calculator.py (修正版)
import math
import requests

def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """
    外部APIからリアルタイムの為替レートを取得します。
    """
    api_url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        rate = data['rates'].get(to_currency)
        if rate:
            print(f"DEBUG: Fetched exchange rate {from_currency} to {to_currency}: {rate}")
            return rate
        else:
            print(f"WARNING: Could not find exchange rate for {to_currency}. Using default.")
            return 150.0
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch exchange rate from API: {e}. Using default.")
        return 150.0
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}. Using default.")
        return 150.0


def calculate_profit(
    buyma_price: float,
    source_price: float,
    shipping_cost: float,
    customs_duty_rate: float = 0.1,
    buyma_commission_rate: float = 0.073,
    buyma_system_fee: float = 200,
    source_currency: str = "USD"
) -> dict:
    """
    BUYMA出品における推定利益と利益率を計算します。
    """
    exchange_rate = get_exchange_rate(source_currency, "JPY")
    source_price_jpy = source_price * exchange_rate
    shipping_cost_jpy = shipping_cost * exchange_rate
    cost_before_customs = source_price_jpy + shipping_cost_jpy
    customs_duty = cost_before_customs * customs_duty_rate
    total_cost_jpy = cost_before_customs + customs_duty
    buyma_commission = buyma_price * buyma_commission_rate
    total_buyma_fees = buyma_commission + buyma_system_fee
    net_revenue = buyma_price - total_buyma_fees
    profit_estimate = net_revenue - total_cost_jpy
    profit_rate = (profit_estimate / buyma_price) * 100 if buyma_price > 0 else 0

    return {
        "profit_estimate": round(profit_estimate, 2),
        "profit_rate": round(profit_rate, 2)
    }

# テスト用の使用例 (このファイルが直接実行された場合のみ)
if __name__ == "__main__":
    # 例: BUYMA価格 26800円, SSENSE価格 120 USD, 送料 20 USD
    result = calculate_profit(
        buyma_price=26800,
        source_price=120,
        shipping_cost=20,
        source_currency="USD"
    )
    print(f"推定利益: {result['profit_estimate']} 円")
    print(f"利益率: {result['profit_rate']}%") # ★★★ ここを修正しました ★★★

    # 例2: 利益が出ないケース
    result_low = calculate_profit(
        buyma_price=15000,
        source_price=120,
        shipping_cost=20,
        source_currency="USD"
    )
    print(f"推定利益 (低): {result_low['profit_estimate']} 円")
    print(f"利益率 (低): {result_low['profit_rate']}%") # ★★★ ここを修正しました ★★★
