
import math
import requests

def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """
    外部APIからリアルタイムの為替レートを取得します。
    ここでは、簡略化のため固定値を返すか、無料の公開APIを想定します。
    実際の運用では、APIキーが必要な場合や、より信頼性の高いサービスを利用することを推奨します。
    """
    # 例: ExchangeRate-API (無料プランで利用可能なAPI)
    # APIキーが必要な場合は 'YOUR_API_KEY' を置き換えてください
    # APIのURLは変更される可能性があるため、最新のドキュメントを確認してください
    api_url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる
        data = response.json()
        rate = data['rates'].get(to_currency)
        if rate:
            print(f"DEBUG: Fetched exchange rate {from_currency} to {to_currency}: {rate}")
            return rate
        else:
            print(f"WARNING: Could not find exchange rate for {to_currency} in API response. Using default.")
            return 150.0 # フォールバック値
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch exchange rate from API: {e}. Using default.")
        return 150.0 # API呼び出し失敗時のフォールバック値
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while fetching exchange rate: {e}. Using default.")
        return 150.0 # その他のエラー時のフォールバック値


def calculate_profit(
    buyma_price: float,
    source_price: float,
    shipping_cost: float,
    customs_duty_rate: float = 0.1, # 例: 関税率10%
    buyma_commission_rate: float = 0.073, # 例: BUYMA手数料7.3%
    buyma_system_fee: float = 200, # 例: BUYMAシステム利用料200円
    source_currency: str = "USD" # 仕入れ元の通貨を追加
) -> dict:
    """
    BUYMA出品における推定利益と利益率を計算します。

    Args:
        buyma_price (float): BUYMAでの目標販売価格 (日本円)。
        source_price (float): 海外仕入れ先からの商品価格 (元の通貨、例: USD/EUR)。
        shipping_cost (float): 仕入れ先からの送料 (元の通貨)。
        customs_duty_rate (float): 推定関税率 (例: 0.1 は10%)。
        buyma_commission_rate (float): BUYMA手数料率 (例: 0.073 は7.3%)。
        buyma_system_fee (float): 取引ごとの固定BUYMAシステム利用料。
        source_currency (str): 仕入れ元の通貨コード (例: "USD", "EUR")。

    Returns:
        dict: 'profit_estimate' (推定利益) と 'profit_rate' (利益率) を含む辞書。
    """
    # 為替レートを動的に取得
    exchange_rate = get_exchange_rate(source_currency, "JPY")

    # 仕入れ価格と送料を日本円に換算
    source_price_jpy = source_price * exchange_rate
    shipping_cost_jpy = shipping_cost * exchange_rate

    # 関税を含む総コストを計算
    # 関税は通常 (商品価格 + 送料) に対して計算されます
    cost_before_customs = source_price_jpy + shipping_cost_jpy
    customs_duty = cost_before_customs * customs_duty_rate
    total_cost_jpy = cost_before_customs + customs_duty

    # BUYMA手数料を計算
    buyma_commission = buyma_price * buyma_commission_rate
    total_buyma_fees = buyma_commission + buyma_system_fee

    # BUYMAからの純収益を計算
    net_revenue = buyma_price - total_buyma_fees

    # 利益を計算
    profit_estimate = net_revenue - total_cost_jpy

    # 利益率を計算
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
    print(f"利益率: {result['profit_rate']}%")

    # 例2: 利益が出ないケース
    result_low = calculate_profit(
        buyma_price=15000,
        source_price=120,
        shipping_cost=20,
        source_currency="USD"
    )
    print(f"推定利益 (低): {result_low['profit_estimate']} 円")
    print(f"利益率 (低): {result_low['profit_rate']}%")
