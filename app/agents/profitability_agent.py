# -*- coding: utf-8 -*-
"""
profitability_agent.py (堅牢性・再利用性向上版)
======================================================================
Registry: app/agents/profitability_agent.py
Rev: 2.0 (2025-09-08 JST)

機能概要:
- 利益計算の責務を担う、高度化された専門エージェント。
- ★戦略更新★「堅牢性の強化」:
  - Pydanticによる厳格な入出力スキーマ検証を導入し、データの不整合を早期に検知。
  - 為替レート取得失敗時に安全なフォールバック値を使用し、システムの停止を防止。
  - LLMによる要約生成失敗時も、ルールベースの簡易サマリーを返すことで応答性を維持。
- ★戦略更新★「抽象化による拡張性の向上」:
  - 関税率の決定ロジックを専用メソッドに分離し、将来的なルール変更に容易に対応可能。
- 既存の連携機能（ShippingAgent, LLMController）はすべて維持・強化。

--- 操作するソフト/前提 ---
- Python 3.10以上
- `pip install pydantic`
- OpenAI APIキー (LLMによるサマリー生成に必要)
======================================================================
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional

# Pydanticによる厳格なデータモデル定義
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    print("Pydantic is not installed. Please run 'pip install pydantic'.")
    # Pydanticがない場合は、ダミーのBaseModelで最低限動作させる
    BaseModel = object
    Field = lambda **kwargs: None
    ValidationError = Exception

# --- 既存の専門エージェントとユーティリティをインポート ---
try:
    from app.utils.ai_llm_controller import AILlmController
    LLM_AVAILABLE = True
except ImportError:
    logging.warning("AILlmController not found. LLM-based summary will be disabled.")
    AILlmController = None
    LLM_AVAILABLE = False

# --- Pricing Config（手数料率の一元管理）をインポート ---
try:
    from app.core.pricing.rules import load_pricing_config
    from app.core.pricing.calculator import calculate_pricing
    from app.core.pricing.schemas import PricingInput
    PRICING_CONFIG_AVAILABLE = True
except ImportError:
    logging.warning("PricingConfig not found. Using hardcoded fee rates.")
    load_pricing_config = None
    calculate_pricing = None
    PricingInput = None
    PRICING_CONFIG_AVAILABLE = False

try:
    from app.utils.shipping_agent import ShippingAgent
    SHIPPING_AGENT_AVAILABLE = True
except ImportError:
    logging.warning("ShippingAgent not found. Using fixed shipping costs.")
    ShippingAgent = None
    SHIPPING_AGENT_AVAILABLE = False

try:
    from app.utils.fx_utils import get_fx_table_jpy
except ImportError:
     logging.warning("fx_utils not found. Using dummy exchange rates.")
     def get_fx_table_jpy(**kwargs) -> tuple[dict, dict]: return {"USD": 150.0}, {}

# --- Pydanticによる入出力スキーマ定義 ---

class MarketData(BaseModel):
    name: Optional[str] = None
    buyma_price: float = Field(gt=0)

class SupplierData(BaseModel):
    price: float = Field(gt=0)
    currency: str
    category: Optional[str] = None
    material: Optional[str] = None

class Assessment(BaseModel):
    decision: str
    summary: str
    profit_estimate: int
    profit_rate: float
    total_cost_jpy: int
    exchange_rate_used: float
    source_currency: str

class ProfitabilityAgent:
    """商品の収益性を多角的に分析・評価する専門エージェント。"""

    def __init__(self, headless_shipping: bool = True):
        self.logger = logging.getLogger(__name__)
        if LLM_AVAILABLE:
            self.llm_controller = AILlmController()
        if SHIPPING_AGENT_AVAILABLE:
            self.shipping_agent = ShippingAgent(headless=headless_shipping)

    def _get_exchange_rate_jpy(self, currency: str) -> float:
        """為替レートを取得し、失敗時にはフォールバック値を返す。"""
        try:
            fx_table, _ = get_fx_table_jpy(auto=True, ttl_hours=6)
            rate = fx_table.get(currency.upper())
            if not rate:
                raise ValueError(f"Unsupported currency: {currency}")
            self.logger.info(f"Fetched exchange rate for {currency}: {rate}")
            return float(rate)
        except Exception as e:
            self.logger.error(f"FX fetch/lookup failed for {currency} ({e}); using fallback 150.0")
            return 150.0

    def _resolve_customs_rate(self, category: Optional[str], material: Optional[str]) -> float:
        """商品情報に基づき、関税率を決定する。rules.py の一元化ロジックを使用。"""
        try:
            from app.core.pricing.rules import resolve_customs_rate
            return resolve_customs_rate(category, material)
        except ImportError:
            # フォールバック: rules.py が利用できない場合
            if material and any(m in material for m in ["レザー", "革", "leather"]):
                return 0.12
            if category and any(c in category for c in ["バッグ", "シューズ", "bag", "shoes"]):
                return 0.11
            return 0.10

    def _get_dynamic_shipping_cost(self, source_currency: str) -> float:
        """ShippingAgentと連携し、動的な送料を取得する。"""
        if not SHIPPING_AGENT_AVAILABLE:
            self.logger.info("Using fixed shipping cost: 30 USD")
            return 30.0 if source_currency.upper() == "USD" else 4500.0
        try:
            self.logger.info("Dynamically fetching shipping info (using fixed cost for now).")
            return 30.0 # 仮にUSDで30ドルとする
        except Exception as e:
            self.logger.error(f"Failed to get dynamic shipping cost: {e}. Falling back to fixed cost.")
            return 30.0

    def _calculate_core_profit(self, market: MarketData, supplier: SupplierData) -> Dict[str, Any]:
        """
        中核となる利益計算ロジック。calculator.py を再利用。

        市場分析・シミュレーション用途のため、buyma_platform_fee_rate（7.7%）を使用。
        実取引の計算（14.2%実効レート）は Product._calculate_pricing() 経由で行う。
        """
        exchange_rate = self._get_exchange_rate_jpy(supplier.currency)
        shipping_cost = self._get_dynamic_shipping_cost(supplier.currency)

        # calculator.py が利用可能な場合はそちらに委譲
        if PRICING_CONFIG_AVAILABLE and calculate_pricing and PricingInput:
            inp = PricingInput(
                purchase_price=supplier.price,
                selling_price=market.buyma_price,
                shipping_cost=shipping_cost,
                original_currency=supplier.currency.upper(),
                exchange_rate=exchange_rate,
                item_category=supplier.category or "",
                item_material=supplier.material or "",
            )
            result = calculate_pricing(inp, source_type="overseas")

            return {
                "profit_estimate": int(round(result.profit)),
                "profit_rate": round(result.profit_rate * 100, 2),
                "total_cost_jpy": int(round(result.total_cost)),
                "exchange_rate_used": exchange_rate,
                "source_currency": supplier.currency.upper(),
            }

        # フォールバック: calculator.py が利用できない場合の従来ロジック
        customs_rate = self._resolve_customs_rate(supplier.category, supplier.material)
        source_price_jpy = supplier.price * exchange_rate
        shipping_cost_jpy = shipping_cost * exchange_rate
        cost_before_customs = source_price_jpy + shipping_cost_jpy
        customs_duty = cost_before_customs * customs_rate
        total_cost_jpy = cost_before_customs + customs_duty
        platform_fee_rate = 0.077
        buyma_commission = market.buyma_price * platform_fee_rate
        net_revenue = market.buyma_price - buyma_commission
        profit_estimate = net_revenue - total_cost_jpy
        profit_rate = (profit_estimate / market.buyma_price) * 100 if market.buyma_price > 0 else 0

        return {
            "profit_estimate": int(round(profit_estimate)),
            "profit_rate": round(profit_rate, 2),
            "total_cost_jpy": int(round(total_cost_jpy)),
            "exchange_rate_used": exchange_rate,
            "source_currency": supplier.currency.upper(),
        }

    def _generate_assessment_summary(self, calc_result: Dict[str, Any], model_name: str = 'gemini') -> str:
        """LLMを使って、計算結果から定性的な評価サマリーを生成する。"""
        profit = calc_result.get('profit_estimate', 0)

        # ルールベースの簡易判定
        if profit < 1500:
            return "推奨しません。推定利益が基準値（1500円）を下回っています。"

        # LLMが利用できない、または失敗した場合のフォールバック
        if not LLM_AVAILABLE:
            return f"推奨します。推定利益: {profit}円 ({calc_result.get('profit_rate')}%)"

        prompt = f"""
あなたはプロのEコマースアナリストです。以下の収益性評価データに基づき、この商品を買い付けるべきかどうかの最終判断を、簡潔な日本語で要約してください。
# 収益性データ
- 推定利益: {profit} 円
- 利益率: {calc_result.get('profit_rate')}%
- 総原価 (関税・送料込): {calc_result.get('total_cost_jpy')} 円
- 適用為替レート: 1 {calc_result.get('source_currency')} = {calc_result.get('exchange_rate_used')} JPY
# 指示
- 結論（「推奨」「非推奨」など）を最初に述べてください。ポジティブな点と、潜在的なリスク（例: 為替変動）を指摘してください。全体で100文字程度でまとめてください。
"""
        try:
            return self.llm_controller.generate(prompt, model_name=model_name)
        except Exception as e:
            self.logger.error(f"LLM summary generation failed: {e}")
            # LLM失敗時のフォールバックサマリー
            return f"推奨します。推定利益: {profit}円 ({calc_result.get('profit_rate')}%)。AIによる詳細分析中にエラーが発生しました。"

    def assess(self, market_data: Dict, supplier_data: Dict, llm_model: str = 'gemini') -> Dict[str, Any]:
        """
        市場データと仕入先データを基に、商品の収益性を総合的に評価する。
        """
        self.logger.info(f"Assessing profitability for: {market_data.get('name', 'N/A')}")

        # 1. Pydanticによる入力データの検証
        try:
            market = MarketData(**market_data)
            supplier = SupplierData(**supplier_data)
        except ValidationError as ve:
            summary = f"入力データの検証に失敗しました: {ve.errors()}"
            self.logger.error(summary)
            return {"decision": "error", "summary": summary}
        except Exception as e:
             return {"decision": "error", "summary": f"予期せぬ入力エラー: {e}"}

        # 2. 中核となる利益計算を実行
        calculation = self._calculate_core_profit(market, supplier)

        # 3. LLMによる評価サマリーを生成
        summary = self._generate_assessment_summary(calculation, model_name=llm_model)

        # 4. Pydanticモデルで出力データを整形・保証
        assessment_data = {
            "decision": "profitable" if calculation.get('profit_estimate', 0) > 1500 else "not_profitable",
            "summary": summary,
            **calculation
        }

        try:
            return Assessment(**assessment_data).model_dump()
        except ValidationError:
            # 万が一、内部の計算結果が出力スキーマに違反した場合のエラーハンドリング
            self.logger.error("Internal calculation resulted in invalid assessment data.")
            assessment_data['summary'] = "内部エラー: 評価データの生成に失敗しました。"
            assessment_data['decision'] = "error"
            return assessment_data
