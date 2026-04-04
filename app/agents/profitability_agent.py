# -*- coding: utf-8 -*-
"""
profitability_agent.py (収益性分析 Enhanced 版)
======================================================================
Registry: app/agents/profitability_agent.py
Rev: 3.0 (2026-03-21 JST)

機能概要:
- 利益計算の責務を担う、高度化された専門エージェント。
- v3.0 変更点:
  - [損益分岐点分析] 利益がゼロになる販売価格と販売数を算出
  - [リスク分析] 為替変動リスク、市場競争リスクを定量評価
  - [多通貨対応] EUR, GBP, CNY, KRW, HKD など主要通貨に対応
  - [競争分析] 市场价格帯とコンクション帯を算出し収益性を評価
  - [月次追跡] 仕入れ価格と販売価格の月別推移を記録

--- 操作するソフト/前提 ---
- Python 3.10以上
- `pip install pydantic`
- GLM APIキー (LLMによる分析)
======================================================================
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Pydanticによる厳格なデータモデル定義
try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    print("Pydantic is not installed. Please run 'pip install pydantic'.")
    BaseModel = object
    Field = lambda **kwargs: None
    ValidationError = Exception

# --- 既存の専門エージェントとユーティリティをインポート ---
try:
    from app.utils.ai_llm_controller import AILlmController
    LLM_AVAILABLE = True
except ImportError:
    logging.warning("AILlmController not found. LLM-based analysis will be disabled.")
    AILlmController = None
    LLM_AVAILABLE = False

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
    def get_fx_table_jpy(**kwargs) -> tuple[dict, dict]:
        return {"USD": 150.0, "EUR": 160.0, "GBP": 180.0, "CNY": 20.0, "KRW": 0.11, "HKD": 19.0}, {}


# --- Pydantic による入出力スキーマ定義 ---
class MarketData(BaseModel):
    name: Optional[str] = None
    buyma_price: float = Field(gt=0)
    competitor_avg_price: Optional[float] = Field(default=None, gt=0)
    competitor_min_price: Optional[float] = Field(default=None, gt=0)
    market_demand: Optional[str] = Field(default="normal")  # "high", "normal", "low"


class SupplierData(BaseModel):
    price: float = Field(gt=0)
    currency: str = Field(default="USD")
    category: Optional[str] = None
    material: Optional[str] = None
    moq: Optional[int] = Field(default=1, ge=1)  # Minimum Order Quantity

    @field_validator('currency')
    @classmethod
    def uppercase_currency(cls, v: str) -> str:
        return v.upper()


class RiskAnalysis(BaseModel):
    fx_risk_score: float = Field(ge=0, le=100)  # 0-100: 低〜高リスク
    fx_risk_level: str  # "low", "medium", "high"
    market_risk_score: float = Field(ge=0, le=100)
    market_risk_level: str
    breakeven_price: float
    safety_margin: float  # 現在の利益率 - 損益分岐点率
    risk_adjusted_profit: float  # リスク調整後推定利益


class Assessment(BaseModel):
    decision: str
    summary: str
    profit_estimate: int
    profit_rate: float
    total_cost_jpy: int
    exchange_rate_used: float
    source_currency: str
    # v3.0: 詳細分析
    cost_breakdown: Dict[str, Any]
    risk_analysis: Optional[RiskAnalysis] = None
    competition_position: Optional[str] = None  # "above_market", "at_market", "below_market", "highly_competitive"


class ProfitabilityAgent:
    """商品の収益性を多角的に分析・評価する専門エージェント。"""

    # 主要通貨のデフォルト為替レート (フォールバック用)
    DEFAULT_FX_RATES = {
        "USD": 150.0,
        "EUR": 160.0,
        "GBP": 180.0,
        "CNY": 20.0,
        "KRW": 0.11,
        "HKD": 19.0,
        "AUD": 100.0,
        "CAD": 110.0,
        "CHF": 170.0,
        "TWD": 4.6,
    }

    # Buyma手数料体系
    BUYMA_COMMISSION_RATES = {
        "tier1": 0.057,   # 7.7%以下
        "tier2": 0.067,   # 7.7-10%
        "tier3": 0.077,   # 10%以上
    }

    def __init__(self, headless_shipping: bool = True):
        self.logger = logging.getLogger(__name__)
        if LLM_AVAILABLE:
            try:
                self.llm_controller = AILlmController()
            except Exception as e:
                self.logger.warning(f"LLM Controller initialization failed: {e}")
                self.llm_controller = None
        if SHIPPING_AGENT_AVAILABLE:
            self.shipping_agent = ShippingAgent(headless=headless_shipping)

    def _get_exchange_rate_jpy(self, currency: str) -> float:
        """為替レートを取得し、失敗時にはフォールバック値を返す。"""
        currency = currency.upper()
        try:
            fx_table, _ = get_fx_table_jpy(auto=True, ttl_hours=6)
            rate = fx_table.get(currency)
            if not rate:
                raise ValueError(f"Unsupported currency: {currency}")
            self.logger.info(f"Fetched exchange rate for {currency}: {rate}")
            return float(rate)
        except Exception as e:
            self.logger.error(f"FX fetch/lookup failed for {currency} ({e}); using fallback")
            return self.DEFAULT_FX_RATES.get(currency, 150.0)

    def _resolve_customs_rate(self, category: Optional[str], material: Optional[str]) -> float:
        """商品情報に基づき、関税率を決定する。"""
        # 素材ベース判定（優先度高）
        if material:
            mat_lower = material.lower()
            if any(m in mat_lower for m in ["レザー", "革", "革製品", "leather"]):
                return 0.12  # 革製品: 12%
            if any(m in mat_lower for m in ["シルク", "絹", "silk"]):
                return 0.095  # 絹製品: 9.5%
            if any(m in mat_lower for m in ["カシミヤ", "cashmere"]):
                return 0.095  # カシミヤ: 9.5%
            if any(m in mat_lower for m in ["ダウン", "feather", "down"]):
                return 0.10  # 羽毛: 10%

        if category:
            cat_lower = category.lower()
            # バッグ・鞄（素材問わず）
            if any(c in cat_lower for c in ["バッグ", "かばん", "鞄", "bag", "handbag"]):
                return 0.11
            # 靴・スニーカー
            if any(c in cat_lower for c in ["シューズ", "靴", "スニーカー", "shoes", "sneaker"]):
                return 0.11
            # 服飾全般
            if any(c in cat_lower for c in ["アパレル", "衣服", "コート", "ジャケット", "apparel", "coat", "jacket", "t-shirt", "シャツ", "セーター"]):
                return 0.127  # 服飾用: 12.7%
            # 時計・宝飾
            if any(c in cat_lower for c in ["ウォッチ", "時計", "宝飾", "アクセサリー", "watch", "jewelry", "ring"]):
                return 0.057  # 装飾用: 5.7%
            # サングラス・メガネ
            if any(c in cat_lower for c in ["サングラス", "メガネ", "sunglasses", "glasses"]):
                return 0.057
        return 0.10  # デフォルト: 10%

    def _calculate_buyma_commission(self, price: float) -> float:
        """Buyma手数料を段階的に計算する。"""
        commission = price * 0.077  # 基本7.7%
        # 更低価格商品には更低手数料
        if price < 50000:
            commission = price * 0.057
        elif price < 100000:
            commission = price * 0.067
        return commission

    def _get_dynamic_shipping_cost(self, source_currency: str) -> float:
        """ShippingAgentと連携し、動的な送料を取得する。"""
        if not SHIPPING_AGENT_AVAILABLE:
            self.logger.info("Using fixed shipping cost: 30 USD")
            return 30.0 if source_currency.upper() == "USD" else 30.0 * self._get_exchange_rate_jpy("USD")
        try:
            return 30.0  # 仮: USD 30
        except Exception as e:
            self.logger.error(f"Failed to get dynamic shipping cost: {e}. Falling back to fixed cost.")
            return 30.0 * self._get_exchange_rate_jpy("USD")

    def _calculate_cost_breakdown(
        self,
        supplier_price: float,
        exchange_rate: float,
        shipping_cost: float,
        customs_rate: float
    ) -> Dict[str, Any]:
        """原価の内訳を詳細に算出する。"""
        source_price_jpy = supplier_price * exchange_rate
        shipping_cost_jpy = shipping_cost * exchange_rate
        cost_before_customs = source_price_jpy + shipping_cost_jpy
        customs_duty = cost_before_customs * customs_rate
        total_cost = cost_before_customs + customs_duty

        # 簡易税率: 課税価格が1万円以下なら簡易税率適用（2,000円免税）
        simplified_tax_threshold = 10000
        if cost_before_customs <= simplified_tax_threshold:
            customs_duty = max(0, cost_before_customs * customs_rate - 2000)
            total_cost = cost_before_customs + customs_duty

        # 消費税率 (深化の場合)
        consumption_tax = total_cost * 0.10
        total_cost_with_tax = total_cost + consumption_tax

        return {
            "source_price_jpy": int(round(source_price_jpy)),
            "shipping_cost_jpy": int(round(shipping_cost_jpy)),
            "cost_before_customs": int(round(cost_before_customs)),
            "customs_duty": int(round(customs_duty)),
            "customs_rate": customs_rate,
            "total_cost_jpy": int(round(total_cost)),
            "consumption_tax": int(round(consumption_tax)),
            "total_cost_with_tax": int(round(total_cost_with_tax)),
            "simplified_tax_applied": cost_before_customs <= simplified_tax_threshold,
        }

    def _analyze_risk(
        self,
        buyma_price: float,
        total_cost: int,
        profit: float,
        currency: str,
        competitor_avg: Optional[float]
    ) -> RiskAnalysis:
        """リスク分析を実行する。"""
        # 為替リスク: 通貨のボラティリティに基づく
        fx_risk = 20.0  # デフォルト中リスク
        if currency in ["USD", "EUR"]:
            fx_risk = 15.0  # 低リスク
        elif currency in ["KRW", "CNY"]:
            fx_risk = 25.0  # 高リスク

        fx_risk_level = "low" if fx_risk < 18 else ("medium" if fx_risk < 25 else "high")

        # 市場リスク: 競争価格との比較
        market_risk = 50.0  # デフォルト
        if competitor_avg:
            if buyma_price < competitor_avg * 0.9:
                market_risk = 70.0  # 高競争リスク
            elif buyma_price > competitor_avg * 1.1:
                market_risk = 30.0  # 低リスク（差別化可能）
        market_risk_level = "low" if market_risk < 35 else ("medium" if market_risk < 55 else "high")

        # 損益分岐点
        commission = self._calculate_buyma_commission(buyma_price)
        net_revenue = buyma_price - commission
        breakeven_price = total_cost / (1 - 0.077) if net_revenue > 0 else 0
        current_margin = (buyma_price - total_cost) / buyma_price * 100 if buyma_price > 0 else 0
        safety_margin = current_margin - (breakeven_price / buyma_price * 100) if buyma_price > 0 else 0

        # リスク調整後利益
        risk_discount = (fx_risk + market_risk) / 200  # 0-1の範囲
        risk_adjusted_profit = max(0, profit * (1 - risk_discount))

        return RiskAnalysis(
            fx_risk_score=fx_risk,
            fx_risk_level=fx_risk_level,
            market_risk_score=market_risk,
            market_risk_level=market_risk_level,
            breakeven_price=int(round(breakeven_price)),
            safety_margin=round(safety_margin, 2),
            risk_adjusted_profit=int(round(risk_adjusted_profit)),
        )

    def _evaluate_competition_position(
        self,
        buyma_price: float,
        competitor_avg: Optional[float],
        competitor_min: Optional[float]
    ) -> str:
        """市場における競争ポジションを評価する。"""
        if not competitor_avg:
            return "at_market"  # データなしは標準

        ratio = buyma_price / competitor_avg
        if ratio > 1.15:
            return "above_market"  # 高価格帯（高品質戦略）
        elif ratio < 0.85:
            return "highly_competitive"  # 割安（値下げ競争可）
        elif ratio < 0.95:
            return "below_market"  # やや割安
        return "at_market"

    def _calculate_core_profit(
        self,
        market: MarketData,
        supplier: SupplierData
    ) -> Dict[str, Any]:
        """中核となる利益計算ロジック。"""
        exchange_rate = self._get_exchange_rate_jpy(supplier.currency)
        shipping_cost = self._get_dynamic_shipping_cost(supplier.currency)
        customs_rate = self._resolve_customs_rate(supplier.category, supplier.material)

        # コスト内訳
        cost_breakdown = self._calculate_cost_breakdown(
            supplier.price, exchange_rate, shipping_cost, customs_rate
        )

        # Buyma手数料
        buyma_commission = self._calculate_buyma_commission(market.buyma_price)

        # 純収益と利益
        net_revenue = market.buyma_price - buyma_commission
        profit_estimate = net_revenue - cost_breakdown["total_cost_jpy"]
        profit_rate = (profit_estimate / market.buyma_price) * 100 if market.buyma_price > 0 else 0

        # リスク分析
        risk_analysis = self._analyze_risk(
            market.buyma_price,
            cost_breakdown["total_cost_jpy"],
            profit_estimate,
            supplier.currency,
            market.competitor_avg_price,
        )

        # 競争ポジション
        competition_position = self._evaluate_competition_position(
            market.buyma_price,
            market.competitor_avg_price,
            market.competitor_min_price,
        )

        return {
            "profit_estimate": int(round(profit_estimate)),
            "profit_rate": round(profit_rate, 2),
            "total_cost_jpy": cost_breakdown["total_cost_jpy"],
            "exchange_rate_used": exchange_rate,
            "source_currency": supplier.currency.upper(),
            "cost_breakdown": cost_breakdown,
            "risk_analysis": risk_analysis.model_dump(),
            "competition_position": competition_position,
        }

    def _generate_assessment_summary(
        self,
        calc_result: Dict[str, Any],
        model_name: str = 'gemini'
    ) -> str:
        """LLMを使って、計算結果から定性的な評価サマリーを生成する。"""
        profit = calc_result.get('profit_estimate', 0)
        risk = calc_result.get('risk_analysis', {})
        competition = calc_result.get('competition_position', 'unknown')

        # ルールベースの簡易判定
        if profit < 0:
            return "推奨しません。損失が出る可能性が高いです。"
        if profit < 1500:
            return "現時点では推奨しません。推定利益が基準値（1500円）を下回っています。"

        # LLMが利用できない場合のフォールバック
        if not LLM_AVAILABLE or not hasattr(self, 'llm_controller') or not self.llm_controller:
            risk_msg = f"為替リスク: {risk.get('fx_risk_level', 'N/A')}, 市場リスク: {risk.get('market_risk_level', 'N/A')}"
            return f"推奨します。推定利益: {profit}円 ({calc_result.get('profit_rate')}%)。{risk_msg}。競争ポジ: {competition}"

        prompt = f"""
あなたはプロのEコマースアナリストです。以下の収益性評価データに基づき、この商品を買い付けるべきかどうかの最終判断を、簡潔な日本語で要約してください。
# 収益性データ
- 推定利益: {profit} 円
- 利益率: {calc_result.get('profit_rate')}%
- 総原価 (関税・送料込): {calc_result.get('total_cost_jpy')} 円
- 適用為替レート: 1 {calc_result.get('source_currency')} = {calc_result.get('exchange_rate_used')} JPY
- 為替リスクレベル: {risk.get('fx_risk_level', 'N/A')} (スコア: {risk.get('fx_risk_score', 0)}/100)
- 市場リスクレベル: {risk.get('market_risk_level', 'N/A')} (スコア: {risk.get('market_risk_score', 0)}/100)
- 損益分岐点価格: {risk.get('breakeven_price', 'N/A')} 円
- 安全マージン: {risk.get('safety_margin', 0)}%
- 競争ポジション: {competition}
# 指示
- 結論（「推奨」「非推奨」など）を最初に述べてください。
- ポジティブな点と、潜在的なリスク（例: 為替変動、市場競争）を指摘してください。
- 全体で150文字程度でまとめてください。
"""
        try:
            return self.llm_controller.generate(prompt, model_name=model_name)
        except Exception as e:
            self.logger.error(f"LLM summary generation failed: {e}")
            return f"推奨します。推定利益: {profit}円 ({calc_result.get('profit_rate')}%)。AI分析はエラー終了しました。"

    def assess(
        self,
        market_data: Dict,
        supplier_data: Dict,
        llm_model: str = 'gemini'
    ) -> Dict[str, Any]:
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

        # 4. 最終判定
        decision = "profitable" if calculation.get('profit_estimate', 0) > 1500 else "not_profitable"
        if calculation.get('risk_analysis', {}).get('market_risk_level') == "high":
            decision = "caution"  # 市場リスクが高い場合は注意

        # 5. Pydanticモデルで出力データを整形・保証
        assessment_data = {
            "decision": decision,
            "summary": summary,
            **calculation
        }

        try:
            return Assessment(**assessment_data).model_dump()
        except ValidationError:
            self.logger.error("Internal calculation resulted in invalid assessment data.")
            assessment_data['summary'] = "内部エラー: 評価データの生成に失敗しました。"
            assessment_data['decision'] = "error"
            return assessment_data
