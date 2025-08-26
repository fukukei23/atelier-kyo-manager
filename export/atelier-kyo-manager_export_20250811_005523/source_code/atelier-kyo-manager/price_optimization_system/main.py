# main.py - 価格最適化システム（GBDT + 時系列分析）
import pandas as pd
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel, Field
import time
from typing import List
import uvicorn

# FastAPIの日本語タイトル・説明
app = FastAPI(
    title="価格最適化API",
    description="GBDTと時系列分析による高速価格最適化システム",
    version="1.0.0"
)

# 入力データのスキーマ（日本語説明付き）
class PriceOptimizationRequest(BaseModel):
    product_id: int = Field(..., description="商品ID")
    cost: float = Field(..., description="原価")
    current_price: float = Field(..., description="現在価格")
    sales_history: List[float] = Field(..., description="過去7日間の売上データ")
    competitor_prices: List[float] = Field(..., description="競合価格データ")

# 出力データのスキーマ（日本語説明付き）
class PriceOptimizationResponse(BaseModel):
    product_id: int = Field(..., description="商品ID")
    optimal_price: float = Field(..., description="最適価格")
    predicted_demand: float = Field(..., description="予測需要")
    confidence_score: float = Field(..., description="信頼度スコア")
    processing_time_ms: int = Field(..., description="処理時間（ミリ秒）")

# ダミーのLightGBMモデル（実運用時は学習済みモデルを読み込む）
def create_dummy_model():
    X_dummy = np.random.rand(1000, 10)
    y_dummy = np.random.rand(1000)
    model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100
    )
    model.fit(X_dummy, y_dummy)
    return model

model = create_dummy_model()

# 特徴量生成（時系列・競合価格などをまとめて10次元に変換）
def create_features(request: PriceOptimizationRequest):
    features = []
    # 基本特徴量
    features.append(request.current_price / request.cost)  # 価格コスト比
    features.append(request.current_price)                 # 現在価格
    # 時系列特徴量
    if len(request.sales_history) >= 7:
        features.append(np.mean(request.sales_history[-7:]))  # 7日平均
        features.append(np.std(request.sales_history[-7:]))   # 7日標準偏差
        features.append(request.sales_history[-1])            # 直近売上
    else:
        features.extend([0, 0, 0])
    # 競合価格特徴量
    if request.competitor_prices:
        features.append(np.mean(request.competitor_prices))   # 競合平均価格
        features.append(request.current_price / np.mean(request.competitor_prices))  # 相対価格
    else:
        features.extend([0, 1])
    # パディング（10次元に調整）
    while len(features) < 10:
        features.append(0)
    return features[:10]

# 価格最適化エンドポイント
@app.post(
    "/optimize",
    response_model=PriceOptimizationResponse,
    summary="価格最適化",
    description="商品の最適価格を計算します"
)
async def optimize_price(request: PriceOptimizationRequest):
    start_time = time.time()
    features = create_features(request)
    predicted_demand = model.predict([features])[0]
    # 利益最大化のための最適価格計算（例：最低10%マージン＋需要ベース）
    optimal_price = max(
        request.cost * 1.1,
        request.cost + (predicted_demand * 0.1)
    )
    # 信頼度スコア（例：0.75～0.92の範囲で算出）
    confidence_score = min(0.92, max(0.75, 1 - abs(optimal_price - request.current_price) / request.current_price))
    processing_time = int((time.time() - start_time) * 1000)
    return PriceOptimizationResponse(
        product_id=request.product_id,
        optimal_price=round(optimal_price, 2),
        predicted_demand=round(predicted_demand, 2),
        confidence_score=round(confidence_score, 3),
        processing_time_ms=processing_time
    )

# ヘルスチェックエンドポイント
@app.get("/health", summary="ヘルスチェック", description="APIの稼働状況を返します")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)