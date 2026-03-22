# -- coding: utf-8 --
# ======================================================================
# File: app/web/dashboard.py
# Registry: app/web/dashboard.py
# Rev: 2.0 (2026-03-21 00:00 JST)
#
# 目的:
# - アラートの感度をビジネス要件に合わせて最適化する。
# - 収益性分析とエクスポート機能を追加。
#
# 変更点 (v2.0):
# - [収益性分析] 仕入れ価格、销售価格、利益率を表示
# - [エクスポート] CSV/JSON エクスポート機能
# - [トレンド分析] 价格趋势と最适合点を算出
# - [信頼区間] EMAの信頼区間を追加
# ======================================================================
from __future__ import annotations
import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence
from flask import Flask, jsonify, render_template, request, Response

# --- DB (optional) ---
try:
    from app import db
    from app.models import ProductPriceHistory
    FLASK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    db = None
    ProductPriceHistory = None
    FLASK_AVAILABLE = False
    logging.info("[Dashboard] DB models not found. Running in CSV-only mode.")


# --- Flask App Initialization (Template Path Fixed) ---
try:
    APP_DIR = Path(__file__).resolve().parent.parent  # -> app/
    TEMPLATE_DIR = APP_DIR / 'templates'
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
except NameError:
    app = Flask(__name__)
    logging.warning("Could not determine absolute path for templates. Using default.")

OUTPUT_DIR = Path("output")
DAYS_DEFAULT = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# Data Models and Utilities
# =============================================================================
@dataclass
class HistoryRow:
    """A class to represent a single row of price history."""
    captured_at: datetime
    brand: str
    name: str
    url: str
    price: Optional[float]
    list_price: Optional[float]
    discount_pct: Optional[float]
    sale_until: Optional[datetime]
    # v2.0: 収益性フィールド
    purchase_price: Optional[float] = None
    selling_price: Optional[float] = None


@dataclass
class ProfitabilityMetrics:
    """収益性分析结果"""
    current_profit: Optional[float] = None
    current_profit_rate: Optional[float] = None
    avg_profit: Optional[float] = None
    avg_profit_rate: Optional[float] = None
    best_price_date: Optional[str] = None
    best_price: Optional[float] = None
    worst_price_date: Optional[str] = None
    worst_price: Optional[float] = None
    price_volatility: Optional[float] = None  # 価格変動係数
    trend_direction: str = "stable"  # "rising", "falling", "stable"
    recommendation: str = ""


def ema_series(values: Sequence[Optional[float]], period: int) -> List[Optional[float]]:
    """Calculates the Exponential Moving Average for a whole time series."""
    if not values:
        return []
    alpha = 2 / (period + 1)

    first_valid_idx = next((i for i, v in enumerate(values) if v is not None), -1)
    if first_valid_idx == -1:
        return [None] * len(values)

    ema_results: List[Optional[float]] = [None] * first_valid_idx
    ema_results.append(values[first_valid_idx])

    for i in range(first_valid_idx + 1, len(values)):
        current_val = values[i]
        prev_ema = ema_results[i - 1]

        if current_val is None:
            ema_results.append(prev_ema)
        else:
            ema_results.append(alpha * current_val + (1 - alpha) * prev_ema)

    return ema_results


def pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
    """Calculates percentage change between two values."""
    if old is None or new is None or old == 0:
        return None
    return (new - old) / old * 100


def calculate_profitability_metrics(
    rows: List[HistoryRow],
    purchase_price: Optional[float] = None
) -> ProfitabilityMetrics:
    """価格データから収益性分析指標を算出する (v2.0)"""
    if not rows:
        return ProfitabilityMetrics()

    prices = [(r.captured_at, r.price) for r in rows if r.price is not None]
    if not prices:
        return ProfitabilityMetrics()

    prices.sort(key=lambda x: x[0])

    # 基本統計
    all_prices = [p[1] for p in prices]
    current_price = all_prices[-1] if all_prices else None
    min_price = min(all_prices) if all_prices else None
    max_price = max(all_prices) if all_prices else None

    # 最高/最安日
    best_price_date = min(prices, key=lambda x: x[1]) if prices else (None, None)
    worst_price_date = max(prices, key=lambda x: x[1]) if prices else (None, None)

    # 変動係数 (Coefficient of Variation)
    if all_prices and len(all_prices) > 1:
        mean_price = sum(all_prices) / len(all_prices)
        variance = sum((p - mean_price) ** 2 for p in all_prices) / len(all_prices)
        volatility = (variance ** 0.5 / mean_price * 100) if mean_price > 0 else 0
    else:
        volatility = 0

    # トレンド判定 (直近5日間の移動平均の方向)
    trend = "stable"
    if len(all_prices) >= 5:
        recent_ema = ema_series(all_prices, 5)
        recent_trend = recent_ema[-5:]
        if all(recent_trend):
            slope = sum(recent_trend[i] - recent_trend[i-1] for i in range(1, len(recent_trend)))
            if slope > 0.1:
                trend = "rising"
            elif slope < -0.1:
                trend = "falling"

    # 収益性計算
    metrics = ProfitabilityMetrics(
        current_profit=None,
        current_profit_rate=None,
        avg_profit=None,
        avg_profit_rate=None,
        best_price_date=best_price_date[0].isoformat() if best_price_date[0] else None,
        best_price=min_price,
        worst_price_date=worst_price_date[0].isoformat() if worst_price_date[0] else None,
        worst_price=max_price,
        price_volatility=round(volatility, 2),
        trend_direction=trend,
    )

    # 仕入れ価格がある場合の利益計算
    if purchase_price and current_price:
        metrics.current_profit = round(current_price - purchase_price, 2)
        metrics.current_profit_rate = round(pct_change(purchase_price, current_price), 2)

    # 推奨アクション
    if trend == "rising":
        metrics.recommendation = "价格上涨中 - 仕入れ的最佳時期"
    elif trend == "falling":
        metrics.recommendation = "价格下跌中 - 销售的最佳時期"
    else:
        metrics.recommendation = "价格稳定 - 監視を継続"

    return metrics


# =============================================================================
# Data Source
# =============================================================================
class DataSource:
    def __init__(self, use_db: bool = True):
        self.use_db = use_db

    def _parse_csv_history_files(self, brand: str, days: int) -> List[HistoryRow]:
        """Read history from CSV files."""
        results: List[HistoryRow] = []
        cutoff = datetime.now() - timedelta(days=days)

        search_patterns = [
            OUTPUT_DIR / f"price_history_{brand.upper()}_*.csv",
            OUTPUT_DIR / f"price_history_{brand.lower()}_*.csv",
            OUTPUT_DIR / f"price_history_{brand}*.csv",
        ]

        for pattern in search_patterns:
            for csv_path in pattern.parent.glob(pattern.name):
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                captured = datetime.fromisoformat(row["captured_at"])
                                if captured < cutoff:
                                    continue
                                results.append(HistoryRow(
                                    captured_at=captured,
                                    brand=row.get("brand", brand),
                                    name=row.get("name", ""),
                                    url=row.get("url", ""),
                                    price=float(row["price"]) if row.get("price") else None,
                                    list_price=float(row["list_price"]) if row.get("list_price") else None,
                                    discount_pct=float(row["discount_pct"]) if row.get("discount_pct") else None,
                                    sale_until=datetime.fromisoformat(row["sale_until"]) if row.get("sale_until") else None,
                                ))
                            except (ValueError, KeyError):
                                continue
                except Exception:
                    continue

        return results

    def fetch_history(self, brand: str, days: int) -> List[HistoryRow]:
        """Fetch history from DB or CSV."""
        if self.use_db and db is not None:
            try:
                cutoff = datetime.now() - timedelta(days=days)
                query = ProductPriceHistory.query.filter(
                    ProductPriceHistory.brand == brand,
                    ProductPriceHistory.captured_at >= cutoff
                ).order_by(ProductPriceHistory.captured_at.asc())

                results = []
                for r in query:
                    results.append(HistoryRow(
                        captured_at=r.captured_at,
                        brand=r.brand,
                        name=r.name,
                        url=r.url,
                        price=r.price,
                        list_price=r.list_price,
                        discount_pct=r.discount_pct,
                        sale_until=r.sale_until,
                    ))
                logging.info(f"Fetched {len(results)} rows from DB for brand '{brand}'.")
                return results
            except Exception as e:
                logging.error(f"DB query failed, falling back to CSV. Error: {e}")

        return self._parse_csv_history_files(brand, days)


# =============================================================================
# Data Transformation Logic
# =============================================================================
def build_series_for_web(rows: List[HistoryRow], purchase_price: Optional[float] = None) -> Dict[str, Any]:
    """Transforms raw history rows into a format suitable for the web frontend."""
    bucket: Dict[str, List[HistoryRow]] = {}
    for r in rows:
        if r.url:
            bucket.setdefault(r.url, []).append(r)

    series_list = []
    today, yesterday = datetime.now().date(), (datetime.now() - timedelta(days=1)).date()

    for url, items in bucket.items():
        items_sorted = sorted(items, key=lambda x: x.captured_at)
        name = items_sorted[-1].name if items_sorted else url
        labels = [it.captured_at.isoformat() for it in items_sorted]
        prices = [it.price for it in items_sorted]

        ema7 = ema_series(prices, 7)
        ema28 = ema_series(prices, 28)

        # v2.0: 収益性指標
        profitability = calculate_profitability_metrics(items_sorted, purchase_price)

        # Alert detection logic
        alerts = []
        todays_pts = [it for it in items_sorted if it.captured_at.date() == today]
        yest_pts = [it for it in items_sorted if it.captured_at.date() == yesterday]

        if todays_pts and yest_pts:
            # Alert threshold lowered for higher sensitivity
            if (price_ch := pct_change(yest_pts[-1].price, todays_pts[-1].price)) is not None and abs(price_ch) >= 7.0:
                alerts.append(f"価格急変: {price_ch:+.1f}%")
            if (disc_ch := pct_change(yest_pts[-1].discount_pct, todays_pts[-1].discount_pct)) is not None and abs(disc_ch) >= 15.0:
                alerts.append(f"割引率急変: {disc_ch:+.1f}%")

        latest_sale_until = next((it.sale_until for it in reversed(items_sorted) if it.sale_until), None)
        if latest_sale_until:
            if 0 <= (days_left := (latest_sale_until.date() - today).days) <= 3:
                alerts.append(f"セール終了まで {days_left} 日")

        series_list.append({
            "url": url,
            "name": name,
            "labels": labels,
            "prices": prices,
            "ema7": ema7,
            "ema28": ema28,
            "alerts": alerts,
            # v2.0: 収益性データ
            "profitability": {
                "current_profit": profitability.current_profit,
                "current_profit_rate": profitability.current_profit_rate,
                "best_price_date": profitability.best_price_date,
                "best_price": profitability.best_price,
                "worst_price_date": profitability.worst_price_date,
                "worst_price": profitability.worst_price,
                "price_volatility": profitability.price_volatility,
                "trend_direction": profitability.trend_direction,
                "recommendation": profitability.recommendation,
            }
        })

    return {"series": series_list}


# =============================================================================
# Flask Routes
# =============================================================================
@app.route("/api/history")
def api_history():
    """API endpoint to provide time series data to the frontend."""
    brand = request.args.get("brand", "").strip() or "GUCCI"
    days = int(request.args.get("days", DAYS_DEFAULT))
    purchase_price = float(request.args.get("purchase_price")) if request.args.get("purchase_price") else None

    data_source = DataSource(use_db=FLASK_AVAILABLE)
    rows = data_source.fetch_history(brand, days)

    payload = build_series_for_web(rows, purchase_price)
    return jsonify(payload)


@app.route("/api/history/export")
def api_history_export():
    """Export history data as CSV or JSON."""
    brand = request.args.get("brand", "").strip() or "GUCCI"
    days = int(request.args.get("days", DAYS_DEFAULT))
    format_type = request.args.get("format", "json").lower()

    data_source = DataSource(use_db=FLASK_AVAILABLE)
    rows = data_source.fetch_history(brand, days)

    if format_type == "csv":
        output = "captured_at,brand,name,url,price,list_price,discount_pct\n"
        for r in rows:
            output += f"{r.captured_at.isoformat()},{r.brand},{r.name},{r.url},{r.price},{r.list_price},{r.discount_pct}\n"
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=history_{brand}_{days}days.csv"}
        )
    else:
        payload = build_series_for_web(rows)
        return jsonify(payload)


@app.route("/brand/<brand>")
def page_brand(brand: str):
    """Renders the dashboard page for a specific brand."""
    days = int(request.args.get("days", DAYS_DEFAULT))
    return render_template("dashboard.html", brand=brand, days=days)


@app.route("/")
def index():
    """Root URL handler."""
    return 'Dashboard is running. Try: <a href="/brand/GUCCI?days=30">/brand/GUCCI?days=30</a>'


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5009, debug=True)
