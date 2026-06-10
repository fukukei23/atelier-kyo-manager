"""SSENSE等のセール品価格監視サービス。

PriceMonitorモデルの商品URLを定期チェックし、
価格変動を検出 → 利益再計算 → アラート通知を行う。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.config.cost_table import get_buyandship_shipping
from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.schemas import PricingInput
from app.core.timezone import _utcnow
from app.extensions import db
from app.models.price_monitor import PriceMonitor
from app.models.source_price_history import SourcePriceHistory
from app.services.price_scraper import PriceScraper
from app.utils.fx_utils import get_fx_table_jpy

logger = logging.getLogger(__name__)

# SSENSE国際送料（USD→JPY換算で約¥4,000）
SSENSE_SHIPPING_USD = 25.0
# アラート再送間隔（時間）
ALERT_INTERVAL_HOURS = 6
# 許可する仕入れ先ドメイン（SSRF対策）
ALLOWED_DOMAINS = {"ssense.com", "www.ssense.com", "yoox.com", "www.yoox.com", "farfetch.com", "www.farfetch.com"}


def _validate_source_url(url: str) -> str:
    """仕入れ先URLのバリデーション（SSRF対策: 許可ドメインのみ）。"""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = (parsed.hostname or "").lower()
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"許可されていないドメインです: {domain}")
    return url


def _safe_log(value: str) -> str:
    """ログ出力用に制御文字を除去。"""
    return value.replace("\n", "").replace("\r", "").replace("\t", " ")[:200]


def add_monitor(
    source_url: str,
    brand: str,
    product_name: str,
    buyma_selling_price: float,
    category: str = "accessory",
    currency: str = "USD",
    brand_price_id: int | None = None,
    current_source_price: float | None = None,
) -> PriceMonitor:
    """監視対象を追加。"""
    # SSRF対策: 許可ドメインのみ
    _validate_source_url(source_url)

    fx_table, _ = get_fx_table_jpy()
    rate = fx_table.get(currency, 1.0)
    price_jpy = (current_source_price or 0) * rate if current_source_price else 0

    monitor = PriceMonitor(
        brand_price_id=brand_price_id,
        source_url=source_url,
        brand=brand,
        product_name=product_name,
        category=category,
        currency=currency,
        current_source_price=current_source_price,
        current_source_price_jpy=round(price_jpy) if price_jpy else None,
        buyma_selling_price=buyma_selling_price,
    )
    db.session.add(monitor)
    db.session.commit()

    # 初回チェック実行
    if current_source_price and price_jpy:
        _update_profitability(monitor, price_jpy, rate)

    logger.info(f"[Monitor] Added: {brand} {product_name}")
    return monitor


def remove_monitor(monitor_id: int) -> bool:
    """監視対象を削除。"""
    monitor = db.session.get(PriceMonitor, monitor_id)
    if not monitor:
        return False
    # 履歴も削除
    db.session.query(SourcePriceHistory).filter_by(monitor_id=monitor_id).delete()
    db.session.delete(monitor)
    db.session.commit()
    logger.info(f"[Monitor] Removed: {monitor_id}")
    return True


def toggle_monitor(monitor_id: int, active: bool) -> PriceMonitor | None:
    """監視の一時停止/再開。"""
    monitor = db.session.get(PriceMonitor, monitor_id)
    if not monitor:
        return None
    monitor.is_active = active
    db.session.commit()
    return monitor


def update_selling_price(monitor_id: int, price: float) -> PriceMonitor | None:
    """BUYMA販売価格を更新。"""
    monitor = db.session.get(PriceMonitor, monitor_id)
    if not monitor:
        return None
    monitor.buyma_selling_price = price
    # 利益再計算
    if monitor.current_source_price_jpy:
        _update_profitability(monitor, monitor.current_source_price_jpy, 1.0)
    db.session.commit()
    return monitor


def check_single_price(monitor: PriceMonitor) -> dict:
    """単一商品の価格チェック。"""
    result: dict = {
        "monitor_id": monitor.id,
        "brand": monitor.brand,
        "product_name": monitor.product_name,
        "price_changed": False,
        "is_profitable": None,
        "error": None,
    }

    try:
        # PriceScraperでURLから価格取得
        scraper = PriceScraper()
        fetch_result = scraper.fetch(monitor.source_url)

        if not fetch_result.get("success"):
            result["error"] = fetch_result.get("error", "fetch failed")
            logger.warning("[Monitor] Fetch failed for %s: %s", _safe_log(monitor.source_url), result["error"])
            return result

        raw_price = fetch_result.get("raw_price") or fetch_result.get("price")
        if raw_price is None:
            result["error"] = "price not found"
            return result

        # 価格を数値に変換
        price = _parse_price(raw_price)
        if price is None or price <= 0:
            result["error"] = f"invalid price: {raw_price}"
            return result

        in_stock = fetch_result.get("in_stock", True)

        # 為替換算
        fx_table, _ = get_fx_table_jpy()
        rate = fx_table.get(monitor.currency, 1.0)
        price_jpy = price * rate

        # 価格変動検出
        old_price = monitor.current_source_price
        price_changed = old_price is not None and abs(price - old_price) > 0.01

        # 利益計算
        was_profitable = monitor.is_profitable
        _update_profitability(monitor, price_jpy, rate)

        # モニター更新
        monitor.current_source_price = price
        monitor.current_source_price_jpy = round(price_jpy)
        monitor.last_checked_at = _utcnow()
        if price_changed:
            monitor.last_price_changed_at = _utcnow()

        # 履歴保存
        history = SourcePriceHistory(
            monitor_id=monitor.id,
            source_price=price,
            source_price_jpy=round(price_jpy),
            exchange_rate=rate,
            in_stock=in_stock,
            is_profitable=monitor.is_profitable,
            profit=monitor.latest_profit,
            profit_rate=monitor.latest_profit_rate,
            price_changed=price_changed,
        )
        db.session.add(history)
        db.session.commit()

        result.update({
            "price_changed": price_changed,
            "old_price": old_price,
            "new_price": price,
            "new_price_jpy": round(price_jpy),
            "is_profitable": monitor.is_profitable,
            "profit": monitor.latest_profit,
            "profit_rate": monitor.latest_profit_rate,
        })

        # 利益なしアラート（利益あり→なしに変化した時）
        if was_profitable and not monitor.is_profitable:
            _send_unprofitable_alert(monitor, old_price, price)

        logger.info(
            f"[Monitor] {monitor.brand} {monitor.product_name}: "
            f"${price} → ¥{round(price_jpy)} "
            f"({'利益あり' if monitor.is_profitable else '赤字'})"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error("[Monitor] Error checking %s: %s", monitor.id, e)

    return result


def run_all_monitors() -> dict:
    """全アクティブ監視対象の価格チェック。"""
    now = _utcnow()
    monitors = db.session.query(PriceMonitor).filter_by(is_active=True).all()

    checked = 0
    alerts = 0
    errors = 0

    for monitor in monitors:
        # チェック間隔の判定
        if monitor.last_checked_at:
            elapsed = (now - monitor.last_checked_at).total_seconds() / 3600
            if elapsed < monitor.check_interval_hours:
                continue

        result = check_single_price(monitor)
        if result.get("error"):
            errors += 1
        else:
            checked += 1
            if not result.get("is_profitable", True):
                alerts += 1

    summary = {"checked": checked, "alerts": alerts, "errors": errors, "total": len(monitors)}
    logger.info(f"[Monitor] Batch: {summary}")
    return summary


def get_dashboard_data() -> list[dict]:
    """ダッシュボード用データ。"""
    monitors = db.session.query(PriceMonitor).order_by(PriceMonitor.is_active.desc(), PriceMonitor.brand).all()
    data = []
    for m in monitors:
        status = "paused"
        if m.is_active:
            status = "profitable" if m.is_profitable else "unprofitable"
        data.append({
            "id": m.id,
            "brand": m.brand,
            "product_name": m.product_name,
            "source_url": m.source_url,
            "category": m.category,
            "current_source_price": m.current_source_price,
            "current_source_price_jpy": m.current_source_price_jpy,
            "currency": m.currency,
            "buyma_selling_price": m.buyma_selling_price,
            "is_profitable": m.is_profitable,
            "latest_profit": m.latest_profit,
            "latest_profit_rate": m.latest_profit_rate,
            "is_active": m.is_active,
            "status": status,
            "last_checked_at": m.last_checked_at,
            "last_price_changed_at": m.last_price_changed_at,
            "check_interval_hours": m.check_interval_hours,
        })
    return data


def get_price_history(monitor_id: int) -> list[dict]:
    """価格履歴データ（JSON API用）。"""
    rows = (
        db.session.query(SourcePriceHistory)
        .filter_by(monitor_id=monitor_id)
        .order_by(SourcePriceHistory.recorded_at)
        .all()
    )
    return [
        {
            "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
            "source_price": r.source_price,
            "source_price_jpy": r.source_price_jpy,
            "is_profitable": r.is_profitable,
            "profit": r.profit,
            "profit_rate": r.profit_rate,
            "price_changed": r.price_changed,
            "in_stock": r.in_stock,
        }
        for r in rows
    ]


def get_ssense_candidates() -> list[dict]:
    """BrandPriceからSSENSE商品を候補として取得。"""
    from app.models.brand_price import BrandPrice

    rows = (
        db.session.query(BrandPrice)
        .filter(BrandPrice.source_site == "ssense")
        .order_by(BrandPrice.brand, BrandPrice.price_jpy)
        .all()
    )
    seen = set()
    candidates = []
    for r in rows:
        key = (r.brand, r.product_name)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "id": r.id,
            "brand": r.brand,
            "product_name": r.product_name,
            "source_url": r.source_url,
            "price_original": r.price_original,
            "currency": r.currency,
            "price_jpy": r.price_jpy,
            "actual_purchase_jpy": r.actual_purchase_jpy,
        })
    return candidates


# ---------------------------------------------------------------------------
# 内部関数
# ---------------------------------------------------------------------------


def _parse_price(raw_price) -> float | None:
    """価格文字列を数値に変換。"""
    if isinstance(raw_price, (int, float)):
        return float(raw_price)
    if isinstance(raw_price, str):
        # "$320.00" → 320.00
        cleaned = raw_price.replace("$", "").replace("€", "").replace("¥", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _update_profitability(monitor: PriceMonitor, source_price_jpy: float, exchange_rate: float) -> None:
    """利益計算してモニターを更新。"""
    # SSENSE送料をJPY換算で追加
    ssense_shipping_jpy = SSENSE_SHIPPING_USD * exchange_rate
    buyandship_shipping = get_buyandship_shipping(monitor.category)

    inp = PricingInput(
        purchase_price=source_price_jpy,
        selling_price=monitor.buyma_selling_price,
        shipping_cost=buyandship_shipping,
        warehouse_shipping_cost=ssense_shipping_jpy,
        original_currency="JPY",
        exchange_rate=1.0,
        item_category=monitor.category,
    )
    result = calculate_pricing(inp, source_type="overseas")

    monitor.is_profitable = result.profit > 0
    monitor.latest_profit = result.profit
    monitor.latest_profit_rate = result.profit_rate


def _send_unprofitable_alert(monitor: PriceMonitor, old_price: float | None, new_price: float) -> None:
    """Slackに利益なしアラートを送信（レートリミット付き）。"""
    # レートリミット: 6時間以内の再送を防止
    if monitor.alert_sent_at:
        elapsed = (_utcnow() - monitor.alert_sent_at).total_seconds() / 3600
        if elapsed < ALERT_INTERVAL_HOURS:
            return

    try:
        from flask import current_app

        notification = getattr(current_app, "notification", None)
        if not notification:
            return

        old_str = f"${old_price}" if old_price else "N/A"
        profit_str = f"¥{monitor.latest_profit:,.0f}" if monitor.latest_profit else "N/A"
        rate_str = f"{monitor.latest_profit_rate:.1%}" if monitor.latest_profit_rate else "N/A"

        message = (
            f":warning: [価格監視] 利益なし検出\n"
            f"商品: {monitor.brand} {monitor.product_name}\n"
            f"SSENSE価格: {old_str} → ${new_price}\n"
            f"BUYMA販売価格: ¥{monitor.buyma_selling_price:,.0f}\n"
            f"利益: {profit_str} ({rate_str})\n"
            f"URL: {monitor.source_url}"
        )
        notification.send(message)
        monitor.alert_sent_at = _utcnow()
        db.session.commit()
        logger.info(f"[Monitor] Alert sent: {monitor.brand} {monitor.product_name}")

    except Exception as e:
        logger.error(f"[Monitor] Alert failed: {e}")
