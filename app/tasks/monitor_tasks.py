"""価格監視 Celery タスク.

SSENSE等の仕入れ先価格を定期チェックし、
価格変動・利益率変化を検出する。
"""

from __future__ import annotations

import logging

from app.core.celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(bind=True, name="monitor_prices_periodic")
def monitor_prices_periodic(self) -> dict:
    """定期タスク: 全アクティブ監視対象の価格チェック。"""
    try:
        from app.services import price_monitor_service

        result = price_monitor_service.run_all_monitors()
        logger.info("monitor_prices_periodic: %s", result)
        return result

    except Exception as exc:
        logger.error("monitor_prices_periodic error: %s", exc, exc_info=True)
        return {"checked": 0, "alerts": 0, "errors": 1, "total": 0}


@celery.task(bind=True, name="monitor_prices_single")
def monitor_prices_single(self, monitor_id: int) -> dict:
    """手動トリガー: 単一監視対象の価格チェック。"""
    try:
        from app.extensions import db
        from app.models.price_monitor import PriceMonitor
        from app.services import price_monitor_service

        monitor = db.session.get(PriceMonitor, monitor_id)
        if not monitor:
            return {"error": "not_found", "monitor_id": monitor_id}

        result = price_monitor_service.check_single_price(monitor)
        return result

    except Exception as exc:
        logger.error("monitor_prices_single error: %s", exc, exc_info=True)
        return {"error": str(exc), "monitor_id": monitor_id}
