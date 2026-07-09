"""互換リダイレクト + 自動リサーチ + API倉庫"""

from __future__ import annotations

import logging
from typing import Any

from flask import redirect, render_template, request, url_for
from flask_login import login_required

from app.forms import AutoResearchForm
from app.utils.errors import safe_error_msg

from . import bp

logger: logging.Logger = logging.getLogger(__name__)

# ── Redirects（旧URL → 新エンドポイント）──────────────────

_REDIRECTS: dict[str, str] = {
    "/dashboard": "main.cashflow_dashboard",
    "/listing-templates": "main.listing_templates",
    "/region-recommendations": "main.region_list",
    "/repeat-customers": "main.customer_list",
    "/auto_research": "main.auto_research",
}


def _make_redirect(endpoint: str):
    def _view() -> str:
        return redirect(url_for(endpoint))

    _view.__name__ = f"redirect_to_{endpoint.replace('.', '_')}"
    return _view


for _path, _endpoint in _REDIRECTS.items():
    bp.get(_path)(_make_redirect(_endpoint))


# ── Research pages ─────────────────────────────────────


@bp.route("/auto-research", methods=["GET", "POST"])
@login_required
def auto_research() -> str:
    form = AutoResearchForm()
    return render_template("auto_research.html", form=form)


@bp.get("/image-crawler")
@login_required
def image_crawler() -> str:
    return render_template("image_crawler.html")


# ── API: 倉庫一覧 ──────────────────────────────────────

try:
    from app.utils.shipping_agent import ShippingAgent

    _shipping_ok = True
except ImportError:
    ShippingAgent = None  # type: ignore[assignment]
    _shipping_ok = False


def _error_json(msg: str, code: int) -> tuple[dict[str, str], int]:
    return {"error": msg}, code


@bp.get("/api/warehouses")
@login_required
def api_warehouses() -> tuple[dict[str, Any], int]:
    country = (request.args.get("country") or "").strip().upper()
    if not country:
        return _error_json("country is required (e.g. HK, TW)", 400)
    if not _shipping_ok:
        return _error_json("ShippingAgent unavailable", 503)
    try:
        warehouses: list[dict[str, Any]] = ShippingAgent().get_warehouses_by_country(country)
        return {"country": country, "warehouses": warehouses}, 200
    except Exception as e:
        logger.error("[api/warehouses] failed: %s", e, exc_info=True)
        return _error_json(safe_error_msg(e, context="api/warehouses"), 500)
