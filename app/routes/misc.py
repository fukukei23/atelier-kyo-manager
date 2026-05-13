"""互換リダイレクト + 自動リサーチ + API倉庫"""
from __future__ import annotations

import logging

from flask import jsonify, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import csrf
from app.forms import AutoResearchForm

from . import bp

logger = logging.getLogger(__name__)

# ── Redirects（旧URL → 新エンドポイント）──────────────────

_REDIRECTS = {
    "/dashboard": "main.cashflow_dashboard",
    "/listing-templates": "main.listing_templates",
    "/region-recommendations": "main.region_list",
    "/repeat-customers": "main.customer_list",
    "/auto_research": "main.auto_research",
}


def _make_redirect(endpoint: str):
    def _view():
        return redirect(url_for(endpoint))
    _view.__name__ = f"redirect_to_{endpoint.replace('.', '_')}"
    return _view


for _path, _endpoint in _REDIRECTS.items():
    bp.get(_path)(_make_redirect(_endpoint))


# ── Research pages ─────────────────────────────────────

@bp.route("/auto-research", methods=["GET", "POST"])
@login_required
def auto_research():
    form = AutoResearchForm()
    return render_template("auto_research.html", form=form)


@bp.get("/image-crawler")
@login_required
def image_crawler():
    return render_template("image_crawler.html")


# ── API: 倉庫一覧 ──────────────────────────────────────

try:
    from app.utils.shipping_agent import ShippingAgent
    _shipping_ok = True
except Exception:
    ShippingAgent = None  # type: ignore
    _shipping_ok = False


@bp.get("/api/warehouses")
@login_required
@csrf.exempt
def api_warehouses():
    country = (request.args.get("country") or "").strip().upper()
    if not country:
        return jsonify({"error": "country is required (e.g. HK, TW)"}), 400
    if not _shipping_ok:
        return jsonify({"error": "ShippingAgent unavailable"}), 503
    try:
        warehouses = ShippingAgent().get_warehouses_by_country(country)
        return jsonify({"country": country, "warehouses": warehouses})
    except Exception as e:
        logger.error(f"[api/warehouses] failed: {e}", exc_info=True)
        return jsonify({"error": f"failed: {e}"}), 500
