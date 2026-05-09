# ======================================================================
# 互換リダイレクト + 自動リサーチ + API倉庫
# ======================================================================
from __future__ import annotations

from flask import jsonify, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import csrf
from app.forms import AutoResearchForm

from . import bp


# ---- 自動リサーチ / 個別リサーチ ----------------------------------------
@bp.route("/auto-research", methods=["GET", "POST"])
@login_required
def auto_research():
    """自動リサーチ（フォーム未使用でも CSRF のため form を渡す）"""
    form = AutoResearchForm()
    return render_template("auto_research.html", form=form)


@bp.get("/image-crawler")
@login_required
def image_crawler():
    """個別リサーチ画面"""
    return render_template("image_crawler.html")


# ---- API: 倉庫一覧（Buyandship） ---------------------------------------
try:
    from app.utils.shipping_agent import ShippingAgent

    _shipping_agent_import_ok = True
except Exception:
    ShippingAgent = None  # type: ignore
    _shipping_agent_import_ok = False


@bp.get("/api/warehouses")
@login_required
@csrf.exempt
def api_warehouses():
    """
    GET /api/warehouses?country=HK
    - 成功: JSON (list of warehouses)
    - エラー:
        400: country 未指定
        503: ShippingAgent 未利用（Playwright 未導入など）
        500: その他例外
    """
    country = (request.args.get("country") or "").strip().upper()
    if not country:
        return jsonify({"error": "country is required (e.g. HK, TW)"}), 400

    if not _shipping_agent_import_ok or ShippingAgent is None:
        return jsonify({"error": "ShippingAgent is unavailable on this environment."}), 503

    try:
        agent = ShippingAgent()
        warehouses = agent.get_warehouses_by_country(country)
        return jsonify({"country": country, "warehouses": warehouses})
    except Exception as e:
        return jsonify({"error": f"failed to fetch warehouses: {e}"}), 500


# ---- 互換リダイレクト（ナビURL → 実際のルート）--------------------------
@bp.get("/dashboard")
def dashboard_redirect():
    """旧: /dashboard → /cashflow"""
    return redirect(url_for("main.cashflow_dashboard"))


@bp.get("/listing-templates")
def listing_templates_redirect():
    """旧: /listing-templates → /templates"""
    return redirect(url_for("main.listing_templates"))


@bp.get("/region-recommendations")
def region_recommendations_redirect():
    """旧: /region-recommendations → /regions"""
    return redirect(url_for("main.region_list"))


@bp.get("/repeat-customers")
def repeat_customers_redirect():
    """旧: /repeat-customers → /customers"""
    return redirect(url_for("main.customer_list"))


@bp.get("/auto_research")
def auto_research_redirect():
    """旧: /auto_research → /auto-research"""
    return redirect(url_for("main.auto_research"))
