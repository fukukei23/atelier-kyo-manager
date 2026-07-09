# ======================================================================
# 価格監視 — SSENSE等のセール品価格追跡
# ======================================================================
from __future__ import annotations

from flask import abort, flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.price_monitor import PriceMonitor
from app.services import price_monitor_service

from . import bp


def _get_owned_monitor(mid: int) -> PriceMonitor:
    """監視レコードを取得。単一ユーザー運用のため存在確認のみ（将来は所有者チェック追加）。"""
    monitor = db.session.get(PriceMonitor, mid)
    if not monitor:
        abort(404)
    # TODO: 多ユーザー化時に所有者チェック追加
    # if monitor.created_by != current_user.id:
    #     abort(403)
    return monitor


@bp.get("/price-monitor")
@login_required
def price_monitor_dashboard():
    """価格監視ダッシュボード"""
    monitors = price_monitor_service.get_dashboard_data()
    active_count = sum(1 for m in monitors if m["is_active"])
    profitable_count = sum(1 for m in monitors if m["is_active"] and m["is_profitable"])
    unprofitable_count = sum(1 for m in monitors if m["is_active"] and not m["is_profitable"])

    # 追加用候補取得
    candidates = price_monitor_service.get_ssense_candidates()

    return render_template(
        "price_monitor.html",
        monitors=monitors,
        active_count=active_count,
        profitable_count=profitable_count,
        unprofitable_count=unprofitable_count,
        candidates=candidates,
    )


@bp.post("/price-monitor/add")
@login_required
def price_monitor_add():
    """監視対象を追加"""
    source_url = request.form.get("source_url", "").strip()
    brand = request.form.get("brand", "").strip()
    product_name = request.form.get("product_name", "").strip()
    buyma_price_str = request.form.get("buyma_selling_price", "0")
    category = request.form.get("category", "accessory")
    currency = request.form.get("currency", "USD")
    brand_price_id = request.form.get("brand_price_id", type=int)

    if not source_url or not brand:
        flash("URLとブランドは必須です。", "error")
        return redirect(url_for("main.price_monitor_dashboard"))

    try:
        buyma_selling_price = float(buyma_price_str)
    except (ValueError, TypeError):
        flash("BUYMA販売価格は数値で入力してください。", "error")
        return redirect(url_for("main.price_monitor_dashboard"))

    try:
        price_monitor_service.add_monitor(
            source_url=source_url,
            brand=brand,
            product_name=product_name,
            buyma_selling_price=buyma_selling_price,
            category=category,
            currency=currency,
            brand_price_id=brand_price_id,
        )
        flash(f"{brand} {product_name} を監視に追加しました。", "success")
    except Exception as e:
        flash(f"追加に失敗しました: {e}", "error")

    return redirect(url_for("main.price_monitor_dashboard"))


@bp.post("/price-monitor/<int:mid>/remove")
@login_required
def price_monitor_remove(mid: int):
    """監視対象を削除"""
    _get_owned_monitor(mid)  # 存在確認（404）
    if price_monitor_service.remove_monitor(mid):
        flash("監視を削除しました。", "success")
    else:
        flash("監視が見つかりません。", "error")
    return redirect(url_for("main.price_monitor_dashboard"))


@bp.post("/price-monitor/<int:mid>/toggle")
@login_required
def price_monitor_toggle(mid: int):
    """監視の一時停止/再開"""
    _get_owned_monitor(mid)  # 存在確認（404）
    active = request.form.get("active", "0") == "1"
    monitor = price_monitor_service.toggle_monitor(mid, active)
    if monitor:
        status = "再開" if active else "停止"
        flash(f"監視を{status}しました。", "success")
    else:
        flash("監視が見つかりません。", "error")
    return redirect(url_for("main.price_monitor_dashboard"))


@bp.post("/price-monitor/<int:mid>/refresh")
@login_required
def price_monitor_refresh(mid: int):
    """単品の価格チェック（手動）"""
    monitor = _get_owned_monitor(mid)

    result = price_monitor_service.check_single_price(monitor)
    if result.get("error"):
        flash(f"チェック失敗: {result['error']}", "error")
    else:
        price_str = f"${result.get('new_price', '?')}"
        profitable = "利益あり" if result.get("is_profitable") else "赤字"
        flash(f"更新完了: {price_str} ({profitable})", "success")

    return redirect(url_for("main.price_monitor_dashboard"))


@bp.post("/price-monitor/refresh-all")
@login_required
def price_monitor_refresh_all():
    """全アクティブ監視の価格チェック（手動）"""
    summary = price_monitor_service.run_all_monitors()
    flash(
        f"全件チェック完了: {summary['checked']}件確認、{summary['alerts']}件アラート、{summary['errors']}件エラー",
        "success",
    )
    return redirect(url_for("main.price_monitor_dashboard"))


@bp.post("/price-monitor/<int:mid>/update-selling-price")
@login_required
def price_monitor_update_selling_price(mid: int):
    """BUYMA販売価格を更新"""
    _get_owned_monitor(mid)  # 存在確認（404）
    price_str = request.form.get("buyma_selling_price", "0")
    try:
        price = float(price_str)
    except (ValueError, TypeError):
        flash("価格は数値で入力してください。", "error")
        return redirect(url_for("main.price_monitor_dashboard"))

    monitor = price_monitor_service.update_selling_price(mid, price)
    if monitor:
        flash(f"販売価格を¥{price:,.0f}に更新しました。", "success")
    else:
        flash("監視が見つかりません。", "error")
    return redirect(url_for("main.price_monitor_dashboard"))


@bp.get("/api/price-monitor/<int:mid>/history")
@login_required
def api_price_history(mid: int):
    """価格履歴JSON API（チャート用）"""
    _get_owned_monitor(mid)  # 存在確認（404）
    history = price_monitor_service.get_price_history(mid)
    return jsonify(history)
