# ======================================================================
# F12: 買付先地域最適化レコメンド
# ======================================================================
from __future__ import annotations

from datetime import datetime

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.region_recommendation import RegionRecommendation
from app.utils.decorators import handle_db_error

from . import bp


@bp.get("/regions")
@login_required
def region_list():
    """買付先地域レコメンド一覧"""
    regions = RegionRecommendation.query.order_by(RegionRecommendation.recommendation_score.desc()).all()
    return render_template("region_recommendations.html", regions=regions)


@bp.route("/regions/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_region():
    """地域登録"""
    if request.method == "POST":
        avg_profit_rate = float(request.form.get("avg_profit_rate", 0) or 0)
        avg_shipping_days = int(request.form.get("avg_shipping_days", 0) or 0)
        risk_score = float(request.form.get("risk_score", 50) or 50)
        reliability_score = float(request.form.get("reliability_score", 50) or 50)
        if avg_shipping_days < 0:
            flash("配送日数に負の値は入力できません。", "error")
            return render_template("region_form.html")
        rr = RegionRecommendation(
            region=request.form.get("region", ""),
            region_name=request.form.get("region_name", ""),
            avg_profit_rate=avg_profit_rate,
            avg_shipping_days=avg_shipping_days,
            avg_customs_rate=float(request.form.get("avg_customs_rate", 0) or 0),
            risk_score=risk_score,
            reliability_score=reliability_score,
            last_updated=datetime.utcnow(),
        )
        rr.recommendation_score = rr.calc_recommendation() or 0
        db.session.add(rr)
        db.session.commit()
        flash("地域を登録しました。", "success")
        return redirect(url_for("main.region_list"))
    return render_template("region_form.html")


@bp.post("/regions/<int:rid>/delete")
@login_required
@handle_db_error("main.region_list")
def delete_region(rid: int):
    """地域削除"""
    rr = RegionRecommendation.query.get_or_404(rid)
    db.session.delete(rr)
    db.session.commit()
    flash("地域を削除しました。", "success")
    return redirect(url_for("main.region_list"))
