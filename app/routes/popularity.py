# ======================================================================
# F11: 人気度トラッキング
# ======================================================================
from __future__ import annotations

from datetime import date as _date

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from app.extensions import db
from app.models.popularity_tracker import PopularityTracker
from app.utils.decorators import handle_db_error

from . import bp


@bp.get("/popularity")
@login_required
def popularity_list():
    """人気度トラッキング一覧"""
    trackers = (
        PopularityTracker.query.options(joinedload(PopularityTracker.product))
        .order_by(PopularityTracker.popularity_score.desc())
        .all()
    )
    avg_score = db.session.query(func.avg(PopularityTracker.popularity_score)).scalar() or 0
    top_count = sum(1 for t in trackers if (t.popularity_score or 0) >= 100)
    low_count = sum(1 for t in trackers if (t.popularity_score or 0) < 20)
    summary = {
        "total_products": len(trackers),
        "avg_score": round(float(avg_score), 1),
        "top_count": top_count,
        "low_count": low_count,
    }
    return render_template("popularity.html", trackers=trackers, summary=summary)


@bp.route("/popularity/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_popularity():
    """人気度記録登録"""
    if request.method == "POST":
        views = int(request.form.get("views", 0) or 0)
        favorites = int(request.form.get("favorites", 0) or 0)
        inquiries = int(request.form.get("inquiries", 0) or 0)
        sold_count = int(request.form.get("sold_count", 0) or 0)
        if any(v < 0 for v in [views, favorites, inquiries, sold_count]):
            flash("閲覧数・お気に入り・問い合わせ・販売数に負の値は入力できません。", "error")
            return render_template("popularity_form.html")
        pt = PopularityTracker(
            product_id=int(request.form.get("product_id", 0)),
            views=views,
            favorites=favorites,
            inquiries=inquiries,
            sold_count=sold_count,
            tracking_date=_date.today(),
        )
        pt.popularity_score = pt.calc_score()
        db.session.add(pt)
        db.session.commit()
        flash("人気度を記録しました。", "success")
        return redirect(url_for("main.popularity_list"))
    return render_template("popularity_form.html")


@bp.post("/popularity/<int:tid>/delete")
@login_required
@handle_db_error("main.popularity_list")
def delete_popularity(tid: int):
    """人気度記録削除"""
    pt = PopularityTracker.query.get_or_404(tid)
    db.session.delete(pt)
    db.session.commit()
    flash("記録を削除しました。", "success")
    return redirect(url_for("main.popularity_list"))
