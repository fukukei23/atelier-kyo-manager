# ======================================================================
# F07: 品出し進捗トラッカー
# ======================================================================
from __future__ import annotations

from datetime import date as _date

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.listing_progress import ListingProgress
from app.utils.decorators import handle_db_error

from . import bp


@bp.get("/listing-progress")
@login_required
def listing_progress_view():
    """品出し進捗一覧"""
    today = _date.today()
    records = ListingProgress.query.filter(
        ListingProgress.record_date >= today.replace(day=1)
    ).order_by(ListingProgress.record_date.desc()).all()
    summary = ListingProgress.get_monthly_summary(today.year, today.month)
    return render_template("listing_progress.html", records=records, summary=summary)


@bp.route("/listing-progress/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_listing_progress():
    """品出し進捗登録"""
    if request.method == "POST":
        record_date_str = request.form.get("record_date", "")
        record_date = _date.fromisoformat(record_date_str) if record_date_str else _date.today()
        listings_count = int(request.form.get("listings_count", 0) or 0)
        target_daily = int(request.form.get("target_daily", 20) or 20)
        target_monthly = int(request.form.get("target_monthly", 600) or 600)
        cumulative_monthly = int(request.form.get("cumulative_monthly", 0) or 0)
        if any(v < 0 for v in [listings_count, target_daily, target_monthly, cumulative_monthly]):
            flash("出品数・目標値に負の値は入力できません。", "error")
            return render_template("listing_progress_form.html", record=None)
        lp = ListingProgress(
            record_date=record_date,
            listings_count=listings_count,
            target_daily=target_daily,
            target_monthly=target_monthly,
            cumulative_monthly=cumulative_monthly,
            notes=request.form.get("notes", ""),
        )
        db.session.add(lp)
        db.session.commit()
        flash("進捗を登録しました。", "success")
        return redirect(url_for("main.listing_progress_view"))
    return render_template("listing_progress_form.html", record=None)


@bp.post("/listing-progress/<int:rid>/delete")
@login_required
@handle_db_error("main.listing_progress_view")
def delete_listing_progress(rid: int):
    """進捗記録削除"""
    r = ListingProgress.query.get_or_404(rid)
    db.session.delete(r)
    db.session.commit()
    flash("進捗記録を削除しました。", "success")
    return redirect(url_for("main.listing_progress_view"))
