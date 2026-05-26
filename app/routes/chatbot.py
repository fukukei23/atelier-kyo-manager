# ======================================================================
# FR-010: 顧客対応AI ChatBot
# ======================================================================
from __future__ import annotations

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.customer_inquiry import CustomerInquiry
from app.services.chatbot_service import ChatBotService
from app.utils.decorators import handle_db_error

from . import bp


@bp.get("/chatbot")
@login_required
def chatbot_dashboard():
    """ChatBot ダッシュボード: 問い合わせ一覧"""
    status_filter = request.args.get("status", "all")
    query = CustomerInquiry.query
    if status_filter != "all":
        query = query.filter_by(status=status_filter)
    inquiries = query.order_by(CustomerInquiry.created_at.desc()).paginate(page=request.args.get("page", 1, type=int), per_page=50, error_out=False).items
    return render_template(
        "chatbot_dashboard.html",
        inquiries=inquiries,
        status_filter=status_filter,
    )


@bp.get("/chatbot/<int:inquiry_id>")
@login_required
def chatbot_detail(inquiry_id):
    """問い合わせ詳細"""
    inquiry = CustomerInquiry.query.get_or_404(inquiry_id)
    return render_template("chatbot_detail.html", inquiry=inquiry)


@bp.route("/chatbot/create", methods=["GET", "POST"])
@login_required
@handle_db_error()
def chatbot_create():
    """新規問い合わせ作成"""
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()
        customer_name = request.form.get("customer_name", "").strip()
        customer_email = request.form.get("customer_email", "").strip()
        if not subject or not message:
            flash("件名と本文は必須です。", "error")
            return render_template("chatbot_create.html")
        inquiry = ChatBotService.process_inquiry(
            subject=subject,
            message=message,
            customer_name=customer_name,
            customer_email=customer_email,
        )
        flash("問い合わせを処理しました。", "success")
        return redirect(url_for("main.chatbot_detail", inquiry_id=inquiry.id))
    return render_template("chatbot_create.html")


@bp.post("/chatbot/<int:inquiry_id>/approve")
@login_required
def chatbot_approve(inquiry_id):
    """回答を承認"""
    inquiry = CustomerInquiry.query.get_or_404(inquiry_id)
    if inquiry.can_approve():
        inquiry.mark_approved()
        db.session.commit()
        flash("回答を承認しました。", "success")
    else:
        flash("この問い合わせは承認できません。", "error")
    return redirect(url_for("main.chatbot_dashboard"))


@bp.post("/chatbot/<int:inquiry_id>/reject")
@login_required
def chatbot_reject(inquiry_id):
    """回答を却下"""
    inquiry = CustomerInquiry.query.get_or_404(inquiry_id)
    reason = request.form.get("reason", "")
    inquiry.mark_rejected(reason=reason)
    db.session.commit()
    flash("回答を却下しました。", "success")
    return redirect(url_for("main.chatbot_dashboard"))
