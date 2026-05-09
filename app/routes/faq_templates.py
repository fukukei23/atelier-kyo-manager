# ======================================================================
# FR-010基盤: FAQテンプレート管理
# ======================================================================
from __future__ import annotations

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.faq_template import FaqTemplate
from app.utils.decorators import handle_db_error

from . import bp


@bp.get("/faq-templates")
@login_required
def faq_templates():
    """FAQテンプレート一覧"""
    faqs = FaqTemplate.query.order_by(FaqTemplate.category, FaqTemplate.id.asc()).all()
    return render_template("faq_templates.html", faqs=faqs)


@bp.route("/faq-templates/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_faq_template():
    """FAQテンプレート新規作成"""
    if request.method == "POST":
        category = request.form.get("category", "general")
        question_pattern = request.form.get("question_pattern", "").strip()
        answer_template = request.form.get("answer_template", "").strip()
        if not question_pattern or not answer_template:
            flash("キーワードパターンと返答テンプレートは必須です。", "error")
            return render_template("faq_template_form.html", faq=None)
        faq = FaqTemplate(
            category=category,
            question_pattern=question_pattern,
            answer_template=answer_template,
        )
        db.session.add(faq)
        db.session.commit()
        flash("FAQテンプレートを登録しました。", "success")
        return redirect(url_for("main.faq_templates"))
    return render_template("faq_template_form.html", faq=None)


@bp.post("/faq-templates/<int:fid>/delete")
@login_required
@handle_db_error("main.faq_templates")
def delete_faq_template(fid: int):
    """FAQテンプレート削除"""
    faq = FaqTemplate.query.get_or_404(fid)
    db.session.delete(faq)
    db.session.commit()
    flash("FAQテンプレートを削除しました。", "success")
    return redirect(url_for("main.faq_templates"))


@bp.post("/api/faq-match")
@login_required
def api_faq_match():
    """問い合わせ内容からFAQテンプレートを検索"""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"success": False, "error": "text required"}), 400
    faqs = FaqTemplate.query.filter_by(is_active=True).all()
    matches = []
    for faq in faqs:
        if faq.match(text):
            matches.append(
                {
                    "id": faq.id,
                    "category": faq.category,
                    "answer_template": faq.answer_template,
                }
            )
    return jsonify({"success": True, "matches": matches})
