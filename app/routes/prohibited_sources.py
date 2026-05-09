# ======================================================================
# F04: 禁制品買付先チェックAPI
# ======================================================================
from __future__ import annotations

from flask import jsonify, request
from flask_login import login_required

from app.extensions import db
from app.models.prohibited_source import ProhibitedSource

from . import bp


@bp.get("/api/check-source")
@login_required
def api_check_source():
    """買付先URLが禁止対象かチェック"""
    url = (request.args.get("url") or "").strip()
    source_type = (request.args.get("source_type") or "domestic").strip()
    prohibited, reason = ProhibitedSource.is_prohibited(url, source_type)
    return jsonify({"prohibited": prohibited, "reason": reason, "url": url})


@bp.get("/api/prohibited-sources")
@login_required
def api_list_prohibited_sources():
    items = ProhibitedSource.query.order_by(ProhibitedSource.id.asc()).all()
    return jsonify([{"id": s.id, "domain": s.domain, "reason": s.reason,
                     "severity": s.severity, "source_type": s.source_type} for s in items])


@bp.post("/api/prohibited-sources")
@login_required
def api_add_prohibited_source():
    data = request.get_json(force=True)
    domain = (data.get("domain") or "").strip()
    if not domain:
        return jsonify({"error": "domain is required"}), 400
    existing = ProhibitedSource.query.filter_by(domain=domain).first()
    if existing:
        return jsonify({"error": "already exists"}), 409
    src = ProhibitedSource(
        domain=domain,
        reason=data.get("reason", ""),
        severity=data.get("severity", "blocked"),
        source_type=data.get("source_type", "domestic"),
    )
    db.session.add(src)
    db.session.commit()
    return jsonify({"id": src.id, "domain": src.domain}), 201


@bp.delete("/api/prohibited-sources/<int:sid>")
@login_required
def api_delete_prohibited_source(sid: int):
    src = ProhibitedSource.query.get_or_404(sid)
    db.session.delete(src)
    db.session.commit()
    return jsonify({"deleted": True})
