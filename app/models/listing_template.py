from __future__ import annotations
from datetime import datetime

from app.extensions import db


class ListingTemplate(db.Model):
    __tablename__ = "listing_template"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    template_text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), default="general")
    is_default = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def render(self, product) -> str:
        mapping = {
            "brand": product.brand or "",
            "productName": product.name or "",
            "color": product.color or "",
            "material": product.material or "",
            "size": product.size or "",
            "retailPrice": f"¥{int(product.retail_price or 0):,}" if product.retail_price else "",
            "sourceCountry": product.source_region or "",
            "description": product.description or "",
            "sellingPrice": f"¥{int(product.selling_price or 0):,}" if product.selling_price else "",
        }
        text = self.template_text
        for key, value in mapping.items():
            text = text.replace("{{" + key + "}}", str(value))
        return text

    def __repr__(self) -> str:
        return f"<ListingTemplate {self.name}>"
