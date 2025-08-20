# app/forms.py
from __future__ import annotations
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, URLField, SubmitField
from wtforms.validators import DataRequired, Optional, URL

class ProductForm(FlaskForm):
    name = StringField("商品名", validators=[DataRequired()])
    price = StringField("価格（文字列でOK）", validators=[Optional()])
    source_url = URLField("元ページURL", validators=[Optional(), URL()])
    image_urls = TextAreaField("画像URL（JSON配列 or 改行区切り）", validators=[Optional()])
    note = StringField("メモ", validators=[Optional()])
    submit = SubmitField("保存")

    def parse_image_urls(self):
        raw = (self.image_urls.data or "").strip()
        if not raw:
            return []
        import json
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
        return [line.strip() for line in raw.splitlines() if line.strip()]
