# app/routes.py
from __future__ import annotations

from flask import (
    Blueprint, current_app, request, redirect, url_for,
    flash, jsonify, render_template_string
)
from werkzeug.exceptions import NotFound, BadRequest
from sqlalchemy.exc import SQLAlchemyError

from app import db

# --- models / forms は存在するものだけ import ---
#   どちらか未定義でも落ちないように try にしています
try:
    from app.models import Product
except Exception:
    Product = None  # type: ignore
try:
    from app.forms import ProductForm
except Exception:
    ProductForm = None  # type: ignore

# 省依存：存在すれば使う
try:
    from app.utils.ai_background_remover import remove_background_from_url
except Exception:
    remove_background_from_url = None  # type: ignore

try:
    from app.utils.ai_image_crawler import CrawlerService
except Exception:
    CrawlerService = None  # type: ignore


bp = Blueprint("main", __name__)


# ------------------------------
# HTML テンプレ（最小限・自己完結）
# ------------------------------
_LAYOUT = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>{{ title or "Atelier KYO Manager" }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;margin:24px;}
    header{margin-bottom:24px}
    nav a{margin-right:12px}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #ccc;padding:8px}
    .flash{padding:8px 12px;background:#f0f7ff;border-left:4px solid #58a; margin:12px 0}
    form div{margin:8px 0}
    input[type=text],input[type=number]{width:100%;max-width:420px;padding:8px}
    button{cursor:pointer;padding:8px 12px}
    .muted{color:#777}
  </style>
</head>
<body>
  <header>
    <nav>
      <a href="{{ url_for('main.index') }}">Home</a>
      <a href="{{ url_for('main.product_list') }}">Products</a>
      <a href="{{ url_for('main.add_product') }}">Add Product</a>
      <a href="{{ url_for('main.healthz') }}">Health</a>
    </nav>
  </header>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
      {% for c,m in messages %}
        <div class="flash">{{ m }}</div>
      {% endfor %}
    {% endif %}
  {% endwith %}

  <main>
    {% block body %}{% endblock %}
  </main>
</body>
</html>
"""

_T_INDEX = """
{% extends "_layout" %}
{% block body %}
  <h1>Atelier KYO Manager</h1>
  <p class="muted">最小ルーティングのデモ（WTForms + SQLAlchemy）。</p>
  <ul>
    <li><a href="{{ url_for('main.product_list') }}">/products</a>（一覧）</li>
    <li><a href="{{ url_for('main.add_product') }}">/products/add</a>（追加）</li>
    <li><a href="{{ url_for('main.tools_bg_remove') }}">/tools/bg-remove</a>（背景除去デモ）</li>
    <li><a href="{{ url_for('main.tools_crawler') }}">/tools/crawler</a>（クローラ実行デモ）</li>
  </ul>
{% endblock %}
"""

_T_PRODUCTS = """
{% extends "_layout" %}
{% block body %}
  <h1>Products</h1>
  <p><a href="{{ url_for('main.add_product') }}"><button>+ Add</button></a></p>
  {% if products %}
    <table>
      <thead><tr><th>ID</th><th>Name</th><th>Actions</th></tr></thead>
      <tbody>
      {% for p in products %}
        <tr>
          <td>{{ p.id }}</td>
          <td><a href="{{ url_for('main.product_detail', product_id=p.id) }}">{{ p.name }}</a></td>
          <td>
            <a href="{{ url_for('main.edit_product', product_id=p.id) }}">Edit</a> /
            <form method="post" action="{{ url_for('main.delete_product', product_id=p.id) }}" style="display:inline" onsubmit="return confirm('Delete?')">
              <button>Delete</button>
            </form>
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p class="muted">No products yet.</p>
  {% endif %}
{% endblock %}
"""

_T_PRODUCT_FORM = """
{% extends "_layout" %}
{% block body %}
  <h1>{{ title }}</h1>
  {% if not form %}
    <p class="muted">forms.py の ProductForm が見つかりませんでした。</p>
  {% else %}
  <form method="post">
    {{ form.hidden_tag() }}
    <div>
      {{ form.name.label }}<br>
      {{ form.name(size=40) }}
      {% for e in form.name.errors %}<div class="muted">{{ e }}</div>{% endfor %}
    </div>
    <div>{{ form.submit() }}</div>
  </form>
  {% endif %}
{% endblock %}
"""

_T_PRODUCT_DETAIL = """
{% extends "_layout" %}
{% block body %}
  <h1>Product #{{ product.id }}</h1>
  <p><b>Name:</b> {{ product.name }}</p>
  <p>
    <a href="{{ url_for('main.edit_product', product_id=product.id) }}"><button>Edit</button></a>
    <a href="{{ url_for('main.product_list') }}">Back</a>
  </p>
{% endblock %}
"""

_T_BG_REMOVE = """
{% extends "_layout" %}
{% block body %}
  <h1>Background Removal (Demo)</h1>
  <form method="post">
    <div>
      <label>Image URL</label><br>
      <input type="text" name="image_url" placeholder="https://..." required>
    </div>
    <div><button>Remove Background</button></div>
  </form>
  {% if result %}
    <h2>Result</h2>
    <pre>{{ result|tojson(indent=2) }}</pre>
  {% endif %}
  {% if not remover %}
    <p class="muted">※ ai_background_remover.py が見つからなかったため、ダミー応答になります。</p>
  {% endif %}
{% endblock %}
"""

_T_CRAWLER = """
{% extends "_layout" %}
{% block body %}
  <h1>AI Image Crawler (Demo)</h1>
  <form method="post">
    <div>
      <label>Target URL</label><br>
      <input type="text" name="url" placeholder="https://www.farfetch.com/..." required>
    </div>
    <div>
      <label>Master Image URL</label><br>
      <input type="text" name="master_image_url" placeholder="https://cdn-images....jpg">
    </div>
    <div><button>Run</button></div>
  </form>
  {% if result %}
    <h2>Result</h2>
    <pre>{{ result|tojson(indent=2, ensure_ascii=False) }}</pre>
  {% endif %}
  {% if not crawler %}
    <p class="muted">※ ai_image_crawler.py が見つからなかったため、ダミー応答になります。</p>
  {% endif %}
{% endblock %}
"""


@bp.app_template_global("_layout")
def _inject_layout():  # noqa: D401
    """render_template_string 用：ベースレイアウト注入"""
    return _LAYOUT


# ------------------------------
# 基本ページ
# ------------------------------
@bp.route("/")
def index():
    return render_template_string(_T_INDEX, title="Home")


@bp.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


# ------------------------------
# Product CRUD（存在すれば有効）
# ------------------------------
@bp.route("/products")
def product_list():
    if Product is None:
        return render_template_string(
            _T_PRODUCTS, title="Products", products=[]
        )
    products = Product.query.order_by(Product.id.desc()).all()
    return render_template_string(_T_PRODUCTS, title="Products", products=products)


@bp.route("/products/<int:product_id>")
def product_detail(product_id: int):
    if Product is None:
        raise NotFound("Product model not found.")
    product = Product.query.get_or_404(product_id)
    return render_template_string(_T_PRODUCT_DETAIL, title=f"Product #{product.id}", product=product)


@bp.route("/products/add", methods=["GET", "POST"])
def add_product():
    form = ProductForm() if ProductForm else None
    if request.method == "POST":
        if not (Product and form):
            raise BadRequest("Product model or ProductForm not available.")
        if form.validate_on_submit():
            try:
                p = Product(name=form.name.data)
                db.session.add(p)
                db.session.commit()
                flash("Product created.", "success")
                return redirect(url_for("main.product_list"))
            except SQLAlchemyError as e:
                current_app.logger.exception("DB error on create: %s", e)
                db.session.rollback()
                flash("DB error.", "error")
        else:
            flash("Validation error.", "error")
    return render_template_string(_T_PRODUCT_FORM, title="Add Product", form=form)


@bp.route("/products/<int:product_id>/edit", methods=["GET", "POST"])
def edit_product(product_id: int):
    if Product is None:
        raise BadRequest("Product model not available.")
    p = Product.query.get_or_404(product_id)
    form = ProductForm(obj=p) if ProductForm else None
    if request.method == "POST":
        if not form:
            raise BadRequest("ProductForm not available.")
        if form.validate_on_submit():
            try:
                p.name = form.name.data
                db.session.commit()
                flash("Product updated.", "success")
                return redirect(url_for("main.product_detail", product_id=p.id))
            except SQLAlchemyError as e:
                current_app.logger.exception("DB error on update: %s", e)
                db.session.rollback()
                flash("DB error.", "error")
        else:
            flash("Validation error.", "error")
    return render_template_string(_T_PRODUCT_FORM, title=f"Edit Product #{p.id}", form=form)


@bp.route("/products/<int:product_id>/delete", methods=["POST"])
def delete_product(product_id: int):
    if Product is None:
        raise BadRequest("Product model not available.")
    p = Product.query.get_or_404(product_id)
    try:
        db.session.delete(p)
        db.session.commit()
        flash("Product deleted.", "success")
    except SQLAlchemyError as e:
        current_app.logger.exception("DB error on delete: %s", e)
        db.session.rollback()
        flash("DB error.", "error")
    return redirect(url_for("main.product_list"))


# ------------------------------
# Tools: 背景除去（デモ）
# ------------------------------
@bp.route("/tools/bg-remove", methods=["GET", "POST"])
def tools_bg_remove():
    result = None
    if request.method == "POST":
        image_url = request.form.get("image_url", "").strip()
        if not image_url:
            flash("image_url is required.", "error")
        else:
            if remove_background_from_url:
                try:
                    result = remove_background_from_url(image_url)
                except Exception as e:
                    current_app.logger.exception("Background remover error: %s", e)
                    flash("Background remover error.", "error")
            else:
                # ダミー応答
                result = {"status": "skipped", "reason": "ai_background_remover.py not available"}
    return render_template_string(
        _T_BG_REMOVE, title="Background Removal", result=result, remover=bool(remove_background_from_url)
    )


# ------------------------------
# Tools: クローラ実行（デモ）
# ------------------------------
@bp.route("/tools/crawler", methods=["GET", "POST"])
def tools_crawler():
    result = None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        master_image_url = (request.form.get("master_image_url") or "").strip()
        if not url:
            flash("url is required.", "error")
        else:
            if CrawlerService:
                try:
                    crawler = CrawlerService()
                    try:
                        # master_image_url が空でも動く実装に寄り添う
                        result = crawler.get_product_info(url, master_image_url)  # type: ignore[arg-type]
                    finally:
                        crawler.quit()
                except Exception as e:
                    current_app.logger.exception("Crawler error: %s", e)
                    result = {"error": str(e)}
            else:
                result = {"status": "skipped", "reason": "ai_image_crawler.py not available"}
    return render_template_string(
        _T_CRAWLER, title="Crawler", result=result, crawler=bool(CrawlerService)
    )
