# -- coding: utf-8 --
# ======================================================================
# File: app/web/dashboard.py
# Registry: app/web/dashboard.py
# Rev: 1.4 (2025-09-11 12:21 JST)
#
# 目的:
# - アラートの感度をビジネス要件に合わせて最適化する。
#
# 変更点 (v1.4):
# - 「価格急変」アラートの閾値を、実際の観測データに基づき、
#   従来の 10.0% から 7.0% へと引き下げ、感度を向上。
# ======================================================================
from __future__ import annotations
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence
from flask import Flask, jsonify, render_template, request

# --- DB (optional) ---
try:
    from app import db
    from app.models import ProductPriceHistory
    FLASK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    db = None
    ProductPriceHistory = None
    FLASK_AVAILABLE = False
    logging.info("[Dashboard] DB models not found. Running in CSV-only mode.")


# --- Flask App Initialization (Template Path Fixed) ---
try:
    APP_DIR = Path(__file__).resolve().parent.parent # -> app/
    TEMPLATE_DIR = APP_DIR / 'templates'
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
except NameError:
    app = Flask(__name__)
    logging.warning("Could not determine absolute path for templates. Using default.")

OUTPUT_DIR = Path("output")
DAYS_DEFAULT = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Data Models and Utilities
# =============================================================================
@dataclass
class HistoryRow:
    """A class to represent a single row of price history."""
    captured_at: datetime
    brand: str
    name: str
    url: str
    price: Optional[float]
    list_price: Optional[float]
    discount_pct: Optional[float]
    sale_until: Optional[datetime]

def ema_series(values: Sequence[Optional[float]], period: int) -> List[Optional[float]]:
    """Calculates the Exponential Moving Average for a whole time series."""
    if not values: return []
    alpha = 2 / (period + 1)

    first_valid_idx = next((i for i, v in enumerate(values) if v is not None), -1)
    if first_valid_idx == -1: return [None] * len(values)

    ema_results: List[Optional[float]] = [None] * first_valid_idx
    ema_results.append(values[first_valid_idx])

    for i in range(first_valid_idx + 1, len(values)):
        current_val = values[i]
        prev_ema = ema_results[i-1]

        if current_val is None:
            ema_results.append(prev_ema)
        else:
            new_ema = alpha * current_val + (1 - alpha) * (prev_ema or current_val)
            ema_results.append(round(new_ema, 2))

    return ema_results

def pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
    """Calculates the percentage change."""
    if old is None or new is None or old == 0: return None
    return round((new - old) * 100.0 / old, 2)

# =============================================================================
# Data Source Abstraction
# =============================================================================
class DataSource:
    """Fetches price history data from either DB or CSV files."""
    def __init__(self, use_db: bool = False, input_dir: Path = Path("output")):
        self.use_db = use_db and FLASK_AVAILABLE
        self.input_dir = input_dir
        if self.use_db:
            logging.info("[DataSource] Using DB (ProductPriceHistory).")
        else:
            logging.info("[DataSource] Using CSV files from output.")

    def _parse_csv_history_files(self, brand: str, days: int) -> List[HistoryRow]:
        """Robustly parses CSV files with various naming conventions."""
        cutoff = datetime.now() - timedelta(days=days)
        rows: List[HistoryRow] = []
        candidates = [
            f"price_history_{brand}_*.csv", f"price_history_{brand.replace(' ', '_')}_*.csv",
            f"price_history_{brand.upper()}_*.csv", f"price_history_{brand.title()}_*.csv", "price_history_*_*.csv",
        ]
        seen = set()
        files_to_process: List[Path] = []
        for pat in candidates:
            for fp in self.input_dir.glob(pat):
                if fp not in seen:
                    seen.add(fp); files_to_process.append(fp)

        for fp in sorted(files_to_process):
            try:
                with fp.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for rec in reader:
                        try:
                            dt = datetime.fromisoformat(rec.get("captured_at") or "")
                            if dt < cutoff: continue
                            rec_brand = (rec.get("brand") or "").strip()
                            if rec_brand and rec_brand.lower() != brand.lower(): continue
                            rows.append(HistoryRow(
                                captured_at=dt, brand=rec_brand or brand, name=rec.get("name") or "",
                                url=rec.get("buyma_url") or "",
                                price=float(rec["buyma_price"]) if rec.get("buyma_price") else None,
                                list_price=float(rec["buyma_list_price"]) if rec.get("buyma_list_price") else None,
                                discount_pct=float(rec["buyma_discount_pct"]) if rec.get("buyma_discount_pct") else None,
                                sale_until=datetime.fromisoformat(rec["buyma_sale_until"]) if rec.get("buyma_sale_until") else None,
                            ))
                        except (ValueError, KeyError): continue
            except Exception as e:
                logging.warning(f"Failed to process CSV {fp}: {e}")
        return rows

    def fetch_history(self, brand: str, days: int) -> List[HistoryRow]:
        """Fetches data from DB if available, otherwise falls back to CSV."""
        if self.use_db and ProductPriceHistory:
            try:
                cutoff = datetime.now() - timedelta(days=days)
                query = (
                    ProductPriceHistory.query.filter(ProductPriceHistory.brand.ilike(f"%{brand}%"))
                    .filter(ProductPriceHistory.captured_at >= cutoff)
                    .order_by(ProductPriceHistory.url.asc(), ProductPriceHistory.captured_at.asc())
                )
                results = []
                for r in query.all():
                    results.append(HistoryRow(
                        captured_at=r.captured_at, brand=r.brand or brand, name=r.name or "",
                        url=r.url or "",
                        price=float(r.price) if r.price is not None else None,
                        list_price=float(r.list_price) if r.list_price is not None else None,
                        discount_pct=float(r.discount_pct) if r.discount_pct is not None else None,
                        sale_until=getattr(r, "sale_until", None),
                    ))
                logging.info(f"Fetched {len(results)} rows from DB for brand '{brand}'.")
                return results
            except Exception as e:
                logging.error(f"DB query failed, falling back to CSV. Error: {e}")

        return self._parse_csv_history_files(brand, days)

# =============================================================================
# Data Transformation Logic
# =============================================================================
def build_series_for_web(rows: List[HistoryRow]) -> Dict[str, Any]:
    """Transforms raw history rows into a format suitable for the web frontend."""
    bucket: Dict[str, List[HistoryRow]] = {}
    for r in rows:
        if r.url: bucket.setdefault(r.url, []).append(r)

    series_list = []
    today, yesterday = datetime.now().date(), (datetime.now() - timedelta(days=1)).date()

    for url, items in bucket.items():
        items_sorted = sorted(items, key=lambda x: x.captured_at)
        name = items_sorted[-1].name if items_sorted else url
        labels = [it.captured_at.isoformat() for it in items_sorted]
        prices = [it.price for it in items_sorted]

        ema7 = ema_series(prices, 7)
        ema28 = ema_series(prices, 28)

        # Alert detection logic
        alerts = []
        todays_pts = [it for it in items_sorted if it.captured_at.date() == today]
        yest_pts = [it for it in items_sorted if it.captured_at.date() == yesterday]

        if todays_pts and yest_pts:
            # --- MODIFIED (v1.4): Alert threshold lowered for higher sensitivity ---
            if (price_ch := pct_change(yest_pts[-1].price, todays_pts[-1].price)) is not None and abs(price_ch) >= 7.0:
                alerts.append(f"価格急変: {price_ch:+.1f}%")
            if (disc_ch := pct_change(yest_pts[-1].discount_pct, todays_pts[-1].discount_pct)) is not None and abs(disc_ch) >= 15.0:
                 alerts.append(f"割引率急変: {disc_ch:+.1f}%")

        latest_sale_until = next((it.sale_until for it in reversed(items_sorted) if it.sale_until), None)
        if latest_sale_until:
            if 0 <= (days_left := (latest_sale_until.date() - today).days) <= 3:
                alerts.append(f"セール終了まで {days_left} 日")

        series_list.append({
            "url": url, "name": name, "labels": labels, "prices": prices,
            "ema7": ema7, "ema28": ema28, "alerts": alerts,
        })
    return {"series": series_list}

# =============================================================================
# Flask Routes
# =============================================================================
@app.route("/api/history")
def api_history():
    """API endpoint to provide time series data to the frontend."""
    brand = request.args.get("brand", "").strip() or "GUCCI"
    days = int(request.args.get("days", DAYS_DEFAULT))

    data_source = DataSource(use_db=FLASK_AVAILABLE)
    rows = data_source.fetch_history(brand, days)

    payload = build_series_for_web(rows)
    return jsonify(payload)

@app.route("/brand/<brand>")
def page_brand(brand: str):
    """Renders the dashboard page for a specific brand."""
    days = int(request.args.get("days", DAYS_DEFAULT))
    return render_template("dashboard.html", brand=brand, days=days)

@app.route("/")
def index():
    """Root URL handler."""
    return 'Dashboard is running. Try: <a href="/brand/GUCCI?days=30">/brand/GUCCI?days=30</a>'

# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    # Runs the local development server.
    app.run(host="127.0.0.1", port=5009, debug=True)
