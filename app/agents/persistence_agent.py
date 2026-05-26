# ======================================================================
# File: app/agents/persistence_agent.py
# Registry: app/agents/persistence_agent.py
# Rev: 1.1 (2025-09-10 13:09 JST)
#
# 目的:
# - 収集データの永続化専任エージェント
# - CSV と DB(Flask/SQLAlchemy) を Strategy で切替
# - DB が有効な場合は ProductPriceHistory への append と EMA7/28/180/365 の更新
#
# 使用ソフト/前提:
# - Python 3.10+
# - Flask アプリ(app, db, ProductPriceHistory が import 可なら DB 使用)
# - DB 未設定時は CSV のみ(output/price_history_*.csv, research_*.{json,csv})
#
# 使い方（例）:
# - from app.agents.persistence_agent import PersistenceAgent
# - pa = PersistenceAgent(use_db=True)
# - pa.ingest_buyma_history(rows) # rows: List[BuymaProduct]
# - pa.snapshot_full_results(final_rows) # rows: List[FinalResult]
#
# 運用メモ:
# - Windows タスクスケジューラ等で日次 --price-history を実行し、CSV/DBへ蓄積
# - DB が落ちた場合でも CSV には保存されるフェイルソフト設計
# ======================================================================
from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable

from app.core.timezone import _utcnow

# --- DB (optional) ---
try:
    from app import db
    from app.models import Product, ProductPriceHistory

    FLASK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    logging.warning("[PersistenceAgent] Could not import 'app' module/models. DB features will be disabled.")
    db = None
    Product = None
    ProductPriceHistory = None
    FLASK_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Type Definitions (Orchestratorと共通) ---
class BuymaProduct(TypedDict, total=False):
    captured_at: str
    brand: str
    name: str
    buyma_url: str
    buyma_price: int
    buyma_list_price: int | None
    buyma_discount_pct: float | None
    buyma_sale_until: str | None


class FinalResult(BuymaProduct, total=False):
    supplier_name: str
    supplier_price: float
    supplier_currency: str
    supplier_url: str
    profit: float
    profit_rate: float


# --- Sink Protocol (永続化戦略のインターフェース) ---
@runtime_checkable
class PersistenceSink(Protocol):
    def save_buyma_history(self, rows: list[BuymaProduct]) -> int: ...
    def save_full_results(self, rows: list[FinalResult]) -> int: ...


# --- Sinks (具体的な永続化戦略) ---
class CsvSink:
    """CSV保存戦略クラス"""

    def __init__(self, out_dir: Path = Path("output")) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True)

    def _get_sanitized_brand(self, rows: list[dict[str, Any]], default: str = "UNKNOWN") -> str:
        brand = default
        if rows:
            brand = rows[0].get("brand", default) or default
        return re.sub(r'[\\/*?:"<>|]', "", brand)

    def save_buyma_history(self, rows: list[BuymaProduct]) -> int:
        if not rows:
            return 0
        brand = self._get_sanitized_brand(rows)
        ts = datetime.now().strftime("%Y%m%d")
        history_file = self.out_dir / f"price_history_{brand}_{ts}.csv"
        file_exists = history_file.exists()
        keys = [
            "captured_at",
            "brand",
            "name",
            "buyma_url",
            "buyma_price",
            "buyma_list_price",
            "buyma_discount_pct",
            "buyma_sale_until",
        ]

        try:
            with open(history_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)
            logging.info(f"[CSV] Saved {len(rows)} price records to {history_file}")
            return len(rows)
        except Exception as e:
            logging.error(f"[CSV] Failed to save history: {e}")
            return 0

    def save_full_results(self, rows: list[FinalResult]) -> int:
        if not rows:
            return 0
        brand = self._get_sanitized_brand(rows)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.out_dir / f"research_{brand}_{ts}.json"
        csv_path = self.out_dir / f"research_{brand}_{ts}.csv"

        try:
            json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            keys = sorted({k for item in rows for k in item})
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            logging.info(f"[CSV] Results dumped to {json_path} and {csv_path}")
            return len(rows)
        except Exception as e:
            logging.error(f"[CSV] Failed to save full results: {e}")
            return 0


class DbSink:
    """DB保存戦略クラス"""

    def __init__(self) -> None:
        if not (FLASK_AVAILABLE and db and ProductPriceHistory):
            raise RuntimeError("DB sink requires Flask app/models to be loaded.")
        self._touched_urls: set[str] = set()

    @staticmethod
    def _ema(series: list[float], period: int) -> float | None:
        if not series:
            return None
        alpha = 2 / (period + 1)
        ema_val = series[0]
        for v in series[1:]:
            ema_val = alpha * v + (1 - alpha) * ema_val
        return float(round(ema_val, 2))

    def _recalc_moving_averages_db(self, url: str) -> None:
        try:
            rows = ProductPriceHistory.query.filter_by(url=url).order_by(ProductPriceHistory.captured_at.asc()).all()
            if not rows:
                return
            prices = [float(r.price) for r in rows if r.price is not None]
            if not prices:
                return

            emas = {
                "ema7": self._ema(prices[-7:], 7),
                "ema28": self._ema(prices[-28:], 28),
                "ema180": self._ema(prices[-180:], 180),
                "ema365": self._ema(prices[-365:], 365),
            }
            latest_record = rows[-1]
            for k, v in emas.items():
                if hasattr(latest_record, k) and v is not None:
                    setattr(latest_record, k, v)
            db.session.commit()
        except Exception as e:
            logging.error(f"[DB] EMA recompute failed for {url}: {e}")
            db.session.rollback()

    def save_buyma_history(self, rows: list[BuymaProduct]) -> int:
        if not rows:
            return 0
        saved = 0
        try:
            for r in rows:
                captured_at_dt = datetime.fromisoformat(r["captured_at"]) if r.get("captured_at") else _utcnow()
                rec = ProductPriceHistory(
                    brand=r.get("brand"),
                    name=r.get("name"),
                    url=r.get("buyma_url"),
                    price=r.get("buyma_price"),
                    list_price=r.get("buyma_list_price"),
                    discount_pct=r.get("buyma_discount_pct"),
                    sale_until=r.get("buyma_sale_until"),
                    captured_at=captured_at_dt,
                )
                db.session.add(rec)
                if r.get("buyma_url"):
                    self._touched_urls.add(r["buyma_url"])
            db.session.commit()
            saved = len(rows)
            logging.info(f"[DB] inserted {saved} rows into ProductPriceHistory")
        except Exception as e:
            logging.error(f"[DB] transaction failed: {e}")
            db.session.rollback()

        for url in list(self._touched_urls):
            self._recalc_moving_averages_db(url)
        self._touched_urls.clear()
        return saved

    def save_full_results(self, rows: list[FinalResult]) -> int:
        logging.info("[DB] save_full_results is not implemented (no-op).")
        return 0


# --- PersistenceAgent 本体 ---
class PersistenceAgent:
    """
    - CSV は常に保存（バックアップ/可搬性）
    - DB が利用可能なときのみ DB にも保存し、EMA を更新
    """

    def __init__(self, use_db: bool = False) -> None:
        self.csv_sink = CsvSink()
        self.db_sink: DbSink | None = None
        if use_db:
            try:
                self.db_sink = DbSink()
                logging.info("[PersistenceAgent] Initialized with DB+CSV sinks.")
            except Exception as e:
                logging.warning(f"[PersistenceAgent] DbSink init failed, falling back to CSV only. reason={e}")
        else:
            logging.info("[PersistenceAgent] Initialized with CSV-only sink.")

    def ingest_buyma_history(self, rows: list[BuymaProduct]) -> None:
        """BUYMAの価格履歴（listページ→PDP抽出合成済み）を保存"""
        self.csv_sink.save_buyma_history(rows)
        if self.db_sink:
            self.db_sink.save_buyma_history(rows)

    def snapshot_full_results(self, rows: list[FinalResult]) -> None:
        """フルリサーチ結果（仕入先/利益情報含む）をスナップショット保存"""
        self.csv_sink.save_full_results(rows)
        if self.db_sink:
            self.db_sink.save_full_results(rows)
