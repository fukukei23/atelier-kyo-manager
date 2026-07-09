"""Phase -1 検証B: 安値出品の滞留時間トラッカー（純粋ロジック・使い捨て）.

why: 「相場より安い出品が、何分で売れるか（＝人間が買う前に消えるか）」を測る。
新着リストから消えた=売れた、は交絡（新出品に押し出されるだけ）なので、
個別商品ページの SOLD 判定で確定した売却だけを滞留時間として記録する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class WatchedItem:
    """監視中の安値候補1件の状態."""

    item_id: str
    keyword: str
    title: str
    price: int
    first_seen: datetime
    last_seen: datetime
    status: str = "on_sale"  # on_sale | sold | vanished
    sold_at: datetime | None = None


@dataclass
class ListingTracker:
    """安値候補の発見・SOLD確定・滞留時間算出を担う純粋ロジック."""

    items: dict[str, WatchedItem] = field(default_factory=dict)
    sold_records: list[dict] = field(default_factory=list)

    def observe_candidate(
        self,
        now: datetime,
        item_id: str,
        keyword: str,
        title: str,
        price: int,
    ) -> bool:
        """安値候補を登録/更新する. 新規登録なら True を返す."""
        existing = self.items.get(item_id)
        if existing is None:
            self.items[item_id] = WatchedItem(
                item_id=item_id,
                keyword=keyword,
                title=title,
                price=price,
                first_seen=now,
                last_seen=now,
            )
            return True
        existing.last_seen = now
        existing.price = price
        return False

    def mark_sold(self, now: datetime, item_id: str) -> dict | None:
        """個別ページで SOLD 確定した候補を売却済みにし, 滞留記録を返す."""
        item = self.items.get(item_id)
        if item is None or item.status != "on_sale":
            return None
        item.status = "sold"
        item.sold_at = now
        record = {
            "item_id": item.item_id,
            "keyword": item.keyword,
            "title": item.title,
            "price": item.price,
            "first_seen": item.first_seen.isoformat(timespec="seconds"),
            "sold_at": now.isoformat(timespec="seconds"),
            "residence_min": round((now - item.first_seen).total_seconds() / 60.0, 1),
        }
        self.sold_records.append(record)
        return record

    def active_ids(self) -> list[str]:
        """まだ on_sale の候補ID一覧（次回SOLD再チェック対象）."""
        return [i.item_id for i in self.items.values() if i.status == "on_sale"]

    def summary(self) -> dict:
        """検証B判定用のサマリー指標を返す."""
        sold = self.sold_records
        residences = [r["residence_min"] for r in sold]
        residences.sort()
        median = residences[len(residences) // 2] if residences else None
        sold_within_10min = sum(1 for r in residences if r <= 10)
        return {
            "watched_total": len(self.items),
            "sold_total": len(sold),
            "median_residence_min": median,
            "sold_within_10min": sold_within_10min,
        }
