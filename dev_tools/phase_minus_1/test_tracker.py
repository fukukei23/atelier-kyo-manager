"""tracker.py の純粋ロジック検証（ネットワーク不要）."""

from __future__ import annotations

from datetime import datetime, timedelta

from tracker import ListingTracker


def _t(minute: int) -> datetime:
    """基準時刻 + minute 分 のヘルパ."""
    return datetime(2026, 7, 3, 9, 0) + timedelta(minutes=minute)


def test_observe_candidate_registers_new_and_refreshes() -> None:
    tr = ListingTracker()
    assert tr.observe_candidate(_t(0), "m1", "SEL55F18Z", "レンズ美品", 42000) is True
    # 同一IDの再観測は新規ではない（last_seen更新のみ）
    assert tr.observe_candidate(_t(5), "m1", "SEL55F18Z", "レンズ美品", 42000) is False
    assert tr.items["m1"].last_seen == _t(5)
    assert tr.active_ids() == ["m1"]


def test_mark_sold_records_residence_time() -> None:
    tr = ListingTracker()
    tr.observe_candidate(_t(0), "m1", "kw", "t", 42000)
    rec = tr.mark_sold(_t(25), "m1")
    assert rec is not None
    assert rec["residence_min"] == 25.0
    # 売却後は on_sale ではないので再チェック対象から外れる
    assert tr.active_ids() == []


def test_mark_sold_ignores_unknown_or_already_sold() -> None:
    tr = ListingTracker()
    tr.observe_candidate(_t(0), "m1", "kw", "t", 100)
    assert tr.mark_sold(_t(1), "unknown") is None
    tr.mark_sold(_t(10), "m1")
    # 二重SOLDは記録しない
    assert tr.mark_sold(_t(20), "m1") is None
    assert len(tr.sold_records) == 1


def test_summary_reports_median_and_fast_sales() -> None:
    tr = ListingTracker()
    for i, resi in enumerate([3, 8, 40]):
        tr.observe_candidate(_t(0), f"m{i}", "kw", "t", 1000)
        tr.mark_sold(_t(resi), f"m{i}")
    s = tr.summary()
    assert s["watched_total"] == 3
    assert s["sold_total"] == 3
    assert s["median_residence_min"] == 8
    assert s["sold_within_10min"] == 2
