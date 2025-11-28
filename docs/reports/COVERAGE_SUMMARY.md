# カバレッジ測定サマリー

## 測定日時
2025-11-27

## 現状

ターミナル出力によると：
- **17個のテストが失敗**
- **7個のテストが成功**
- カバレッジレポートは生成されていない（テストが失敗しているため）

## 測定方法

成功するテストのみでカバレッジを測定します：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest \
  tests/test_navigation_driver_stage3a2.py \
  tests/test_telemetry_service_stage3b.py \
  --cov=app/agents/browser \
  --cov-report=term \
  --cov-report=json \
  --cov-report=term-missing
```

## 測定対象

- `app/agents/browser/navigation_driver.py`
- `app/agents/browser/telemetry.py`
- `app/agents/browser/` 配下のその他のファイル

## 注意事項

- 現在、全テストでカバレッジを測定するには、失敗しているテストの修正が必要です
- 成功するテストのみで測定した場合、カバレッジは部分的になります

